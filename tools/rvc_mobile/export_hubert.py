"""Export RVC's HuBERT/ContentVec feature extractor for the app.

Input is RVC's Transformers-format checkpoint directory (``assets/hubert_base`` from
huggingface.co/lj1995/VoiceConversionWebUI).  Writes::

    <out>/hubert/contentvec.bin        fp16 parameters (shared by every graph below)
    <out>/hubert/hubert_v2.onnx        audio[1,N] (16 kHz) -> feats[1,F,768]  (layer 12)
    <out>/hubert/hubert_v1.onnx        audio[1,N] -> feats[1,F,256] (layer 9 + final_proj)
    <out>/hubert/manifest.json

The v1 graph stops after layer 9, so v1 voice models do not pay for layers 10-12.
"""

import argparse
import os

import numpy as np
import torch
from torch import nn

from .onnx_tools import DataLayout, externalize, save_graph, torch_export, write_json


def load_hf_hubert(path):
    from transformers import HubertModel

    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    model = HubertModelWithFinalProj.from_pretrained(path, torch_dtype=torch.float32,
                                                     attn_implementation="eager", local_files_only=True)
    return model.eval()


class HubertGraph(nn.Module):
    def __init__(self, hf, n_layers, final_proj):
        super().__init__()
        self.hf = hf
        self.n_layers = n_layers
        self.use_proj = final_proj

    def forward(self, audio):
        x = self.hf.feature_extractor(audio).transpose(1, 2)
        x = self.hf.feature_projection(x)
        enc = self.hf.encoder
        x = enc.layer_norm(x + enc.pos_conv_embed(x))
        for layer in enc.layers[: self.n_layers]:
            x = layer(x)[0]
        if self.use_proj:
            x = self.hf.final_proj(x)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubert", required=True, help="RVC assets/hubert_base directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hf = load_hf_hubert(args.hubert)
    with torch.no_grad():
        for p in hf.parameters():
            p.copy_(p.half().float())  # checkpoint is fp16 already; make it explicit
    params = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    out_dir = os.path.join(args.out, "hubert")
    os.makedirs(out_dir, exist_ok=True)
    layout = DataLayout("contentvec.bin")

    audio = torch.randn(1, 16000) * 0.1
    graphs = {}
    for name, n_layers, proj in (("hubert_v2", 12, False), ("hubert_v1", 9, True)):
        g = HubertGraph(hf, n_layers, proj).eval()
        # parameter names inside the wrapper are prefixed by "hf."
        prm = {"hf." + k: v for k, v in params.items()}
        m = torch_export(g, (audio,), ["audio"], ["feats"], {"audio": {1: "N"}, "feats": {1: "F"}})
        used = externalize(m, prm, layout)  # both graphs share one fp16 parameter file
        save_graph(m, os.path.join(out_dir, name + ".onnx"))
        graphs[name] = {"fp32": name + ".onnx",
                        "dim": 256 if proj else 768, "layers": n_layers, "tensors": len(used)}
        print(f"[hubert] {name}: {len(used)} tensors")
    layout.write(os.path.join(out_dir, "contentvec.bin"))
    write_json({"format": 1, "data_file": "contentvec.bin", "data_size": layout.size, "sample_rate": 16000,
                "hop": 320, "graphs": graphs}, os.path.join(out_dir, "manifest.json"))
    print(f"[hubert] contentvec.bin {layout.size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
