"""Export FCPE (torchfcpe bundled model ``fcpe_c_v001``) for the app.

The STFT is done natively in the app (hann 1024, hop 160, reflect padding exactly as
torchfcpe's MelModule); the graph takes the magnitude spectrogram::

    mag[1,T,513] -> mel (librosa slaney basis, 128 bins, 0-8 kHz) -> log(clamp 1e-5)
                 -> CFNaiveMelPE -> latent[1,T,360] (sigmoid activations)

The cent table needed for "local_argmax" decoding is written to manifest.json.
Writes ``<out>/fcpe/{fcpe.onnx, fcpe.bin, manifest.json}``.
"""

import argparse
import os
import pathlib

import numpy as np
import torch
from torch import nn

from .onnx_tools import DataLayout, externalize, save_graph, torch_export, write_json


def load_fcpe():
    import torchfcpe
    from torchfcpe.mel_fn_librosa import mel as librosa_mel_fn
    from torchfcpe.models import CFNaiveMelPE

    ckpt = pathlib.Path(torchfcpe.__file__).parent / "assets" / "fcpe_c_v001.pt"
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    cfg = ck["config_dict"]
    mc = cfg["model"]
    model = CFNaiveMelPE(input_channels=cfg["mel"]["num_mels"], out_dims=mc["out_dims"],
                         hidden_dims=mc["hidden_dims"], n_layers=mc["n_layers"], n_heads=mc["n_heads"],
                         f0_max=mc["f0_max"], f0_min=mc["f0_min"], use_fa_norm=mc["use_fa_norm"],
                         conv_only=mc["conv_only"], conv_dropout=0.0, atten_dropout=0.0)
    model.load_state_dict(ck["model"])
    model.eval()
    m = cfg["mel"]
    basis = librosa_mel_fn(sr=m["sr"], n_fft=m["n_fft"], n_mels=m["num_mels"], fmin=m["fmin"], fmax=m["fmax"])
    return model, np.asarray(basis, dtype=np.float32), cfg


class FcpeGraph(nn.Module):
    def __init__(self, model, basis):
        super().__init__()
        self.model = model
        self.register_buffer("mel_basis_t", torch.from_numpy(basis.T.copy()))  # [513,128]

    def forward(self, mag):
        mel = torch.matmul(mag, self.mel_basis_t)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return self.model(mel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, basis, cfg = load_fcpe()
    graph = FcpeGraph(model, basis).eval()
    params = {k: v.detach().numpy() for k, v in graph.state_dict().items()
              if k.startswith("model.") and not k.endswith(("cent_table", "gaussian_blurred_cent_mask"))}
    out_dir = os.path.join(args.out, "fcpe")
    os.makedirs(out_dir, exist_ok=True)
    layout = DataLayout("fcpe.bin")
    mag = torch.rand(1, 40, 513)
    m = torch_export(graph, (mag,), ["mag"], ["latent"], {"mag": {1: "T"}, "latent": {1: "T"}})
    used = externalize(m, params, layout, store_dtype="f32")
    save_graph(m, os.path.join(out_dir, "fcpe.onnx"))
    layout.write(os.path.join(out_dir, "fcpe.bin"))
    mel = cfg["mel"]
    write_json({
        "format": 1, "graph": "fcpe.onnx", "data_file": "fcpe.bin",
        "sample_rate": mel["sr"], "n_fft": mel["n_fft"], "win_size": mel["win_size"], "hop": mel["hop_size"],
        "f0_min": cfg["model"]["f0_min"], "f0_max": cfg["model"]["f0_max"],
        "cent_table": [float(x) for x in model.cent_table.numpy()],
    }, os.path.join(out_dir, "manifest.json"))
    print(f"[fcpe] {len(used)} tensors, {layout.size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
