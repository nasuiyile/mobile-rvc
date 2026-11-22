"""Export RVC synthesizer *templates* (graph structure only, no weights).

For each standard RVC architecture this writes::

    <out>/synth/<key>/manifest.json   tensor layout of weights.bin + architecture signature
    <out>/synth/<key>/enc.onnx        enc_p   : phone[1,T,C], pitch[1,T]  -> m_p, logs_p [1,192,T]
    <out>/synth/<key>/flow.onnx       flow^-1 : z_p[1,192,T], g[1,256,1]  -> z[1,192,T]
    <out>/synth/<key>/dec.onnx        NSF-HiFiGAN decoder: z, har (sine source), g -> audio

All graphs reference ``weights.bin`` (fp16, raw RVC tensors).  The Android app builds
that file from a user's ``.pth`` by copying tensor bytes at the manifest offsets, so no
desktop conversion step is needed for standard RVC models.

Usage:
    python -m rvc_mobile.export_synth --out build/assets
    python -m rvc_mobile.export_synth --out build/assets --pth some_custom_arch.pth
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from .onnx_tools import DataLayout, externalize, save_graph, torch_export, write_json
from .synth_modules import DecGraph, EncGraph, FlowGraph, MobileSynth

# (upsample_rates, upsample_kernel_sizes) of RVC's configs/{v1,v2}/*.json
DEC_CONFIGS = {
    "32k_v1": ([10, 4, 2, 2, 2], [16, 16, 4, 4, 4], 32000),
    "40k": ([10, 10, 2, 2], [16, 16, 4, 4], 40000),
    "48k_v1": ([10, 6, 2, 2, 2], [16, 16, 4, 4, 4], 48000),
    "32k_v2": ([10, 8, 2, 2], [20, 16, 4, 4], 32000),
    "48k_v2": ([12, 10, 2, 2], [24, 20, 4, 4], 48000),
}


def make_cfg(upsample_rates, upsample_kernel_sizes, sr):
    return [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates, 512, upsample_kernel_sizes, 109, 256, sr]


def signature(cfg, phone_dim):
    return {
        "phone_dim": phone_dim,
        "inter_channels": cfg[2], "hidden_channels": cfg[3], "filter_channels": cfg[4],
        "n_heads": cfg[5], "n_layers": cfg[6], "kernel_size": cfg[7],
        "resblock": str(cfg[9]), "resblock_kernel_sizes": list(cfg[10]),
        "resblock_dilation_sizes": [list(d) for d in cfg[11]],
        "upsample_rates": list(cfg[12]), "upsample_initial_channel": cfg[13],
        "upsample_kernel_sizes": list(cfg[14]), "gin_channels": cfg[16],
    }


def sig_key(sig):
    h = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:8]
    v = "v1" if sig["phone_dim"] == 256 else "v2"
    ups = "x".join(map(str, sig["upsample_rates"]))
    return f"{v}_u{ups}_{h}"


def export_template(cfg, phone_dim, out_root, state_dict=None, verbose=True):
    torch.manual_seed(0)
    synth = MobileSynth(cfg, phone_dim).eval()
    if state_dict is not None:
        sd = {k: v.float() for k, v in state_dict.items()
              if k.startswith(("enc_p.", "flow.", "dec.")) and not k.startswith("dec.m_source")}
        missing, unexpected = synth.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"checkpoint lacks tensors: {missing[:8]}")
    # Exact fp16 values so the externalized fp16 data reproduces the exported graph.
    with torch.no_grad():
        for p in synth.parameters():
            p.copy_(p.half().float())
    params = {k: v.detach().numpy() for k, v in synth.state_dict().items()}

    sig = signature(cfg, phone_dim)
    key = sig_key(sig)
    out_dir = os.path.join(out_root, "synth", key)
    os.makedirs(out_dir, exist_ok=True)
    layout = DataLayout("weights.bin")
    # deterministic layout: enc_p, flow, dec in state-dict order
    for k in params:
        layout.add(k, params[k], "f16")

    T = 64
    inter = cfg[2]
    phone = torch.randn(1, T, phone_dim)
    pitch = torch.randint(1, 255, (1, T), dtype=torch.long)
    z = torch.randn(1, inter, T)
    g = torch.randn(1, cfg[16], 1)
    har = torch.randn(1, 1, T * synth.upp)

    graphs = {}
    m = torch_export(EncGraph(synth), (phone, pitch), ["phone", "pitch"], ["m_p", "logs_p"],
                     {"phone": {1: "T"}, "pitch": {1: "T"}, "m_p": {2: "T"}, "logs_p": {2: "T"}})
    used = externalize(m, params, layout)
    save_graph(m, os.path.join(out_dir, "enc.onnx"))
    graphs["enc"] = ("enc.onnx", used)

    m = torch_export(FlowGraph(synth), (z, g), ["z_p", "g"], ["z"], {"z_p": {2: "T"}, "z": {2: "T"}})
    used = externalize(m, params, layout)
    save_graph(m, os.path.join(out_dir, "flow.onnx"))
    graphs["flow"] = ("flow.onnx", used)

    m = torch_export(DecGraph(synth), (z, har, g), ["z", "har", "g"], ["audio"],
                     {"z": {2: "T"}, "har": {2: "S"}, "audio": {2: "S"}})
    used = externalize(m, params, layout)
    save_graph(m, os.path.join(out_dir, "dec.onnx"))
    graphs["dec"] = ("dec.onnx", used)

    all_used = set(u for _, us in graphs.values() for u in us)
    unused = [k for k in params if k not in all_used]
    if unused:
        raise RuntimeError(f"parameters not referenced by any graph: {unused[:8]}")

    manifest = {
        "format": 1,
        "key": key,
        "signature": sig,
        "upp": synth.upp,
        "data_file": "weights.bin",
        "data_size": layout.size,
        "graphs": {"enc": "enc.onnx", "flow": "flow.onnx", "dec": "dec.onnx"},
        "tensors": layout.manifest(),
    }
    write_json(manifest, os.path.join(out_dir, "manifest.json"))
    if verbose:
        print(f"[synth] {key}: {len(layout.order)} tensors, {layout.size / 1e6:.1f} MB weights")
    return out_dir, layout, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pth", help="export a template matching this (possibly non-standard) checkpoint")
    args = ap.parse_args()
    if args.pth:
        cpt = torch.load(args.pth, map_location="cpu", weights_only=True)
        cfg = list(cpt["config"])
        phone_dim = cpt["weight"]["enc_p.emb_phone.weight"].shape[1]
        export_template(cfg, phone_dim, args.out, cpt["weight"])
        return
    for name, (ups, ks, sr) in DEC_CONFIGS.items():
        for phone_dim in (256, 768):
            export_template(make_cfg(ups, ks, sr), phone_dim, args.out)


if __name__ == "__main__":
    main()
