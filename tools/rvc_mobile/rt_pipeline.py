"""Reference (numpy + ONNX Runtime) implementation of the app's realtime engine.

This file is the executable specification of ``android/app/src/main/cpp/engine``:
the C++ engine is a line-by-line port and is tested against it.  It reproduces RVC's
realtime path (``realtime_gui.py`` audio_callback + ``infer/rtrvc.py``, FCPE f0) using
the exported ONNX graphs, plus two optional mobile improvements:

* ``phase_lock``: NSF sine phase continuous across blocks (RVC restarts it at 0).
* sessions with fixed shapes (accelerators) - shapes here are always identical per block.

CLI (desktop testing):
    python -m rvc_mobile.rt_pipeline import  --assets build/assets --pth m.pth [--index m.index] --out models/m
    python -m rvc_mobile.rt_pipeline process --assets build/assets --model models/m --in in.wav --out out.wav
"""

import argparse
import json
import math
import os
import shutil
import struct
from typing import Optional

import numpy as np

# ============================================================================ DSP
class SincResampler:
    """torchaudio.transforms.Resample (sinc_interp_hann, width 6, rolloff 0.99)."""

    def __init__(self, orig: int, new: int, lowpass_filter_width=6, rolloff=0.99):
        g = math.gcd(orig, new)
        self.o, self.n = orig // g, new // g
        base = min(self.o, self.n) * rolloff
        self.width = math.ceil(lowpass_filter_width * self.o / base)
        idx = np.arange(-self.width, self.width + self.o, dtype=np.float64)[None, :] / self.o
        t = np.arange(0, -self.n, -1, dtype=np.float64)[:, None] / self.n + idx
        t *= base
        t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
        window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
        t *= math.pi
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(t == 0, 1.0, np.sin(t) / t)
        self.kernel = (k * window * base / self.o).astype(np.float32)  # [n, K]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.o == self.n:
            return x.astype(np.float32)
        L = len(x)
        xp = np.pad(x.astype(np.float32), (self.width, self.width + self.o))
        K = self.kernel.shape[1]
        nout = (len(xp) - K) // self.o + 1
        frames = np.lib.stride_tricks.as_strided(xp, (nout, K), (xp.strides[0] * self.o, xp.strides[0]))
        y = (frames @ self.kernel.T).reshape(-1)
        return y[: int(math.ceil(self.n * L / self.o))].astype(np.float32)


def rms_frames(y: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """librosa.feature.rms(y, frame_length, hop_length) with center=True, constant pad."""
    yp = np.pad(y.astype(np.float32), (frame_length // 2, frame_length // 2))
    n = 1 + (len(yp) - frame_length) // hop
    fr = np.lib.stride_tricks.as_strided(yp, (n, frame_length), (yp.strides[0] * hop, yp.strides[0]))
    return np.sqrt(np.mean(fr.astype(np.float32) ** 2, axis=1, dtype=np.float32))


def interp_linear_align(x: np.ndarray, size: int) -> np.ndarray:
    """F.interpolate(mode='linear', align_corners=True) for a 1-D signal."""
    if len(x) == 1:
        return np.full(size, x[0], np.float32)
    pos = np.arange(size, dtype=np.float64) * (len(x) - 1) / (size - 1)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


def interp_linear(x: np.ndarray, size: int) -> np.ndarray:
    """F.interpolate(mode='linear', align_corners=False) along the last axis."""
    n = x.shape[-1]
    scale = n / size
    src = (np.arange(size, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.maximum(src, 0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0).astype(np.float32)
    return x[..., i0] * (1 - w) + x[..., i1] * w


def stft_mag_fcpe(y: np.ndarray, n_fft=1024, win=1024, hop=160) -> np.ndarray:
    """Magnitude spectrogram exactly as torchfcpe MelModule (center=False, reflect pad)."""
    pad_l = (win - hop) // 2
    pad_r = max((win - hop + 1) // 2, win - len(y) - pad_l)
    yp = np.pad(y, (pad_l, pad_r), mode="reflect" if pad_r < len(y) else "constant")
    w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)).astype(np.float32)
    n = 1 + (len(yp) - n_fft) // hop
    fr = np.lib.stride_tricks.as_strided(yp, (n, n_fft), (yp.strides[0] * hop, yp.strides[0])) * w
    sp = np.fft.rfft(fr, axis=1)
    m = np.sqrt(sp.real ** 2 + sp.imag ** 2 + 1e-9).astype(np.float32)
    nf = len(y) // hop + 1
    if nf > n:
        m = np.concatenate([m, m[-1:]], 0)
    return m[:nf]


# ============================================================================ index
def read_faiss_ivfflat(path: str):
    """Minimal reader for faiss IndexIVFFlat files (RVC's added_*.index)."""
    with open(path, "rb") as f:
        buf = f.read()
    p = [0]

    def rd(fmt):
        v = struct.unpack_from("<" + fmt, buf, p[0])
        p[0] += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def header():
        d = rd("i"); ntotal = rd("q"); rd("q"); rd("q"); rd("B"); metric = rd("i")
        if metric > 1:
            rd("f")
        return d, ntotal, metric

    if buf[0:4] != b"IwFl":
        raise ValueError(f"unsupported faiss index type {buf[0:4]!r} (need IVF*,Flat)")
    p[0] = 4
    d, ntotal, metric = header()
    if metric != 1:
        raise ValueError("only L2 metric is supported")
    nlist = rd("Q"); nprobe = rd("Q")
    qh = buf[p[0]:p[0] + 4]; p[0] += 4
    if qh not in (b"IxF2", b"IxFl"):
        raise ValueError(f"unsupported quantizer {qh!r}")
    qd, qn, _ = header()
    n = rd("Q")
    cent = np.frombuffer(buf, np.float32, n, p[0]).reshape(qn, qd); p[0] += 4 * n
    dm_type = rd("b"); n = rd("Q"); p[0] += 8 * n
    if dm_type == 2:
        n = rd("Q"); p[0] += 16 * n
    if buf[p[0]:p[0] + 4] != b"ilar":
        raise ValueError("unsupported inverted list storage")
    p[0] += 4
    nl = rd("Q"); code_size = rd("Q")
    lt = buf[p[0]:p[0] + 4]; p[0] += 4
    sizes = np.zeros(nl, np.int64)
    if lt == b"full":
        n = rd("Q"); sizes[:] = np.frombuffer(buf, np.uint64, n, p[0]); p[0] += 8 * n
    elif lt == b"sprs":
        n = rd("Q"); v = np.frombuffer(buf, np.uint64, n, p[0]); p[0] += 8 * n
        sizes[v[0::2].astype(np.int64)] = v[1::2].astype(np.int64)
    else:
        raise ValueError("bad list type")
    vecs, offs = [], [0]
    for i in range(nl):
        k = int(sizes[i])
        if k:
            vecs.append(np.frombuffer(buf, np.float32, k * d, p[0]).reshape(k, d)); p[0] += k * code_size
            p[0] += 8 * k  # ids (unused: RVC reads big_npy[ids], i.e. the vectors themselves)
        offs.append(offs[-1] + k)
    return {"dim": d, "nlist": nl, "nprobe": int(nprobe), "centroids": cent.copy(),
            "offsets": np.array(offs, np.int64), "vectors": np.concatenate(vecs, 0) if vecs else np.zeros((0, d), np.float32)}


def write_index_bin(idx, path, fp16=True):
    with open(path, "wb") as f:
        f.write(b"RVCIDX1\0")
        f.write(struct.pack("<IIIIQ", idx["dim"], idx["nlist"], idx["nprobe"], 1 if fp16 else 0, len(idx["vectors"])))
        f.write(idx["centroids"].astype(np.float32).tobytes())
        f.write(idx["offsets"].astype(np.uint64).tobytes())
        f.write(idx["vectors"].astype(np.float16 if fp16 else np.float32).tobytes())


def read_index_bin(path):
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:8] == b"RVCIDX1\0"
    d, nl, nprobe, f16, nt = struct.unpack_from("<IIIIQ", buf, 8)
    p = 8 + 24
    cent = np.frombuffer(buf, np.float32, nl * d, p).reshape(nl, d); p += nl * d * 4
    offs = np.frombuffer(buf, np.uint64, nl + 1, p).astype(np.int64); p += (nl + 1) * 8
    vec = np.frombuffer(buf, np.float16 if f16 else np.float32, nt * d, p).reshape(nt, d).astype(np.float32)
    return {"dim": d, "nlist": nl, "nprobe": nprobe, "centroids": cent, "offsets": offs, "vectors": vec}


def ivf_search_blend(idx, q: np.ndarray, k=8, nprobe=None):
    """faiss IVFFlat search (L2) + RVC's inverse-square-distance weighted average."""
    nprobe = nprobe or idx["nprobe"]
    cent, offs, vec = idx["centroids"], idx["offsets"], idx["vectors"]
    out = np.empty_like(q)
    cn = (cent * cent).sum(1)
    for i, x in enumerate(q):
        dc = cn - 2 * cent @ x
        lists = np.argpartition(dc, min(nprobe, len(dc) - 1))[:nprobe]
        cand = np.concatenate([vec[offs[l]:offs[l + 1]] for l in lists])
        dist = ((cand - x) ** 2).sum(1)
        kk = min(k, len(dist))
        sel = np.argpartition(dist, kk - 1)[:kk]
        w = np.square(1.0 / np.maximum(dist[sel], 1e-12))
        w /= w.sum()
        out[i] = (cand[sel] * w[:, None]).sum(0)
    return out


# ============================================================================ model import
def read_pth(path):
    import torch  # desktop only; the app has its own zip/pickle reader
    return torch.load(path, map_location="cpu", weights_only=True)


def match_template(assets, cpt):
    cfg = cpt["config"]
    sd = cpt["weight"]
    phone_dim = int(sd["enc_p.emb_phone.weight"].shape[1])
    from .export_synth import signature
    sig = signature(cfg, phone_dim)
    root = os.path.join(assets, "synth")
    for key in sorted(os.listdir(root)):
        man = json.load(open(os.path.join(root, key, "manifest.json")))
        if man["signature"] == sig:
            return os.path.join(root, key), man
    raise ValueError("no template for this architecture; run export_synth --pth")


def import_model(assets, pth, index, out_dir):
    cpt = read_pth(pth)
    if int(cpt.get("f0", 1)) != 1:
        raise ValueError("only f0 (pitch) models are supported")
    sd = cpt["weight"]
    tpl, man = match_template(assets, cpt)
    os.makedirs(out_dir, exist_ok=True)
    buf = bytearray(man["data_size"])
    for t in man["tensors"]:
        a = sd[t["name"]].half().contiguous().numpy()
        if list(a.shape) != t["shape"]:
            raise ValueError(f"shape mismatch {t['name']}")
        buf[t["offset"]:t["offset"] + t["nbytes"]] = a.tobytes()
    with open(os.path.join(out_dir, man["data_file"]), "wb") as f:
        f.write(buf)
    for g in man["graphs"].values():
        shutil.copy(os.path.join(tpl, g), out_dir)
    shutil.copy(os.path.join(tpl, "manifest.json"), out_dir)
    sr = cpt["config"][-1]
    if isinstance(sr, str):
        sr = {"32k": 32000, "40k": 40000, "48k": 48000}[sr]
    meta = {
        "name": os.path.splitext(os.path.basename(pth))[0], "version": cpt.get("version", "v1"),
        "tgt_sr": int(sr), "upp": man["upp"], "template": man["key"],
        "phone_dim": man["signature"]["phone_dim"], "graphs": man["graphs"],
        "n_spk": int(sd["emb_g.weight"].shape[0]),
        "emb_g": sd["emb_g.weight"].float().numpy().tolist(),
        "l_linear_w": float(sd["dec.m_source.l_linear.weight"].float().reshape(-1)[0]),
        "l_linear_b": float(sd["dec.m_source.l_linear.bias"].float().reshape(-1)[0]),
        "has_index": False,
    }
    if index:
        write_index_bin(read_faiss_ivfflat(index), os.path.join(out_dir, "index.bin"))
        meta["has_index"] = True
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f)
    return meta


# ============================================================================ engine
class Params:
    def __init__(self, **kw):
        self.pitch = 0.0
        self.formant = 0.0
        self.index_rate = 0.0
        self.rms_mix_rate = 0.0
        self.threshold = -60.0
        self.block_time = 0.25
        self.crossfade_time = 0.05
        self.extra_time = 2.5
        self.sid = 0
        self.phase_lock = False
        self.deterministic = False
        self.__dict__.update(kw)


class RealtimeVC:
    F0_MIN, F0_MAX = 50.0, 1100.0

    def __init__(self, assets, model_dir, p: Params, samplerate=48000, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        self.p = p
        self.sr = samplerate
        self.meta = json.load(open(os.path.join(model_dir, "model.json")))
        self.tgt_sr, self.upp = self.meta["tgt_sr"], self.meta["upp"]
        so = ort.SessionOptions()
        mk = lambda path: ort.InferenceSession(path, so, providers=list(providers))
        hman = json.load(open(os.path.join(assets, "hubert", "manifest.json")))
        hname = "hubert_v1" if self.meta["phone_dim"] == 256 else "hubert_v2"
        self.hubert = mk(os.path.join(assets, "hubert", hman["graphs"][hname]["fp32"]))
        fman = json.load(open(os.path.join(assets, "fcpe", "manifest.json")))
        self.fcpe = mk(os.path.join(assets, "fcpe", fman["graph"]))
        self.cent_table = np.array(fman["cent_table"], np.float32)
        self.fcpe_f0_min = fman["f0_min"]
        g = self.meta["graphs"]
        self.enc = mk(os.path.join(model_dir, g["enc"]))
        self.flow = mk(os.path.join(model_dir, g["flow"]))
        self.dec = mk(os.path.join(model_dir, g["dec"]))
        self.g = np.array(self.meta["emb_g"][p.sid], np.float32).reshape(1, -1, 1)
        self.index = read_index_bin(os.path.join(model_dir, "index.bin")) if self.meta.get("has_index") else None
        self.rng = np.random.default_rng(1234)
        self.f0_mel_min = 1127 * np.log(1 + self.F0_MIN / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.F0_MAX / 700)
        self._setup()

    # ---- sizes (realtime_gui.start_vc)
    def _setup(self):
        p, sr = self.p, self.sr
        zc = self.zc = sr // 100
        r = lambda t: int(np.round(t * sr / zc)) * zc
        self.block_frame = r(p.block_time)
        self.block_frame_16k = 160 * self.block_frame // zc
        self.crossfade_frame = r(p.crossfade_time)
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = r(p.extra_time)
        n = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = np.zeros(n, np.float32)
        self.input_wav_res = np.zeros(160 * n // zc, np.float32)
        self.rms_buffer = np.zeros(4 * zc, np.float32)
        self.sola_buffer = np.zeros(self.sola_buffer_frame, np.float32)
        self.skip_head = self.extra_frame // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc
        self.fade_in = (np.sin(0.5 * np.pi * np.linspace(0, 1, self.sola_buffer_frame, dtype=np.float32)) ** 2).astype(np.float32)
        self.fade_out = 1 - self.fade_in
        self.res_in = SincResampler(sr, 16000)
        self.res_out = SincResampler(self.tgt_sr, sr) if self.tgt_sr != sr else None
        self.cache_pitch = np.zeros(1024, np.int64)
        self.cache_pitchf = np.zeros(1024, np.float32)
        self.phase_frames = 0  # absolute 100 Hz frame counter for phase_lock
        self.phase_acc = {}    # absolute frame -> phase at frame start (phase_lock)

    # ---- f0 (rtrvc.get_f0_fcpe + get_f0_post)
    def _fcpe(self, x16):
        lat = self.fcpe.run(None, {"mag": stft_mag_fcpe(x16)[None]})[0][0]
        imax = np.argmax(lat, axis=1)
        conf = lat[np.arange(len(lat)), imax]
        idx = np.clip(imax[:, None] + np.arange(-4, 5)[None, :], 0, lat.shape[1] - 1)
        yl = np.take_along_axis(lat, idx, 1)
        cents = (self.cent_table[idx] * yl).sum(1) / yl.sum(1)
        f0 = np.where(conf > 0.006, 10.0 * 2 ** (cents / 1200.0), 0.0).astype(np.float32)
        f0[f0 < self.fcpe_f0_min] = 0
        return f0

    def _f0_post(self, f0, key):
        uv = f0 == 0
        if np.any(~uv):
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0 = (f0 * pow(2, key / 12)).astype(np.float32)
        f0_mel = (1127 * np.log(1 + f0 / 700)).astype(np.float32)
        pos = f0_mel > 0
        f0_mel[pos] = (f0_mel[pos] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.round(f0_mel).astype(np.int64), f0

    # ---- NSF source (SineGen + l_linear + tanh)
    def _har(self, f0, upp, start_frame):
        sr = self.tgt_sr
        L = len(f0)
        k = np.arange(1, upp + 1, dtype=np.float32)
        rad = (f0[:, None] / np.float32(sr)) * k[None, :]
        rad2 = np.fmod(rad[:, -1] + np.float32(0.5), np.float32(1.0)) - np.float32(0.5)
        if self.p.phase_lock:
            # phase at the start of each absolute frame is carried across blocks
            start = np.empty(L, np.float64)
            ph = self.phase_acc.get(start_frame, 0.0)
            for i in range(L):
                start[i] = ph
                ph = math.fmod(ph + float(rad2[i]), 1.0)
                self.phase_acc[start_frame + i + 1] = ph
            rad = rad + start[:, None].astype(np.float32)
        else:
            acc = np.fmod(np.cumsum(rad2.astype(np.float64)).astype(np.float32), np.float32(1.0))
            rad[1:] += acc[:-1, None]
        sine = np.sin(np.float32(2 * np.pi) * rad.reshape(-1)) * np.float32(0.1)
        uv = np.repeat((f0 > 0).astype(np.float32), upp)
        noise_amp = uv * np.float32(0.003) + (1 - uv) * np.float32(0.1 / 3)
        noise = np.zeros_like(sine) if self.p.deterministic else self.rng.standard_normal(len(sine)).astype(np.float32)
        sine = sine * uv + noise_amp * noise
        return np.tanh(self.meta["l_linear_w"] * sine + self.meta["l_linear_b"]).astype(np.float32)

    # ---- rtrvc.RVC.infer
    def _infer(self):
        p = self.p
        x = self.input_wav_res
        feats = self.hubert.run(None, {"audio": x[None]})[0][0]
        feats = np.concatenate([feats, feats[-1:]], 0)
        if self.index is not None and p.index_rate > 0:
            s = self.skip_head // 2
            npy = ivf_search_blend(self.index, feats[s:].astype(np.float32))
            feats[s:] = npy * p.index_rate + (1 - p.index_rate) * feats[s:]
        p_len = len(x) // 160
        factor = pow(2, p.formant / 12)
        rl, rl2 = self.return_length, int(np.ceil(self.return_length * factor))
        n16 = self.block_frame_16k + 800
        f0 = self._fcpe(x[-n16:])
        pitch, pitchf = self._f0_post(f0, p.pitch - p.formant)
        shift = self.block_frame_16k // 160
        self.cache_pitch[:-shift] = self.cache_pitch[shift:].copy()
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].copy()
        self.cache_pitch[4 - len(pitch):] = pitch[3:-1]
        self.cache_pitchf[4 - len(pitch):] = pitchf[3:-1]
        cp = self.cache_pitch[-p_len:]
        cpf = self.cache_pitchf[-p_len:] * rl2 / rl
        phone = np.repeat(feats, 2, axis=0)[:p_len]
        m_p, logs_p = self.enc.run(None, {"phone": phone[None].astype(np.float32), "pitch": cp[None]})
        flow_head = max(self.skip_head - 24, 0)
        dec_head = self.skip_head - flow_head
        m_p, logs_p = m_p[:, :, flow_head:], logs_p[:, :, flow_head:]
        noise = np.zeros_like(m_p) if p.deterministic else self.rng.standard_normal(m_p.shape).astype(np.float32)
        z_p = m_p + np.exp(logs_p) * noise * np.float32(0.66666)
        z = self.flow.run(None, {"z_p": z_p.astype(np.float32), "g": self.g})[0]
        z = z[:, :, dec_head:dec_head + rl]
        nsff0 = cpf[self.skip_head:self.skip_head + rl].astype(np.float32)
        start_frame = self.phase_frames + self.skip_head
        har = self._har(nsff0, self.upp, start_frame)[None, None]
        if rl2 != rl:
            har = interp_linear(har, rl2 * self.upp).astype(np.float32)
            z = interp_linear(z, rl2).astype(np.float32)
        audio = self.dec.run(None, {"z": z, "har": har, "g": self.g})[0][0, 0]
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            audio = SincResampler(upp_res, self.tgt_sr // 100)(audio[: rl * upp_res])
        return audio.astype(np.float32)

    # ---- realtime_gui.audio_callback
    def process_block(self, indata: np.ndarray) -> np.ndarray:
        p, zc = self.p, self.zc
        bf = self.block_frame
        indata = indata.astype(np.float32).copy()
        if p.threshold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = rms_frames(indata, 4 * zc, zc)[2:]
            self.rms_buffer[:] = indata[-4 * zc:]
            indata = indata[2 * zc - zc // 2:]
            db = 20 * np.log10(np.maximum(rms, 1e-5))
            for i in np.where(db < p.threshold)[0]:
                indata[i * zc:(i + 1) * zc] = 0
            indata = indata[zc // 2:]
        self.input_wav[:-bf] = self.input_wav[bf:].copy()
        self.input_wav[-len(indata):] = indata
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].copy()
        res = self.res_in(self.input_wav[-len(indata) - 2 * zc:])
        self.input_wav_res[-160 * (len(indata) // zc + 1):] = res[160:]
        infer_wav = self._infer()
        self.phase_frames += self.block_frame // zc
        if self.res_out is not None:
            infer_wav = self.res_out(infer_wav)
        if p.rms_mix_rate < 1:
            iw = self.input_wav[self.extra_frame:][: len(infer_wav)]
            rms1 = interp_linear_align(rms_frames(iw, 4 * zc, zc), len(infer_wav) + 1)[:-1]
            rms2 = interp_linear_align(rms_frames(infer_wav, 4 * zc, zc), len(infer_wav) + 1)[:-1]
            rms2 = np.maximum(rms2, 1e-3)
            infer_wav = infer_wav * np.power(rms1 / rms2, np.float32(1.0 - p.rms_mix_rate))
        sb, ss = self.sola_buffer_frame, self.sola_search_frame
        ci = infer_wav[: sb + ss]
        nom = np.correlate(ci, self.sola_buffer, mode="valid")
        den = np.sqrt(np.correlate(ci * ci, np.ones(sb, np.float32), mode="valid") + 1e-8)
        off = int(np.argmax(nom / den))
        self.last_offset = off
        infer_wav = infer_wav[off:].copy()
        infer_wav[:sb] = infer_wav[:sb] * self.fade_in + self.sola_buffer * self.fade_out
        self.sola_buffer[:] = infer_wav[bf:bf + sb]
        return infer_wav[:bf].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("import")
    a.add_argument("--assets", required=True); a.add_argument("--pth", required=True)
    a.add_argument("--index"); a.add_argument("--out", required=True)
    b = sub.add_parser("process")
    b.add_argument("--assets", required=True); b.add_argument("--model", required=True)
    b.add_argument("--in", dest="inp", required=True); b.add_argument("--out", required=True)
    for k, v in vars(Params()).items():
        b.add_argument("--" + k.replace("_", "-"), type=type(v) if not isinstance(v, bool) else int, default=v)
    args = ap.parse_args()
    if args.cmd == "import":
        print(import_model(args.assets, args.pth, args.index, args.out)["template"])
        return
    import soundfile as sf
    x, sr = sf.read(args.inp, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    pr = Params(**{k: getattr(args, k) for k in vars(Params())})
    vc = RealtimeVC(args.assets, args.model, pr, samplerate=sr)
    out = [vc.process_block(x[i:i + vc.block_frame]) for i in range(0, len(x) - vc.block_frame + 1, vc.block_frame)]
    sf.write(args.out, np.concatenate(out), sr, subtype="FLOAT")


if __name__ == "__main__":
    main()
