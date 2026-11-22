"""Export-friendly re-implementation of the RVC synthesizer (inference path only).

The module tree and parameter names mirror RVC's ``SynthesizerTrnMs{256,768}NSFsid``
state dict exactly (``enc_p.*``, ``flow.*``, ``dec.*``), so an original RVC ``.pth``
loads with ``load_state_dict`` unchanged.  Differences from RVC's own classes:

* No masks: in realtime inference every sequence is full length, so RVC's masks are
  all ones and multiplying by them is a no-op.
* Weight norm is computed explicitly from ``weight_g``/``weight_v`` inside ``forward``.
  When exported with ``do_constant_folding=False`` the ONNX graph keeps the original
  ``weight_g``/``weight_v`` tensors as initializers and ONNX Runtime folds the weight
  norm at session-creation time.  That is what lets the Android app import an
  unmodified RVC ``.pth`` by copying raw tensor bytes.
* The NSF sine source (``dec.m_source``) is not part of the decoder graph; the app
  computes it natively (phase continuous across blocks) and feeds ``har`` directly.
* No ``int()`` on traced sizes, so dynamic time axes survive ONNX export.

Derived from RVC-Project/Retrieval-based-Voice-Conversion-WebUI (MIT License).
"""

import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

LRELU_SLOPE = 0.1


def _wn(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """torch.nn.utils.weight_norm with dim=0: w = v * g / ||v|| (norm over dims != 0)."""
    dims = list(range(1, v.dim()))
    return v * (g / torch.sqrt(torch.sum(v * v, dim=dims, keepdim=True)))


class Conv1d(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, k) * 0.05)
        self.bias = nn.Parameter(torch.randn(cout) * 0.05) if bias else None
        self.stride, self.padding, self.dilation = stride, padding, dilation

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class WNConv1d(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, dilation=1):
        super().__init__()
        self.weight_g = nn.Parameter(torch.rand(cout, 1, 1) + 0.5)
        self.weight_v = nn.Parameter(torch.randn(cout, cin, k) * 0.05)
        self.bias = nn.Parameter(torch.randn(cout) * 0.05)
        self.stride, self.padding, self.dilation = stride, padding, dilation

    def forward(self, x):
        w = _wn(self.weight_g, self.weight_v)
        return F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation)


class WNConvTranspose1d(nn.Module):
    def __init__(self, cin, cout, k, stride, padding):
        super().__init__()
        self.weight_g = nn.Parameter(torch.rand(cin, 1, 1) + 0.5)
        self.weight_v = nn.Parameter(torch.randn(cin, cout, k) * 0.05)
        self.bias = nn.Parameter(torch.randn(cout) * 0.05)
        self.stride, self.padding = stride, padding

    def forward(self, x):
        w = _wn(self.weight_g, self.weight_v)
        return F.conv_transpose1d(x, w, self.bias, self.stride, self.padding)


class Linear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin) * 0.05)
        self.bias = nn.Parameter(torch.randn(cout) * 0.05)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Embedding(nn.Module):
    def __init__(self, n, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, dim) * 0.05)

    def forward(self, idx):
        return F.embedding(idx, self.weight)


class LayerNorm(nn.Module):
    """RVC channels-first LayerNorm (parameters named gamma/beta)."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels, self.eps = channels, eps
        self.gamma = nn.Parameter(torch.rand(channels) + 0.5)
        self.beta = nn.Parameter(torch.randn(channels) * 0.05)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# --------------------------------------------------------------------------- enc_p
class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, window_size=10):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.conv_q = Conv1d(channels, channels, 1)
        self.conv_k = Conv1d(channels, channels, 1)
        self.conv_v = Conv1d(channels, channels, 1)
        self.conv_o = Conv1d(channels, out_channels, 1)
        std = self.k_channels ** -0.5
        self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * std)
        self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * std)

    def forward(self, x):
        q, k, v = self.conv_q(x), self.conv_k(x), self.conv_v(x)
        b, d, t = k.size()
        q = q.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        qs = q / math.sqrt(self.k_channels)
        scores = torch.matmul(qs, k.transpose(-2, -1))
        rel_k = self._get_relative_embeddings(self.emb_rel_k, t)
        rel_logits = torch.matmul(qs, rel_k.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._rel_to_abs(rel_logits)
        p_attn = F.softmax(scores, dim=-1)
        out = torch.matmul(p_attn, v)
        rel_w = self._abs_to_rel(p_attn)
        rel_v = self._get_relative_embeddings(self.emb_rel_v, t)
        out = out + torch.matmul(rel_w, rel_v.unsqueeze(0))
        out = out.transpose(2, 3).contiguous().view(b, d, t)
        return self.conv_o(out)

    def _get_relative_embeddings(self, emb, length):
        # Realtime windows are always longer than window_size + 1 frames, so the RVC
        # branch "pad_length > 0" is the only one taken; pad with traced sizes.
        pad = length - (self.window_size + 1)
        padded = F.pad(emb, [0, 0, pad, pad, 0, 0])
        return padded[:, 0 : 2 * length - 1]

    def _rel_to_abs(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.reshape(batch, heads, length * 2 * length)
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        return x_flat.reshape(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1 :]

    def _abs_to_rel(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.reshape(batch, heads, length * length + length * (length - 1))
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        return x_flat.reshape(batch, heads, length, 2 * length)[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(self, cin, cout, filter_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = Conv1d(cin, filter_channels, kernel_size)
        self.conv_2 = Conv1d(filter_channels, cout, kernel_size)

    def _pad(self, x):
        if self.kernel_size == 1:
            return x
        return F.pad(x, [(self.kernel_size - 1) // 2, self.kernel_size // 2, 0, 0, 0, 0])

    def forward(self, x):
        x = torch.relu(self.conv_1(self._pad(x)))
        return self.conv_2(self._pad(x))


class Encoder(nn.Module):
    def __init__(self, hidden, filter_channels, n_heads, n_layers, kernel_size):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden, hidden, n_heads))
            self.norm_layers_1.append(LayerNorm(hidden))
            self.ffn_layers.append(FFN(hidden, hidden, filter_channels, kernel_size))
            self.norm_layers_2.append(LayerNorm(hidden))

    def forward(self, x):
        for attn, n1, ffn, n2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            x = n1(x + attn(x))
            x = n2(x + ffn(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, filter_channels, n_heads, n_layers, kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.hidden = hidden
        self.emb_phone = Linear(in_channels, hidden)
        self.emb_pitch = Embedding(256, hidden)
        self.encoder = Encoder(hidden, filter_channels, n_heads, n_layers, kernel_size)
        self.proj = Conv1d(hidden, out_channels * 2, 1)

    def forward(self, phone, pitch):
        x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = F.leaky_relu(x * math.sqrt(self.hidden), 0.1)
        x = self.encoder(x.transpose(1, -1))
        stats = self.proj(x)
        return stats[:, : self.out_channels], stats[:, self.out_channels :]


# --------------------------------------------------------------------------- flow
class WN(nn.Module):
    def __init__(self, hidden, kernel_size, dilation_rate, n_layers, gin_channels):
        super().__init__()
        self.hidden, self.n_layers = hidden, n_layers
        self.cond_layer = WNConv1d(gin_channels, 2 * hidden * n_layers, 1)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for i in range(n_layers):
            d = dilation_rate ** i
            self.in_layers.append(WNConv1d(hidden, 2 * hidden, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2))
            ch = 2 * hidden if i < n_layers - 1 else hidden
            self.res_skip_layers.append(WNConv1d(hidden, ch, 1))

    def forward(self, x, g):
        out = None
        g = self.cond_layer(g)
        h = self.hidden
        for i in range(self.n_layers):
            a = self.in_layers[i](x) + g[:, i * 2 * h : (i + 1) * 2 * h, :]
            acts = torch.tanh(a[:, :h]) * torch.sigmoid(a[:, h:])
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = x + rs[:, :h]
                skip = rs[:, h:]
            else:
                skip = rs
            out = skip if out is None else out + skip
        return out


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden, kernel_size, dilation_rate, n_layers, gin_channels):
        super().__init__()
        self.half = channels // 2
        self.pre = Conv1d(self.half, hidden, 1)
        self.enc = WN(hidden, kernel_size, dilation_rate, n_layers, gin_channels)
        self.post = Conv1d(hidden, self.half, 1)

    def reverse(self, x, g):
        x0, x1 = x[:, : self.half], x[:, self.half :]
        m = self.post(self.enc(self.pre(x0), g))
        return torch.cat([x0, x1 - m], 1)


class Flip(nn.Module):
    pass


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden, kernel_size, dilation_rate, n_layers, gin_channels, n_flows=4):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden, kernel_size, dilation_rate, n_layers, gin_channels))
            self.flows.append(Flip())

    def reverse(self, x, g):
        for f in reversed(self.flows):
            x = torch.flip(x, [1]) if isinstance(f, Flip) else f.reverse(x, g)
        return x


# --------------------------------------------------------------------------- dec
class ResBlock1(nn.Module):
    def __init__(self, ch, k, dil):
        super().__init__()
        self.convs1 = nn.ModuleList([WNConv1d(ch, ch, k, dilation=d, padding=(k * d - d) // 2) for d in dil])
        self.convs2 = nn.ModuleList([WNConv1d(ch, ch, k, dilation=1, padding=(k - 1) // 2) for _ in dil])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, ch, k, dil):
        super().__init__()
        self.convs = nn.ModuleList([WNConv1d(ch, ch, k, dilation=d, padding=(k * d - d) // 2) for d in dil])

    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


class GeneratorNSF(nn.Module):
    """NSF-HiFiGAN decoder; ``har`` is the (already tanh(linear)-merged) sine source."""

    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(WNConvTranspose1d(upsample_initial_channel // (2 ** i), c_cur, k, u, (k - u) // 2))
            if i + 1 < len(upsample_rates):
                s = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(Conv1d(1, c_cur, s * 2, stride=s, padding=s // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, 1))
        rb = ResBlock1 if str(resblock) == "1" else ResBlock2
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(rb(ch, k, d))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, har, g):
        x = self.conv_pre(x) + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x = x + self.noise_convs[i](har)
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = F.leaky_relu(x)  # default slope 0.01, as in RVC
        return torch.tanh(self.conv_post(x))


# --------------------------------------------------------------------------- full
class MobileSynth(nn.Module):
    """Holds enc_p / flow / dec with RVC parameter names."""

    def __init__(self, cfg: List, phone_dim: int):
        super().__init__()
        (spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers,
         kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
         upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr) = cfg[:18]
        self.inter_channels = inter_channels
        self.enc_p = TextEncoder(phone_dim, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels)
        self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                                upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels)
        self.upp = math.prod(upsample_rates)


class EncGraph(nn.Module):
    def __init__(self, s: MobileSynth):
        super().__init__()
        self.enc_p = s.enc_p

    def forward(self, phone, pitch):
        return self.enc_p(phone, pitch)


class FlowGraph(nn.Module):
    def __init__(self, s: MobileSynth):
        super().__init__()
        self.flow = s.flow

    def forward(self, z_p, g):
        return self.flow.reverse(z_p, g)


class DecGraph(nn.Module):
    def __init__(self, s: MobileSynth):
        super().__init__()
        self.dec = s.dec

    def forward(self, z, har, g):
        return self.dec(z, har, g)
