#include "native_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

#include "common.hpp"
#include "conv_kernels.hpp"
#include "pth_reader.hpp"  // MappedFile
#include "thread_pool.hpp"

namespace rvc {

namespace {

constexpr int kMargin = 64;  // zero columns on both sides of every conv input row

inline int round_up(int v, int m) { return (v + m - 1) / m * m; }
inline int ld_for(int L) { return round_up(L, kConvTileT) + 2 * kMargin; }

struct Tensor {
  std::vector<float> v;
  std::vector<int64_t> shape;
};

class WeightFile {
 public:
  explicit WeightFile(const std::string& dir)
      : man_(Json::parse(read_file(join_path(dir, "manifest.json")))), file_(join_path(dir, "weights.bin")) {
    for (auto& t : man_["tensors"].items()) idx_[t["name"].str()] = &t;
  }
  const Json& manifest() const { return man_; }
  bool has(const std::string& n) const { return idx_.count(n) > 0; }
  Tensor get(const std::string& n) const {
    auto it = idx_.find(n);
    if (it == idx_.end()) throw Error("weights: missing " + n);
    const Json& t = *it->second;
    Tensor r;
    for (auto& s : t["shape"].items()) r.shape.push_back(s.i64());
    size_t off = static_cast<size_t>(t["offset"].i64()), nb = static_cast<size_t>(t["nbytes"].i64());
    if (off + nb > file_.size() || t["dtype"].str() != "f16") throw Error("weights: bad entry " + n);
    r.v.resize(nb / 2);
    const uint8_t* p = file_.data() + off;
    for (size_t i = 0; i < r.v.size(); ++i) {
      uint16_t h;
      memcpy(&h, p + 2 * i, 2);
      r.v[i] = half_to_float(h);
    }
    return r;
  }
  // weight_norm(dim=0): w = v * g / ||v||
  Tensor wn(const std::string& prefix) const {
    Tensor g = get(prefix + ".weight_g"), v = get(prefix + ".weight_v");
    size_t c0 = static_cast<size_t>(v.shape[0]), inner = v.v.size() / c0;
    for (size_t c = 0; c < c0; ++c) {
      double s = 0;
      for (size_t i = 0; i < inner; ++i) s += static_cast<double>(v.v[c * inner + i]) * v.v[c * inner + i];
      double k = g.v[c] / std::sqrt(s);
      for (size_t i = 0; i < inner; ++i) v.v[c * inner + i] = static_cast<float>(v.v[c * inner + i] * k);
    }
    return v;
  }

 private:
  Json man_;
  MappedFile file_;
  std::map<std::string, const Json*> idx_;
};

std::vector<int> ints(const Json& a) {
  std::vector<int> v;
  for (auto& x : a.items()) v.push_back(x.i32());
  return v;
}

}  // namespace

NativeDecoder::NativeDecoder(const std::string& model_dir, const std::vector<float>& g, int threads, bool want_fp16,
                             const std::vector<int>& affinity) {
  fp16_ = want_fp16 && fp16_supported();
  pool_.reset(new ThreadPool(threads, affinity));
  WeightFile wf(model_dir);
  const Json& sig = wf.manifest()["signature"];
  std::vector<int> rates = ints(sig["upsample_rates"]), kernels = ints(sig["upsample_kernel_sizes"]);
  std::vector<int> rk = ints(sig["resblock_kernel_sizes"]);
  std::vector<std::vector<int>> rd;
  for (auto& d : sig["resblock_dilation_sizes"].items()) rd.push_back(ints(d));
  int c0 = sig["upsample_initial_channel"].i32();
  cin_ = sig["inter_channels"].i32();
  resblock1_ = sig["resblock"].str() == "1";
  num_kernels_ = static_cast<int>(rk.size());
  upp_ = 1;
  for (int r : rates) upp_ *= r;

  // conv_pre with the speaker-conditioning term folded into its bias
  {
    Tensor w = wf.get("dec.conv_pre.weight"), b = wf.get("dec.conv_pre.bias");
    Tensor cw = wf.get("dec.cond.weight"), cb = wf.get("dec.cond.bias");
    int Co = static_cast<int>(w.shape[0]), Ci = static_cast<int>(w.shape[1]), K = static_cast<int>(w.shape[2]);
    int gin = static_cast<int>(cw.shape[1]);
    if (static_cast<int>(g.size()) != gin) throw Error("speaker embedding size mismatch");
    std::vector<float> bias(Co);
    for (int co = 0; co < Co; ++co) {
      double s = b.v[co] + cb.v[co];
      for (int k = 0; k < gin; ++k) s += static_cast<double>(cw.v[static_cast<size_t>(co) * gin + k]) * g[k];
      bias[co] = static_cast<float>(s);
    }
    std::vector<int> kidx, off;
    for (int k = 0; k < K; ++k) { kidx.push_back(k); off.push_back(k - (K - 1) / 2); }
    pack(conv_pre_, w.v, Co, Ci, K, kidx, off, bias.data());
  }

  int ch = c0;
  for (size_t i = 0; i < rates.size(); ++i) {
    Stage st;
    st.rate = rates[i];
    st.cin = ch;
    st.cout = ch / 2;
    int u = rates[i], K = kernels[i], p = (K - u) / 2;
    Tensor wt = wf.wn("dec.ups." + std::to_string(i));  // [cin][cout][K]
    Tensor bt = wf.get("dec.ups." + std::to_string(i) + ".bias");
    // transposed conv -> one ordinary conv per output phase (dense [cout][cin][K] view)
    std::vector<float> dense(static_cast<size_t>(st.cout) * st.cin * K);
    for (int ci = 0; ci < st.cin; ++ci)
      for (int co = 0; co < st.cout; ++co)
        for (int k = 0; k < K; ++k)
          dense[(static_cast<size_t>(co) * st.cin + ci) * K + k] = wt.v[(static_cast<size_t>(ci) * st.cout + co) * K + k];
    for (int ph = 0; ph < u; ++ph) {
      std::vector<int> kidx, off;
      for (int k = 0; k < K; ++k) {
        int d = k - p - ph;
        if (((d % u) + u) % u == 0) { kidx.push_back(k); off.push_back(-(d / u)); }
      }
      Conv c;
      pack(c, dense, st.cout, st.cin, K, kidx, off, bt.v.data());
      st.phases.push_back(std::move(c));
    }
    // noise conv (strided) -> polyphase channels
    {
      Tensor nw = wf.get("dec.noise_convs." + std::to_string(i) + ".weight");
      Tensor nb = wf.get("dec.noise_convs." + std::to_string(i) + ".bias");
      int kn = static_cast<int>(nw.shape[2]);
      if (i + 1 < rates.size()) {
        int s = 1;
        for (size_t r = i + 1; r < rates.size(); ++r) s *= rates[r];
        st.noise_s = s;
        // W'[co][phi][q] = W[co][0][q*s + phi + s/2], q in {-1,0,1}
        std::vector<float> wp(static_cast<size_t>(st.cout) * s * 3, 0.f);
        for (int co = 0; co < st.cout; ++co)
          for (int ph = 0; ph < s; ++ph)
            for (int q = -1; q <= 1; ++q) {
              int k = q * s + ph + s / 2;
              if (k >= 0 && k < kn) wp[(static_cast<size_t>(co) * s + ph) * 3 + (q + 1)] = nw.v[static_cast<size_t>(co) * kn + k];
            }
        pack(st.noise, wp, st.cout, s, 3, {0, 1, 2}, {-1, 0, 1}, nb.v.data());
      } else {
        st.noise_s = 1;
        pack(st.noise, nw.v, st.cout, 1, kn, {0}, {0}, nb.v.data());
      }
    }
    // resblocks
    for (int j = 0; j < num_kernels_; ++j) {
      int idx = static_cast<int>(i) * num_kernels_ + j;
      std::string pre = "dec.resblocks." + std::to_string(idx);
      int k = rk[j];
      std::vector<Conv> convs;
      auto add = [&](const std::string& name, int dil) {
        Tensor w = wf.wn(name);
        Tensor b = wf.get(name + ".bias");
        std::vector<int> kidx, off;
        for (int t = 0; t < k; ++t) { kidx.push_back(t); off.push_back((t - (k - 1) / 2) * dil); }
        Conv c;
        pack(c, w.v, st.cout, st.cout, k, kidx, off, b.v.data());
        convs.push_back(std::move(c));
      };
      for (size_t m = 0; m < rd[j].size(); ++m) {
        if (resblock1_) {
          add(pre + ".convs1." + std::to_string(m), rd[j][m]);
          add(pre + ".convs2." + std::to_string(m), 1);
        } else {
          add(pre + ".convs." + std::to_string(m), rd[j][m]);
        }
      }
      st.res.push_back(std::move(convs));
    }
    ch = st.cout;
    stages_.push_back(std::move(st));
  }
  {
    Tensor w = wf.get("dec.conv_post.weight");  // [1][ch][7], no bias
    int K = static_cast<int>(w.shape[2]), Ci = static_cast<int>(w.shape[1]);
    std::vector<float> w8(static_cast<size_t>(kConvCoTile) * Ci * K, 0.f);
    std::copy(w.v.begin(), w.v.end(), w8.begin());
    std::vector<int> kidx, off;
    for (int k = 0; k < K; ++k) { kidx.push_back(k); off.push_back(k - (K - 1) / 2); }
    pack(conv_post_, w8, kConvCoTile, Ci, K, kidx, off, nullptr);
  }
  RVC_LOGI("native decoder: %zu stages, upp %d, %s", stages_.size(), upp_, fp16_ ? "fp16" : "fp32");
}

NativeDecoder::~NativeDecoder() = default;

std::vector<int> NativeDecoder::tids() const { return pool_->tids(); }

void NativeDecoder::pack(Conv& c, const std::vector<float>& w, int Co, int Ci, int K, const std::vector<int>& kidx,
                         const std::vector<int>& off, const float* bias) {
  c.Co = Co;
  c.Ci = Ci;
  c.ntaps = static_cast<int>(kidx.size());
  c.off = off;
  c.bias.assign(Co, 0.f);
  if (bias) std::copy(bias, bias + Co, c.bias.begin());
  std::vector<float> p(static_cast<size_t>(Co) * Ci * c.ntaps);
  size_t q = 0;
  for (int cot = 0; cot < Co / kConvCoTile; ++cot)
    for (int ci = 0; ci < Ci; ++ci)
      for (int t = 0; t < c.ntaps; ++t)
        for (int l = 0; l < kConvCoTile; ++l)
          p[q++] = w[(static_cast<size_t>(cot * kConvCoTile + l) * Ci + ci) * K + kidx[t]];
  if (fp16_) {
    c.w16.resize(p.size());
    for (size_t i = 0; i < p.size(); ++i) c.w16[i] = float_to_half(p[i]);
  } else {
    c.w32.swap(p);
  }
}

void NativeDecoder::conv(const Conv& c, const void* x, int ldx, int T, float* y, int ldy, bool accumulate) {
  ConvArgs a;
  a.Co = c.Co;
  a.Ci = c.Ci;
  a.ntaps = c.ntaps;
  a.off = c.off.data();
  a.w = fp16_ ? static_cast<const void*>(c.w16.data()) : static_cast<const void*>(c.w32.data());
  a.bias = c.bias.data();
  a.x = x;
  a.ldx = ldx;
  a.T = T;
  a.y = y;
  a.ldy = ldy;
  a.accumulate = accumulate;
  if (fp16_) conv_f16(a, *pool_);
  else conv_f32(a, *pool_);
}

void NativeDecoder::run(const float* z, int T, const float* har, float* out) {
  // buffer sizing: largest channels x row stride over all stages
  size_t need = static_cast<size_t>(std::max(cin_, conv_pre_.Co)) * ld_for(T);
  {
    int L = T;
    for (auto& st : stages_) {
      int Lo = L * st.rate;
      need = std::max(need, static_cast<size_t>(std::max(st.cin, st.cout)) * ld_for(Lo));
      need = std::max(need, static_cast<size_t>(st.noise_s) * ld_for(Lo));
      L = Lo;
    }
    need = std::max(need, static_cast<size_t>(kConvCoTile) * ld_for(L));
  }
  for (auto& b : fbuf_)
    if (b.size() < need) b.assign(need, 0.f);
  if (fp16_) {
    if (hbuf_.size() < 2 * need) hbuf_.assign(2 * need, 0);
  } else if (abuf32_.size() < 2 * need) {
    abuf32_.assign(2 * need, 0.f);
  }
  ThreadPool& pool = *pool_;
  const size_t es = fp16_ ? 2 : 4;
  uint8_t* abase[2] = {fp16_ ? reinterpret_cast<uint8_t*>(hbuf_.data()) : reinterpret_cast<uint8_t*>(abuf32_.data()),
                       fp16_ ? reinterpret_cast<uint8_t*>(hbuf_.data() + need)
                             : reinterpret_cast<uint8_t*>(abuf32_.data() + need)};
  // conv-input view: column 0 of row 0
  auto A = [&](int i) -> void* { return abase[i] + kMargin * es; };
  auto F = [&](int i) -> float* { return fbuf_[i].data() + kMargin; };

  // write rows of a conv input: dst[c][0..L) = act(src[c][t]), zero margins
  auto make_input = [&](int dst, const float* src, int lds, int C, int L, float slope) {
    int ld = ld_for(L);
    pool.parallel_for(C, [&](int c) {
      thread_local std::vector<float> tmp;
      if (tmp.size() < static_cast<size_t>(L)) tmp.resize(static_cast<size_t>(L));
      const float* s = src + static_cast<ptrdiff_t>(c) * lds;
      for (int t = 0; t < L; ++t) {
        float v = s[t];
        tmp[t] = v < 0 ? v * slope : v;
      }
      uint8_t* row = abase[dst] + static_cast<size_t>(c) * ld * es;
      memset(row, 0, kMargin * es);
      size_t tail = static_cast<size_t>(ld - kMargin - L);
      if (fp16_) {
        uint16_t* r = reinterpret_cast<uint16_t*>(row) + kMargin;
        f32_to_f16(tmp.data(), r, L);
        memset(r + L, 0, tail * 2);
      } else {
        float* r = reinterpret_cast<float*>(row) + kMargin;
        memcpy(r, tmp.data(), sizeof(float) * L);
        memset(r + L, 0, tail * 4);
      }
    });
  };

  // conv_pre(z) (+ cond(g) folded in bias)
  int L = T;
  make_input(0, z, T, cin_, L, 1.f);
  int x = 0;  // index of fp32 buffer holding the current activation
  conv(conv_pre_, A(0), ld_for(L), L, F(x), ld_for(L), false);

  for (auto& st : stages_) {
    const int Lo = L * st.rate, ldi = ld_for(L), ldo = ld_for(Lo);
    const int xn = (x + 1) % 5, r = (x + 2) % 5, b = (x + 3) % 5, tmp = (x + 4) % 5;
    // leaky_relu -> transposed conv (polyphase) -> interleave
    make_input(0, F(x), ldi, st.cin, L, 0.1f);
    for (int ph = 0; ph < st.rate; ++ph) {
      conv(st.phases[ph], A(0), ldi, L, F(tmp), ldi, false);
      float* dst = F(xn);
      const float* src = F(tmp);
      pool.parallel_for(st.cout, [&](int c) {
        const float* s = src + static_cast<size_t>(c) * ldi;
        float* d = dst + static_cast<size_t>(c) * ldo + ph;
        for (int m = 0; m < L; ++m) d[static_cast<size_t>(m) * st.rate] = s[m];
      });
    }
    // + noise_conv(har): har as noise_s polyphase channels
    {
      const int s = st.noise_s;
      std::vector<float> poly(static_cast<size_t>(s) * Lo);
      for (int m = 0; m < Lo; ++m)
        for (int ph = 0; ph < s; ++ph) poly[static_cast<size_t>(ph) * Lo + m] = har[static_cast<size_t>(m) * s + ph];
      make_input(1, poly.data(), Lo, s, Lo, 1.f);
      conv(st.noise, A(1), ldo, Lo, F(xn), ldo, true);
    }
    // resblocks; xs accumulates in buffer x (previous activation is dead now)
    make_input(1, F(xn), ldo, st.cout, Lo, 0.1f);  // shared first input of every resblock
    const size_t plane = static_cast<size_t>(st.cout) * ldo;
    for (int j = 0; j < num_kernels_; ++j) {
      auto& convs = st.res[j];
      float* R = F(r);
      memcpy(R - kMargin, F(xn) - kMargin, plane * sizeof(float));
      if (resblock1_) {
        for (size_t m = 0; m + 1 < convs.size(); m += 2) {
          if (m > 0) make_input(0, R, ldo, st.cout, Lo, 0.1f);
          conv(convs[m], m == 0 ? A(1) : A(0), ldo, Lo, F(b), ldo, false);
          make_input(0, F(b), ldo, st.cout, Lo, 0.1f);
          conv(convs[m + 1], A(0), ldo, Lo, R, ldo, true);
        }
      } else {
        for (size_t m = 0; m < convs.size(); ++m) {
          if (m > 0) make_input(0, R, ldo, st.cout, Lo, 0.1f);
          conv(convs[m], m == 0 ? A(1) : A(0), ldo, Lo, R, ldo, true);
        }
      }
      float* XS = F(x);
      if (j == 0) memcpy(XS - kMargin, R - kMargin, plane * sizeof(float));
      else {
        pool.parallel_for(st.cout, [&](int c) {
          float* d = XS + static_cast<size_t>(c) * ldo;
          const float* sr = R + static_cast<size_t>(c) * ldo;
          for (int t = 0; t < Lo; ++t) d[t] += sr[t];
        });
      }
    }
    {
      float inv = 1.f / num_kernels_;
      float* XS = F(x);
      pool.parallel_for(st.cout, [&](int c) {
        float* d = XS + static_cast<size_t>(c) * ldo;
        for (int t = 0; t < Lo; ++t) d[t] *= inv;
      });
    }
    L = Lo;
  }
  // leaky_relu(0.01) -> conv_post -> tanh
  const int ld = ld_for(L);
  make_input(0, F(x), ld, stages_.back().cout, L, 0.01f);
  int y = (x + 1) % 5;
  conv(conv_post_, A(0), ld, L, F(y), ld, false);
  const float* yp = F(y);
  for (int t = 0; t < L; ++t) out[t] = std::tanh(yp[t]);
}

}  // namespace rvc
