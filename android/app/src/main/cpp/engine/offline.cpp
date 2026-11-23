#include "offline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>

#include "common.hpp"
#include "ivf_index.hpp"
#include "native_decoder.hpp"
#include "rvc_engine.hpp"

namespace rvc {

static size_t round_up(size_t v, size_t m) { return (v + m - 1) / m * m; }

void OfflineProgress::check() const {
  if (cancel.load()) throw Error("cancelled");
}

// ------------------------------------------------------------------ StreamResampler
//
// SincResampler::process makes output frame f (outputs f*n .. f*n+n-1, ratio o:n after the
// gcd) from inputs [f*o - width, f*o + width + o). Pieces start at multiples of o, so a piece
// run over [B, B + 2 ctx + chunk) gives exactly the whole-signal outputs for the inputs
// [B + ctx, B + ctx + chunk) whenever ctx >= width; outside the signal both see zeros.
StreamResampler::StreamResampler(int orig, int target) : r_(orig, target) {
  identity_ = orig == target;
  const size_t g = std::gcd(orig, target);
  o_ = orig / g;
  n_ = target / g;
  ctx_ = round_up(std::max(1, r_.width()), o_);
  chunk_ = round_up(std::max(orig / 10, 1), o_);  // ~0.1 s
  buf_.assign(ctx_, 0.f);
}

void StreamResampler::push(const float* x, size_t n, std::vector<float>& out) {
  in_total_ += n;
  if (identity_) {
    out.insert(out.end(), x, x + n);
    out_total_ += n;
    return;
  }
  buf_.insert(buf_.end(), x, x + n);
  const size_t win = 2 * ctx_ + chunk_, emit = chunk_ / o_ * n_, skip = ctx_ / o_ * n_;
  size_t start = 0;
  while (buf_.size() - start >= win) {
    y_.resize(static_cast<size_t>(r_.out_len(static_cast<int>(win))));
    r_.process(buf_.data() + start, static_cast<int>(win), y_.data());
    out.insert(out.end(), y_.begin() + skip, y_.begin() + skip + emit);
    out_total_ += emit;
    start += chunk_;
  }
  if (start) buf_.erase(buf_.begin(), buf_.begin() + start);
}

void StreamResampler::finish(std::vector<float>& out) {
  if (identity_) return;
  const size_t total = (in_total_ * n_ + o_ - 1) / o_;  // SincResampler::out_len of the whole signal
  if (total > out_total_) {
    const size_t rest = total - out_total_, skip = ctx_ / o_ * n_;
    const size_t len = round_up(buf_.size() + ctx_ + 2 * o_, o_);
    buf_.resize(len, 0.f);
    y_.resize(static_cast<size_t>(r_.out_len(static_cast<int>(len))));
    r_.process(buf_.data(), static_cast<int>(len), y_.data());
    out.insert(out.end(), y_.begin() + skip, y_.begin() + skip + rest);
  }
  buf_.assign(ctx_, 0.f);
  in_total_ = out_total_ = 0;
}

// ------------------------------------------------------------------ params
OfflineParams OfflineParams::from_json(const Json& j) {
  OfflineParams p;
  p.pitch = static_cast<int>(std::lround(j.num_or("pitch", p.pitch)));
  p.index_rate = static_cast<float>(j.num_or("index_rate", p.index_rate));
  p.rms_mix_rate = static_cast<float>(j.num_or("rms_mix_rate", p.rms_mix_rate));
  p.protect = static_cast<float>(j.num_or("protect", p.protect));
  p.deterministic = j.bool_or("deterministic", p.deterministic);
  p.x_pad = static_cast<int>(j.num_or("x_pad", p.x_pad));
  p.x_query = static_cast<int>(j.num_or("x_query", p.x_query));
  p.x_center = static_cast<int>(j.num_or("x_center", p.x_center));
  p.x_max = static_cast<int>(j.num_or("x_max", p.x_max));
  p.dec_chunk = static_cast<int>(j.num_or("dec_chunk", p.dec_chunk));
  if (p.x_pad < 1 || p.x_query < 1 || p.x_center <= p.x_query || p.x_max < p.x_center) throw Error("bad x_pad / x_query / x_center / x_max");
  return p;
}

// ------------------------------------------------------------------ helpers
namespace {

// numpy.pad(mode="reflect") index, also for pads longer than the signal
int64_t reflect(int64_t j, int64_t n) {
  if (n == 1) return 0;
  const int64_t p = 2 * (n - 1);
  j = (j < 0 ? -j : j) % p;
  return j >= n ? p - j : j;
}

// scipy.signal.filtfilt(bh, ah, x) with bh, ah = butter(N=5, Wn=48, btype="high", fs=16000):
// odd extension by 3 * 6 samples, lfilter (direct form II transposed, float64) forwards and
// backwards from the step-response steady state.
std::vector<double> highpass_filtfilt(const std::vector<float>& x) {
  static const double b[6] = {0.9699606451838447, -4.849803225919223, 9.699606451838447,
                              -9.699606451838447, 4.849803225919223,  -0.9699606451838447};
  static const double a[6] = {1.0, -4.939001819168364, 9.757863526739541, -9.63954484941346, 4.76150679735621, -0.9408236532054606};
  static const double zi[5] = {-0.9699610662701577, 3.879844239395132, -5.81976632134609, 3.879844189572757, -0.9699610413518082};
  const size_t n = x.size(), pad = 18;
  if (n <= pad) throw Error("音频太短");
  std::vector<double> y(n + 2 * pad);
  for (size_t i = 0; i < pad; ++i) y[i] = 2.f * x[0] - x[pad - i];  // the extension is float32, as numpy makes it
  for (size_t i = 0; i < n; ++i) y[pad + i] = x[i];
  for (size_t i = 0; i < pad; ++i) y[pad + n + i] = 2.f * x[n - 1] - x[n - 2 - i];
  auto lfilter = [&](auto idx) {
    const double x0 = y[idx(0)];
    double z[5];
    for (int k = 0; k < 5; ++k) z[k] = zi[k] * x0;
    for (size_t i = 0; i < y.size(); ++i) {
      double& v = y[idx(i)];
      const double xn = v, yn = z[0] + b[0] * xn;
      for (int k = 0; k < 4; ++k) z[k] = z[k + 1] + xn * b[k + 1] - yn * a[k + 1];
      z[4] = xn * b[5] - yn * a[5];
      v = yn;
    }
  };
  const size_t L = y.size();
  lfilter([](size_t i) { return i; });
  lfilter([L](size_t i) { return L - 1 - i; });
  return std::vector<double>(y.begin() + pad, y.begin() + pad + n);
}

// librosa.feature.rms(frame_length = 2 hop, hop_length = hop, center=True, zero padding),
// fed in order: frame k covers [(k-1) hop, (k+1) hop).
class HalfRms {
 public:
  explicit HalfRms(size_t hop) : hop_(hop) {}
  template <typename T>
  void add(const T* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      cur_ += static_cast<double>(y[i]) * y[i];
      if (++fill_ == hop_) {
        sums_.push_back(cur_);
        cur_ = 0;
        fill_ = 0;
      }
    }
    len_ += n;
  }
  std::vector<double> frames() const {
    std::vector<double> s = sums_;
    s.push_back(cur_);
    const size_t nf = 1 + len_ / hop_;
    std::vector<double> r(nf);
    for (size_t k = 0; k < nf; ++k) {
      double e = (k > 0 ? s[k - 1] : 0.0) + (k < s.size() ? s[k] : 0.0);
      r[k] = std::sqrt(e / static_cast<double>(2 * hop_));
    }
    return r;
  }

 private:
  size_t hop_, fill_ = 0, len_ = 0;
  double cur_ = 0;
  std::vector<double> sums_;
};

// get_f0's np.interp over the unvoiced frames (edges take the nearest voiced value); an
// all-unvoiced track stays as it is (np.interp raises and RVC carries on).
void interp_unvoiced(std::vector<float>& f0) {
  const int L = static_cast<int>(f0.size());
  std::vector<int> voiced;
  for (int i = 0; i < L; ++i)
    if (f0[i] != 0) voiced.push_back(i);
  if (voiced.empty() || static_cast<int>(voiced.size()) == L) return;
  size_t vi = 0;
  for (int i = 0; i < L; ++i) {
    if (f0[i] != 0) continue;
    while (vi < voiced.size() && voiced[vi] < i) ++vi;
    if (vi == 0) f0[i] = f0[voiced[0]];
    else if (vi >= voiced.size()) f0[i] = f0[voiced.back()];
    else {
      const int a = voiced[vi - 1], b = voiced[vi];
      const double slope = (static_cast<double>(f0[b]) - f0[a]) / (b - a);
      f0[i] = static_cast<float>(slope * (i - a) + f0[a]);
    }
  }
}

}  // namespace

// ------------------------------------------------------------------ OfflinePipeline
OfflinePipeline::OfflinePipeline(const std::string& assets, const std::string& model_dir, const BackendConfig& cpu_in, bool fp16, int sid)
    : model_dir_(model_dir) {
  const auto t0 = Clock::now();
  BackendConfig cpu = cpu_in;
  cpu.backend = Backend::CPU;
  cpu.mem_arena = false;  // every segment has another length
  meta_ = Json::parse(read_file(join_path(model_dir, "model.json")));
  refresh_model_graphs(assets, model_dir, meta_);
  tgt_sr_ = meta_["tgt_sr"].i32();
  upp_ = meta_["upp"].i32();
  phone_dim_ = meta_["phone_dim"].i32();
  l_w_ = static_cast<float>(meta_["l_linear_w"].num());
  l_b_ = static_cast<float>(meta_["l_linear_b"].num());
  const int n_spk = meta_["n_spk"].i32(), gin = static_cast<int>(meta_.num_or("gin_channels", 256));
  const std::string eg = read_file(join_path(model_dir, "emb_g.bin"));
  if (eg.size() != static_cast<size_t>(n_spk) * gin * 4) throw Error("bad emb_g.bin");
  sid = std::max(0, std::min(sid, n_spk - 1));
  g_.assign(reinterpret_cast<const float*>(eg.data()) + static_cast<size_t>(sid) * gin,
            reinterpret_cast<const float*>(eg.data()) + static_cast<size_t>(sid + 1) * gin);

  // any length: no fixed dimensions, CPU only (accelerators need static shapes)
  const Json hman = Json::parse(read_file(join_path(assets, "hubert/manifest.json")));
  const std::string hdir = join_path(assets, "hubert");
  hubert_.reset(new OnnxModel(join_path(hdir, hman["graphs"][phone_dim_ == 256 ? "hubert_v1" : "hubert_v2"]["fp32"].str()), cpu, {},
                              "offline_hubert_" + std::to_string(phone_dim_)));
  const Json fman = Json::parse(read_file(join_path(assets, "fcpe/manifest.json")));
  for (auto& v : fman["cent_table"].items()) cent_table_.push_back(static_cast<float>(v.num()));
  fcpe_f0_min_ = static_cast<float>(fman["f0_min"].num());
  fcpe_.reset(new OnnxModel(join_path(assets, "fcpe/" + fman["graph"].str()), cpu, {}, "offline_fcpe"));
  const Json& g = meta_["graphs"];
  const std::string tmpl = meta_["template"].str();
  enc_.reset(new OnnxModel(join_path(model_dir, g["enc"].str()), cpu, {}, "offline_" + tmpl + "_enc"));
  flow_.reset(new OnnxModel(join_path(model_dir, g["flow"].str()), cpu, {}, "offline_" + tmpl + "_flow"));
  if (file_exists(join_path(model_dir, "manifest.json")))
    native_dec_.reset(new NativeDecoder(model_dir, g_, cpu.threads, fp16, cpu.affinity));
  else
    dec_.reset(new OnnxModel(join_path(model_dir, g["dec"].str()), cpu, {}, "offline_" + tmpl + "_dec"));
  if (meta_.bool_or("has_index", false) && file_exists(join_path(model_dir, "index.bin")))
    index_.reset(new IvfIndex(join_path(model_dir, "index.bin")));
  load_ms_ = ms_since(t0);
}

OfflinePipeline::~OfflinePipeline() = default;

Json OfflinePipeline::info() const {
  Json j = Json::object();
  j.set("model", meta_.str_or("name", ""));
  j.set("version", meta_.str_or("version", ""));
  j.set("tgt_sr", tgt_sr_);
  j.set("has_index", index_ != nullptr);
  Json be = Json::object();
  be.set("hubert", hubert_->backend_used());
  be.set("dec", native_dec_ ? (native_dec_->fp16() ? "native_fp16" : "native_fp32") : dec_->backend_used());
  j.set("backends", be);
  j.set("load_ms", load_ms_);
  return j;
}

// infer/fcpe.py FCPEInfer.infer(sr=16000, decoder_mode="local_argmax", threshold=0.006):
// frames = n / 160 + 1. Same decoding as RvcEngine::run_fcpe.
std::vector<float> OfflinePipeline::fcpe(const float* x, int n) {
  std::vector<float> mag;
  stft_.process(x, n, mag);
  const int T = stft_.frames(n), nb = stft_.bins();
  std::vector<Ort::Value> in;
  in.push_back(make_tensor<float>(mag.data(), mag.size(), {1, T, nb}));
  auto out = fcpe_->run({"mag"}, in, {"latent"});
  const float* lat = out[0].GetTensorData<float>();
  const int C = static_cast<int>(out[0].GetTensorTypeAndShapeInfo().GetShape()[2]);
  std::vector<float> f0(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) {
    const float* y = lat + static_cast<size_t>(t) * C;
    const int imax = static_cast<int>(std::max_element(y, y + C) - y);
    const float conf = y[imax];
    float num = 0, den = 0;
    for (int k = imax - 4; k <= imax + 4; ++k) {
      const int kk = std::max(0, std::min(C - 1, k));
      num += cent_table_[kk] * y[kk];
      den += y[kk];
    }
    const float v = conf > 0.006f ? 10.f * std::pow(2.f, (num / den) / 1200.f) : 0.f;
    f0[t] = v < fcpe_f0_min_ ? 0.f : v;
  }
  return f0;
}

// The whole file in one call, as RVC does, up to 5 minutes (~0.8 MB per second of audio).
// FCPE's attention is global, so pieces would shift f0 slightly everywhere (and the NSF phase
// with it): only longer files go in 2-minute pieces that see 10 s on either side.
std::vector<float> OfflinePipeline::fcpe_long(const float* x, int n) {
  const int hop = 160, total = n / hop + 1;
  if (total <= 30000) return fcpe(x, n);
  const int core = 12000, ctx = 1000;
  std::vector<float> f0(static_cast<size_t>(total));
  for (int c0 = 0; c0 < total; c0 += core) {
    const int c1 = std::min(total, c0 + core), a = std::max(0, c0 - ctx);
    const int end = c1 + ctx >= total ? n : (c1 + ctx) * hop;
    std::vector<float> part = fcpe(x + static_cast<size_t>(a) * hop, end - a * hop);
    std::copy(part.begin() + (c0 - a), part.begin() + (c1 - a), f0.begin() + c0);
  }
  return f0;
}

void OfflinePipeline::make_har(const float* f0, int L, bool det, float* har) {
  // SineGen + SourceModuleHnNSF over one segment (same arithmetic as RvcEngine::make_har)
  const float sr = static_cast<float>(tgt_sr_);
  std::vector<float> start(static_cast<size_t>(L), 0.f), rad2(static_cast<size_t>(L));
  for (int i = 0; i < L; ++i) rad2[i] = std::fmod((f0[i] / sr) * static_cast<float>(upp_) + 0.5f, 1.0f) - 0.5f;
  double acc = 0;
  for (int i = 1; i < L; ++i) {
    acc += rad2[i - 1];
    start[i] = std::fmod(static_cast<float>(acc), 1.0f);
  }
  const float two_pi = static_cast<float>(2.0 * M_PI);
  for (int i = 0; i < L; ++i) {
    const float step = f0[i] / sr;
    const bool voiced = f0[i] > 0;
    const float namp = voiced ? 0.003f : static_cast<float>(0.1 / 3);
    float* h = har + static_cast<size_t>(i) * upp_;
    for (int k = 0; k < upp_; ++k) {
      const float rad = step * static_cast<float>(k + 1) + start[i];
      float s = std::sin(two_pi * rad) * 0.1f;
      const float noise = det ? 0.f : normal_(rng_);
      s = (voiced ? s : 0.f) + namp * noise;
      h[k] = std::tanh(l_w_ * s + l_b_);
    }
  }
}

// Vocoder output for frames [k0, k1) of a T-frame segment, in pieces of `chunk` frames that
// see 24 frames on either side (beyond the decoder's receptive field), or whole (chunk 0).
void OfflinePipeline::decode(const std::vector<float>& z, int C, int T, const std::vector<float>& har, int k0, int k1, int chunk,
                             std::vector<float>& out) {
  constexpr int kMargin = 24;
  const bool whole = chunk <= 0;
  if (whole) chunk = k1 - k0;
  std::vector<float> zc, y;
  for (int c0 = k0; c0 < k1; c0 += chunk) {
    const int c1 = std::min(k1, c0 + chunk);
    const int a = whole ? 0 : std::max(0, c0 - kMargin), b = whole ? T : std::min(T, c1 + kMargin), L = b - a;
    zc.resize(static_cast<size_t>(C) * L);
    for (int c = 0; c < C; ++c) memcpy(zc.data() + static_cast<size_t>(c) * L, z.data() + static_cast<size_t>(c) * T + a, sizeof(float) * L);
    const float* h = har.data() + static_cast<size_t>(a) * upp_;
    y.resize(static_cast<size_t>(L) * upp_);
    if (native_dec_) {
      native_dec_->run(zc.data(), L, h, y.data());
    } else {
      std::vector<Ort::Value> din;
      din.push_back(make_tensor<float>(zc.data(), zc.size(), {1, C, L}));
      din.push_back(make_tensor<float>(const_cast<float*>(h), y.size(), {1, 1, static_cast<int64_t>(y.size())}));
      din.push_back(make_tensor<float>(g_.data(), g_.size(), {1, static_cast<int64_t>(g_.size()), 1}));
      auto dout = dec_->run({"z", "har", "g"}, din, {"audio"});
      const float* au = dout[0].GetTensorData<float>();
      std::copy(au, au + y.size(), y.begin());
    }
    out.insert(out.end(), y.begin() + static_cast<ptrdiff_t>(c0 - a) * upp_, y.begin() + static_cast<ptrdiff_t>(c1 - a) * upp_);
    segment_progress(0.3 + 0.7 * (c1 - k0) / std::max(1, k1 - k0));
  }
}

void OfflinePipeline::segment_progress(double f) {
  if (!prog_) return;
  prog_->check();
  prog_->fraction = static_cast<float>(seg_from_ + (seg_to_ - seg_from_) * std::max(0.0, std::min(1.0, f)));
}

// Pipeline.vc: one segment; keeps the frames after `keep` frames of padding on either side
// (the [t_pad_tgt : -t_pad_tgt] slice).
void OfflinePipeline::vc(const float* seg, int n, const int64_t* pitch, const float* pitchf, int np, const OfflineParams& p, int keep,
                         std::vector<float>& out) {
  // HuBERT / ContentVec
  std::vector<Ort::Value> hin;
  hin.push_back(make_tensor<float>(const_cast<float*>(seg), static_cast<size_t>(n), {1, n}));
  auto hout = hubert_->run({"audio"}, hin, {"feats"});
  const auto hs = hout[0].GetTensorTypeAndShapeInfo().GetShape();
  const int F = static_cast<int>(hs[1]), D = static_cast<int>(hs[2]);
  const float* hf = hout[0].GetTensorData<float>();
  std::vector<float> feats(hf, hf + static_cast<size_t>(F) * D);
  hout.clear();
  const bool protect = p.protect < 0.5f;
  std::vector<float> feats0;
  if (protect) feats0 = feats;
  // feature retrieval: k = 8, 1/d^2 weights, over every frame
  if (index_ && p.index_rate != 0 && index_->dim() == D) {
    std::vector<float> blend(feats.size());
    index_->search_blend(feats.data(), F, blend.data(), 8, 0);
    const float r = p.index_rate;
    for (size_t i = 0; i < feats.size(); ++i) feats[i] = blend[i] * r + (1 - r) * feats[i];
  }
  // 50 -> 100 frames/s (nearest), then protect: unvoiced frames lean on the unindexed features
  const int p_len = std::min({n / 160, 2 * F, np});
  if (p_len <= 0) return;
  segment_progress(0.15);
  std::vector<float> phone(static_cast<size_t>(p_len) * D);
  for (int t = 0; t < p_len; ++t) {
    const float* f = feats.data() + static_cast<size_t>(t / 2) * D;
    float* o = phone.data() + static_cast<size_t>(t) * D;
    if (!protect) {
      memcpy(o, f, sizeof(float) * D);
      continue;
    }
    const float w = pitchf[t] < 1 ? p.protect : 1.f, w0 = 1.f - w;
    const float* f0r = feats0.data() + static_cast<size_t>(t / 2) * D;
    for (int d = 0; d < D; ++d) o[d] = f[d] * w + f0r[d] * w0;
  }

  // net_g.infer: enc_p, z_p = m + e^logs * eps * 0.66666, flow^-1, NSF-HiFiGAN
  std::vector<int64_t> cp(pitch, pitch + p_len);
  std::vector<Ort::Value> ein;
  ein.push_back(make_tensor<float>(phone.data(), phone.size(), {1, p_len, D}));
  ein.push_back(make_tensor<int64_t>(cp.data(), cp.size(), {1, p_len}));
  auto eout = enc_->run({"phone", "pitch"}, ein, {"m_p", "logs_p"});
  const float* m_p = eout[0].GetTensorData<float>();
  const float* logs_p = eout[1].GetTensorData<float>();
  const int C = static_cast<int>(eout[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
  std::vector<float> z_p(static_cast<size_t>(C) * p_len);
  for (size_t i = 0; i < z_p.size(); ++i) {
    const float noise = p.deterministic ? 0.f : normal_(rng_);
    z_p[i] = m_p[i] + std::exp(logs_p[i]) * noise * 0.66666f;
  }
  eout.clear();
  std::vector<Ort::Value> fin;
  fin.push_back(make_tensor<float>(z_p.data(), z_p.size(), {1, C, p_len}));
  fin.push_back(make_tensor<float>(g_.data(), g_.size(), {1, static_cast<int64_t>(g_.size()), 1}));
  auto fout = flow_->run({"z_p", "g"}, fin, {"z"});
  const float* zf = fout[0].GetTensorData<float>();
  std::vector<float> z(zf, zf + z_p.size());
  fout.clear();

  segment_progress(0.3);
  std::vector<float> har(static_cast<size_t>(p_len) * upp_);
  make_har(pitchf, p_len, p.deterministic, har.data());
  const int k0 = keep, k1 = p_len - keep;
  if (k1 > k0) decode(z, C, p_len, har, k0, k1, p.dec_chunk, out);
}

OfflineResult OfflinePipeline::run(std::vector<float> audio, const OfflineParams& p,
                                   const std::function<void(const float*, size_t)>& sink, OfflineProgress* prog, float p0, float p1) {
  const auto t_start = Clock::now();
  constexpr int64_t sr = 16000, window = 160;
  const int64_t t_pad = sr * p.x_pad, t_pad2 = 2 * t_pad, t_query = sr * p.x_query, t_center = sr * p.x_center,
                t_max = sr * p.x_max;
  const int64_t N = static_cast<int64_t>(audio.size());
  if (N < sr / 10) throw Error("音频太短（不到 0.1 秒）");
  auto at = [&](double f) { return p0 + (p1 - p0) * std::max(0.0, std::min(1.0, f)); };
  auto progress = [&](double f) {
    if (prog) prog->fraction = static_cast<float>(at(f));
  };
  prog_ = prog;
  struct Unset {
    OfflineProgress*& p;
    ~Unset() { p = nullptr; }
  } unset{prog_};
  auto check = [&] {
    if (prog) prog->check();
  };

  // vc_single: peak at most 0.95
  float peak = 0;
  for (float v : audio) peak = std::max(peak, std::fabs(v));
  const double audio_max = peak / 0.95;
  if (audio_max > 1) {
    const float d = static_cast<float>(audio_max);
    for (float& v : audio) v /= d;
  }
  const std::vector<double> x = highpass_filtfilt(audio);
  std::vector<float>().swap(audio);

  // split points: the quietest 10 ms near every t_center (|x| summed over 160 samples)
  std::vector<int64_t> opt_ts;
  if (N + window > t_max) {
    std::vector<double> mag;  // |audio_pad| around the search range (audio_pad: 80 reflected each side)
    for (int64_t t = t_center; t < N; t += t_center) {
      const int64_t lo = t - t_query, hi = std::min(N, t + t_query);
      mag.resize(static_cast<size_t>(hi - lo + window));
      for (int64_t k = 0; k < hi - lo + window; ++k) mag[k] = std::fabs(x[reflect(lo + k - window / 2, N)]);
      int64_t best = lo;
      double best_v = 0;
      for (int64_t j = lo; j < hi; ++j) {
        const double* m = mag.data() + (j - lo);
        double v = 0;  // summed in numpy's order
        for (int64_t i = 0; i < window; ++i) v += m[i];
        if (j == lo || v < best_v) {
          best_v = v;
          best = j;
        }
      }
      opt_ts.push_back(best);
      check();
    }
  }
  check();

  // t_pad of reflection on either side; FCPE over all of it
  const int64_t Np = N + 2 * t_pad;
  std::vector<float> pad(static_cast<size_t>(Np));
  for (int64_t i = 0; i < Np; ++i) pad[i] = static_cast<float>(x[reflect(i - t_pad, N)]);
  const int64_t p_len_all = Np / window;
  progress(0.02);
  std::vector<float> f0 = fcpe_long(pad.data(), static_cast<int>(Np));
  check();
  // get_f0: fill unvoiced frames, shift, coarse pitch (1..255 on the mel scale)
  interp_unvoiced(f0);
  const float mul = static_cast<float>(std::pow(2.0, p.pitch / 12.0));
  const float mel_min = static_cast<float>(1127.0 * std::log(1.0 + 50.0 / 700.0));
  const float mel_max = static_cast<float>(1127.0 * std::log(1.0 + 1100.0 / 700.0));
  f0.resize(static_cast<size_t>(std::min<int64_t>(p_len_all, static_cast<int64_t>(f0.size()))));
  std::vector<int64_t> pitch(f0.size());
  for (size_t i = 0; i < f0.size(); ++i) {
    f0[i] *= mul;
    float m = 1127.f * std::log(1.f + f0[i] / 700.f);
    if (m > 0) m = (m - mel_min) * 254.f / (mel_max - mel_min) + 1.f;
    if (m <= 1) m = 1;
    if (m > 255) m = 255;
    pitch[i] = static_cast<int64_t>(std::nearbyint(m));
  }
  const int64_t nf0 = static_cast<int64_t>(f0.size());
  progress(0.08);

  OfflineResult r;
  {
    HalfRms in_rms(sr / 2);
    in_rms.add(x.data(), x.size());
    r.rms_in = in_rms.frames();
  }
  HalfRms out_rms(static_cast<size_t>(tgt_sr_ / 2));
  const int keep = 100 * p.x_pad;  // t_pad_tgt in 100 Hz frames
  std::vector<float> y;
  auto segment = [&](int64_t s, int64_t e, int64_t pitch_end) {
    check();
    seg_from_ = at(0.08 + 0.92 * static_cast<double>(s) / static_cast<double>(N));  // kept output: audio [s, e - 2 t_pad)
    seg_to_ = at(0.08 + 0.92 * static_cast<double>(e - t_pad2) / static_cast<double>(N));
    y.clear();
    const int64_t f = s / window, np = std::min(pitch_end, nf0) - f;
    vc(pad.data() + s, static_cast<int>(e - s), pitch.data() + f, f0.data() + f, static_cast<int>(np), p, keep, y);
    if (!y.empty()) sink(y.data(), y.size());
    out_rms.add(y.data(), y.size());
    r.samples += y.size();
    ++r.segments;
    progress(0.08 + 0.92 * static_cast<double>(e - t_pad2) / static_cast<double>(N));
  };
  int64_t s = 0, t = -1;
  for (int64_t t0 : opt_ts) {
    t = t0 / window * window;
    segment(s, std::min(t + t_pad2 + window, Np), (t + t_pad2) / window);
    s = t;
  }
  const int64_t st = t < 0 ? 0 : t;
  segment(st, Np, p_len_all);
  r.rms_out = out_rms.frames();
  r.seconds = ms_since(t_start) / 1000.0;
  progress(1.0);
  return r;
}

// ------------------------------------------------------------------ RmsMix
RmsMix::RmsMix(const OfflineResult& r, float rate) : in_(r.rms_in), out_(r.rms_out), len_(r.samples), rate_(rate) {}

// F.interpolate(mode="linear", align_corners=False) of v to len_ points, at point i
double RmsMix::at(const std::vector<double>& v, size_t i) const {
  const size_t n = v.size();
  const double scale = static_cast<double>(n) / static_cast<double>(len_);
  double src = scale * (static_cast<double>(i) + 0.5) - 0.5;
  if (src < 0) src = 0;
  const size_t i0 = static_cast<size_t>(src), i1 = i0 + (i0 + 1 < n ? 1 : 0);
  const double l1 = src - static_cast<double>(i0);
  return (1.0 - l1) * v[i0] + l1 * v[i1];
}

void RmsMix::apply(float* y, size_t n, size_t pos) const {
  if (!active() || in_.empty() || out_.empty()) return;
  const double e1 = 1.0 - rate_, e2 = static_cast<double>(rate_) - 1.0;
  for (size_t k = 0; k < n; ++k) {
    const size_t i = pos + k;
    const double g = std::pow(at(in_, i), e1) * std::pow(std::max(at(out_, i), 1e-6), e2);
    y[k] = static_cast<float>(y[k] * g);
  }
}

}  // namespace rvc
