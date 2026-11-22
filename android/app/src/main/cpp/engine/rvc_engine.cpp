#include "rvc_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>

#include "common.hpp"
#include "ivf_index.hpp"
#include "native_decoder.hpp"

namespace rvc {

// ------------------------------------------------------------------ helper thread
struct RvcEngine::Worker {
  std::thread th;
  std::mutex mu;
  std::condition_variable cv;
  std::function<void()> job;
  std::exception_ptr err;
  bool pending = false, done = true, quit = false;

  Worker() { th = std::thread([this] { loop(); }); }
  ~Worker() {
    {
      std::lock_guard<std::mutex> lk(mu);
      quit = true;
    }
    cv.notify_all();
    th.join();
  }
  void submit(std::function<void()> f) {
    {
      std::lock_guard<std::mutex> lk(mu);
      job = std::move(f);
      pending = true;
      done = false;
      err = nullptr;
    }
    cv.notify_all();
  }
  void wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return done; });
    if (err) std::rethrow_exception(err);
  }
  void loop() {
    std::unique_lock<std::mutex> lk(mu);
    for (;;) {
      cv.wait(lk, [&] { return pending || quit; });
      if (quit) return;
      auto f = std::move(job);
      pending = false;
      lk.unlock();
      std::exception_ptr e;
      try {
        f();
      } catch (...) {
        e = std::current_exception();
      }
      lk.lock();
      err = e;
      done = true;
      cv.notify_all();
    }
  }
};

// ------------------------------------------------------------------ config
static BackendConfig backend_from_json(const Json& j, const BackendConfig& def) {
  BackendConfig b = def;
  if (!j.is_object()) return b;
  b.backend = backend_from_name(j.str_or("backend", backend_name(def.backend)));
  b.threads = static_cast<int>(j.num_or("threads", def.threads));
  b.fp16 = j.bool_or("fp16", def.fp16);
  b.spinning = j.bool_or("spinning", def.spinning);
  b.cache_dir = j.str_or("cache_dir", def.cache_dir);
  if (j.get("affinity").is_array()) {
    b.affinity.clear();
    for (auto& v : j["affinity"].items()) b.affinity.push_back(v.i32());
  }
  return b;
}

static Json backend_to_json(const BackendConfig& b) {
  Json j = Json::object();
  j.set("backend", backend_name(b.backend));
  j.set("threads", b.threads);
  j.set("fp16", b.fp16);
  j.set("spinning", b.spinning);
  return j;
}

EngineConfig EngineConfig::from_json(const Json& j) {
  EngineConfig c;
  c.sample_rate = static_cast<int>(j.num_or("sample_rate", c.sample_rate));
  c.block_time = j.num_or("block_time", c.block_time);
  c.crossfade_time = j.num_or("crossfade_time", c.crossfade_time);
  c.extra_time = j.num_or("extra_time", c.extra_time);
  c.sid = static_cast<int>(j.num_or("sid", c.sid));
  c.phase_lock = j.bool_or("phase_lock", c.phase_lock);
  c.deterministic = j.bool_or("deterministic", c.deterministic);
  c.index_nprobe = static_cast<int>(j.num_or("index_nprobe", c.index_nprobe));
  c.parallel_f0 = j.bool_or("parallel_f0", c.parallel_f0);
  BackendConfig def;
  def.threads = static_cast<int>(j.num_or("threads", def.threads));
  def.cache_dir = j.str_or("cache_dir", "");
  if (j.get("affinity").is_array())
    for (auto& v : j["affinity"].items()) def.affinity.push_back(v.i32());
  BackendConfig small = def;
  small.threads = std::max(1, std::min(2, def.threads));
  c.hubert = backend_from_json(j.get("hubert"), def);
  c.fcpe = backend_from_json(j.get("fcpe"), small);
  c.synth = backend_from_json(j.get("synth"), def);
  c.dec = backend_from_json(j.get("dec"), def);
  return c;
}

Json EngineConfig::to_json() const {
  Json j = Json::object();
  j.set("sample_rate", sample_rate);
  j.set("block_time", block_time);
  j.set("crossfade_time", crossfade_time);
  j.set("extra_time", extra_time);
  j.set("sid", sid);
  j.set("phase_lock", phase_lock);
  j.set("hubert", backend_to_json(hubert));
  j.set("fcpe", backend_to_json(fcpe));
  j.set("synth", backend_to_json(synth));
  j.set("dec", backend_to_json(dec));
  return j;
}

// ------------------------------------------------------------------ engine
RvcEngine::RvcEngine(std::string assets_dir) : assets_(std::move(assets_dir)) {}
RvcEngine::~RvcEngine() = default;

static bool is_static_backend(Backend b) { return b != Backend::CPU && b != Backend::XNNPACK; }

// A model's graph files are copies of the app's templates, made when the model was imported;
// its weights.bin is laid out as the template's manifest says. When an app update ships
// corrected graphs for the same layout (same manifest), the imported model takes them over:
// nothing has to be imported again.
static void refresh_graphs(const std::string& assets, const std::string& model_dir, const Json& meta) {
  try {
    const std::string tdir = join_path(join_path(assets, "synth"), meta.str_or("template", ""));
    const std::string tman = join_path(tdir, "manifest.json"), mman = join_path(model_dir, "manifest.json");
    if (!file_exists(tman) || !file_exists(mman)) return;
    // same layout = same tensors at the same offsets (other fields of the manifest may differ)
    const Json a = Json::parse(read_file(tman)), b = Json::parse(read_file(mman));
    if (a.num_or("data_size", -1) != b.num_or("data_size", -2) || a["tensors"].dump() != b["tensors"].dump()) return;
    const Json& graphs = meta["graphs"];
    for (const char* k : {"enc", "flow", "dec"}) {
      if (!graphs.get(k).is_string()) continue;
      const std::string name = graphs[k].str(), src = join_path(tdir, name), dst = join_path(model_dir, name);
      if (!file_exists(src)) continue;
      std::string want = read_file(src);
      if (file_exists(dst) && read_file(dst) == want) continue;
      write_file(dst, want);
      RVC_LOGI("model graph %s updated from the app's template", name.c_str());
    }
  } catch (const std::exception& e) {
    RVC_LOGW("graph refresh skipped: %s", e.what());
  }
}


void RvcEngine::load(const std::string& model_dir, const EngineConfig& cfg) {
  loaded_ = false;
  hubert_.reset();
  fcpe_.reset();
  enc_.reset();
  flow_.reset();
  dec_.clear();
  native_dec_.reset();
  index_.reset();
  cfg_ = cfg;
  model_dir_ = model_dir;
  dec_warning_.clear();
  meta_ = Json::parse(read_file(join_path(model_dir, "model.json")));
  refresh_graphs(assets_, model_dir, meta_);
  weights_salt_ = file_fingerprint(join_path(model_dir, "weights.bin"));
  tgt_sr_ = meta_["tgt_sr"].i32();
  upp_ = meta_["upp"].i32();
  phone_dim_ = meta_["phone_dim"].i32();
  l_w_ = static_cast<float>(meta_["l_linear_w"].num());
  l_b_ = static_cast<float>(meta_["l_linear_b"].num());
  int n_spk = meta_["n_spk"].i32(), gin = static_cast<int>(meta_.num_or("gin_channels", 256));
  std::string eg = read_file(join_path(model_dir, "emb_g.bin"));
  if (eg.size() != static_cast<size_t>(n_spk) * gin * 4) throw Error("bad emb_g.bin");
  int sid = std::max(0, std::min(cfg.sid, n_spk - 1));
  g_.assign(reinterpret_cast<const float*>(eg.data()) + static_cast<size_t>(sid) * gin,
            reinterpret_cast<const float*>(eg.data()) + static_cast<size_t>(sid + 1) * gin);

  // sizes (realtime_gui.start_vc)
  int sr = cfg.sample_rate;
  zc_ = sr / 100;
  auto rnd = [&](double t) { return static_cast<int>(std::lround(t * sr / zc_)) * zc_; };
  block_frame_ = std::max(zc_, rnd(cfg.block_time));
  block_frame_16k_ = 160 * block_frame_ / zc_;
  crossfade_frame_ = rnd(cfg.crossfade_time);
  sola_buffer_frame_ = std::min(crossfade_frame_, 4 * zc_);
  sola_search_frame_ = zc_;
  extra_frame_ = rnd(cfg.extra_time);
  int n = extra_frame_ + crossfade_frame_ + sola_search_frame_ + block_frame_;
  input_wav_.assign(static_cast<size_t>(n), 0.f);
  input_wav_res_.assign(static_cast<size_t>(160) * n / zc_, 0.f);
  skip_head_ = extra_frame_ / zc_;
  return_length_ = (block_frame_ + sola_buffer_frame_ + sola_search_frame_) / zc_;
  if (static_cast<int>(input_wav_res_.size()) / 160 > 1024) throw Error("window too long (block+extra must be < 10 s)");
  if (sola_buffer_frame_ <= 0) throw Error("crossfade must be at least 10 ms");
  fade_in_.resize(sola_buffer_frame_);
  fade_out_.resize(sola_buffer_frame_);
  for (int i = 0; i < sola_buffer_frame_; ++i) {
    double x = sola_buffer_frame_ > 1 ? static_cast<double>(i) / (sola_buffer_frame_ - 1) : 0.0;
    double s = std::sin(0.5 * M_PI * x);
    fade_in_[i] = static_cast<float>(s * s);
    fade_out_[i] = 1.f - fade_in_[i];
  }
  gate_ = SpectralGate(sr, 4 * zc_, 0.9f);  // as realtime_gui: TorchGate(sr, n_fft=4*zc, prop_decrease=0.9)
  res_in_ = SincResampler(sr, 16000);
  res_out_ = SincResampler(tgt_sr_, sr);
  res_formant_.clear();

  // models
  Json hman = Json::parse(read_file(join_path(assets_, "hubert/manifest.json")));
  const Json& hg = hman["graphs"][phone_dim_ == 256 ? "hubert_v1" : "hubert_v2"];
  std::string hdir = join_path(assets_, "hubert");
  int N16 = static_cast<int>(input_wav_res_.size());
  hubert_.reset(new OnnxModel(join_path(hdir, hg["fp32"].str()), cfg.hubert, {{"N", N16}}, "hubert_" + std::to_string(phone_dim_),
                              file_fingerprint(join_path(hdir, hman.str_or("data_file", "contentvec.bin")))));
  Json fman = Json::parse(read_file(join_path(assets_, "fcpe/manifest.json")));
  cent_table_.clear();
  for (auto& v : fman["cent_table"].items()) cent_table_.push_back(static_cast<float>(v.num()));
  fcpe_f0_min_ = static_cast<float>(fman["f0_min"].num());
  int n16 = block_frame_16k_ + 800;
  fcpe_.reset(new OnnxModel(join_path(assets_, "fcpe/" + fman["graph"].str()), cfg.fcpe, {{"T", n16 / 160 + 1}}, "fcpe"));
  const Json& g = meta_["graphs"];
  int p_len = N16 / 160;
  int flow_head = std::max(skip_head_ - 24, 0);
  enc_.reset(new OnnxModel(join_path(model_dir, g["enc"].str()), cfg.synth, {{"T", p_len}}, meta_["template"].str() + "_enc", weights_salt_));
  flow_.reset(new OnnxModel(join_path(model_dir, g["flow"].str()), cfg.synth, {{"T", p_len - flow_head}},
                            meta_["template"].str() + "_flow", weights_salt_));
  if (cfg.dec.backend == Backend::NATIVE && file_exists(join_path(model_dir, "manifest.json"))) {
    auto t = Clock::now();
    native_dec_.reset(new NativeDecoder(model_dir, g_, cfg.dec.threads, cfg.dec.fp16, cfg.dec.affinity));
    native_ms_ = ms_since(t);
  } else {
    dec_for(return_length_);
  }
  if (meta_.bool_or("has_index", false) && file_exists(join_path(model_dir, "index.bin")))
    index_.reset(new IvfIndex(join_path(model_dir, "index.bin")));
  worker_.reset(cfg.parallel_f0 ? new Worker() : nullptr);
  reset_state();
  loaded_ = true;
}

OnnxModel* RvcEngine::dec_for(int rl2) {
  if (native_dec_) return nullptr;
  bool fixed = is_static_backend(cfg_.dec.backend);
  int key = fixed ? rl2 : 0;
  auto it = dec_.find(key);
  if (it != dec_.end()) return it->second.get();
  const Json& g = meta_["graphs"];
  std::vector<std::pair<std::string, int64_t>> dims;
  if (fixed) dims = {{"T", rl2}, {"S", static_cast<int64_t>(rl2) * upp_}};
  const std::string path = join_path(model_dir_, g["dec"].str()), tag = meta_["template"].str() + "_dec";
  // An accelerator that cannot take the vocoder: ORT's CPU kernels would be the automatic
  // fallback, but they run it several times slower than the NEON decoder (the vocoder is most
  // of the work), usually too slow for realtime. Use the NEON decoder instead when it can
  // run this model.
  const bool native_ok = fixed && file_exists(join_path(model_dir_, "manifest.json"));
  try {
    auto* m = new OnnxModel(path, cfg_.dec, dims, tag, weights_salt_, !native_ok);
    dec_[key].reset(m);
    return m;
  } catch (const std::exception& e) {
    if (!native_ok) throw;
    auto t = Clock::now();
    native_dec_.reset(new NativeDecoder(model_dir_, g_, cfg_.dec.threads, cfg_.dec.fp16, cfg_.dec.affinity));
    native_ms_ = ms_since(t);
    dec_.clear();
    dec_warning_ = std::string("声码器无法在所选的加速器上运行，已改用 CPU 原生 NEON（") + (native_dec_->fp16() ? "fp16" : "fp32") +
                   "）。加速器给出的原因：\n" + e.what();
    RVC_LOGW("vocoder: accelerator unavailable, using the native decoder");
    return nullptr;
  }
}

void RvcEngine::reset_state() {
  std::fill(input_wav_.begin(), input_wav_.end(), 0.f);
  std::fill(input_wav_res_.begin(), input_wav_res_.end(), 0.f);
  input_wav_denoise_.assign(input_wav_.size(), 0.f);
  output_buffer_.assign(input_wav_.size(), 0.f);
  nr_buffer_.assign(static_cast<size_t>(sola_buffer_frame_), 0.f);
  rms_buffer_.assign(static_cast<size_t>(4 * zc_), 0.f);
  sola_buffer_.assign(static_cast<size_t>(sola_buffer_frame_), 0.f);
  cache_pitch_.assign(1024, 0);
  cache_pitchf_.assign(1024, 0.f);
  phase_frames_ = 0;
  phase_acc_.clear();
  rng_.seed(1234);
  std::lock_guard<std::mutex> lk(stats_mu_);
  stats_ = BlockStats();
}

double RvcEngine::algorithmic_latency() const {
  return static_cast<double>(block_frame_ + sola_buffer_frame_ + sola_search_frame_) / cfg_.sample_rate;
}

BlockStats RvcEngine::stats() const {
  std::lock_guard<std::mutex> lk(stats_mu_);
  return stats_;
}

Json RvcEngine::info() const {
  Json j = Json::object();
  j.set("loaded", loaded_);
  if (!loaded_) return j;
  j.set("model", meta_.str_or("name", ""));
  j.set("version", meta_.str_or("version", ""));
  j.set("tgt_sr", tgt_sr_);
  j.set("block_frame", block_frame_);
  j.set("sample_rate", cfg_.sample_rate);
  j.set("latency_ms", algorithmic_latency() * 1000.0);
  j.set("has_index", index_ != nullptr);
  Json be = Json::object(), warn = Json::array(), load = Json::object();
  auto add = [&](const char* k, const OnnxModel* m) {
    if (!m) return;
    be.set(k, m->backend_used());
    load.set(k, m->load_ms());
    if (!m->warning().empty()) warn.push(m->warning());
  };
  add("hubert", hubert_.get());
  add("fcpe", fcpe_.get());
  add("enc", enc_.get());
  add("flow", flow_.get());
  if (native_dec_) {
    be.set("dec", native_dec_->fp16() ? "native_fp16" : "native_fp32");
    load.set("dec", native_ms_);
    if (!dec_warning_.empty()) warn.push(dec_warning_);
  } else if (!dec_.empty()) {
    add("dec", dec_.begin()->second.get());
  }
  j.set("backends", be);
  j.set("load_ms", load);
  j.set("warnings", warn);
  return j;
}

// ------------------------------------------------------------------ f0
std::vector<float> RvcEngine::run_fcpe(const float* x16, int n) {
  std::vector<float> mag;
  stft_.process(x16, n, mag);
  int T = stft_.frames(n), nb = stft_.bins();
  std::vector<Ort::Value> in;
  in.push_back(make_tensor<float>(mag.data(), mag.size(), {1, T, nb}));
  auto out = fcpe_->run({"mag"}, in, {"latent"});
  const float* lat = out[0].GetTensorData<float>();
  int C = static_cast<int>(out[0].GetTensorTypeAndShapeInfo().GetShape()[2]);
  std::vector<float> f0(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) {
    const float* y = lat + static_cast<size_t>(t) * C;
    int imax = static_cast<int>(std::max_element(y, y + C) - y);
    float conf = y[imax];
    float num = 0, den = 0;
    for (int k = imax - 4; k <= imax + 4; ++k) {
      int kk = std::max(0, std::min(C - 1, k));
      num += cent_table_[kk] * y[kk];
      den += y[kk];
    }
    float v = conf > 0.006f ? 10.f * std::pow(2.f, (num / den) / 1200.f) : 0.f;
    f0[t] = v < fcpe_f0_min_ ? 0.f : v;
  }
  return f0;
}

// ------------------------------------------------------------------ NSF source
void RvcEngine::make_har(const float* f0, int L, int64_t start_frame, float* har) {
  const float sr = static_cast<float>(tgt_sr_);
  std::vector<float> start(static_cast<size_t>(L), 0.f);
  std::vector<float> rad2(static_cast<size_t>(L));
  for (int i = 0; i < L; ++i) rad2[i] = std::fmod((f0[i] / sr) * static_cast<float>(upp_) + 0.5f, 1.0f) - 0.5f;
  if (cfg_.phase_lock) {
    auto it = phase_acc_.find(start_frame);
    double ph = it == phase_acc_.end() ? 0.0 : it->second;
    for (int i = 0; i < L; ++i) {
      start[i] = static_cast<float>(ph);
      ph = std::fmod(ph + rad2[i], 1.0);
      phase_acc_[start_frame + i + 1] = ph;
    }
    phase_acc_.erase(phase_acc_.begin(), phase_acc_.lower_bound(start_frame));
  } else {
    double acc = 0;
    for (int i = 1; i < L; ++i) {
      acc += rad2[i - 1];
      start[i] = std::fmod(static_cast<float>(acc), 1.0f);
    }
  }
  const float two_pi = static_cast<float>(2.0 * M_PI);
  for (int i = 0; i < L; ++i) {
    float step = f0[i] / sr;
    bool voiced = f0[i] > 0;
    float namp = voiced ? 0.003f : static_cast<float>(0.1 / 3);
    float* h = har + static_cast<size_t>(i) * upp_;
    for (int k = 0; k < upp_; ++k) {
      float rad = step * static_cast<float>(k + 1) + start[i];
      float s = std::sin(two_pi * rad) * 0.1f;
      float noise = cfg_.deterministic ? 0.f : normal_(rng_);
      s = (voiced ? s : 0.f) + namp * noise;
      h[k] = std::tanh(l_w_ * s + l_b_);
    }
  }
}

// ------------------------------------------------------------------ rtrvc.RVC.infer
std::vector<float> RvcEngine::infer(BlockStats& st) {
  const int N16 = static_cast<int>(input_wav_res_.size());
  const float* x = input_wav_res_.data();
  const int n16 = block_frame_16k_ + 800;
  std::vector<float> f0raw;
  auto f0job = [&] {
    auto t = Clock::now();
    f0raw = run_fcpe(x + N16 - n16, n16);
    st.f0_ms = ms_since(t);
  };
  if (worker_) worker_->submit(f0job);

  // HuBERT / ContentVec
  auto t0 = Clock::now();
  std::vector<Ort::Value> hin;
  hin.push_back(make_tensor<float>(const_cast<float*>(x), static_cast<size_t>(N16), {1, N16}));
  auto hout = hubert_->run({"audio"}, hin, {"feats"});
  auto hs = hout[0].GetTensorTypeAndShapeInfo().GetShape();
  const int F = static_cast<int>(hs[1]), D = static_cast<int>(hs[2]);
  std::vector<float> feats(static_cast<size_t>(F + 1) * D);
  memcpy(feats.data(), hout[0].GetTensorData<float>(), sizeof(float) * F * D);
  memcpy(feats.data() + static_cast<size_t>(F) * D, feats.data() + static_cast<size_t>(F - 1) * D, sizeof(float) * D);
  hout.clear();
  st.hubert_ms = ms_since(t0);

  if (worker_) worker_->wait();
  else f0job();

  // feature retrieval
  float index_rate = params.index_rate.load();
  if (index_ && index_rate > 0 && index_->dim() == D) {
    auto t = Clock::now();
    int s = skip_head_ / 2, rows = F + 1 - s;
    if (rows > 0) {
      std::vector<float> blend(static_cast<size_t>(rows) * D);
      index_->search_blend(feats.data() + static_cast<size_t>(s) * D, rows, blend.data(), 8, cfg_.index_nprobe);
      float* f = feats.data() + static_cast<size_t>(s) * D;
      for (size_t i = 0; i < blend.size(); ++i) f[i] = blend[i] * index_rate + (1 - index_rate) * f[i];
    }
    st.index_ms = ms_since(t);
  }

  // pitch: get_f0_fcpe + get_f0_post + realtime cache
  const int p_len = N16 / 160;
  const float formant = params.formant.load();
  const double factor = std::pow(2.0, formant / 12.0);
  const int rl = return_length_;
  const int rl2 = static_cast<int>(std::ceil(rl * factor));
  {
    std::vector<float>& f0 = f0raw;
    const int L = static_cast<int>(f0.size());
    std::vector<int> voiced;
    for (int i = 0; i < L; ++i) if (f0[i] != 0) voiced.push_back(i);
    if (!voiced.empty() && static_cast<int>(voiced.size()) < L) {
      size_t vi = 0;
      for (int i = 0; i < L; ++i) {
        if (f0[i] != 0) continue;
        while (vi < voiced.size() && voiced[vi] < i) ++vi;
        if (vi == 0) f0[i] = f0[voiced[0]];
        else if (vi >= voiced.size()) f0[i] = f0[voiced.back()];
        else {
          int a = voiced[vi - 1], b = voiced[vi];
          double w = static_cast<double>(i - a) / (b - a);
          f0[i] = static_cast<float>(f0[a] + (f0[b] - f0[a]) * w);
        }
      }
    }
    const float key = params.pitch.load() - formant;
    const float mul = static_cast<float>(std::pow(2.0, key / 12.0));
    const float mel_min = static_cast<float>(1127.0 * std::log(1.0 + 50.0 / 700.0));
    const float mel_max = static_cast<float>(1127.0 * std::log(1.0 + 1100.0 / 700.0));
    std::vector<int64_t> pitch(static_cast<size_t>(L));
    for (int i = 0; i < L; ++i) {
      f0[i] *= mul;
      float m = 1127.f * std::log(1.f + f0[i] / 700.f);
      if (m > 0) m = (m - mel_min) * 254.f / (mel_max - mel_min) + 1.f;
      if (m <= 1) m = 1;
      if (m > 255) m = 255;
      pitch[i] = static_cast<int64_t>(std::nearbyint(m));
    }
    const int shift = block_frame_16k_ / 160;
    std::memmove(cache_pitch_.data(), cache_pitch_.data() + shift, sizeof(int64_t) * (1024 - shift));
    std::memmove(cache_pitchf_.data(), cache_pitchf_.data() + shift, sizeof(float) * (1024 - shift));
    const int w = L - 4;
    for (int i = 0; i < w; ++i) {
      cache_pitch_[1024 - w + i] = pitch[3 + i];
      cache_pitchf_[1024 - w + i] = f0[3 + i];
    }
  }

  // synthesizer: enc_p
  t0 = Clock::now();
  std::vector<float> phone(static_cast<size_t>(p_len) * D);
  for (int t = 0; t < p_len; ++t) memcpy(phone.data() + static_cast<size_t>(t) * D, feats.data() + static_cast<size_t>(t / 2) * D, sizeof(float) * D);
  std::vector<int64_t> cp(cache_pitch_.end() - p_len, cache_pitch_.end());
  std::vector<Ort::Value> ein;
  ein.push_back(make_tensor<float>(phone.data(), phone.size(), {1, p_len, D}));
  ein.push_back(make_tensor<int64_t>(cp.data(), cp.size(), {1, p_len}));
  auto eout = enc_->run({"phone", "pitch"}, ein, {"m_p", "logs_p"});
  const float* m_p = eout[0].GetTensorData<float>();
  const float* logs_p = eout[1].GetTensorData<float>();
  const int C = static_cast<int>(eout[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
  const int flow_head = std::max(skip_head_ - 24, 0), dec_head = skip_head_ - flow_head, Tf = p_len - flow_head;
  std::vector<float> z_p(static_cast<size_t>(C) * Tf);
  for (int c = 0; c < C; ++c) {
    for (int t = 0; t < Tf; ++t) {
      size_t si = static_cast<size_t>(c) * p_len + flow_head + t;
      float noise = cfg_.deterministic ? 0.f : normal_(rng_);
      z_p[static_cast<size_t>(c) * Tf + t] = m_p[si] + std::exp(logs_p[si]) * noise * 0.66666f;
    }
  }
  eout.clear();
  st.enc_ms = ms_since(t0);

  // flow (reverse)
  t0 = Clock::now();
  std::vector<Ort::Value> fin;
  fin.push_back(make_tensor<float>(z_p.data(), z_p.size(), {1, C, Tf}));
  fin.push_back(make_tensor<float>(g_.data(), g_.size(), {1, static_cast<int64_t>(g_.size()), 1}));
  auto fout = flow_->run({"z_p", "g"}, fin, {"z"});
  const float* zf = fout[0].GetTensorData<float>();
  std::vector<float> z(static_cast<size_t>(C) * rl);
  for (int c = 0; c < C; ++c) memcpy(z.data() + static_cast<size_t>(c) * rl, zf + static_cast<size_t>(c) * Tf + dec_head, sizeof(float) * rl);
  fout.clear();
  st.flow_ms = ms_since(t0);

  // NSF source + decoder
  t0 = Clock::now();
  std::vector<float> nsff0(static_cast<size_t>(rl));
  for (int i = 0; i < rl; ++i) nsff0[i] = cache_pitchf_[1024 - p_len + skip_head_ + i] * static_cast<float>(rl2) / static_cast<float>(rl);
  std::vector<float> har(static_cast<size_t>(rl) * upp_);
  make_har(nsff0.data(), rl, phase_frames_ + skip_head_, har.data());
  if (rl2 != rl) {
    std::vector<float> h2(static_cast<size_t>(rl2) * upp_), z2(static_cast<size_t>(C) * rl2);
    interp_linear_rows(har.data(), 1, rl * upp_, h2.data(), rl2 * upp_);
    interp_linear_rows(z.data(), C, rl, z2.data(), rl2);
    har.swap(h2);
    z.swap(z2);
  }
  std::vector<float> audio(static_cast<size_t>(rl2) * upp_);
  OnnxModel* dec = dec_for(rl2);  // null: the native decoder is in use (possibly as of this call)
  if (!dec) {
    native_dec_->run(z.data(), rl2, har.data(), audio.data());
  } else {
    std::vector<Ort::Value> din;
    din.push_back(make_tensor<float>(z.data(), z.size(), {1, C, rl2}));
    din.push_back(make_tensor<float>(har.data(), har.size(), {1, 1, static_cast<int64_t>(rl2) * upp_}));
    din.push_back(make_tensor<float>(g_.data(), g_.size(), {1, static_cast<int64_t>(g_.size()), 1}));
    auto dout = dec->run({"z", "har", "g"}, din, {"audio"});
    const float* au = dout[0].GetTensorData<float>();
    std::copy(au, au + audio.size(), audio.begin());
  }
  st.dec_ms = ms_since(t0);

  const int upp_res = static_cast<int>(std::floor(factor * tgt_sr_ / 100.0));
  if (upp_res != tgt_sr_ / 100) {
    auto it = res_formant_.find(upp_res);
    if (it == res_formant_.end()) it = res_formant_.emplace(upp_res, SincResampler(upp_res, tgt_sr_ / 100)).first;
    int nuse = std::min(static_cast<int>(audio.size()), rl * upp_res);
    audio = it->second(audio.data(), nuse);
  }
  return audio;
}

// ------------------------------------------------------------------ realtime_gui.audio_callback
void RvcEngine::process_block(const float* in, float* out) {
  if (!loaded_) throw Error("engine not loaded");
  BlockStats st;
  auto t_all = Clock::now();
  const int bf = block_frame_, zc = zc_;
  std::vector<float> indata(in, in + bf);
  const float thr = params.threshold.load();
  if (thr > -60) {
    std::vector<float> buf(rms_buffer_);
    buf.insert(buf.end(), indata.begin(), indata.end());
    std::vector<float> rms = rms_frames(buf.data(), static_cast<int>(buf.size()), 4 * zc, zc);
    std::copy(buf.end() - 4 * zc, buf.end(), rms_buffer_.begin());
    std::vector<float> data(buf.begin() + (2 * zc - zc / 2), buf.end());
    for (size_t i = 2; i < rms.size(); ++i) {
      float db = 20.f * std::log10(std::max(rms[i], 1e-5f));
      if (db < thr) {
        size_t a = (i - 2) * zc, b = std::min(data.size(), a + zc);
        for (size_t k = a; k < b; ++k) data[k] = 0;
      }
    }
    indata.assign(data.begin() + zc / 2, data.end());
  }
  const int len = static_cast<int>(indata.size());
  std::memmove(input_wav_.data(), input_wav_.data() + bf, sizeof(float) * (input_wav_.size() - bf));
  std::copy(indata.begin(), indata.end(), input_wav_.end() - len);
  std::memmove(input_wav_res_.data(), input_wav_res_.data() + block_frame_16k_, sizeof(float) * (input_wav_res_.size() - block_frame_16k_));
  const bool inr = params.input_nr.load();
  const int sb = sola_buffer_frame_;
  if (inr) {
    // input noise reduction (realtime_gui I_noise_reduce)
    std::memmove(input_wav_denoise_.data(), input_wav_denoise_.data() + bf, sizeof(float) * (input_wav_denoise_.size() - bf));
    std::vector<float> iw = gate_.process(input_wav_.data() + input_wav_.size() - sb - bf, sb + bf, input_wav_.data(),
                                          static_cast<int>(input_wav_.size()));
    for (int i = 0; i < sb; ++i) iw[i] = iw[i] * fade_in_[i] + nr_buffer_[i] * fade_out_[i];
    std::copy(iw.begin(), iw.begin() + bf, input_wav_denoise_.end() - bf);
    std::copy(iw.begin() + bf, iw.begin() + bf + sb, nr_buffer_.begin());
    int rn = bf + 2 * zc;
    std::vector<float> res = res_in_(input_wav_denoise_.data() + input_wav_denoise_.size() - rn, rn);
    int nw = block_frame_16k_ + 160;
    std::copy(res.begin() + 160, res.begin() + 160 + nw, input_wav_res_.end() - nw);
  } else {
    int rn = len + 2 * zc;
    std::vector<float> res = res_in_(input_wav_.data() + input_wav_.size() - rn, rn);
    int nw = 160 * (len / zc + 1);
    std::copy(res.begin() + 160, res.begin() + 160 + nw, input_wav_res_.end() - nw);
  }

  std::vector<float> infer_wav = infer(st);
  phase_frames_ += block_frame_ / zc_;
  auto t_post = Clock::now();
  if (!res_out_.identity()) infer_wav = res_out_(infer_wav.data(), static_cast<int>(infer_wav.size()));
  if (params.output_nr.load()) {
    // output noise reduction (realtime_gui O_noise_reduce)
    std::memmove(output_buffer_.data(), output_buffer_.data() + bf, sizeof(float) * (output_buffer_.size() - bf));
    std::copy(infer_wav.end() - bf, infer_wav.end(), output_buffer_.end() - bf);
    infer_wav = gate_.process(infer_wav.data(), static_cast<int>(infer_wav.size()), output_buffer_.data(),
                              static_cast<int>(output_buffer_.size()));
  }
  const int n = static_cast<int>(infer_wav.size());

  const float rmr = params.rms_mix_rate.load();
  if (rmr < 1) {
    const float* iw = (inr ? input_wav_denoise_.data() : input_wav_.data()) + extra_frame_;
    int niw = std::min(n, static_cast<int>(input_wav_.size()) - extra_frame_);
    std::vector<float> r1 = interp_linear_align(rms_frames(iw, niw, 4 * zc, zc), n + 1);
    std::vector<float> r2 = interp_linear_align(rms_frames(infer_wav.data(), n, 4 * zc, zc), n + 1);
    const float e = 1.f - rmr;
    for (int i = 0; i < n; ++i) infer_wav[i] *= std::pow(r1[i] / std::max(r2[i], 1e-3f), e);
  }

  // SOLA (DDSP-SVC)
  const int ss = sola_search_frame_;
  int best = 0;
  double best_v = -1e300;
  for (int k = 0; k <= ss && k + sb <= n; ++k) {
    double nom = 0, den = 0;
    for (int i = 0; i < sb; ++i) {
      double v = infer_wav[k + i];
      nom += v * sola_buffer_[i];
      den += v * v;
    }
    double r = nom / std::sqrt(den + 1e-8);
    if (r > best_v) {
      best_v = r;
      best = k;
    }
  }
  float* w = infer_wav.data() + best;
  for (int i = 0; i < sb; ++i) w[i] = w[i] * fade_in_[i] + sola_buffer_[i] * fade_out_[i];
  std::copy(w + bf, w + bf + sb, sola_buffer_.begin());
  std::copy(w, w + bf, out);

  st.post_ms = ms_since(t_post);
  st.total_ms = ms_since(t_all);
  st.sola_offset = best;
  std::lock_guard<std::mutex> lk(stats_mu_);
  st.blocks = stats_.blocks + 1;
  stats_ = st;
}

}  // namespace rvc
