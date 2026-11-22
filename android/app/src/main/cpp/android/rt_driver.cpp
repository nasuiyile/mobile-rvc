#include "rt_driver.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/system_properties.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>

#include "../engine/common.hpp"
#include "../engine/rvc_engine.hpp"
#include "../engine/wav.hpp"

namespace rvc {

// ------------------------------------------------------------------ ADPF (Android 13+), loaded lazily
namespace {
struct Adpf {
  void* (*get_manager)() = nullptr;
  void* (*create)(void*, const int32_t*, size_t, int64_t) = nullptr;
  int (*report)(void*, int64_t) = nullptr;
  int (*update)(void*, int64_t) = nullptr;
  void (*close)(void*) = nullptr;
  int (*workload_increase)(void*, bool, bool, const char*) = nullptr;  // Android 16+
  bool ok = false;
  Adpf() {
    void* h = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) return;
    get_manager = reinterpret_cast<void* (*)()>(dlsym(h, "APerformanceHint_getManager"));
    create = reinterpret_cast<void* (*)(void*, const int32_t*, size_t, int64_t)>(dlsym(h, "APerformanceHint_createSession"));
    report = reinterpret_cast<int (*)(void*, int64_t)>(dlsym(h, "APerformanceHint_reportActualWorkDuration"));
    update = reinterpret_cast<int (*)(void*, int64_t)>(dlsym(h, "APerformanceHint_updateTargetWorkDuration"));
    close = reinterpret_cast<void (*)(void*)>(dlsym(h, "APerformanceHint_closeSession"));
    workload_increase = reinterpret_cast<int (*)(void*, bool, bool, const char*)>(
        dlsym(h, "APerformanceHint_notifyWorkloadIncrease"));
    ok = get_manager && create && report && close;
  }
};
Adpf& adpf() {
  static Adpf a;
  return a;
}
}  // namespace

// ------------------------------------------------------------------ CPU topology
static long read_long(const std::string& path) {
  std::ifstream f(path);
  long v = -1;
  if (f) f >> v;
  return v;
}

std::vector<int> big_cores() {
  int n = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
  std::vector<std::pair<long, int>> f;
  long mx = 0;
  for (int i = 0; i < n; ++i) {
    long v = read_long("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq");
    f.emplace_back(v, i);
    mx = std::max(mx, v);
  }
  std::vector<int> big;
  for (auto& p : f)
    if (mx <= 0 || p.first >= mx * 0.7) big.push_back(p.second);
  if (big.empty())
    for (int i = 0; i < n; ++i) big.push_back(i);
  return big;
}

Json cpu_info() {
  Json j = Json::object();
  int n = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
  Json freqs = Json::array();
  for (int i = 0; i < n; ++i)
    freqs.push(Json(static_cast<int64_t>(read_long("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq"))));
  j.set("cores", n);
  j.set("max_freq_khz", freqs);
  Json big = Json::array();
  for (int c : big_cores()) big.push(c);
  j.set("big_cores", big);
  return j;
}

// ------------------------------------------------------------------ options
DriverOptions DriverOptions::from_json(const Json& j) {
  DriverOptions o;
  o.local_input = j.bool_or("local_input", o.local_input);
  o.local_output = j.bool_or("local_output", o.local_output);
  o.input_preset = static_cast<int>(j.num_or("input_preset", o.input_preset));
  o.input_exclusive = j.bool_or("input_exclusive", o.input_exclusive);
  o.passthrough = j.bool_or("passthrough", o.passthrough);
  o.output_gain = static_cast<float>(j.num_or("output_gain", o.output_gain));
  o.max_queue_blocks = static_cast<int>(j.num_or("max_queue_blocks", o.max_queue_blocks));
  o.sample_rate = static_cast<int>(j.num_or("sample_rate", o.sample_rate));
  o.debug_input = j.str_or("debug_input", "");
  o.virtual_capture = j.bool_or("virtual_capture", o.virtual_capture);
  o.warmup = j.bool_or("warmup", o.warmup);
  return o;
}

// ------------------------------------------------------------------ driver
RealtimeDriver::RealtimeDriver(RvcEngine* engine) : engine_(engine) {}

void RealtimeDriver::set_error(const std::string& e) {
  std::lock_guard<std::mutex> lk(err_mu_);
  last_error_ = e;
}
RealtimeDriver::~RealtimeDriver() { stop(); }

const float* RealtimeDriver::input_samples(const float* captured, size_t n) {
  if (debug_wav_.empty() || n > scratch_.size()) return captured;
  size_t p = debug_pos_.fetch_add(n) % debug_wav_.size();
  for (size_t i = 0; i < n; ++i) scratch_[i] = debug_wav_[(p + i) % debug_wav_.size()];
  return scratch_.data();
}

aaudio_data_callback_result_t RealtimeDriver::in_cb(AAudioStream*, void* user, void* audio, int32_t frames) {
  auto* d = static_cast<RealtimeDriver*>(user);
  d->in_cb_frames_.fetch_add(frames, std::memory_order_relaxed);
  int m = d->capture_mode_.load(std::memory_order_relaxed);
  if (m == 0 || m == 2) return AAUDIO_CALLBACK_RESULT_CONTINUE;  // virtual mic idle / shell captures
  const float* src = d->input_samples(static_cast<const float*>(audio), static_cast<size_t>(frames));
  size_t w = d->in_ring_.write(src, static_cast<size_t>(frames));
  if (w < static_cast<size_t>(frames)) d->in_overflows_.fetch_add(1, std::memory_order_relaxed);
  return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

aaudio_data_callback_result_t RealtimeDriver::out_cb(AAudioStream*, void* user, void* audio, int32_t frames) {
  auto* d = static_cast<RealtimeDriver*>(user);
  d->out_cb_calls_.fetch_add(1, std::memory_order_relaxed);
  float* out = static_cast<float*>(audio);
  // bound latency: never keep more than max_queue_blocks blocks queued
  size_t cap = static_cast<size_t>(d->block_) * std::max(1, d->opt_.max_queue_blocks);
  size_t avail = d->out_ring_.available();
  if (avail > cap + static_cast<size_t>(frames)) {
    size_t drop = avail - cap;
    d->out_ring_.skip(drop);
    d->dropped_.fetch_add(static_cast<int64_t>(drop), std::memory_order_relaxed);
  }
  size_t r = d->out_ring_.read(out, static_cast<size_t>(frames));
  if (r < static_cast<size_t>(frames)) {
    std::fill(out + r, out + frames, 0.f);
    if (d->blocks_.load(std::memory_order_relaxed) > 1) d->out_underruns_.fetch_add(1, std::memory_order_relaxed);
  }
  return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

void RealtimeDriver::err_cb(AAudioStream*, void* user, aaudio_result_t err) {
  auto* d = static_cast<RealtimeDriver*>(user);
  RVC_LOGW("aaudio error %s", AAudio_convertResultToText(err));
  d->restart_.store(true);  // reopened from the worker thread (not allowed inside the callback)
}

AAudioStream* RealtimeDriver::open_stream(bool input) {
  AAudioStreamBuilder* b = nullptr;
  if (AAudio_createStreamBuilder(&b) != AAUDIO_OK) throw Error("AAudio unavailable");
  AAudioStreamBuilder_setDirection(b, input ? AAUDIO_DIRECTION_INPUT : AAUDIO_DIRECTION_OUTPUT);
  AAudioStreamBuilder_setSampleRate(b, sr_);
  AAudioStreamBuilder_setChannelCount(b, 1);
  AAudioStreamBuilder_setFormat(b, AAUDIO_FORMAT_PCM_FLOAT);
  AAudioStreamBuilder_setPerformanceMode(b, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
  AAudioStreamBuilder_setSharingMode(b, (!input || opt_.input_exclusive) ? AAUDIO_SHARING_MODE_EXCLUSIVE
                                                                       : AAUDIO_SHARING_MODE_SHARED);
  if (input) {
    AAudioStreamBuilder_setInputPreset(b, opt_.input_preset);
    AAudioStreamBuilder_setDataCallback(b, &RealtimeDriver::in_cb, this);
  } else {
    AAudioStreamBuilder_setUsage(b, AAUDIO_USAGE_MEDIA);
    AAudioStreamBuilder_setContentType(b, AAUDIO_CONTENT_TYPE_SPEECH);
    AAudioStreamBuilder_setDataCallback(b, &RealtimeDriver::out_cb, this);
  }
  AAudioStreamBuilder_setErrorCallback(b, &RealtimeDriver::err_cb, this);
  AAudioStream* s = nullptr;
  aaudio_result_t r = AAudioStreamBuilder_openStream(b, &s);
  AAudioStreamBuilder_delete(b);
  if (r != AAUDIO_OK) throw Error(std::string("open ") + (input ? "input" : "output") + " stream: " + AAudio_convertResultToText(r));
  if (AAudioStream_getSampleRate(s) != sr_) RVC_LOGW("stream rate %d != %d", AAudioStream_getSampleRate(s), sr_);
  int32_t burst = AAudioStream_getFramesPerBurst(s);
  AAudioStream_setBufferSizeInFrames(s, burst * 2);
  if (input) { in_burst_ = burst; in_rate_ = AAudioStream_getSampleRate(s); in_exclusive_ = AAudioStream_getSharingMode(s) == AAUDIO_SHARING_MODE_EXCLUSIVE; }
  else { out_burst_ = burst; out_rate_ = AAudioStream_getSampleRate(s); out_exclusive_ = AAudioStream_getSharingMode(s) == AAUDIO_SHARING_MODE_EXCLUSIVE; }
  return s;
}

bool RealtimeDriver::input_should_run() const {
  int m = capture_mode_.load();
  return m == 1 || (m < 0 && opt_.local_input);
}

static int debug_prop(const char* name) {
  char v[PROP_VALUE_MAX] = {0};
  return __system_property_get(name, v) > 0 ? atoi(v) : 0;
}

static int64_t steady_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

// Starts the capture stream. A stream that was disconnected while it was stopped (an audio
// route change while idle: nobody tells a stopped stream) does not start any more; it is then
// replaced by a new one. Returns false, with the error set, when no stream could be started.
bool RealtimeDriver::start_input_locked() {
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (!in_) {
      try {
        in_ = open_stream(true);
      } catch (const std::exception& e) {
        set_error(e.what());
        RVC_LOGE("%s", e.what());
        return false;
      }
    }
    aaudio_result_t r;
    if (dbg_fail_start_ > 0) {
      --dbg_fail_start_;
      r = AAUDIO_ERROR_DISCONNECTED;
    } else if (dbg_mute_start_ > 0) {
      --dbg_mute_start_;
      r = AAUDIO_OK;  // pretend: the stream stays stopped and delivers nothing
    } else {
      r = AAudioStream_requestStart(in_);
    }
    if (r == AAUDIO_OK) return true;
    RVC_LOGW("capture stream did not start (%s): replacing it", AAudio_convertResultToText(r));
    AAudioStream_close(in_);
    in_ = nullptr;
    input_reopens_.fetch_add(1);
  }
  set_error("microphone stream cannot be started");
  return false;
}

bool RealtimeDriver::start_output_locked() {
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (!out_) {
      try {
        out_ = open_stream(false);
      } catch (const std::exception& e) {
        set_error(e.what());
        RVC_LOGE("%s", e.what());
        return false;
      }
    }
    aaudio_result_t r = AAudioStream_requestStart(out_);
    if (r == AAUDIO_OK) return true;
    RVC_LOGW("playback stream did not start (%s): replacing it", AAudio_convertResultToText(r));
    AAudioStream_close(out_);
    out_ = nullptr;
    output_reopens_.fetch_add(1);
  }
  set_error("playback stream cannot be started");
  return false;
}

// Opens whatever is missing. The two streams are independent: a failure of one must not keep
// the other from coming (back) up.
void RealtimeDriver::open_streams() {
  std::lock_guard<std::mutex> lk(stream_mu_);
  bool ok = true;
  if (opt_.local_output && !out_) ok = start_output_locked() && ok;
  // virtual-mic mode keeps the input opened but stopped, so waking up only needs a start
  if ((opt_.local_input || opt_.virtual_capture) && !in_) {
    if (input_should_run()) {
      ok = start_input_locked() && ok;
    } else {
      try {
        in_ = open_stream(true);
      } catch (const std::exception& e) {
        set_error(e.what());
        ok = false;
      }
    }
  }
  if (!ok) {
    std::lock_guard<std::mutex> el(err_mu_);
    throw Error(last_error_.empty() ? "audio stream error" : last_error_);
  }
}

// Streams can die without telling anyone: replace those that should be delivering but are
// not. Called from the worker every 50 ms.
void RealtimeDriver::watch_streams() {
  const auto now = Clock::now();
  auto since = [&](Clock::time_point t) { return std::chrono::duration<double, std::milli>(now - t).count(); };
  // capture: the callback must keep coming while the stream is supposed to run
  const int64_t nin = in_cb_frames_.load(std::memory_order_relaxed);
  if (!input_should_run() || nin != wd_in_frames_) {
    wd_in_frames_ = nin;
    wd_in_time_ = now;
    wd_in_tries_ = 0;
  } else if (since(wd_in_time_) > (wd_in_tries_ == 0 ? 500.0 : 1500.0)) {
    RVC_LOGW("no microphone data for %.0f ms: replacing the capture stream (attempt %d)", since(wd_in_time_), wd_in_tries_ + 1);
    {
      std::lock_guard<std::mutex> lk(stream_mu_);
      if (in_) {
        AAudioStream_requestStop(in_);
        AAudioStream_close(in_);
        in_ = nullptr;
      }
      if (input_should_run() && start_input_locked()) set_error("");
    }
    input_reopens_.fetch_add(1);
    wd_in_time_ = Clock::now();
    ++wd_in_tries_;
  }
  // playback (monitor): its callback runs all the time, data or not
  if (opt_.local_output) {
    const int64_t nout = out_cb_calls_.load(std::memory_order_relaxed);
    if (nout != wd_out_calls_) {
      wd_out_calls_ = nout;
      wd_out_time_ = now;
      wd_out_tries_ = 0;
    } else if (since(wd_out_time_) > (wd_out_tries_ == 0 ? 1000.0 : 3000.0)) {
      RVC_LOGW("playback stream idle for %.0f ms: replacing it (attempt %d)", since(wd_out_time_), wd_out_tries_ + 1);
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
        if (out_) {
          AAudioStream_requestStop(out_);
          AAudioStream_close(out_);
          out_ = nullptr;
        }
        out_ring_.skip(out_ring_.available());  // no callback is running now: stale audio goes
        start_output_locked();
      }
      output_reopens_.fetch_add(1);
      wd_out_time_ = Clock::now();
      ++wd_out_tries_;
    }
  }
}

void RealtimeDriver::close_streams() {
  std::lock_guard<std::mutex> lk(stream_mu_);
  for (AAudioStream** s : {&in_, &out_}) {
    if (*s) {
      AAudioStream_requestStop(*s);
      AAudioStream_close(*s);
      *s = nullptr;
    }
  }
}

// Virtual-mic mode, decided by the shell service: 0 = no app is recording (capture nothing),
// 1 = capture here (AAudio, accessibility exemption), 2 = the shell captures (pipe).
void RealtimeDriver::set_capture_mode(int m) {
  std::lock_guard<std::mutex> lk(stream_mu_);
  int prev = capture_mode_.exchange(m);
  if (!running_.load() || prev == m) return;
  if (m == 1) start_input_locked();  // also replaces a stream that died while it was idle
  else if (in_) AAudioStream_requestStop(in_);
  if (m != 0 && prev == 0) wake_pending_ = true;  // fresh recording: start from a clean state
  if (m == 0) idle_pending_ = true;
  cv_.notify_one();
}

void RealtimeDriver::start(const DriverOptions& opt) {
  stop();
  opt_ = opt;  // takes ownership of the fds (closed by stop())
  if (!engine_ || !engine_->loaded()) {
    stop();
    throw Error("no model loaded");
  }
  sr_ = engine_->sample_rate();
  block_ = engine_->block_frame();
  in_ring_.reset(static_cast<size_t>(block_) * 8 + 8192);
  out_ring_.reset(static_cast<size_t>(block_) * 8 + 8192);
  in_overflows_ = out_underruns_ = dropped_ = blocks_ = late_blocks_ = 0;
  last_ms_ = avg_ms_ = max_ms_ = 0;
  set_error("");
  debug_wav_.clear();
  scratch_.assign(16384, 0.f);
  debug_pos_ = 0;
  if (!opt_.debug_input.empty()) {
    try {
      int dsr = 0;
      debug_wav_ = read_wav_mono(opt_.debug_input, &dsr);
      RVC_LOGW("debug input %s (%zu samples @ %d Hz) replaces the microphone", opt_.debug_input.c_str(), debug_wav_.size(), dsr);
    } catch (const std::exception& e) {
      RVC_LOGE("debug input: %s", e.what());
    }
  }
  engine_->reset_state();
  capture_mode_ = opt_.virtual_capture ? 0 : -1;
  wake_pending_ = idle_pending_ = false;
  in_cb_frames_ = out_cb_calls_ = input_reopens_ = output_reopens_ = 0;
  wd_in_frames_ = wd_out_calls_ = 0;
  wd_in_tries_ = wd_out_tries_ = 0;
  wd_checked_ = wd_in_time_ = wd_out_time_ = Clock::now();
  in_peak_ = 0.f;
  in_zero_ms_ = 0;
  last_block_ms_ = steady_ms();
  first_block_pending_ = true;
  dbg_fail_start_ = debug_prop("debug.rvc.fail_start");
  dbg_mute_start_ = debug_prop("debug.rvc.mute_start");
  if (opt_.playback_fd >= 0) fcntl(opt_.playback_fd, F_SETFL, fcntl(opt_.playback_fd, F_GETFL) | O_NONBLOCK);
  try {
    open_streams();
  } catch (...) {
    stop();
    throw;
  }
  running_ = true;
  worker_ = std::thread([this] { worker_loop(); });
  if (opt_.capture_fd >= 0) reader_ = std::thread([this] { reader_loop(); });
}

void RealtimeDriver::stop() {
  if (running_.exchange(false)) {
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (reader_.joinable()) reader_.join();
  }
  close_streams();
  for (int* fd : {&opt_.capture_fd, &opt_.playback_fd}) {
    if (*fd >= 0) {
      close(*fd);
      *fd = -1;
    }
  }
}

void RealtimeDriver::reader_loop() {
  setpriority(PRIO_PROCESS, 0, -16);
  std::vector<float> buf(4096);
  while (running_.load()) {
    pollfd p{opt_.capture_fd, POLLIN, 0};
    int pr = poll(&p, 1, 50);
    if (pr <= 0) continue;
    if (p.revents & (POLLHUP | POLLERR)) {
      set_error("capture stream closed");
      break;
    }
    ssize_t n = read(opt_.capture_fd, buf.data(), buf.size() * sizeof(float));
    if (n <= 0) {
      if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      set_error("capture stream closed");
      break;
    }
    size_t frames = static_cast<size_t>(n) / sizeof(float);
    int m = capture_mode_.load();
    if (m >= 0 && m != 2) continue;  // stale shell capture after a mode change: discard
    if (in_ring_.write(input_samples(buf.data(), frames), frames) < frames) in_overflows_.fetch_add(1);
    cv_.notify_one();
  }
}

void RealtimeDriver::worker_loop() {
  setpriority(PRIO_PROCESS, 0, -19);
  Adpf& hint = adpf();
  void* session = nullptr;
  const int64_t block_ns = static_cast<int64_t>(1e9 * block_ / sr_);
  if (hint.ok) {
    void* mgr = hint.get_manager();
    int32_t tid = gettid();
    if (mgr) session = hint.create(mgr, &tid, 1, block_ns * 8 / 10);
  }
  std::vector<float> in(block_), out(block_);
  double avg = 0;
  while (running_.load()) {
    if (restart_.exchange(false)) {
      close_streams();
      try {
        open_streams();
      } catch (const std::exception& e) {
        set_error(e.what());
        RVC_LOGE("reopen failed: %s", e.what());
      }
    }
    if (std::chrono::duration<double, std::milli>(Clock::now() - wd_checked_).count() > 50.0) {
      wd_checked_ = Clock::now();
      watch_streams();
    }
    if (idle_pending_.exchange(false)) in_ring_.skip(in_ring_.available());
    if (wake_pending_.exchange(false)) {
      // A new recording started: drop leftovers of the previous one, clear the model state and,
      // while the first real block is still being captured, run one silent block so caches,
      // thread pools and CPU clocks are warm when it arrives.
      in_ring_.skip(in_ring_.available());
      if (session && hint.workload_increase) hint.workload_increase(session, true, false, "rvc wake");
      engine_->reset_state();
      if (opt_.warmup && !opt_.passthrough) {
        try {
          std::fill(in.begin(), in.end(), 0.f);
          auto tw = Clock::now();
          engine_->process_block(in.data(), out.data());
          RVC_LOGI("wake warm-up block %.0f ms", ms_since(tw));
        } catch (...) {
        }
        engine_->reset_state();
      }
      in_zero_ms_ = 0;
      last_block_ms_ = steady_ms();
      first_block_pending_ = true;
      continue;
    }
    if (in_ring_.available() < static_cast<size_t>(block_)) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait_for(lk, std::chrono::milliseconds(1));
      continue;
    }
    // too much backlog (e.g. after a stall): drop whole blocks to catch up
    while (in_ring_.available() >= static_cast<size_t>(block_) * (opt_.max_queue_blocks + 1)) {
      in_ring_.skip(block_);
      dropped_.fetch_add(block_);
    }
    in_ring_.read(in.data(), block_);
    {
      // what arrived: a silenced capture delivers exact zeros, which no real microphone does
      float pk = 0.f;
      for (float v : in) pk = std::max(pk, std::fabs(v));
      in_peak_ = pk;
      in_zero_ms_ = pk == 0.f ? in_zero_ms_.load() + 1000.0 * block_ / sr_ : 0.0;
      last_block_ms_ = steady_ms();
    }
    auto t0 = Clock::now();
    if (opt_.passthrough) {
      out = in;
    } else {
      try {
        engine_->process_block(in.data(), out.data());
      } catch (const std::exception& e) {
        set_error(e.what());
        RVC_LOGE("process_block: %s", e.what());
        std::fill(out.begin(), out.end(), 0.f);
      }
    }
    double ms = ms_since(t0);
    if (session) hint.report(session, static_cast<int64_t>(ms * 1e6));
    if (opt_.output_gain != 1.f)
      for (auto& v : out) v *= opt_.output_gain;
    if (opt_.local_output) out_ring_.write(out.data(), out.size());
    if (opt_.playback_fd >= 0) {
      const char* p = reinterpret_cast<const char*>(out.data());
      size_t left = out.size() * sizeof(float);
      while (left > 0) {
        ssize_t w = write(opt_.playback_fd, p, left);
        if (w < 0) {
          if (errno == EAGAIN) dropped_.fetch_add(static_cast<int64_t>(left / sizeof(float)));
          else if (errno != EINTR) set_error("playback stream closed");
          if (errno != EINTR) break;
          continue;
        }
        p += w;
        left -= static_cast<size_t>(w);
      }
    }
    int64_t n = blocks_.fetch_add(1) + 1;
    if (first_block_pending_.exchange(false))
      RVC_LOGI("first converted block out (%.0f ms), input peak %.1f dB", ms, 20.0 * std::log10(std::max(in_peak_.load(), 1e-6f)));
    avg = n == 1 ? ms : avg * 0.9 + ms * 0.1;
    last_ms_ = ms;
    avg_ms_ = avg;
    if (n > 2) max_ms_ = std::max(max_ms_.load(), ms);
    if (ms * 1e6 > block_ns) late_blocks_.fetch_add(1);
  }
  if (session) hint.close(session);
}

Json RealtimeDriver::status() const {
  Json j = Json::object();
  j.set("running", running_.load());
  j.set("blocks", blocks_.load());
  j.set("block_ms", 1000.0 * block_ / std::max(1, sr_));
  j.set("infer_ms", last_ms_.load());
  j.set("infer_avg_ms", avg_ms_.load());
  j.set("infer_max_ms", max_ms_.load());
  j.set("late_blocks", late_blocks_.load());
  j.set("in_overflows", in_overflows_.load());
  j.set("out_underruns", out_underruns_.load());
  j.set("dropped_samples", dropped_.load());
  j.set("in_queue", static_cast<int64_t>(in_ring_.available()));
  j.set("out_queue", static_cast<int64_t>(out_ring_.available()));
  j.set("in_burst", in_burst_);
  j.set("out_burst", out_burst_);
  j.set("in_exclusive", in_exclusive_);
  j.set("out_exclusive", out_exclusive_);
  j.set("adpf", adpf().ok);
  // capture health: is audio expected right now, when did the converter last get a block, and
  // was it real sound
  const int cm = capture_mode_.load();
  const bool expecting = running_.load() && (cm < 0 ? opt_.local_input || opt_.capture_fd >= 0 : cm != 0);
  j.set("capture_expected", expecting);
  j.set("input_age_ms", last_block_ms_.load() > 0 ? static_cast<double>(steady_ms() - last_block_ms_.load()) : -1.0);
  j.set("input_zero_ms", in_zero_ms_.load());
  j.set("input_level_db", 20.0 * std::log10(std::max(in_peak_.load(), 1e-6f)));
  j.set("input_reopens", input_reopens_.load());
  j.set("output_reopens", output_reopens_.load());
  if (engine_ && engine_->loaded()) {
    BlockStats s = engine_->stats();
    Json st = Json::object();
    st.set("hubert", s.hubert_ms);
    st.set("f0", s.f0_ms);
    st.set("index", s.index_ms);
    st.set("enc", s.enc_ms);
    st.set("flow", s.flow_ms);
    st.set("dec", s.dec_ms);
    st.set("post", s.post_ms);
    st.set("sola_offset", s.sola_offset);
    j.set("stages", st);
    // device-side latency estimate: block accumulation + inference + output buffering
    double lat = 1000.0 * (block_ + engine_->algorithmic_latency() * sr_ - block_) / sr_ + avg_ms_.load() +
                 1000.0 * (in_burst_ * 2 + out_burst_ * 2) / std::max(1, sr_);
    j.set("latency_ms", lat);
  }
  {
    std::lock_guard<std::mutex> lk(err_mu_);
    if (!last_error_.empty()) j.set("error", last_error_);
  }
  return j;
}

}  // namespace rvc
