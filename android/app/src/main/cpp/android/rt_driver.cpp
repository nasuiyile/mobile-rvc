#include "rt_driver.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
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
  o.pipeline = static_cast<int>(j.num_or("pipeline", o.pipeline));
  if (j.bool_or("pipeline", false)) o.pipeline = 1;        // also accepts true / false
  else if (!j.bool_or("pipeline", true)) o.pipeline = 0;
  o.playout_control = j.bool_or("playout_control", o.playout_control);
  o.sim_output = j.bool_or("sim_output", o.sim_output);
  o.sim_record = j.str_or("sim_record", "");
  o.debug_stall = j.str_or("debug_stall", "");
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
  int m = d->capture_mode_.load(std::memory_order_relaxed);
  if (m == 0 || m == 2) return AAUDIO_CALLBACK_RESULT_CONTINUE;  // virtual mic idle / shell captures
  const float* src = d->input_samples(static_cast<const float*>(audio), static_cast<size_t>(frames));
  size_t w = d->in_ring_.write(src, static_cast<size_t>(frames));
  if (w < static_cast<size_t>(frames)) d->in_overflows_.fetch_add(1, std::memory_order_relaxed);
  return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

aaudio_data_callback_result_t RealtimeDriver::out_cb(AAudioStream*, void* user, void* audio, int32_t frames) {
  auto* d = static_cast<RealtimeDriver*>(user);
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
  const bool dry = r < static_cast<size_t>(frames);
  if (dry) {
    std::fill(out + r, out + frames, 0.f);
    // one count per dropout (the moment the queue runs dry), not per silent callback
    if (!d->out_dry_.load(std::memory_order_relaxed) && d->blocks_.load(std::memory_order_relaxed) > 1)
      d->out_underruns_.fetch_add(1, std::memory_order_relaxed);
  }
  d->out_dry_.store(dry, std::memory_order_relaxed);
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

void RealtimeDriver::open_streams() {
  std::lock_guard<std::mutex> lk(stream_mu_);
  if (opt_.local_output && !opt_.sim_output && !out_) {
    out_ = open_stream(false);
    AAudioStream_requestStart(out_);
  }
  // virtual-mic mode keeps the input opened but stopped, so waking up only needs a start
  if ((opt_.local_input || opt_.virtual_capture) && !in_) {
    in_ = open_stream(true);
    if (input_should_run()) AAudioStream_requestStart(in_);
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
  if (in_) {
    if (m == 1) AAudioStream_requestStart(in_);
    else AAudioStream_requestStop(in_);
  } else if (m == 1) {
    try {
      in_ = open_stream(true);
      AAudioStream_requestStart(in_);
    } catch (const std::exception& e) {
      set_error(e.what());
    }
  }
  if (m != 0 && prev == 0) wake_pending_ = true;  // fresh recording: start from a clean state
  if (m == 0) idle_pending_ = true;
  cv_.notify_one();
}

// The conversion threads compute on the big cores (as the engine's own pools do).
static void pin_to_big_cores() {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : big_cores()) CPU_SET(c, &set);
  sched_setaffinity(0, sizeof(set), &set);
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
  out_dry_ = true;
  last_ms_ = avg_ms_ = max_ms_ = 0;
  avg_acc_ = 0;
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
  stalls_.clear();
  for (size_t p = 0; p < opt_.debug_stall.size();) {
    size_t e = opt_.debug_stall.find(',', p);
    if (e == std::string::npos) e = opt_.debug_stall.size();
    std::string item = opt_.debug_stall.substr(p, e - p);
    size_t c = item.find(':');
    if (c != std::string::npos) stalls_.emplace_back(atoll(item.substr(0, c).c_str()), atoi(item.substr(c + 1).c_str()));
    p = e + 1;
  }
  sim_played_.clear();
  play_local_.configure(sr_, block_, true);
  play_pipe_.configure(sr_, block_, false);
  pipelined_ = !opt_.passthrough && (opt_.pipeline > 0 || (opt_.pipeline < 0 && engine_->stages_independent()));
  slot_full_ = synth_busy_ = false;
  pipe_level_ok_ = true;
  in_device_ms_ = out_device_ms_ = -1;
  worker_tid_ = synth_tid_ = 0;
  engine_->reset_state();
  // The first runs of a session are much slower than the rest (GPU pipelines are compiled, the
  // NPU and the CPU clocks ramp up, buffers are allocated). Get them out of the way before
  // audio flows: a slow first block would otherwise sit in the output queue as extra delay.
  if (opt_.warmup && !opt_.passthrough) warm_up(2);
  capture_mode_ = opt_.virtual_capture ? 0 : -1;
  wake_pending_ = idle_pending_ = false;
  if (opt_.playback_fd >= 0) fcntl(opt_.playback_fd, F_SETFL, fcntl(opt_.playback_fd, F_GETFL) | O_NONBLOCK);
  try {
    open_streams();
  } catch (...) {
    stop();
    throw;
  }
  if (opt_.sim_output) out_burst_ = sr_ / 250;
  play_local_.configure(sr_, block_, true, out_burst_);
  running_ = true;
  if (pipelined_) synth_ = std::thread([this] { synth_loop(); });
  worker_ = std::thread([this] { worker_loop(); });
  if (opt_.capture_fd >= 0) reader_ = std::thread([this] { reader_loop(); });
  if (opt_.sim_output && opt_.local_output) sim_ = std::thread([this] { sim_loop(); });
  RVC_LOGI("driver started: block %d, %s", block_, pipelined_ ? "two-stage pipeline" : "single stage");
}

void RealtimeDriver::stop() {
  if (running_.exchange(false)) {
    cv_.notify_all();
    {
      std::lock_guard<std::mutex> lk(hand_mu_);
    }
    hand_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (synth_.joinable()) synth_.join();
    if (reader_.joinable()) reader_.join();
    if (sim_.joinable()) sim_.join();
    hint_close();
    if (!opt_.sim_record.empty() && !sim_played_.empty()) {
      try {
        write_wav_f32(opt_.sim_record, sim_played_, sr_);
      } catch (const std::exception& e) {
        RVC_LOGE("sim record: %s", e.what());
      }
    }
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

// Test hook: plays the local output like an audio device would (4 ms bursts on a steady clock).
void RealtimeDriver::sim_loop() {
  setpriority(PRIO_PROCESS, 0, -19);
  const int frames = sr_ / 250;
  std::vector<float> buf(static_cast<size_t>(frames));
  const auto t0 = Clock::now();
  for (int64_t k = 1; running_.load(); ++k) {
    out_cb(nullptr, this, buf.data(), frames);
    if (!opt_.sim_record.empty()) sim_played_.insert(sim_played_.end(), buf.begin(), buf.end());
    std::this_thread::sleep_until(t0 + std::chrono::microseconds(k * 4000));
  }
}

// ------------------------------------------------------------------ ADPF session
void RealtimeDriver::hint_open() {
  Adpf& a = adpf();
  if (!a.ok) return;
  void* mgr = a.get_manager();
  if (!mgr) return;
  // every thread that computes part of a block: both stages and the engine's helpers
  std::vector<int32_t> tids;
  for (int t : {worker_tid_.load(), synth_tid_.load()})
    if (t > 0) tids.push_back(t);
  size_t own = tids.size();
  for (int t : engine_->helper_tids()) tids.push_back(t);
  // Latency is block + inference time, so every millisecond of inference counts: ask for much
  // less than the deadline instead of "just in time".
  const int64_t target = static_cast<int64_t>(1e9 * block_ / sr_) / 4;
  std::lock_guard<std::mutex> lk(hint_mu_);
  hint_ = a.create(mgr, tids.data(), tids.size(), target);
  if (!hint_ && own > 0) hint_ = a.create(mgr, tids.data(), own, target);
}

void RealtimeDriver::hint_report(double ms) {
  std::lock_guard<std::mutex> lk(hint_mu_);
  if (hint_) adpf().report(hint_, static_cast<int64_t>(ms * 1e6));
}

void RealtimeDriver::hint_boost(const char* why) {
  std::lock_guard<std::mutex> lk(hint_mu_);
  if (hint_ && adpf().workload_increase) adpf().workload_increase(hint_, true, false, why);
}

void RealtimeDriver::hint_close() {
  std::lock_guard<std::mutex> lk(hint_mu_);
  if (hint_) adpf().close(hint_);
  hint_ = nullptr;
}

// ------------------------------------------------------------------ conversion
void RealtimeDriver::warm_up(int blocks) {
  std::vector<float> in(static_cast<size_t>(block_), 0.f), out(static_cast<size_t>(block_));
  try {
    for (int i = 0; i < blocks; ++i) {
      auto t = Clock::now();
      engine_->process_block(in.data(), out.data());
      RVC_LOGI("warm-up block %.0f ms", ms_since(t));
    }
  } catch (const std::exception& e) {
    RVC_LOGW("warm-up: %s", e.what());
  }
  engine_->reset_state();
}

// How long a frame spends between the stream's buffer and the transducer, from the streams' own
// timestamps. Only used for the delay shown to the user.
void RealtimeDriver::measure_device_latency() {
  std::unique_lock<std::mutex> lk(stream_mu_, std::try_to_lock);
  if (!lk.owns_lock()) return;
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const int64_t now = static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
  int64_t pos = 0, t = 0;
  if (out_ && AAudioStream_getTimestamp(out_, CLOCK_MONOTONIC, &pos, &t) == AAUDIO_OK) {
    // the frame written last is presented (written - pos) frames after the one presented at t
    double ms = 1000.0 * (AAudioStream_getFramesWritten(out_) - pos) / sr_ - (now - t) / 1e6;
    if (ms >= 0 && ms < 2000) out_device_ms_ = ms;
  }
  if (in_ && input_should_run() && AAudioStream_getTimestamp(in_, CLOCK_MONOTONIC, &pos, &t) == AAUDIO_OK) {
    // frames captured by now, minus the ones already handed to us
    double ms = 1000.0 * (pos - AAudioStream_getFramesRead(in_)) / sr_ + (now - t) / 1e6;
    if (ms >= 0 && ms < 2000) in_device_ms_ = ms;
  }
}

void RealtimeDriver::settle() {
  if (!pipelined_) return;
  std::unique_lock<std::mutex> lk(hand_mu_);
  hand_cv_.wait(lk, [&] { return (!slot_full_ && !synth_busy_) || !running_.load(); });
}

// Queues one converted block for every consumer. t_in: when its last input sample arrived;
// t_read: when the converter started on it.
void RealtimeDriver::emit(float* out, Clock::time_point t_in, Clock::time_point t_read) {
  const size_t n = static_cast<size_t>(block_);
  if (opt_.output_gain != 1.f)
    for (size_t i = 0; i < n; ++i) out[i] *= opt_.output_gain;
  if (opt_.local_output) {
    shaped_.assign(out, out + n);
    if (opt_.playout_control) play_local_.shape(shaped_, out_ring_.available(), ms_since(t_in), ms_since(t_read));
    out_ring_.write(shaped_.data(), shaped_.size());
  }
  if (opt_.playback_fd >= 0) {
    shaped_.assign(out, out + n);
    if (opt_.playout_control && pipe_level_ok_) {
      int pending = 0;  // bytes the injection service has not read yet
      if (ioctl(opt_.playback_fd, FIONREAD, &pending) == 0) {
        play_pipe_.shape(shaped_, static_cast<size_t>(std::max(pending, 0)) / sizeof(float), ms_since(t_in), ms_since(t_read));
      } else {
        pipe_level_ok_ = false;  // not permitted here: deliver as is, the service bounds the backlog
        RVC_LOGW("FIONREAD on the playback pipe failed (%s): output delay control off for the pipe", strerror(errno));
      }
    }
    const char* p = reinterpret_cast<const char*>(shaped_.data());
    size_t left = shaped_.size() * sizeof(float);
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
}

// ms: input complete -> output queued. A block is late when a stage cannot keep up with realtime.
void RealtimeDriver::finish_block(double ms, double stage_a_ms, double stage_b_ms) {
  hint_report(ms);
  const double block_ms = 1000.0 * block_ / sr_;
  int64_t n = blocks_.fetch_add(1) + 1;
  avg_acc_ = n == 1 ? ms : avg_acc_ * 0.9 + ms * 0.1;
  last_ms_ = ms;
  avg_ms_ = avg_acc_;
  if (n > 2) max_ms_ = std::max(max_ms_.load(), ms);
  if ((pipelined_ ? std::max(stage_a_ms, stage_b_ms) : ms) > block_ms) late_blocks_.fetch_add(1);
}

void RealtimeDriver::worker_loop() {
  setpriority(PRIO_PROCESS, 0, -19);
  pin_to_big_cores();
  worker_tid_ = gettid();
  if (!pipelined_) hint_open();
  std::vector<float> in(block_), out(block_);
  Job job;
  int64_t seq = 0;  // blocks read since the stream / the recording started
  while (running_.load()) {
    if (restart_.exchange(false)) {
      settle();
      close_streams();
      try {
        open_streams();
      } catch (const std::exception& e) {
        set_error(e.what());
        RVC_LOGE("reopen failed: %s", e.what());
      }
      play_local_.configure(sr_, block_, true, out_burst_);  // the new stream may use another burst size
    }
    if (idle_pending_.exchange(false)) in_ring_.skip(in_ring_.available());
    if (wake_pending_.exchange(false)) {
      // A new recording started: drop leftovers of the previous one, clear the model state and,
      // while the first real block is still being captured, run one silent block so caches,
      // thread pools and CPU clocks are warm when it arrives.
      in_ring_.skip(in_ring_.available());
      settle();
      hint_boost("rvc wake");
      engine_->reset_state();
      if (opt_.warmup && !opt_.passthrough) warm_up(1);
      play_local_.reset();
      play_pipe_.reset();
      seq = 0;
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
    auto t0 = Clock::now();
    // what is still waiting was captured after this block: that is how old its last sample is
    auto t_in = t0 - std::chrono::microseconds(static_cast<int64_t>(1e6 * in_ring_.available() / sr_));
    if (++seq % 8 == 1) measure_device_latency();
    for (auto& st : stalls_)
      if (st.first == seq) std::this_thread::sleep_for(std::chrono::milliseconds(st.second));
    if (opt_.passthrough) {
      out = in;
      emit(out.data(), t_in, t0);
      finish_block(ms_since(t0), 0, 0);
      continue;
    }
    if (!pipelined_) {
      try {
        engine_->process_block(in.data(), out.data());
      } catch (const std::exception& e) {
        set_error(e.what());
        RVC_LOGE("process_block: %s", e.what());
        std::fill(out.begin(), out.end(), 0.f);
      }
      emit(out.data(), t_in, t0);
      finish_block(ms_since(t0), 0, 0);
      continue;
    }
    // two-stage: analyse here, hand the block to synth_loop (waits while it is still busy with
    // the block before the previous one, which only happens when a stage is slower than realtime)
    job.t_read = t0;
    job.t_in = t_in;
    job.ok = true;
    try {
      engine_->analyze(in.data(), job.frame);
    } catch (const std::exception& e) {
      job.ok = false;
      set_error(e.what());
      RVC_LOGE("analyze: %s", e.what());
    }
    {
      std::unique_lock<std::mutex> lk(hand_mu_);
      hand_cv_.wait(lk, [&] { return !slot_full_ || !running_.load(); });
      if (!running_.load()) break;
      std::swap(slot_, job);
      slot_full_ = true;
    }
    hand_cv_.notify_all();
  }
}

void RealtimeDriver::synth_loop() {
  setpriority(PRIO_PROCESS, 0, -19);
  pin_to_big_cores();
  synth_tid_ = gettid();
  for (int i = 0; i < 200 && worker_tid_.load() == 0 && running_.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  hint_open();
  std::vector<float> out(block_);
  Job job;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(hand_mu_);
      hand_cv_.wait(lk, [&] { return slot_full_ || !running_.load(); });
      if (!running_.load()) break;
      std::swap(job, slot_);
      slot_full_ = false;
      synth_busy_ = true;
    }
    hand_cv_.notify_all();
    bool ok = job.ok;
    if (ok) {
      try {
        engine_->synthesize(job.frame, out.data());
      } catch (const std::exception& e) {
        ok = false;
        set_error(e.what());
        RVC_LOGE("synthesize: %s", e.what());
      }
    }
    if (!ok) std::fill(out.begin(), out.end(), 0.f);
    emit(out.data(), job.t_in, job.t_read);
    finish_block(ms_since(job.t_read), job.frame.st.analyze_ms, job.frame.st.synth_ms);
    {
      std::lock_guard<std::mutex> lk(hand_mu_);
      synth_busy_ = false;
    }
    hand_cv_.notify_all();
  }
}

Json RealtimeDriver::status() const {
  Json j = Json::object();
  j.set("running", running_.load());
  j.set("blocks", blocks_.load());
  const double block_ms = 1000.0 * block_ / std::max(1, sr_);
  j.set("block_ms", block_ms);
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
  j.set("pipeline", pipelined_);
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
    st.set("analyze", s.analyze_ms);
    st.set("synth", s.synth_ms);
    st.set("sola_offset", s.sola_offset);
    j.set("stages", st);
    // End-to-end delay, from its parts: a sample waits for its block to fill, for the block to
    // be converted, in the output queue, for the SOLA overlap that is held back, and in the
    // device buffers. The queue is measured, not assumed.
    const PlayoutControl& pc = opt_.local_output ? play_local_ : play_pipe_;
    const bool measured = opt_.playout_control && pc.delay_ms() > 0;
    const double infer = avg_ms_.load();
    const double queue = measured ? std::max(0.0, pc.delay_ms() - infer) : 1000.0 * out_ring_.available() / std::max(1, sr_);
    const double sola = engine_->algorithmic_latency() * 1000.0 - block_ms;
    // device buffers: as the streams report them, else two bursts each
    const double in_io = in_device_ms_.load() >= 0 ? in_device_ms_.load() : 2000.0 * in_burst_ / std::max(1, sr_);
    const double out_io = out_device_ms_.load() >= 0 ? out_device_ms_.load() : 2000.0 * out_burst_ / std::max(1, sr_);
    const double io = in_io + out_io;
    Json lat = Json::object();
    lat.set("block", block_ms);
    lat.set("infer", infer);
    lat.set("queue", queue);
    lat.set("sola", sola);
    lat.set("io", io);
    j.set("latency_parts", lat);
    j.set("latency_ms", block_ms + infer + queue + sola + io);
    j.set("trimmed_ms", 1000.0 * (play_local_.trimmed_frames() + play_pipe_.trimmed_frames()) / std::max(1, sr_));
    j.set("padded_ms", 1000.0 * (play_local_.padded_frames() + play_pipe_.padded_frames()) / std::max(1, sr_));
  }
  {
    std::lock_guard<std::mutex> lk(err_mu_);
    if (!last_error_.empty()) j.set("error", last_error_);
  }
  return j;
}

}  // namespace rvc
