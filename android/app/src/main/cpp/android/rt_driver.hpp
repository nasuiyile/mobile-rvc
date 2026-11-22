// Realtime audio driver: AAudio (or pipes from the Shizuku shell service) <-> RvcEngine.
#pragma once

#include <aaudio/AAudio.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../engine/json.hpp"
#include "../engine/ring_buffer.hpp"

namespace rvc {

class RvcEngine;

struct DriverOptions {
  bool local_input = true;    // capture the microphone in this process (AAudio)
  bool local_output = true;   // play converted audio in this process (monitor)
  int capture_fd = -1;        // float32 mono stream from the shell service (virtual-mic mode)
  int playback_fd = -1;       // converted float32 mono stream to the shell service
  int input_preset = AAUDIO_INPUT_PRESET_VOICE_PERFORMANCE;
  bool input_exclusive = true;  // MMAP exclusive capture (local mode); shared in virtual-mic mode
  bool passthrough = false;   // RVC "input monitor": bypass conversion
  float output_gain = 1.f;
  int max_queue_blocks = 2;   // drop audio beyond this backlog to bound latency
  int sample_rate = 48000;
  std::string debug_input;    // test hook: replace captured samples with this wav (looped)
  bool virtual_capture = false; // capture driven by set_capture_mode() (virtual-mic mode)
  bool warmup = true;         // run a silent block while the first block of a recording fills

  static DriverOptions from_json(const Json& j);
};

class RealtimeDriver {
 public:
  explicit RealtimeDriver(RvcEngine* engine);
  ~RealtimeDriver();
  void start(const DriverOptions& opt);
  void stop();
  bool running() const { return running_.load(); }
  void set_capture_mode(int mode);
  Json status() const;

 private:
  RvcEngine* engine_;
  DriverOptions opt_;
  std::atomic<bool> running_{false}, restart_{false};
  std::thread worker_, reader_;
  AAudioStream* in_ = nullptr;
  AAudioStream* out_ = nullptr;
  SpscRing in_ring_, out_ring_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<int64_t> in_overflows_{0}, out_underruns_{0}, dropped_{0}, blocks_{0}, late_blocks_{0};
  std::atomic<double> last_ms_{0}, avg_ms_{0}, max_ms_{0};
  mutable std::mutex err_mu_;
  std::string last_error_;
  void set_error(const std::string& e);
  int block_ = 0, sr_ = 48000;
  int in_burst_ = 0, out_burst_ = 0, in_rate_ = 0, out_rate_ = 0;
  bool in_exclusive_ = false, out_exclusive_ = false;

  std::mutex stream_mu_;
  std::atomic<int> capture_mode_{-1};  // -1 = not virtual-mic mode
  std::atomic<bool> wake_pending_{false}, idle_pending_{false};
  bool input_should_run() const;
  std::vector<float> debug_wav_, scratch_;
  std::atomic<size_t> debug_pos_{0};
  // returns the samples to queue: the captured ones, or the debug wav in their place
  const float* input_samples(const float* captured, size_t n);
  AAudioStream* open_stream(bool input);
  void open_streams();
  void close_streams();
  void worker_loop();
  void reader_loop();
  static aaudio_data_callback_result_t in_cb(AAudioStream*, void*, void*, int32_t);
  static aaudio_data_callback_result_t out_cb(AAudioStream*, void*, void*, int32_t);
  static void err_cb(AAudioStream*, void*, aaudio_result_t);
};

// CPU topology helpers (big-core detection from cpufreq).
Json cpu_info();
std::vector<int> big_cores();

}  // namespace rvc
