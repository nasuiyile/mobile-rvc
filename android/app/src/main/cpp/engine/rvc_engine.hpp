// Realtime RVC voice conversion engine (block processor).
//
// Port of tools/rvc_mobile/rt_pipeline.py, which itself reproduces RVC's
// realtime_gui.py audio_callback + infer/rtrvc.py with FCPE pitch extraction.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "dsp.hpp"
#include "noise_gate.hpp"
#include "json.hpp"
#include "ort_model.hpp"

namespace rvc {

class IvfIndex;
class NativeDecoder;
class ThreadPool;

struct EngineConfig {
  int sample_rate = 48000;      // device I/O rate
  double block_time = 0.25;     // s, multiple of 10 ms
  double crossfade_time = 0.05; // s
  double extra_time = 0.5;      // s of left context fed to HuBERT/enc_p
  int sid = 0;                  // speaker id
  bool phase_lock = false;      // experimental: NSF sine phase continuous across blocks
  bool deterministic = false;   // zero noise (tests)
  int index_nprobe = 0;         // 0 = value stored in the index
  bool parallel_f0 = true;      // FCPE concurrently with HuBERT
  BackendConfig hubert, fcpe, synth, dec;

  static EngineConfig from_json(const Json& j);
  Json to_json() const;
};

struct LiveParams {
  std::atomic<float> pitch{0.f};         // semitones
  std::atomic<float> formant{0.f};       // semitones
  std::atomic<float> index_rate{0.f};    // 0..1
  std::atomic<float> rms_mix_rate{0.f};  // 0..1 (1 = keep converted loudness)
  std::atomic<float> threshold{-60.f};   // dB; > -60 enables the response gate
  std::atomic<bool> input_nr{false};     // RVC "输入降噪" (TorchGate on the input)
  std::atomic<bool> output_nr{false};    // RVC "输出降噪" (TorchGate on the output)
};

struct BlockStats {
  double total_ms = 0, hubert_ms = 0, f0_ms = 0, index_ms = 0, enc_ms = 0, flow_ms = 0, dec_ms = 0, post_ms = 0;
  double analyze_ms = 0, synth_ms = 0;  // the two halves of a block (total = analyze + synth)
  int sola_offset = 0;
  int64_t blocks = 0;
};

// Everything the second half of a block needs from the first one. A block is
// analyze() -> synthesize(); the realtime driver may run the halves on two threads (block k+1
// is analysed while block k is still being synthesised) when they use different processors.
struct Frame {
  std::vector<float> phone;    // [p_len][dim] HuBERT features after retrieval, 100 frames/s
  std::vector<int64_t> pitch;  // [p_len] coarse pitch
  std::vector<float> nsff0;    // [return_length] f0 driving the NSF source
  std::vector<float> rms_ref;  // input aligned with the converted audio (loudness mixing)
  int dim = 0;
  float formant = 0.f;         // the value the pitch above was computed with
  BlockStats st;
};

class RvcEngine {
 public:
  explicit RvcEngine(std::string assets_dir);
  ~RvcEngine();

  void load(const std::string& model_dir, const EngineConfig& cfg);
  bool loaded() const { return loaded_; }
  const EngineConfig& config() const { return cfg_; }
  int block_frame() const { return block_frame_; }
  int sample_rate() const { return cfg_.sample_rate; }
  // Algorithmic delay in seconds (block + sola buffer + search), as RVC's GUI reports.
  double algorithmic_latency() const;
  void reset_state();
  // in/out: block_frame() mono float samples at sample_rate().
  void process_block(const float* in, float* out);
  // The same block in two halves. analyze(): input conditioning, HuBERT + FCPE, retrieval, pitch
  // cache. synthesize(): enc_p, flow, NSF decoder, loudness mixing, SOLA. The halves share no
  // mutable state, so synthesize(k) may overlap analyze(k+1); each half must see the blocks in
  // order. The output is bit-identical to process_block().
  void analyze(const float* in, Frame& f);
  void synthesize(Frame& f, float* out);
  // True when the halves' heavy models run on different processors (e.g. HuBERT on the GPU and
  // the vocoder on the NPU or CPU), i.e. when overlapping them raises throughput.
  bool stages_independent() const;
  // Kernel thread ids of the engine's own compute helpers (for ADPF performance hints).
  std::vector<int> helper_tids() const;

  LiveParams params;
  BlockStats stats() const;
  Json info() const;

 private:
  struct Worker;
  std::string assets_;
  EngineConfig cfg_;
  bool loaded_ = false;
  Json meta_;
  int tgt_sr_ = 40000, upp_ = 400, phone_dim_ = 768;
  float l_w_ = 1.f, l_b_ = 0.f;
  std::vector<float> g_;  // speaker embedding [gin]
  std::vector<float> cent_table_;
  float fcpe_f0_min_ = 32.7f;
  std::string model_dir_;
  double native_ms_ = 0;

  std::unique_ptr<OnnxModel> hubert_, fcpe_, enc_, flow_;
  std::map<int, std::unique_ptr<OnnxModel>> dec_;  // by decoder length (formant changes it)
  std::unique_ptr<NativeDecoder> native_dec_;
  std::unique_ptr<IvfIndex> index_;
  std::unique_ptr<Worker> worker_;
  std::unique_ptr<ThreadPool> pool_;  // retrieval rows
  Frame frame_;                       // process_block() scratch

  // sizes (realtime_gui.start_vc)
  int zc_ = 480, block_frame_ = 0, block_frame_16k_ = 0, crossfade_frame_ = 0, sola_buffer_frame_ = 0,
      sola_search_frame_ = 0, extra_frame_ = 0, skip_head_ = 0, return_length_ = 0;
  std::vector<float> input_wav_, input_wav_res_, rms_buffer_, sola_buffer_, fade_in_, fade_out_;
  std::vector<float> input_wav_denoise_, nr_buffer_, output_buffer_;
  SpectralGate gate_, gate_out_;  // input / output noise reduction (one per half: they keep scratch state)
  std::vector<int64_t> cache_pitch_;
  std::vector<float> cache_pitchf_;
  SincResampler res_in_, res_out_;
  std::map<int, SincResampler> res_formant_;
  FcpeStft stft_;
  std::mt19937 rng_{1234};
  std::normal_distribution<float> normal_{0.f, 1.f};
  int64_t phase_frames_ = 0;
  std::map<int64_t, double> phase_acc_;

  mutable std::mutex stats_mu_;
  BlockStats stats_;

  OnnxModel& dec_for(int rl2);
  std::vector<float> run_fcpe(const float* x16, int n);
  void make_har(const float* f0, int L, int64_t start_frame, float* har);
};

}  // namespace rvc
