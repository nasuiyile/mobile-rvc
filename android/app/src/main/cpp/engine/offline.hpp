// Offline (whole-file) conversion: port of RVC's 模型推理 path, infer/vc/modules.py vc_single +
// infer/vc/pipeline.py Pipeline.pipeline, with FCPE pitch.
//
//   audio (16 kHz, peak <= 0.95) -> 48 Hz high-pass (filtfilt) -> split at the quietest points
//   near every x_center s -> FCPE over the whole file -> per segment (x_pad s of reflected
//   context each side): HuBERT, index retrieval, protect, enc_p / flow / NSF-HiFiGAN -> concat
//   -> RMS mix (half-second envelopes) -> scaled down if it would clip.
//
// Sizes differ from the PC only where memory demands it: segments are shorter (RVC itself picks
// x_* per GPU memory), FCPE runs in long overlapping pieces on files over 5 minutes, and the
// vocoder runs in overlapping pieces (its receptive field is ~12 frames, the overlap 24, so the
// result is the same as decoding whole segments).
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "dsp.hpp"
#include "json.hpp"
#include "ort_model.hpp"

namespace rvc {

class IvfIndex;
class NativeDecoder;

// SincResampler over an unbounded signal, piece by piece. The output is identical to
// resampling the whole signal in one call (zero padding at both ends, same length), with
// memory bounded by a short chunk.
class StreamResampler {
 public:
  StreamResampler(int orig, int target);
  void push(const float* x, size_t n, std::vector<float>& out);  // appends to out
  void finish(std::vector<float>& out);                           // appends the rest

 private:
  SincResampler r_;
  bool identity_ = false;
  size_t o_ = 1, n_ = 1, ctx_ = 0, chunk_ = 0;
  std::vector<float> buf_;  // ctx_ samples of left context, then input not resampled yet
  std::vector<float> y_;
  size_t in_total_ = 0, out_total_ = 0;
};

// A decoded audio stream, mono.
struct AudioSource {
  virtual ~AudioSource() = default;
  // Appends the next samples to `out`; false once the stream has ended (and nothing was added).
  // rate() is valid as soon as samples have been returned and may change between reads.
  virtual bool read(std::vector<float>& out) = 0;
  virtual int rate() const = 0;
  virtual double duration() const { return 0; }  // seconds, 0 = unknown
};

// Shared with another thread: progress for the UI, cancel request from it.
struct OfflineProgress {
  std::atomic<float> fraction{0.f};
  std::atomic<bool> cancel{false};
  void check() const;  // throws Error("cancelled")
};

// The WebUI's inference settings (defaults as in its 模型推理 tab).
struct OfflineParams {
  int pitch = 0;               // 变调 (f0_up_key), semitones
  float index_rate = 0.75f;    // 检索特征占比
  float rms_mix_rate = 0.25f;  // 输入源音量包络替换输出音量包络融合比例 (1 = keep the output's)
  float protect = 0.33f;       // 保护清辅音和呼吸声 (0.5 = off)
  bool deterministic = false;  // zero noise (tests)
  // configs/config.py x_pad / x_query / x_center / x_max, seconds (PC CPU: 1 / 6 / 38 / 41)
  int x_pad = 1, x_query = 4, x_center = 12, x_max = 14;
  int dec_chunk = 256;  // vocoder frames per piece; 0 = whole segments (tests)

  static OfflineParams from_json(const Json& j);
};

struct OfflineResult {
  size_t samples = 0;           // output length at tgt_sr
  int segments = 0;
  double seconds = 0;           // wall time
  std::vector<double> rms_in;   // RVC change_rms: half-second rms of the (filtered) input ...
  std::vector<double> rms_out;  // ... and of the output, for RmsMix
};

class OfflinePipeline {
 public:
  // Loads its own sessions (ORT CPU, any length) and vocoder (native NEON when the model has a
  // manifest, else ORT CPU). cpu: threads / affinity; fp16: NEON vocoder precision.
  OfflinePipeline(const std::string& assets, const std::string& model_dir, const BackendConfig& cpu, bool fp16, int sid);
  ~OfflinePipeline();
  int tgt_sr() const { return tgt_sr_; }
  Json info() const;

  // audio: the whole input at 16 kHz, mono. sink(x, n) receives the converted audio at
  // tgt_sr() in order, before the RMS mix and normalisation (see RmsMix); progress runs
  // from p0 to p1.
  OfflineResult run(std::vector<float> audio, const OfflineParams& p, const std::function<void(const float*, size_t)>& sink,
                    OfflineProgress* prog, float p0 = 0.f, float p1 = 1.f);

 private:
  std::vector<float> fcpe(const float* x, int n);  // FCPEInfer.infer (local_argmax, 0.006)
  std::vector<float> fcpe_long(const float* x, int n);
  // Pipeline.vc for one segment; appends the output frames [keep0, keep1) to out.
  void vc(const float* seg, int n, const int64_t* pitch, const float* pitchf, int np, const OfflineParams& p, int keep_frames_each_side,
          std::vector<float>& out);
  void decode(const std::vector<float>& z, int C, int T, const std::vector<float>& har, int k0, int k1, int chunk, std::vector<float>& out);
  void make_har(const float* f0, int L, bool deterministic, float* har);  // NSF sine source, phase from 0

  std::string model_dir_;
  Json meta_;
  int tgt_sr_ = 40000, upp_ = 400, phone_dim_ = 768;
  float l_w_ = 1.f, l_b_ = 0.f;
  std::vector<float> g_;
  std::vector<float> cent_table_;
  float fcpe_f0_min_ = 32.7f;
  FcpeStft stft_;
  std::unique_ptr<OnnxModel> hubert_, fcpe_, enc_, flow_, dec_;
  std::unique_ptr<NativeDecoder> native_dec_;
  std::unique_ptr<IvfIndex> index_;
  std::mt19937 rng_{1234};
  std::normal_distribution<float> normal_{0.f, 1.f};
  double load_ms_ = 0;
  // progress within the current segment (the vocoder pieces dominate its time); cancel checks
  OfflineProgress* prog_ = nullptr;
  double seg_from_ = 0, seg_to_ = 0;
  void segment_progress(double f);
};

// RVC pipeline.change_rms as a gain per output sample, applied while streaming:
// out *= rms_in^(1-rate) * max(rms_out, 1e-6)^(rate-1), both envelopes linearly interpolated
// (align_corners=False) to the output length.
class RmsMix {
 public:
  RmsMix(const OfflineResult& r, float rate);
  bool active() const { return rate_ != 1.f; }
  void apply(float* y, size_t n, size_t pos) const;  // y holds output samples [pos, pos + n)

 private:
  double at(const std::vector<double>& v, size_t i) const;
  std::vector<double> in_, out_;
  size_t len_;
  float rate_;
};

}  // namespace rvc
