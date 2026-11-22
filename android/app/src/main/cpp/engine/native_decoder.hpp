// Native NSF-HiFiGAN decoder (RVC "dec" without the sine source) on NEON fp16/fp32.
//
// The decoder dominates RVC's cost (~90 GFLOP per second of 40 kHz audio).  ONNX Runtime's
// CPU EP runs it in fp32 only; this implementation uses ARMv8.2 fp16 FMA (8 lanes) with
// periodic fp32 flushes, roughly doubling arithmetic throughput on phone CPUs.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "json.hpp"

namespace rvc {

class ThreadPool;

class NativeDecoder {
 public:
  // model_dir must contain weights.bin and manifest.json (synth template manifest).
  NativeDecoder(const std::string& model_dir, const std::vector<float>& g, int threads, bool want_fp16,
                const std::vector<int>& affinity);
  ~NativeDecoder();
  bool fp16() const { return fp16_; }
  int upp() const { return upp_; }
  int in_channels() const { return cin_; }
  // z: [in_channels][T] (row-major), har: [T*upp]; out: [T*upp]
  void run(const float* z, int T, const float* har, float* out);

 private:
  struct Conv {
    int Co = 0, Ci = 0, ntaps = 0;
    std::vector<int> off;
    std::vector<float> w32;
    std::vector<uint16_t> w16;
    std::vector<float> bias;
  };
  struct Stage {
    int rate = 1, cin = 0, cout = 0, noise_s = 1;
    std::vector<Conv> phases;  // transposed conv, one per output phase
    Conv noise;
    std::vector<std::vector<Conv>> res;  // per resblock: convs1[0],convs2[0],convs1[1],... (ResBlock1)
  };
  bool fp16_ = false;
  bool resblock1_ = true;
  int upp_ = 1, cin_ = 192, num_kernels_ = 3;
  Conv conv_pre_, conv_post_;
  std::vector<Stage> stages_;
  std::unique_ptr<ThreadPool> pool_;

  // scratch
  std::vector<float> fbuf_[5];
  std::vector<uint16_t> hbuf_;
  std::vector<float> abuf32_;

  void pack(Conv& c, const std::vector<float>& w, int Co, int Ci, int K, const std::vector<int>& kidx,
            const std::vector<int>& off, const float* bias);
  void conv(const Conv& c, const void* x, int ldx, int T, float* y, int ldy, bool accumulate);
};

}  // namespace rvc
