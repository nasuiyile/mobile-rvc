// Stationary spectral noise gate, a port of RVC's tools/torchgate (TorchGate with
// nonstationary=False), used for the "input / output noise reduction" switches.
#pragma once

#include <complex>
#include <vector>

#include "dsp.hpp"

namespace rvc {

// Complex DFT of any length (radix-2 directly, Bluestein otherwise).
class AnyFFT {
 public:
  explicit AnyFFT(int n = 1024);
  int size() const { return n_; }
  void forward(std::complex<float>* a);
  void inverse(std::complex<float>* a);  // unnormalized

 private:
  int n_, m_ = 0;
  bool pow2_;
  FFT fft_;
  std::vector<std::complex<float>> chirp_, bfft_, buf_;
};

class SpectralGate {
 public:
  SpectralGate() = default;
  // TorchGate(sr, n_fft, prop_decrease) as constructed by realtime_gui (n_fft = 4 * sr/100).
  SpectralGate(int sr, int n_fft, float prop_decrease = 0.9f, float n_std = 1.5f, float freq_smooth_hz = 500.f,
               float time_smooth_ms = 50.f);
  bool valid() const { return n_fft_ > 0; }
  // y = gate(x) with noise statistics from `noise`; returns n samples.
  std::vector<float> process(const float* x, int n, const float* noise, int nn);

 private:
  int n_fft_ = 0, hop_ = 0, bins_ = 0, sf_ = 1, st_ = 1;
  float prop_ = 0.9f, n_std_ = 1.5f;
  std::vector<float> window_, smooth_;
  AnyFFT fft_;
  void stft(const float* x, int n, std::vector<std::complex<float>>& X, int& frames);
  static void to_db(const std::vector<std::complex<float>>& X, int frames, int bins, std::vector<float>& db);
};

}  // namespace rvc
