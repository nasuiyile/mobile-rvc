// DSP building blocks matching the Python/torch reference bit-for-bit where it matters.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace rvc {

// torchaudio.transforms.Resample(orig, new) with sinc_interp_hann, width 6, rolloff 0.99.
class SincResampler {
 public:
  SincResampler() = default;
  SincResampler(int orig, int target);
  bool identity() const { return o_ == n_; }
  // Taps reaching beyond an output's input position on either side (plus orig/gcd on the right).
  int width() const { return width_; }
  int out_len(int in_len) const;
  // y must have out_len(n) elements.
  void process(const float* x, int n, float* y) const;
  std::vector<float> operator()(const float* x, int n) const;

 private:
  int o_ = 1, n_ = 1, width_ = 0, K_ = 0;
  std::vector<float> kernel_;  // [n_][K_]
  mutable std::vector<float> pad_;
};

// librosa.feature.rms(y, frame_length, hop_length) (center=True, zero padding)
std::vector<float> rms_frames(const float* y, int n, int frame_length, int hop);
// F.interpolate(linear, align_corners=True) of a 1-D signal
std::vector<float> interp_linear_align(const std::vector<float>& x, int size);
// F.interpolate(linear, align_corners=False) of rows x len  -> rows x size
void interp_linear_rows(const float* x, int rows, int len, float* y, int size);

// Radix-2 complex FFT (in place).
class FFT {
 public:
  explicit FFT(int n = 1024);
  int size() const { return n_; }
  void forward(std::complex<float>* a) const;

 private:
  int n_;
  std::vector<std::complex<float>> tw_;
  std::vector<int> rev_;
};

// Magnitude spectrogram as torchfcpe MelModule (hann 1024 periodic, hop 160, center=False,
// reflect padding). Output: frames x (n_fft/2+1), frames = n/hop + 1.
class FcpeStft {
 public:
  FcpeStft(int n_fft = 1024, int win = 1024, int hop = 160);
  int frames(int n) const { return n / hop_ + 1; }
  int bins() const { return n_fft_ / 2 + 1; }
  void process(const float* y, int n, std::vector<float>& mag) const;

 private:
  int n_fft_, win_, hop_;
  std::vector<float> window_;
  FFT fft_;
};

}  // namespace rvc
