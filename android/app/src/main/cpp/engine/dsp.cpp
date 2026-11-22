#include "dsp.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rvc {

// ------------------------------------------------------------------ resampler
SincResampler::SincResampler(int orig, int target) {
  int g = std::gcd(orig, target);
  o_ = orig / g;
  n_ = target / g;
  const double lw = 6.0, rolloff = 0.99;
  double base = std::min(o_, n_) * rolloff;
  width_ = static_cast<int>(std::ceil(lw * o_ / base));
  K_ = 2 * width_ + o_;
  kernel_.resize(static_cast<size_t>(n_) * K_);
  for (int j = 0; j < n_; ++j) {
    for (int i = 0; i < K_; ++i) {
      double t = (-static_cast<double>(j) / n_ + static_cast<double>(i - width_) / o_) * base;
      t = std::max(-lw, std::min(lw, t));
      double w = std::cos(t * M_PI / lw / 2);
      w *= w;
      double tp = t * M_PI;
      double k = tp == 0 ? 1.0 : std::sin(tp) / tp;
      kernel_[static_cast<size_t>(j) * K_ + i] = static_cast<float>(k * w * base / o_);
    }
  }
}

int SincResampler::out_len(int n) const {
  return static_cast<int>(std::ceil(static_cast<double>(n_) * n / o_));
}

void SincResampler::process(const float* x, int n, float* y) const {
  int L = out_len(n);
  if (o_ == n_) {
    std::copy(x, x + n, y);
    return;
  }
  pad_.assign(static_cast<size_t>(n + 2 * width_ + o_), 0.f);
  std::copy(x, x + n, pad_.begin() + width_);
  int nframes = (static_cast<int>(pad_.size()) - K_) / o_ + 1;
  int idx = 0;
  for (int f = 0; f < nframes && idx < L; ++f) {
    const float* src = pad_.data() + static_cast<size_t>(f) * o_;
    for (int j = 0; j < n_ && idx < L; ++j, ++idx) {
      const float* k = kernel_.data() + static_cast<size_t>(j) * K_;
      float s = 0;
      int i = 0;
#if defined(__aarch64__)
      float32x4_t acc = vdupq_n_f32(0);
      for (; i + 4 <= K_; i += 4) acc = vfmaq_f32(acc, vld1q_f32(src + i), vld1q_f32(k + i));
      s = vaddvq_f32(acc);
#endif
      for (; i < K_; ++i) s += src[i] * k[i];
      y[idx] = s;
    }
  }
  for (; idx < L; ++idx) y[idx] = 0;
}

std::vector<float> SincResampler::operator()(const float* x, int n) const {
  std::vector<float> y(static_cast<size_t>(out_len(n)));
  process(x, n, y.data());
  return y;
}

// ------------------------------------------------------------------ rms / interp
std::vector<float> rms_frames(const float* y, int n, int fl, int hop) {
  int pad = fl / 2;
  int total = n + 2 * pad;
  int frames = 1 + (total - fl) / hop;
  std::vector<float> out(static_cast<size_t>(std::max(frames, 0)));
  for (int f = 0; f < frames; ++f) {
    int start = f * hop - pad;
    double acc = 0;
    for (int i = std::max(start, 0); i < std::min(start + fl, n); ++i) acc += static_cast<double>(y[i]) * y[i];
    out[f] = static_cast<float>(std::sqrt(acc / fl));
  }
  return out;
}

std::vector<float> interp_linear_align(const std::vector<float>& x, int size) {
  std::vector<float> y(static_cast<size_t>(size));
  int n = static_cast<int>(x.size());
  if (n == 1 || size == 1) {
    std::fill(y.begin(), y.end(), x.empty() ? 0.f : x[0]);
    return y;
  }
  for (int i = 0; i < size; ++i) {
    double pos = static_cast<double>(i) * (n - 1) / (size - 1);
    int i0 = std::min(static_cast<int>(pos), n - 2);
    double w = pos - i0;
    y[i] = static_cast<float>(x[i0] * (1 - w) + x[i0 + 1] * w);
  }
  return y;
}

void interp_linear_rows(const float* x, int rows, int len, float* y, int size) {
  double scale = static_cast<double>(len) / size;
  for (int i = 0; i < size; ++i) {
    double src = std::max(0.0, (i + 0.5) * scale - 0.5);
    int i0 = std::min(static_cast<int>(src), len - 1);
    int i1 = std::min(i0 + 1, len - 1);
    float w = static_cast<float>(src - i0);
    for (int r = 0; r < rows; ++r) {
      const float* xr = x + static_cast<size_t>(r) * len;
      y[static_cast<size_t>(r) * size + i] = xr[i0] * (1 - w) + xr[i1] * w;
    }
  }
}

// ------------------------------------------------------------------ FFT
FFT::FFT(int n) : n_(n), tw_(n / 2), rev_(n) {
  for (int i = 0; i < n / 2; ++i) tw_[i] = std::polar(1.0f, static_cast<float>(-2.0 * M_PI * i / n));
  int bits = 0;
  while ((1 << bits) < n) ++bits;
  for (int i = 0; i < n; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    rev_[i] = r;
  }
}

void FFT::forward(std::complex<float>* a) const {
  for (int i = 0; i < n_; ++i) if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
  for (int len = 2; len <= n_; len <<= 1) {
    int step = n_ / len;
    for (int i = 0; i < n_; i += len) {
      for (int j = 0; j < len / 2; ++j) {
        std::complex<float> u = a[i + j], v = a[i + j + len / 2] * tw_[j * step];
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
      }
    }
  }
}

FcpeStft::FcpeStft(int n_fft, int win, int hop) : n_fft_(n_fft), win_(win), hop_(hop), window_(win), fft_(n_fft) {
  for (int i = 0; i < win; ++i) window_[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / win));
}

void FcpeStft::process(const float* y, int n, std::vector<float>& mag) const {
  int pad_l = (win_ - hop_) / 2;
  int pad_r = std::max((win_ - hop_ + 1) / 2, win_ - n - pad_l);
  bool reflect = pad_r < n;
  int total = n + pad_l + pad_r;
  std::vector<float> yp(static_cast<size_t>(total), 0.f);
  for (int i = 0; i < total; ++i) {
    int j = i - pad_l;
    if (j >= 0 && j < n) yp[i] = y[j];
    else if (reflect) {
      if (j < 0) j = -j;
      if (j >= n) j = 2 * (n - 1) - j;
      yp[i] = (j >= 0 && j < n) ? y[j] : 0.f;
    }
  }
  int nfr = 1 + (total - n_fft_) / hop_;
  int want = frames(n), nb = bins();
  mag.assign(static_cast<size_t>(want) * nb, 0.f);
  std::vector<std::complex<float>> buf(n_fft_);
  int last = -1;
  for (int f = 0; f < std::min(nfr, want); ++f) {
    const float* src = yp.data() + static_cast<size_t>(f) * hop_;
    for (int i = 0; i < n_fft_; ++i) buf[i] = {src[i] * window_[i], 0.f};
    fft_.forward(buf.data());
    float* m = mag.data() + static_cast<size_t>(f) * nb;
    for (int k = 0; k < nb; ++k) m[k] = std::sqrt(buf[k].real() * buf[k].real() + buf[k].imag() * buf[k].imag() + 1e-9f);
    last = f;
  }
  for (int f = last + 1; f < want && last >= 0; ++f)
    std::copy(mag.begin() + static_cast<long>(last) * nb, mag.begin() + static_cast<long>(last + 1) * nb, mag.begin() + static_cast<long>(f) * nb);
}

}  // namespace rvc
