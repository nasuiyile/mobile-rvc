#include "noise_gate.hpp"

#include <algorithm>
#include <cmath>

namespace rvc {

// ------------------------------------------------------------------ FFT of any size
AnyFFT::AnyFFT(int n) : n_(n), pow2_((n & (n - 1)) == 0), fft_(1) {
  if (pow2_) {
    fft_ = FFT(n);
    return;
  }
  m_ = 1;
  while (m_ < 2 * n - 1) m_ <<= 1;
  fft_ = FFT(m_);
  chirp_.resize(n);
  for (int k = 0; k < n; ++k) {
    int64_t kk = (static_cast<int64_t>(k) * k) % (2LL * n);
    double ang = -M_PI * static_cast<double>(kk) / n;
    chirp_[k] = {static_cast<float>(std::cos(ang)), static_cast<float>(std::sin(ang))};
  }
  bfft_.assign(m_, {0.f, 0.f});
  bfft_[0] = std::conj(chirp_[0]);
  for (int k = 1; k < n; ++k) bfft_[k] = bfft_[m_ - k] = std::conj(chirp_[k]);
  fft_.forward(bfft_.data());
  buf_.resize(m_);
}

void AnyFFT::forward(std::complex<float>* a) {
  if (pow2_) {
    fft_.forward(a);
    return;
  }
  std::fill(buf_.begin(), buf_.end(), std::complex<float>(0.f, 0.f));
  for (int k = 0; k < n_; ++k) buf_[k] = a[k] * chirp_[k];
  fft_.forward(buf_.data());
  for (int k = 0; k < m_; ++k) buf_[k] = std::conj(buf_[k] * bfft_[k]);
  fft_.forward(buf_.data());  // conj(fft(conj(.))) = m * ifft(.)
  const float inv = 1.f / m_;
  for (int k = 0; k < n_; ++k) a[k] = std::conj(buf_[k]) * inv * chirp_[k];
}

void AnyFFT::inverse(std::complex<float>* a) {
  for (int k = 0; k < n_; ++k) a[k] = std::conj(a[k]);
  forward(a);
  for (int k = 0; k < n_; ++k) a[k] = std::conj(a[k]);
}

// ------------------------------------------------------------------ TorchGate
static std::vector<float> tg_linspace(float a, float b, int num, bool endpoint) {
  int n = endpoint ? num : num + 1;
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = n == 1 ? a : a + (b - a) * static_cast<float>(i) / (n - 1);
  if (!endpoint) v.pop_back();
  return v;
}

SpectralGate::SpectralGate(int sr, int n_fft, float prop_decrease, float n_std, float freq_smooth_hz, float time_smooth_ms)
    : n_fft_(n_fft), hop_(n_fft / 4), bins_(n_fft / 2 + 1), prop_(prop_decrease), n_std_(n_std), fft_(n_fft) {
  window_.resize(n_fft);
  for (int i = 0; i < n_fft; ++i) window_[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / n_fft));
  int ngf = static_cast<int>(freq_smooth_hz / (static_cast<float>(sr) / (n_fft / 2.f)));
  int ngt = static_cast<int>(time_smooth_ms / ((static_cast<float>(hop_) / sr) * 1000.f));
  ngf = std::max(ngf, 1);
  ngt = std::max(ngt, 1);
  auto vec = [&](int g) {
    std::vector<float> a = tg_linspace(0, 1, g + 1, false), b = tg_linspace(1, 0, g + 2, true);
    a.insert(a.end(), b.begin(), b.end());
    return std::vector<float>(a.begin() + 1, a.end() - 1);
  };
  std::vector<float> vf = vec(ngf), vt = vec(ngt);
  sf_ = static_cast<int>(vf.size());
  st_ = static_cast<int>(vt.size());
  smooth_.resize(static_cast<size_t>(sf_) * st_);
  double sum = 0;
  for (int i = 0; i < sf_; ++i)
    for (int j = 0; j < st_; ++j) sum += smooth_[static_cast<size_t>(i) * st_ + j] = vf[i] * vt[j];
  for (auto& v : smooth_) v = static_cast<float>(v / sum);
}

// torch.stft(center=True, pad_mode="constant", hann, onesided): X[t][f]
void SpectralGate::stft(const float* x, int n, std::vector<std::complex<float>>& X, int& frames) {
  int pad = n_fft_ / 2;
  frames = 1 + n / hop_;
  X.assign(static_cast<size_t>(frames) * bins_, {0.f, 0.f});
  std::vector<std::complex<float>> buf(n_fft_);
  for (int f = 0; f < frames; ++f) {
    for (int i = 0; i < n_fft_; ++i) {
      int j = f * hop_ + i - pad;
      buf[i] = {(j >= 0 && j < n) ? x[j] * window_[i] : 0.f, 0.f};
    }
    fft_.forward(buf.data());
    std::copy(buf.begin(), buf.begin() + bins_, X.begin() + static_cast<long>(f) * bins_);
  }
}

// amp_to_db(top_db=40): per frequency, clipped at (max over time - 40)
void SpectralGate::to_db(const std::vector<std::complex<float>>& X, int frames, int bins, std::vector<float>& db) {
  db.resize(static_cast<size_t>(frames) * bins);
  const double eps = 2.220446049250313e-16;
  for (size_t i = 0; i < db.size(); ++i) db[i] = static_cast<float>(20.0 * std::log10(std::abs(X[i]) + eps));
  for (int b = 0; b < bins; ++b) {
    float mx = -1e30f;
    for (int f = 0; f < frames; ++f) mx = std::max(mx, db[static_cast<size_t>(f) * bins + b]);
    for (int f = 0; f < frames; ++f) {
      float& v = db[static_cast<size_t>(f) * bins + b];
      v = std::max(v, mx - 40.f);
    }
  }
}

std::vector<float> SpectralGate::process(const float* x, int n, const float* noise, int nn) {
  int T = 0, TN = 0;
  std::vector<std::complex<float>> X, XN;
  std::vector<float> xdb, ndb;
  stft(x, n, X, T);
  to_db(X, T, bins_, xdb);
  stft(noise, nn, XN, TN);
  to_db(XN, TN, bins_, ndb);
  // stationary threshold: mean + n_std * std (unbiased) over time, per frequency
  std::vector<float> thr(bins_);
  for (int b = 0; b < bins_; ++b) {
    double s = 0, s2 = 0;
    for (int f = 0; f < TN; ++f) s += ndb[static_cast<size_t>(f) * bins_ + b];
    double mean = s / TN;
    for (int f = 0; f < TN; ++f) {
      double d = ndb[static_cast<size_t>(f) * bins_ + b] - mean;
      s2 += d * d;
    }
    double sd = TN > 1 ? std::sqrt(s2 / (TN - 1)) : 0.0;
    thr[b] = static_cast<float>(mean + sd * n_std_);
  }
  // mask -> prop_decrease -> 2-D smoothing (conv2d "same", zero padding) over [freq][time]
  std::vector<float> m(static_cast<size_t>(T) * bins_);
  for (int f = 0; f < T; ++f)
    for (int b = 0; b < bins_; ++b) {
      float sig = xdb[static_cast<size_t>(f) * bins_ + b] > thr[b] ? 1.f : 0.f;
      m[static_cast<size_t>(f) * bins_ + b] = prop_ * (sig - 1.f) + 1.f;
    }
  std::vector<float> ms(m.size(), 0.f);
  const int pf = (sf_ - 1) / 2, pt = (st_ - 1) / 2;
  for (int f = 0; f < T; ++f)
    for (int b = 0; b < bins_; ++b) {
      float acc = 0;
      for (int i = 0; i < sf_; ++i) {
        int bb = b + i - pf;
        if (bb < 0 || bb >= bins_) continue;
        for (int j = 0; j < st_; ++j) {
          int ff = f + j - pt;
          if (ff < 0 || ff >= T) continue;
          acc += smooth_[static_cast<size_t>(i) * st_ + j] * m[static_cast<size_t>(ff) * bins_ + bb];
        }
      }
      ms[static_cast<size_t>(f) * bins_ + b] = acc;
    }
  // torch.istft(center=True): overlap-add of windowed irfft, normalised by sum(window^2)
  int total = n_fft_ + hop_ * (T - 1);
  std::vector<float> y(static_cast<size_t>(total), 0.f), env(static_cast<size_t>(total), 0.f);
  std::vector<std::complex<float>> buf(n_fft_);
  const float inv = 1.f / n_fft_;
  for (int f = 0; f < T; ++f) {
    for (int b = 0; b < bins_; ++b) buf[b] = X[static_cast<size_t>(f) * bins_ + b] * ms[static_cast<size_t>(f) * bins_ + b];
    buf[0] = {buf[0].real(), 0.f};
    if (n_fft_ % 2 == 0) buf[bins_ - 1] = {buf[bins_ - 1].real(), 0.f};
    for (int b = bins_; b < n_fft_; ++b) buf[b] = std::conj(buf[n_fft_ - b]);
    fft_.inverse(buf.data());
    for (int i = 0; i < n_fft_; ++i) {
      y[static_cast<size_t>(f) * hop_ + i] += buf[i].real() * inv * window_[i];
      env[static_cast<size_t>(f) * hop_ + i] += window_[i] * window_[i];
    }
  }
  std::vector<float> out(static_cast<size_t>(n), 0.f);
  int start = n_fft_ / 2;
  for (int i = 0; i < n && start + i < total; ++i) {
    float e = env[start + i];
    out[i] = e > 1e-11f ? y[start + i] / e : 0.f;
  }
  return out;
}

}  // namespace rvc
