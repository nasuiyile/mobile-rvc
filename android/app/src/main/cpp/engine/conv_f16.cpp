// fp16 multi-tap conv kernel. Compiled with -march=armv8.2-a+fp16 and only used after a
// runtime check, so the library still loads on ARMv8.0 CPUs.
#if defined(__aarch64__)

#include <arm_neon.h>

#include <algorithm>
#include <cstring>

#include "conv_kernels.hpp"
#include "thread_pool.hpp"

#if defined(__ANDROID__) || defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1 << 10)
#endif
#ifndef HWCAP_FPHP
#define HWCAP_FPHP (1 << 9)
#endif
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace rvc {

bool fp16_supported() {
#if defined(__ANDROID__) || defined(__linux__)
  unsigned long hw = getauxval(AT_HWCAP);
  return (hw & HWCAP_ASIMDHP) && (hw & HWCAP_FPHP);
#elif defined(__APPLE__)
  int v = 0;
  size_t n = sizeof v;
  return sysctlbyname("hw.optional.arm.FEAT_FP16", &v, &n, nullptr, 0) == 0 && v == 1;
#else
  return false;
#endif
}

void f32_to_f16(const float* src, uint16_t* dst, int n) {
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src + i)), vcvt_f16_f32(vld1q_f32(src + i + 4)));
    vst1q_u16(dst + i, vreinterpretq_u16_f16(h));
  }
  for (; i < n; ++i) {
    __fp16 h = static_cast<__fp16>(src[i]);
    memcpy(dst + i, &h, 2);
  }
}

namespace {

constexpr int kTT16 = 24;  // 8 co x 24 t (3 x float16x8)
constexpr int kFlushTerms = 64;  // fp16 terms per fp32 flush: decoder SNR ~55 dB vs fp32

struct Split {
  int tchunks, cogroups, chunk_t, co_per_group;
};

Split make_split(int Co, int T, int tile, int threads) {
  int cotiles = Co / kConvCoTile;
  int ttiles = (T + tile - 1) / tile;
  int target = std::max(1, threads * 4);
  int tiles_per_chunk = std::max(1, std::min(8, ttiles / std::max(1, target)));
  int tchunks = (ttiles + tiles_per_chunk - 1) / tiles_per_chunk;
  int cogroups = 1;
  while (tchunks * cogroups < target && cogroups < cotiles) cogroups *= 2;
  cogroups = std::min(cogroups, cotiles);
  int co_per_group = (cotiles + cogroups - 1) / cogroups;
  return {tchunks, (cotiles + co_per_group - 1) / co_per_group, tiles_per_chunk * tile, co_per_group};
}

__attribute__((always_inline)) inline void flush(float* y, int ldy, int t0, float16x8_t (&acc)[8][3], bool first,
                                                const ConvArgs& a, int cot) {
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    int co = cot * 8 + c;
    float* yp = y + static_cast<size_t>(co) * ldy + t0;
    float b = (first && a.bias) ? a.bias[co] : 0.f;
    float32x4_t bv = vdupq_n_f32(b);
    bool add_prev = !first || a.accumulate;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      float32x4_t lo = vaddq_f32(vcvt_f32_f16(vget_low_f16(acc[c][k])), bv);
      float32x4_t hi = vaddq_f32(vcvt_high_f32_f16(acc[c][k]), bv);
      if (add_prev) {
        lo = vaddq_f32(lo, vld1q_f32(yp + 8 * k));
        hi = vaddq_f32(hi, vld1q_f32(yp + 8 * k + 4));
      }
      vst1q_f32(yp + 8 * k, lo);
      vst1q_f32(yp + 8 * k + 4, hi);
      acc[c][k] = vdupq_n_f16(0);
    }
  }
}

void tile_f16(const ConvArgs& a, int cot, int t0) {
  const __fp16* W = static_cast<const __fp16*>(a.w);
  const __fp16* X = static_cast<const __fp16*>(a.x);
  const int Ci = a.Ci, nt = a.ntaps;
  const int ci_chunk = std::max(1, kFlushTerms / std::max(1, nt));
  float16x8_t acc[8][3];
#pragma unroll
  for (int c = 0; c < 8; ++c)
#pragma unroll
    for (int k = 0; k < 3; ++k) acc[c][k] = vdupq_n_f16(0);
  const __fp16* wp = W + static_cast<size_t>(cot) * Ci * nt * 8;
  bool first = true;
  for (int c0 = 0; c0 < Ci; c0 += ci_chunk) {
    int c1 = std::min(Ci, c0 + ci_chunk);
    for (int ci = c0; ci < c1; ++ci) {
      const __fp16* xr = X + static_cast<ptrdiff_t>(ci) * a.ldx + t0;
      for (int tp = 0; tp < nt; ++tp, wp += 8) {
        const __fp16* xp = xr + a.off[tp];
        float16x8_t x0 = vld1q_f16(xp), x1 = vld1q_f16(xp + 8), x2 = vld1q_f16(xp + 16);
        float16x8_t w = vld1q_f16(wp);
#define RVC_FMA(l)                                          \
  acc[l][0] = vfmaq_laneq_f16(acc[l][0], x0, w, l);         \
  acc[l][1] = vfmaq_laneq_f16(acc[l][1], x1, w, l);         \
  acc[l][2] = vfmaq_laneq_f16(acc[l][2], x2, w, l);
        RVC_FMA(0) RVC_FMA(1) RVC_FMA(2) RVC_FMA(3) RVC_FMA(4) RVC_FMA(5) RVC_FMA(6) RVC_FMA(7)
#undef RVC_FMA
      }
    }
    flush(a.y, a.ldy, t0, acc, first, a, cot);
    first = false;
  }
}

}  // namespace

void conv_f16(const ConvArgs& a, ThreadPool& pool) {
  Split s = make_split(a.Co, a.T, kTT16, pool.size());
  int cotiles = a.Co / kConvCoTile;
  pool.parallel_for(s.tchunks * s.cogroups, [&](int task) {
    int tc = task % s.tchunks, cg = task / s.tchunks;
    int tbeg = tc * s.chunk_t, tend = std::min(a.T, tbeg + s.chunk_t);
    int cbeg = cg * s.co_per_group, cend = std::min(cotiles, cbeg + s.co_per_group);
    for (int cot = cbeg; cot < cend; ++cot)
      for (int t0 = tbeg; t0 < tend; t0 += kTT16) tile_f16(a, cot, t0);
  });
}

}  // namespace rvc

#endif  // __aarch64__
