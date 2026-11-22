// fp32 multi-tap conv kernel (NEON on arm64, scalar elsewhere) + helpers.
#include <algorithm>
#include <cstring>

#include "common.hpp"
#include "conv_kernels.hpp"
#include "thread_pool.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rvc {

namespace {

constexpr int kTT32 = 12;  // fp32 register tile: 8 co x 12 t

struct Split {
  int tchunks, cogroups, chunk_t, co_per_group;
};

Split make_split(int Co, int T, int tile, int threads) {
  int cotiles = Co / kConvCoTile;
  int ttiles = (T + tile - 1) / tile;
  int target = std::max(1, threads * 4);
  // prefer splitting time (weights stay hot per co tile), then co
  int tiles_per_chunk = std::max(1, std::min(8, ttiles / std::max(1, target)));
  int tchunks = (ttiles + tiles_per_chunk - 1) / tiles_per_chunk;
  int cogroups = 1;
  while (tchunks * cogroups < target && cogroups < cotiles) cogroups *= 2;
  cogroups = std::min(cogroups, cotiles);
  int co_per_group = (cotiles + cogroups - 1) / cogroups;
  return {tchunks, (cotiles + co_per_group - 1) / co_per_group, tiles_per_chunk * tile, co_per_group};
}

void tile_f32(const ConvArgs& a, int cot, int t0) {
  const float* W = static_cast<const float*>(a.w);
  const float* X = static_cast<const float*>(a.x);
  const int Ci = a.Ci, nt = a.ntaps;
#if defined(__aarch64__)
  float32x4_t acc[8][3];
  for (int c = 0; c < 8; ++c)
    for (int k = 0; k < 3; ++k) acc[c][k] = vdupq_n_f32(0.f);
  const float* wp = W + static_cast<size_t>(cot) * Ci * nt * 8;
  for (int ci = 0; ci < Ci; ++ci) {
    const float* xr = X + static_cast<ptrdiff_t>(ci) * a.ldx + t0;
    for (int tp = 0; tp < nt; ++tp, wp += 8) {
      const float* xp = xr + a.off[tp];
      float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp + 4), x2 = vld1q_f32(xp + 8);
      float32x4_t w0 = vld1q_f32(wp), w1 = vld1q_f32(wp + 4);
#define RVC_FMA(c, w, l)                                 \
  acc[c][0] = vfmaq_laneq_f32(acc[c][0], x0, w, l);      \
  acc[c][1] = vfmaq_laneq_f32(acc[c][1], x1, w, l);      \
  acc[c][2] = vfmaq_laneq_f32(acc[c][2], x2, w, l);
      RVC_FMA(0, w0, 0) RVC_FMA(1, w0, 1) RVC_FMA(2, w0, 2) RVC_FMA(3, w0, 3)
      RVC_FMA(4, w1, 0) RVC_FMA(5, w1, 1) RVC_FMA(6, w1, 2) RVC_FMA(7, w1, 3)
#undef RVC_FMA
    }
  }
  for (int c = 0; c < 8; ++c) {
    int co = cot * 8 + c;
    float* yp = a.y + static_cast<size_t>(co) * a.ldy + t0;
    float32x4_t b = vdupq_n_f32(a.bias ? a.bias[co] : 0.f);
    for (int k = 0; k < 3; ++k) {
      float32x4_t v = vaddq_f32(acc[c][k], b);
      if (a.accumulate) v = vaddq_f32(v, vld1q_f32(yp + 4 * k));
      vst1q_f32(yp + 4 * k, v);
    }
  }
#else
  float acc[8][kTT32] = {};
  const float* wp = W + static_cast<size_t>(cot) * Ci * nt * 8;
  for (int ci = 0; ci < Ci; ++ci) {
    const float* xr = X + static_cast<ptrdiff_t>(ci) * a.ldx + t0;
    for (int tp = 0; tp < nt; ++tp, wp += 8) {
      const float* xp = xr + a.off[tp];
      for (int c = 0; c < 8; ++c) {
        float w = wp[c];
        for (int t = 0; t < kTT32; ++t) acc[c][t] += w * xp[t];
      }
    }
  }
  for (int c = 0; c < 8; ++c) {
    int co = cot * 8 + c;
    float* yp = a.y + static_cast<size_t>(co) * a.ldy + t0;
    float b = a.bias ? a.bias[co] : 0.f;
    for (int t = 0; t < kTT32; ++t) yp[t] = acc[c][t] + b + (a.accumulate ? yp[t] : 0.f);
  }
#endif
}

}  // namespace

void conv_f32(const ConvArgs& a, ThreadPool& pool) {
  Split s = make_split(a.Co, a.T, kTT32, pool.size());
  int cotiles = a.Co / kConvCoTile;
  pool.parallel_for(s.tchunks * s.cogroups, [&](int task) {
    int tc = task % s.tchunks, cg = task / s.tchunks;
    int tbeg = tc * s.chunk_t, tend = std::min(a.T, tbeg + s.chunk_t);
    int cbeg = cg * s.co_per_group, cend = std::min(cotiles, cbeg + s.co_per_group);
    for (int cot = cbeg; cot < cend; ++cot)
      for (int t0 = tbeg; t0 < tend; t0 += kTT32) tile_f32(a, cot, t0);
  });
}

#if !defined(__aarch64__)
bool fp16_supported() { return false; }
void conv_f16(const ConvArgs& a, ThreadPool& pool) { (void)a; (void)pool; }
void f32_to_f16(const float* src, uint16_t* dst, int n) {
  for (int i = 0; i < n; ++i) dst[i] = float_to_half(src[i]);
}
#endif

}  // namespace rvc
