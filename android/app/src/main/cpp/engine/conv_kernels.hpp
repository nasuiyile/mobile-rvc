// Multi-tap 1-D convolution kernels used by the native NSF-HiFiGAN decoder.
//
//   Y[co][t] (+)= bias[co] + sum_ci sum_tap W[co][ci][tap] * X[ci][t + off[tap]]
//
// X is channel-major with zero margins (X[ci*ldx + t], t may be negative).  Ordinary,
// dilated, transposed (one call per polyphase component) and strided (input split into
// polyphase channels) convolutions all reduce to this form.  Co must be a multiple of 8.
#pragma once

#include <cstdint>
#include <vector>

namespace rvc {

class ThreadPool;

constexpr int kConvCoTile = 8;

struct ConvArgs {
  int Co = 0, Ci = 0, ntaps = 0;
  const int* off = nullptr;     // ntaps offsets
  const void* w = nullptr;      // packed [Co/8][Ci][ntaps][8] (float or fp16)
  const float* bias = nullptr;  // Co or null
  const void* x = nullptr;      // float or fp16, see above
  int ldx = 0;
  int T = 0;                    // outputs per channel (computed up to round_up(T, tile))
  float* y = nullptr;
  int ldy = 0;
  bool accumulate = false;
};

// Output columns are produced in tiles; buffers must have this many spare columns.
constexpr int kConvTileT = 24;

void conv_f32(const ConvArgs& a, ThreadPool& pool);
// fp16 storage + fp16 FMA (ARMv8.2-A FP16), periodic fp32 flush. Only call if fp16_supported().
void conv_f16(const ConvArgs& a, ThreadPool& pool);
bool fp16_supported();

void f32_to_f16(const float* src, uint16_t* dst, int n);

}  // namespace rvc
