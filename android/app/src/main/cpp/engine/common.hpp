// Shared helpers: logging, file I/O, half-float conversion, timing.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#define RVC_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "rvc", __VA_ARGS__)
#define RVC_LOGW(...) __android_log_print(ANDROID_LOG_WARN, "rvc", __VA_ARGS__)
#define RVC_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "rvc", __VA_ARGS__)
#else
#define RVC_LOGI(...) (fprintf(stderr, "[rvc] " __VA_ARGS__), fputc('\n', stderr))
#define RVC_LOGW(...) (fprintf(stderr, "[rvc][W] " __VA_ARGS__), fputc('\n', stderr))
#define RVC_LOGE(...) (fprintf(stderr, "[rvc][E] " __VA_ARGS__), fputc('\n', stderr))
#endif

namespace rvc {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::string read_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw Error("cannot open " + path);
  std::string s;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  s.resize(static_cast<size_t>(n));
  size_t got = n > 0 ? fread(&s[0], 1, static_cast<size_t>(n), f) : 0;
  fclose(f);
  if (got != static_cast<size_t>(n)) throw Error("short read " + path);
  return s;
}

inline void write_file(const std::string& path, const void* data, size_t n) {
  std::string tmp = path + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) throw Error("cannot write " + path);
  size_t w = n ? fwrite(data, 1, n, f) : 0;
  bool ok = (w == n) && fflush(f) == 0;
  fclose(f);
  if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
    remove(tmp.c_str());
    throw Error("write failed " + path);
  }
}

inline void write_file(const std::string& path, const std::string& s) { write_file(path, s.data(), s.size()); }

inline bool file_exists(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fclose(f);
  return true;
}

inline std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  return a.back() == '/' ? a + b : a + "/" + b;
}

inline void copy_file(const std::string& src, const std::string& dst) {
  std::string d = read_file(src);
  write_file(dst, d);
}

// IEEE 754 binary16 <-> binary32 (round-to-nearest-even), portable.
inline float half_to_float(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, f;
  if (exp == 0) {
    if (mant == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
      mant &= 0x3FFu;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &f, 4);
  return out;
}

inline uint16_t float_to_half(float v) {
  uint32_t x;
  memcpy(&x, &v, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t h = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return static_cast<uint16_t>(sign | h);
  }
  uint32_t h = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
  return static_cast<uint16_t>(sign | h);
}

inline uint16_t bf16_to_half(uint16_t b) {
  uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  memcpy(&f, &x, 4);
  return float_to_half(f);
}

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace rvc
