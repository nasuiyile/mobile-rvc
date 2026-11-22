// Tiny WAV reader/writer (PCM16 / PCM24 / PCM32 / float32, any channel count -> mono).
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"

namespace rvc {

inline std::vector<float> read_wav_mono(const std::string& path, int* sample_rate) {
  std::string d = read_file(path);
  auto u16 = [&](size_t p) { uint16_t v; memcpy(&v, d.data() + p, 2); return v; };
  auto u32 = [&](size_t p) { uint32_t v; memcpy(&v, d.data() + p, 4); return v; };
  if (d.size() < 12 || d.compare(0, 4, "RIFF") != 0 || d.compare(8, 4, "WAVE") != 0) throw Error("not a wav file");
  size_t p = 12;
  int fmt = 0, ch = 0, bits = 0;
  std::vector<float> out;
  while (p + 8 <= d.size()) {
    std::string id = d.substr(p, 4);
    uint32_t sz = u32(p + 4);
    size_t body = p + 8;
    if (id == "fmt ") {
      fmt = u16(body);
      ch = u16(body + 2);
      *sample_rate = static_cast<int>(u32(body + 4));
      bits = u16(body + 14);
      if (fmt == 0xFFFE && sz >= 26) fmt = u16(body + 24);
    } else if (id == "data") {
      if (!ch) throw Error("wav: data before fmt");
      size_t bps = bits / 8, frames = std::min<size_t>(sz, d.size() - body) / (bps * ch);
      out.resize(frames);
      for (size_t i = 0; i < frames; ++i) {
        double acc = 0;
        for (int c = 0; c < ch; ++c) {
          const char* s = d.data() + body + (i * ch + c) * bps;
          double v = 0;
          if (fmt == 3 && bits == 32) { float f; memcpy(&f, s, 4); v = f; }
          else if (fmt == 1 && bits == 16) { int16_t x; memcpy(&x, s, 2); v = x / 32768.0; }
          else if (fmt == 1 && bits == 24) { int32_t x = (static_cast<uint8_t>(s[0]) | (static_cast<uint8_t>(s[1]) << 8) | (static_cast<int8_t>(s[2]) << 16)); v = x / 8388608.0; }
          else if (fmt == 1 && bits == 32) { int32_t x; memcpy(&x, s, 4); v = x / 2147483648.0; }
          else throw Error("wav: unsupported sample format");
          acc += v;
        }
        out[i] = static_cast<float>(acc / ch);
      }
      return out;
    }
    p = body + sz + (sz & 1);
  }
  throw Error("wav: no data chunk");
}

inline void write_wav_f32(const std::string& path, const std::vector<float>& x, int sr) {
  std::string d;
  auto put32 = [&](uint32_t v) { d.append(reinterpret_cast<const char*>(&v), 4); };
  auto put16 = [&](uint16_t v) { d.append(reinterpret_cast<const char*>(&v), 2); };
  uint32_t bytes = static_cast<uint32_t>(x.size() * 4);
  d += "RIFF"; put32(36 + bytes); d += "WAVEfmt "; put32(16);
  put16(3); put16(1); put32(static_cast<uint32_t>(sr)); put32(static_cast<uint32_t>(sr) * 4); put16(4); put16(32);
  d += "data"; put32(bytes);
  d.append(reinterpret_cast<const char*>(x.data()), bytes);
  write_file(path, d);
}

}  // namespace rvc
