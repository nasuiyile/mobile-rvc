#include "ivf_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "common.hpp"
#include "pth_reader.hpp"  // MappedFile

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rvc {

namespace {

struct Cursor {
  const uint8_t* d;
  size_t n, p = 0;
  void need(size_t k) const { if (p + k > n) throw Error("faiss index truncated"); }
  template <typename T> T rd() { need(sizeof(T)); T v; memcpy(&v, d + p, sizeof(T)); p += sizeof(T); return v; }
  std::string fourcc() { need(4); std::string s(reinterpret_cast<const char*>(d + p), 4); p += 4; return s; }
  void skip(size_t k) { need(k); p += k; }
};

void read_header(Cursor& c, int32_t& d, int64_t& ntotal, int32_t& metric) {
  d = c.rd<int32_t>();
  ntotal = c.rd<int64_t>();
  c.rd<int64_t>();
  c.rd<int64_t>();
  c.rd<uint8_t>();  // is_trained
  metric = c.rd<int32_t>();
  if (metric > 1) c.rd<float>();
}

inline float dot(const float* a, const float* b, int n) {
#if defined(__aarch64__)
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  float s = vaddvq_f32(vaddq_f32(s0, s1));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
#else
  float s = 0;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
#endif
}

inline float l2_f16(const float* q, const uint16_t* v, int n) {
#if defined(__aarch64__)
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(v + i));
    float32x4_t d0 = vsubq_f32(vld1q_f32(q + i), vcvt_f32_f16(vget_low_f16(h)));
    float32x4_t d1 = vsubq_f32(vld1q_f32(q + i + 4), vcvt_f32_f16(vget_high_f16(h)));
    s0 = vfmaq_f32(s0, d0, d0);
    s1 = vfmaq_f32(s1, d1, d1);
  }
  float s = vaddvq_f32(vaddq_f32(s0, s1));
  for (; i < n; ++i) { float d = q[i] - half_to_float(v[i]); s += d * d; }
  return s;
#else
  float s = 0;
  for (int i = 0; i < n; ++i) { float d = q[i] - half_to_float(v[i]); s += d * d; }
  return s;
#endif
}

inline float l2_f32(const float* q, const float* v, int n) {
  float s = 0;
  for (int i = 0; i < n; ++i) { float d = q[i] - v[i]; s += d * d; }
  return s;
}

}  // namespace

int64_t convert_faiss_index(const std::string& faiss_path, const std::string& out_path, int* dim_out) {
  MappedFile mf(faiss_path);
  Cursor c{mf.data(), mf.size()};
  std::string h = c.fourcc();
  if (h != "IwFl") {
    throw Error("unsupported index type '" + h + "': use RVC's added_IVF*_Flat*.index (not trained_*.index)");
  }
  int32_t d, metric;
  int64_t ntotal;
  read_header(c, d, ntotal, metric);
  if (metric != 1) throw Error("index metric must be L2");
  uint64_t nlist = c.rd<uint64_t>();
  uint64_t nprobe = c.rd<uint64_t>();
  std::string qh = c.fourcc();
  if (qh != "IxF2" && qh != "IxFl") throw Error("unsupported index quantizer " + qh);
  int32_t qd, qm;
  int64_t qn;
  read_header(c, qd, qn, qm);
  uint64_t nfl = c.rd<uint64_t>();
  if (qd != d || static_cast<uint64_t>(qn) != nlist || nfl != nlist * static_cast<uint64_t>(d)) throw Error("bad index quantizer");
  c.need(nfl * 4);
  const float* cent = reinterpret_cast<const float*>(c.d + c.p);
  std::vector<float> centroids(cent, cent + nfl);  // copy: may be unaligned
  c.skip(nfl * 4);
  int8_t dm_type = c.rd<int8_t>();
  uint64_t dm_n = c.rd<uint64_t>();
  c.skip(dm_n * 8);
  if (dm_type == 2) { uint64_t hn = c.rd<uint64_t>(); c.skip(hn * 16); }
  if (c.fourcc() != "ilar") throw Error("unsupported inverted list storage (need ArrayInvertedLists)");
  uint64_t nl = c.rd<uint64_t>(), code_size = c.rd<uint64_t>();
  if (nl != nlist || code_size != static_cast<uint64_t>(d) * 4) throw Error("bad inverted lists");
  std::vector<uint64_t> sizes(nl, 0);
  std::string lt = c.fourcc();
  if (lt == "full") {
    uint64_t n = c.rd<uint64_t>();
    if (n != nl) throw Error("bad list sizes");
    for (uint64_t i = 0; i < n; ++i) sizes[i] = c.rd<uint64_t>();
  } else if (lt == "sprs") {
    uint64_t n = c.rd<uint64_t>();
    for (uint64_t i = 0; i + 1 < n; i += 2) {
      uint64_t li = c.rd<uint64_t>(), sz = c.rd<uint64_t>();
      if (li < nl) sizes[li] = sz;
    }
    if (n % 2) c.rd<uint64_t>();
  } else {
    throw Error("bad inverted list type");
  }
  uint64_t total = std::accumulate(sizes.begin(), sizes.end(), uint64_t{0});
  std::vector<uint64_t> offsets(nl + 1, 0);
  std::vector<uint16_t> vec(total * static_cast<uint64_t>(d));
  uint64_t row = 0;
  for (uint64_t i = 0; i < nl; ++i) {
    offsets[i] = row;
    uint64_t k = sizes[i];
    if (!k) continue;
    c.need(k * code_size);
    const uint8_t* src = c.d + c.p;
    for (uint64_t r = 0; r < k * static_cast<uint64_t>(d); ++r) {
      float f;
      memcpy(&f, src + r * 4, 4);
      vec[row * d + r] = float_to_half(f);
    }
    c.skip(k * code_size);
    c.skip(k * 8);  // ids: RVC uses big_npy[ids], i.e. these same vectors
    row += k;
  }
  offsets[nl] = row;

  std::string out;
  out.append("RVCIDX1", 8);
  auto put32 = [&](uint32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); };
  put32(static_cast<uint32_t>(d));
  put32(static_cast<uint32_t>(nl));
  put32(static_cast<uint32_t>(std::max<uint64_t>(1, nprobe)));
  put32(1);
  out.append(reinterpret_cast<const char*>(&total), 8);
  out.append(reinterpret_cast<const char*>(centroids.data()), centroids.size() * 4);
  out.append(reinterpret_cast<const char*>(offsets.data()), offsets.size() * 8);
  out.append(reinterpret_cast<const char*>(vec.data()), vec.size() * 2);
  write_file(out_path, out);
  if (dim_out) *dim_out = d;
  return static_cast<int64_t>(total);
}

IvfIndex::IvfIndex(const std::string& path) : file_(new MappedFile(path)) {
  const uint8_t* p = file_->data();
  if (file_->size() < 32 || memcmp(p, "RVCIDX1", 8) != 0) throw Error("bad index.bin");
  uint32_t hdr[4];
  memcpy(hdr, p + 8, 16);
  dim_ = static_cast<int>(hdr[0]);
  nlist_ = static_cast<int>(hdr[1]);
  nprobe_ = static_cast<int>(hdr[2]);
  f16_ = hdr[3] == 1;
  memcpy(&ntotal_, p + 24, 8);
  size_t off = 32;
  centroids_ = reinterpret_cast<const float*>(p + off);
  off += static_cast<size_t>(nlist_) * dim_ * 4;
  offsets_ = reinterpret_cast<const uint64_t*>(p + off);
  off += static_cast<size_t>(nlist_ + 1) * 8;
  size_t need = off + static_cast<size_t>(ntotal_) * dim_ * (f16_ ? 2 : 4);
  if (need > file_->size()) throw Error("index.bin truncated");
  if (f16_) vec16_ = reinterpret_cast<const uint16_t*>(p + off);
  else vec32_ = reinterpret_cast<const float*>(p + off);
  cnorm_.resize(nlist_);
  for (int i = 0; i < nlist_; ++i) cnorm_[i] = dot(centroids_ + static_cast<size_t>(i) * dim_, centroids_ + static_cast<size_t>(i) * dim_, dim_);
}

IvfIndex::~IvfIndex() = default;

void IvfIndex::search_blend(const float* q, int n, float* out, int k, int nprobe) const {
  if (nprobe <= 0) nprobe = nprobe_;
  nprobe = std::min(nprobe, nlist_);
  std::vector<std::pair<float, int>> cd(nlist_);
  std::vector<std::pair<float, int64_t>> best;
  std::vector<float> acc(dim_);
  for (int r = 0; r < n; ++r) {
    const float* x = q + static_cast<size_t>(r) * dim_;
    for (int i = 0; i < nlist_; ++i) cd[i] = {cnorm_[i] - 2.f * dot(centroids_ + static_cast<size_t>(i) * dim_, x, dim_), i};
    std::partial_sort(cd.begin(), cd.begin() + nprobe, cd.end());
    best.clear();
    for (int pi = 0; pi < nprobe; ++pi) {
      int l = cd[pi].second;
      for (uint64_t v = offsets_[l]; v < offsets_[l + 1]; ++v) {
        float dist = f16_ ? l2_f16(x, vec16_ + v * dim_, dim_) : l2_f32(x, vec32_ + v * dim_, dim_);
        if (static_cast<int>(best.size()) < k) {
          best.emplace_back(dist, static_cast<int64_t>(v));
          std::push_heap(best.begin(), best.end());
        } else if (dist < best.front().first) {
          std::pop_heap(best.begin(), best.end());
          best.back() = {dist, static_cast<int64_t>(v)};
          std::push_heap(best.begin(), best.end());
        }
      }
    }
    float* o = out + static_cast<size_t>(r) * dim_;
    if (best.empty()) {
      memcpy(o, x, sizeof(float) * dim_);
      continue;
    }
    double wsum = 0;
    std::vector<double> w(best.size());
    for (size_t i = 0; i < best.size(); ++i) {
      double d = std::max<double>(best[i].first, 1e-12);
      w[i] = 1.0 / (d * d);
      wsum += w[i];
    }
    std::fill(acc.begin(), acc.end(), 0.f);
    for (size_t i = 0; i < best.size(); ++i) {
      float wi = static_cast<float>(w[i] / wsum);
      int64_t v = best[i].second;
      if (f16_) {
        const uint16_t* src = vec16_ + v * dim_;
        for (int j = 0; j < dim_; ++j) acc[j] += wi * half_to_float(src[j]);
      } else {
        const float* src = vec32_ + v * dim_;
        for (int j = 0; j < dim_; ++j) acc[j] += wi * src[j];
      }
    }
    memcpy(o, acc.data(), sizeof(float) * dim_);
  }
}

}  // namespace rvc
