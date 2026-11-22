// Lock-free single-producer / single-consumer float ring buffer (audio callback safe).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>
#include <vector>

namespace rvc {

class SpscRing {
 public:
  explicit SpscRing(size_t min_capacity = 1 << 16) { reset(min_capacity); }

  void reset(size_t min_capacity) {
    size_t cap = 1;
    while (cap < min_capacity) cap <<= 1;
    buf_.assign(cap, 0.f);
    mask_ = cap - 1;
    head_.store(0);
    tail_.store(0);
  }
  size_t capacity() const { return buf_.size(); }
  size_t available() const { return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire); }
  size_t space() const { return capacity() - available(); }

  // producer
  size_t write(const float* src, size_t n) {
    size_t h = head_.load(std::memory_order_relaxed), t = tail_.load(std::memory_order_acquire);
    n = std::min(n, capacity() - (h - t));
    for (size_t i = 0; i < n;) {
      size_t idx = (h + i) & mask_, chunk = std::min(n - i, capacity() - idx);
      memcpy(&buf_[idx], src + i, chunk * sizeof(float));
      i += chunk;
    }
    head_.store(h + n, std::memory_order_release);
    return n;
  }
  // consumer
  size_t read(float* dst, size_t n) {
    size_t t = tail_.load(std::memory_order_relaxed), h = head_.load(std::memory_order_acquire);
    n = std::min(n, h - t);
    for (size_t i = 0; i < n;) {
      size_t idx = (t + i) & mask_, chunk = std::min(n - i, capacity() - idx);
      memcpy(dst + i, &buf_[idx], chunk * sizeof(float));
      i += chunk;
    }
    tail_.store(t + n, std::memory_order_release);
    return n;
  }
  size_t skip(size_t n) {
    size_t t = tail_.load(std::memory_order_relaxed), h = head_.load(std::memory_order_acquire);
    n = std::min(n, h - t);
    tail_.store(t + n, std::memory_order_release);
    return n;
  }

 private:
  std::vector<float> buf_;
  size_t mask_ = 0;
  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};
};

}  // namespace rvc
