// Small fork-join pool for data-parallel kernels (caller thread participates).
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__ANDROID__) || defined(__linux__)
#include <sched.h>
#include <sys/resource.h>
#endif

namespace rvc {

class ThreadPool {
 public:
  explicit ThreadPool(int threads, std::vector<int> affinity = {}) : affinity_(std::move(affinity)) {
    int n = std::max(1, threads) - 1;
    for (int i = 0; i < n; ++i) workers_.emplace_back([this, i] { loop(i); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      quit_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  int size() const { return static_cast<int>(workers_.size()) + 1; }

  // Runs fn(i) for i in [0, n); returns when all are done.
  void parallel_for(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    if (workers_.empty() || n == 1) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = &fn;
      n_ = n;
      next_.store(0);
      active_.store(static_cast<int>(workers_.size()));
      ++gen_;
    }
    cv_.notify_all();
    work();
    // spin briefly, then block until workers finish
    for (int spin = 0; active_.load(std::memory_order_acquire) > 0; ++spin) {
      if (spin > 2000) {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return active_.load() == 0; });
        break;
      }
      std::this_thread::yield();
    }
    fn_ = nullptr;
  }

 private:
  std::vector<std::thread> workers_;
  std::vector<int> affinity_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int)>* fn_ = nullptr;
  int n_ = 0;
  uint64_t gen_ = 0;
  bool quit_ = false;
  std::atomic<int> next_{0}, active_{0};

  void work() {
    for (;;) {
      int i = next_.fetch_add(1, std::memory_order_relaxed);
      if (i >= n_) break;
      (*fn_)(i);
    }
  }

  void loop(int idx) {
#if defined(__ANDROID__) || defined(__linux__)
    if (!affinity_.empty()) {
      cpu_set_t set;
      CPU_ZERO(&set);
      for (int c : affinity_) CPU_SET(c, &set);
      sched_setaffinity(0, sizeof(set), &set);
    }
    setpriority(PRIO_PROCESS, 0, -16);  // compute helpers of the realtime audio worker
#endif
    (void)idx;
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        if (quit_) return;
      }
      work();
      if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lk(mu_);
        done_cv_.notify_all();
      }
    }
  }
};

}  // namespace rvc
