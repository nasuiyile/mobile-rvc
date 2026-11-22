// Playout delay control for the queue between the converter and a realtime consumer.
//
// The converter delivers one block every block period, `work` ms after the block's last input
// sample arrived; the consumer (audio callback, injection track) plays continuously. A block
// that is ready while `q` frames are still queued plays q frames later, so the queue is pure
// delay, and whatever was queued when the stream started (a slow first inference) or was
// inserted by an underrun (a late block) stays there for good. This class keeps that queue at
// the smallest level that still absorbs the measured jitter of the inference time:
//
//   * the consumer is starving (start, late block): a short silence in front of the block
//     rebuilds a jitter margin instead of leaving it to chance, and the controller remembers
//     that blocks can be that slow;
//   * more is queued than the jitter needs: the excess is removed, but only out of pauses;
//   * the margin is about to run out (inference getting slower, e.g. thermal throttling):
//     a pause is lengthened before the consumer underruns in the middle of speech.
//
// Audible content is never touched: edits happen inside stretches whose peak level is at least
// 34 dB below the recent peak (and below -42 dBFS), with one 5 ms guard frame on either side
// and a 5 ms cross-fade, so only the length of a pause changes, by a few milliseconds.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace rvc {

class PlayoutControl {
 public:
  // exact: `queued` passed to shape() is everything between the converter and the speaker side
  // (our own ring buffer). Otherwise it is only the part of the queue we can see (the pipe to
  // the injection service, which additionally holds up to one track buffer of audio).
  // burst_frames: how much the consumer takes at a time (its callback size), 0 if unknown.
  void configure(int sample_rate, int block_frames, bool exact, int burst_frames = 0) {
    sr_ = std::max(8000, sample_rate);
    block_ = std::max(1, block_frames);
    exact_ = exact;
    burst_ms_ = 1000.0 * std::max(0, burst_frames) / sr_;
    frame_ = std::max(16, sr_ / 200);  // 5 ms analysis / cross-fade frame
    fade_.resize(static_cast<size_t>(frame_));
    for (int i = 0; i < frame_; ++i) fade_[i] = static_cast<float>(0.5 - 0.5 * std::cos(M_PI * (i + 0.5) / frame_));
    reset();
    trimmed_ = padded_ = 0;
  }

  // New stream or new recording: nothing is queued downstream any more.
  void reset() {
    hist_.clear();
    peaks_.clear();
    works_.clear();
    primed_ = false;
    window_s_ = base_window_s();
    need_s_ = kSettleS;
    clean_blocks_ = 0;
    since_underrun_ = kNever;
    guard_ms_ = 0;
    guard_period_s_ = 120.0;
    queue_ms_ = delay_ms_ = 0;
  }

  // Call right before `blk` is queued.
  //   queued:  frames the consumer still has to play (see configure);
  //   age_ms:  time since the block's last input sample was captured;
  //   work_ms: the part of that the converter spent on this block (the rest it waited behind
  //            earlier blocks, i.e. the converter is catching up after a stall).
  // The block may come back shorter (pause shortened) or longer (silence in front / pause
  // lengthened).
  void shape(std::vector<float>& blk, size_t queued, double age_ms, double work_ms) {
    if (blk.empty()) return;
    const double per_ms = sr_ / 1000.0, block_s = static_cast<double>(block_) / sr_;
    works_.push_back(work_ms);
    if (works_.size() > 9) works_.pop_front();
    const double typ = typical_work();  // robust against the odd slow block
    // A block slower than this is a stall (the app was preempted, the GPU was busy drawing), not
    // jitter: it is survived, not planned for, unless it keeps happening.
    const double stall_ms = std::max(60.0, 0.6 * typ);
    note_peak(blk);
    if (since_underrun_ < kNever) {
      ++since_underrun_;
      // what an underrun taught us fades: half of it per period without another one
      const int64_t period = std::max<int64_t>(1, static_cast<int64_t>(guard_period_s_ / block_s));
      if (since_underrun_ % period == 0) guard_ms_ *= 0.5;
    }
    const bool waited = age_ms - work_ms > 25.0;

    if (exact_ ? queued == 0 : !primed_) {
      // The consumer has nothing left: this block is late (or the first one). Playing it right
      // away would leave no room for the next block to be any slower than this one.
      const bool first = !primed_;
      double pad_ms = exact_ ? clamp(0.3 * typ + 15.0, 25.0, 120.0) + burst_ms_ : clamp(0.25 * typ + 5.0, 15.0, 60.0);
      size_t pad = static_cast<size_t>(pad_ms * per_ms);
      blk.insert(blk.begin(), pad, 0.f);
      padded_ += static_cast<int64_t>(pad);
      hist_.clear();
      if (!first) {
        // An underrun in mid-stream: blocks can be this slow, so keep room for it from now on.
        // A stall counts in full only when it comes back while the previous one is still
        // remembered (and is then remembered for twice as long); the first time, most of the
        // queue it leaves behind is given back as soon as there are pauses.
        const double over = std::max(0.0, work_ms - typ);
        const bool again = guard_ms_ > 1.0;
        guard_ms_ = std::max(guard_ms_, again ? over : std::min(over, stall_ms));
        if (again) guard_period_s_ = std::min(3600.0, guard_period_s_ * 2);
        window_s_ = std::min(kMaxWindowS, window_s_ * 2);
        need_s_ = window_s_;
        since_underrun_ = 0;
      }
      clean_blocks_ = 0;
      primed_ = true;
      publish(static_cast<double>(pad) / per_ms, age_ms);
      return;
    }

    const size_t cap = std::max<size_t>(4, static_cast<size_t>(std::ceil(window_s_ / block_s)));
    // Stalled blocks, and the ones that waited behind them, say nothing about the margin that
    // ordinary jitter needs: they are survived (and remembered in guard_ms_), not planned for.
    hist_.push_back({static_cast<double>(queued), work_ms, waited});
    while (hist_.size() > cap) hist_.pop_front();
    if (++clean_blocks_ * block_s > 4 * window_s_ && window_s_ > base_window_s()) {
      window_s_ = std::max(base_window_s(), window_s_ / 2);
      clean_blocks_ = 0;
    }
    const double covered_s = hist_.size() * block_s;
    const bool settled = covered_s >= std::min(need_s_, window_s_) - 1e-9;
    // the window: how little was queued when a block arrived, and how slow blocks were
    double low = -1, slowest = typ;
    for (const Sample& h : hist_) {
      if (h.waited || h.work > typ + stall_ms) continue;
      low = low < 0 ? h.q : std::min(low, h.q);
      slowest = std::max(slowest, h.work);
    }
    const bool regular = !waited && work_ms <= typ + stall_ms;

    double cut = 0, add = 0;
    if (exact_) {
      // The slowest block of the window should still find `target` queued: a base amount, half
      // of the jitter seen in the window (what was slow by x can be slow by 1.5 x), and two
      // bursts, because the consumer takes a whole burst at a moment unrelated to ours.
      const double target = std::max(clamp(15.0 + 0.15 * typ, 15.0, 60.0) + 0.5 * (slowest - typ), 2.0 * burst_ms_ + 4.0) * per_ms;
      // This block's playout delay (queue + its age): unlike the queue alone it does not depend
      // on how fast this particular block was. It must stay above what the last underruns
      // asked for.
      const double delay = static_cast<double>(queued) + age_ms * per_ms;
      const double floor = target + (typ + guard_ms_) * per_ms;
      const double room = low < 0 ? delay - floor : std::min(low - target, delay - floor);
      // (1) what a stall left behind is far more than any jitter needs: do not wait for the
      // window. How much is left for (2) depends on how well the jitter is known by now.
      const double ceiling = floor + (covered_s >= kSettleS ? 0.25 : 1.0) * stall_ms * per_ms;
      if (delay > ceiling) cut = std::min(delay - ceiling, room);
      // (2) a whole window never came closer than target + 6 ms to running dry
      if (settled && low >= 0 && room >= 6.0 * per_ms) cut = std::max(cut, room);
      // (3) nearly dry: inference is getting slower; lengthen a pause before speech underruns
      if (cut <= 0 && regular && queued < 0.35 * target) add = target - static_cast<double>(queued);
    } else {
      // only the surplus beyond the consumer's own buffer is visible; that buffer is margin enough
      if (queued > 60.0 * per_ms) cut = static_cast<double>(queued) - 20.0 * per_ms;
      if (settled && low >= 10.0 * per_ms) cut = std::max(cut, low);
    }

    if (cut >= frame_ / 2) {
      size_t done = trim_quiet(blk, static_cast<size_t>(cut));
      if (done) {
        trimmed_ += static_cast<int64_t>(done);
        for (auto& h : hist_) h.q = std::max(0.0, h.q - static_cast<double>(done));
      }
    } else if (add >= 1) {
      size_t done = stretch_quiet(blk, std::max<size_t>(static_cast<size_t>(add), static_cast<size_t>(frame_)));
      if (done) {
        padded_ += static_cast<int64_t>(done);
        for (auto& h : hist_) h.q += static_cast<double>(done);
      }
    }
    publish(static_cast<double>(queued) / per_ms, age_ms);
  }

  // Smoothed over the last blocks; readable from other threads.
  double queue_ms() const { return queue_ms_.load(std::memory_order_relaxed); }  // queued when a block arrives
  double delay_ms() const { return delay_ms_.load(std::memory_order_relaxed); }  // input complete -> starts playing
  int64_t trimmed_frames() const { return trimmed_.load(std::memory_order_relaxed); }
  int64_t padded_frames() const { return padded_.load(std::memory_order_relaxed); }

 private:
  static constexpr double kSettleS = 6.0;     // history needed after a (re)start
  static constexpr double kMaxWindowS = 120.0;
  static constexpr int64_t kNever = INT64_C(1) << 40;
  double base_window_s() const { return exact_ ? 15.0 : 3.0; }
  static double clamp(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

  int sr_ = 48000, block_ = 4800, frame_ = 240;
  double burst_ms_ = 0;
  bool exact_ = true, primed_ = false;
  std::vector<float> fade_;   // raised-cosine fade-in over one frame
  struct Sample {
    double q;     // frames queued when the block arrived, as it would be with the edits made since
    double work;  // what the converter spent on the block, ms
    bool waited;  // it also waited behind earlier blocks
  };
  std::deque<Sample> hist_;   // the window
  std::deque<float> peaks_;   // block peaks of the last ~3 s
  std::deque<double> works_;  // inference time of the last blocks
  double window_s_ = 15.0, need_s_ = kSettleS;
  double guard_ms_ = 0;       // how much slower than typical a block may be: learnt from underruns
  double guard_period_s_ = 120.0;  // half-life of that
  int64_t clean_blocks_ = 0, since_underrun_ = kNever;

  // lower median of the recent inference times
  double typical_work() const {
    std::vector<double> v(works_.begin(), works_.end());
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[(v.size() - 1) / 2];
  }
  std::atomic<double> queue_ms_{0}, delay_ms_{0};
  std::atomic<int64_t> trimmed_{0}, padded_{0};

  void publish(double q_ms, double work_ms) {
    double d = q_ms + work_ms, pq = queue_ms_.load(std::memory_order_relaxed);
    bool first = delay_ms_.load(std::memory_order_relaxed) == 0;
    queue_ms_.store(first ? q_ms : 0.8 * pq + 0.2 * q_ms, std::memory_order_relaxed);
    delay_ms_.store(first ? d : 0.8 * delay_ms_.load(std::memory_order_relaxed) + 0.2 * d, std::memory_order_relaxed);
  }

  void note_peak(const std::vector<float>& blk) {
    float p = 0;
    for (float v : blk) p = std::max(p, std::fabs(v));
    peaks_.push_back(p);
    const size_t keep = std::max<size_t>(2, static_cast<size_t>(std::ceil(3.0 * sr_ / block_)));
    while (peaks_.size() > keep) peaks_.pop_front();
  }

  // Sample ranges [first, second) that may be edited: runs of quiet 5 ms frames, minus one
  // guard frame at either end (the neighbours of a run are the decay / onset of speech).
  std::vector<std::pair<size_t, size_t>> quiet_ranges(const std::vector<float>& blk) const {
    float recent = 0;
    for (float p : peaks_) recent = std::max(recent, p);
    const float thr = std::max(1e-4f, std::min(0.008f, 0.02f * recent));
    std::vector<std::pair<size_t, size_t>> out;
    const size_t F = static_cast<size_t>(frame_), nf = blk.size() / F;
    size_t run = 0;
    for (size_t i = 0; i <= nf; ++i) {
      bool quiet = false;
      if (i < nf) {
        float p = 0;
        for (size_t k = i * F; k < (i + 1) * F; ++k) p = std::max(p, std::fabs(blk[k]));
        quiet = p < thr;
      }
      if (quiet) {
        ++run;
        continue;
      }
      if (run >= 4) out.emplace_back((i - run + 1) * F, (i - 1) * F);
      run = 0;
    }
    return out;
  }

  // Removes up to `want` samples from pauses; returns how many were removed.
  size_t trim_quiet(std::vector<float>& blk, size_t want) const {
    const size_t F = static_cast<size_t>(frame_);
    auto ranges = quiet_ranges(blk);
    size_t done = 0;
    for (size_t r = ranges.size(); r-- > 0 && done < want;) {  // back to front: indices stay valid
      const size_t s = ranges[r].first, e = ranges[r].second;
      const size_t k = std::min(want - done, e - s - F);
      if (k < F / 2) continue;
      // x[s..s+F) fades out while x[s+k..s+k+F) fades in, then everything in between goes
      for (size_t i = 0; i < F; ++i) blk[s + i] = blk[s + i] * (1.f - fade_[i]) + blk[s + k + i] * fade_[i];
      blk.erase(blk.begin() + static_cast<std::ptrdiff_t>(s + F), blk.begin() + static_cast<std::ptrdiff_t>(s + F + k));
      done += k;
    }
    return done;
  }

  // Lengthens one pause by up to `want` samples (at least one frame); returns the amount.
  size_t stretch_quiet(std::vector<float>& blk, size_t want) const {
    const size_t F = static_cast<size_t>(frame_);
    for (auto& rg : quiet_ranges(blk)) {
      const size_t s = rg.first, e = rg.second;
      const size_t k = std::min(want, e - s - F);
      if (k < F) continue;
      // replay x[s..p) after p = s + k; its start is cross-faded with what followed p
      const size_t p = s + k;
      std::vector<float> ins(blk.begin() + static_cast<std::ptrdiff_t>(s), blk.begin() + static_cast<std::ptrdiff_t>(p));
      for (size_t i = 0; i < F; ++i) ins[i] = ins[i] * fade_[i] + blk[p + i] * (1.f - fade_[i]);
      blk.insert(blk.begin() + static_cast<std::ptrdiff_t>(p), ins.begin(), ins.end());
      return k;
    }
    return 0;
  }
};

}  // namespace rvc
