#include "offline_file.hpp"

#include <fcntl.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>
#include <media/NdkMediaMuxer.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "../engine/common.hpp"

namespace rvc {

namespace {

// android.media.AudioFormat
constexpr int kPcm16 = 2, kPcm8 = 3, kPcmFloat = 4, kPcm24Packed = 21, kPcm32 = 22;

// WeChat / QQ voice messages: SILK, which no Android decoder takes.
bool is_silk(const std::string& path) {
  char head[16] = {0};
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  size_t n = fread(head, 1, sizeof(head), f);
  fclose(f);
  return n >= 7 && (memcmp(head, "#!SILK", 6) == 0 || memcmp(head + 1, "#!SILK", 6) == 0);
}

Error unreadable(const std::string& path, const std::string& why) {
  if (is_silk(path))
    return Error("这是微信 / QQ 的语音消息（SILK 编码），系统解码器不支持。请先用其他工具转成 mp3、m4a 或 wav 再导入。");
  return Error("无法读取这个音频文件：" + why);
}

}  // namespace

// ------------------------------------------------------------------ decoder
MediaDecoder::MediaDecoder(const std::string& path) {
  try {
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw Error("cannot open " + path);
    struct stat sb {};
    fstat(fd_, &sb);
    ex_ = AMediaExtractor_new();
    if (AMediaExtractor_setDataSourceFd(ex_, fd_, 0, sb.st_size) != AMEDIA_OK) throw unreadable(path, "不是系统支持的音频或视频格式");
    AMediaFormat* fmt = nullptr;
    const size_t tracks = AMediaExtractor_getTrackCount(ex_);
    for (size_t i = 0; i < tracks && !fmt; ++i) {
      AMediaFormat* f = AMediaExtractor_getTrackFormat(ex_, i);
      const char* m = nullptr;
      if (AMediaFormat_getString(f, AMEDIAFORMAT_KEY_MIME, &m) && m && strncmp(m, "audio/", 6) == 0) {
        mime_ = m;
        fmt = f;
        AMediaExtractor_selectTrack(ex_, i);
      } else {
        AMediaFormat_delete(f);
      }
    }
    if (!fmt) throw unreadable(path, "文件里没有音轨");
    int64_t dur = 0;
    int32_t v = 0;
    if (AMediaFormat_getInt64(fmt, AMEDIAFORMAT_KEY_DURATION, &dur) && dur > 0) duration_ = dur / 1e6;
    // provisional: the decoder's output format is what counts (HE-AAC doubles the rate)
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_SAMPLE_RATE, &v)) rate_ = v;
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_CHANNEL_COUNT, &v)) channels_ = std::max(1, static_cast<int>(v));
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_PCM_ENCODING, &v)) encoding_ = v;
    codec_ = AMediaCodec_createDecoderByType(mime_.c_str());
    media_status_t st = codec_ ? AMediaCodec_configure(codec_, fmt, nullptr, nullptr, 0) : AMEDIA_ERROR_UNSUPPORTED;
    AMediaFormat_delete(fmt);
    if (st != AMEDIA_OK || AMediaCodec_start(codec_) != AMEDIA_OK) throw unreadable(path, "系统没有 " + mime_ + " 的解码器");
  } catch (...) {
    close_all();
    throw;
  }
}

MediaDecoder::~MediaDecoder() { close_all(); }

void MediaDecoder::close_all() {
  if (codec_) {
    AMediaCodec_stop(codec_);
    AMediaCodec_delete(codec_);
    codec_ = nullptr;
  }
  if (ex_) {
    AMediaExtractor_delete(ex_);
    ex_ = nullptr;
  }
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

void MediaDecoder::update_format() {
  AMediaFormat* f = AMediaCodec_getOutputFormat(codec_);
  if (!f) return;
  int32_t v = 0;
  if (AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_SAMPLE_RATE, &v) && v > 0) rate_ = v;
  if (AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_CHANNEL_COUNT, &v) && v > 0) channels_ = v;
  encoding_ = AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_PCM_ENCODING, &v) ? v : kPcm16;
  AMediaFormat_delete(f);
}

void MediaDecoder::append_mono(const uint8_t* p, size_t bytes, std::vector<float>& out) {
  int bps;
  switch (encoding_) {
    case kPcm8: bps = 1; break;
    case kPcm16: bps = 2; break;
    case kPcm24Packed: bps = 3; break;
    case kPcmFloat:
    case kPcm32: bps = 4; break;
    default: throw Error("unsupported PCM encoding " + std::to_string(encoding_));
  }
  const int ch = channels_;
  const size_t frames = bytes / (static_cast<size_t>(bps) * ch);
  const size_t base = out.size();
  out.resize(base + frames);
  const float inv = 1.f / ch;
  for (size_t i = 0; i < frames; ++i) {
    float acc = 0;
    for (int c = 0; c < ch; ++c) {
      const uint8_t* s = p + (i * ch + c) * bps;
      switch (encoding_) {
        case kPcm8: acc += (static_cast<int>(s[0]) - 128) / 128.f; break;
        case kPcm16: { int16_t x; memcpy(&x, s, 2); acc += x / 32768.f; break; }
        case kPcm24Packed: { int32_t x = s[0] | (s[1] << 8) | (static_cast<int8_t>(s[2]) * 65536); acc += x / 8388608.f; break; }
        case kPcmFloat: { float x; memcpy(&x, s, 4); acc += x; break; }
        default: { int32_t x; memcpy(&x, s, 4); acc += static_cast<float>(x / 2147483648.0); break; }
      }
    }
    out[base + i] = acc * inv;
  }
}

bool MediaDecoder::read(std::vector<float>& out) {
  const size_t start = out.size();
  while (!output_eos_) {
    if (!input_eos_) {
      ssize_t ii = AMediaCodec_dequeueInputBuffer(codec_, 2000);
      if (ii >= 0) {
        size_t cap = 0;
        uint8_t* buf = AMediaCodec_getInputBuffer(codec_, static_cast<size_t>(ii), &cap);
        ssize_t n = buf ? AMediaExtractor_readSampleData(ex_, buf, cap) : -1;
        if (n < 0) {
          AMediaCodec_queueInputBuffer(codec_, static_cast<size_t>(ii), 0, 0, 0, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
          input_eos_ = true;
        } else {
          int64_t pts = AMediaExtractor_getSampleTime(ex_);
          AMediaCodec_queueInputBuffer(codec_, static_cast<size_t>(ii), 0, static_cast<size_t>(n), static_cast<uint64_t>(std::max<int64_t>(pts, 0)), 0);
          AMediaExtractor_advance(ex_);
        }
      }
    }
    AMediaCodecBufferInfo info{};
    ssize_t oi = AMediaCodec_dequeueOutputBuffer(codec_, &info, input_eos_ ? 10000 : 0);
    if (oi >= 0) {
      idle_ = 0;
      if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) output_eos_ = true;
      size_t cap = 0;
      uint8_t* buf = AMediaCodec_getOutputBuffer(codec_, static_cast<size_t>(oi), &cap);
      if (buf && info.size > 0) append_mono(buf + info.offset, static_cast<size_t>(info.size), out);
      AMediaCodec_releaseOutputBuffer(codec_, static_cast<size_t>(oi), false);
      if (out.size() > start) return true;
    } else if (oi == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
      update_format();
    } else if (oi == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
      // a decoder that never signals the end after its input ended: give up after ~3 s
      if (input_eos_ && ++idle_ > 300) output_eos_ = true;
    } else if (oi != AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
      throw Error("音频解码出错（" + std::to_string(oi) + "）");
    }
  }
  return out.size() > start;
}

Json probe_media(const std::string& path) {
  MediaDecoder d(path);
  Json j = Json::object();
  j.set("duration_s", d.duration());
  j.set("sample_rate", d.rate());
  j.set("channels", d.channels());
  j.set("mime", d.mime());
  return j;
}

// ------------------------------------------------------------------ writers
namespace {

class AudioWriter {
 public:
  virtual ~AudioWriter() = default;
  virtual void write(const int16_t* x, size_t n) = 0;
  virtual void finish() = 0;
};

class WavWriter : public AudioWriter {
 public:
  WavWriter(const std::string& path, int sr) : sr_(sr) {
    f_ = fopen(path.c_str(), "wb");
    if (!f_) throw Error("cannot write " + path);
    header(0);
  }
  ~WavWriter() override {
    if (f_) fclose(f_);
  }
  void write(const int16_t* x, size_t n) override {
    if (fwrite(x, 2, n, f_) != n) throw Error("写入失败（存储空间不足？）");
    bytes_ += n * 2;
  }
  void finish() override {
    if (bytes_ > 0xFFFFFFF0ull) throw Error("too long for a wav file");
    fseek(f_, 0, SEEK_SET);
    header(static_cast<uint32_t>(bytes_));
    bool ok = fflush(f_) == 0;
    ok = fclose(f_) == 0 && ok;
    f_ = nullptr;
    if (!ok) throw Error("写入失败（存储空间不足？）");
  }

 private:
  void header(uint32_t data) {
    uint8_t h[44];
    auto p32 = [&](int at, uint32_t v) { memcpy(h + at, &v, 4); };
    auto p16 = [&](int at, uint16_t v) { memcpy(h + at, &v, 2); };
    memcpy(h, "RIFF", 4);
    p32(4, 36 + data);
    memcpy(h + 8, "WAVEfmt ", 8);
    p32(16, 16);
    p16(20, 1);  // PCM
    p16(22, 1);  // mono
    p32(24, static_cast<uint32_t>(sr_));
    p32(28, static_cast<uint32_t>(sr_) * 2);
    p16(32, 2);
    p16(34, 16);
    memcpy(h + 36, "data", 4);
    p32(40, data);
    if (fwrite(h, 1, 44, f_) != 44) throw Error("写入失败（存储空间不足？）");
  }
  FILE* f_ = nullptr;
  int sr_;
  uint64_t bytes_ = 0;
};

// AAC-LC through the platform encoder, muxed into MPEG-4 (.m4a).
class AacWriter : public AudioWriter {
 public:
  AacWriter(const std::string& path, int sr, int bitrate) : sr_(sr) {
    try {
      fd_ = open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0644);
      if (fd_ < 0) throw Error("cannot write " + path);
      mux_ = AMediaMuxer_new(fd_, AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4);
      enc_ = AMediaCodec_createEncoderByType("audio/mp4a-latm");
      if (!mux_ || !enc_) throw Error("系统没有 AAC 编码器，请改用 WAV");
      AMediaFormat* f = AMediaFormat_new();
      AMediaFormat_setString(f, AMEDIAFORMAT_KEY_MIME, "audio/mp4a-latm");
      AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_SAMPLE_RATE, sr);
      AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_CHANNEL_COUNT, 1);
      AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_BIT_RATE, bitrate);
      AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_AAC_PROFILE, 2);  // LC
      AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_MAX_INPUT_SIZE, 16384);
      media_status_t st = AMediaCodec_configure(enc_, f, nullptr, nullptr, AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
      AMediaFormat_delete(f);
      if (st != AMEDIA_OK || AMediaCodec_start(enc_) != AMEDIA_OK) throw Error("AAC 编码器启动失败，请改用 WAV");
      started_codec_ = true;
    } catch (...) {
      release();
      throw;
    }
  }
  ~AacWriter() override { release(); }
  void write(const int16_t* x, size_t n) override {
    pending_.insert(pending_.end(), x, x + n);
    feed(false);
  }
  void finish() override {
    feed(true);
    if (!muxing_) throw Error("AAC 编码器没有输出");
    if (AMediaMuxer_stop(mux_) != AMEDIA_OK) throw Error("写入 m4a 失败");
    muxing_ = false;
  }

 private:
  void release() {
    if (enc_) {
      if (started_codec_) AMediaCodec_stop(enc_);
      AMediaCodec_delete(enc_);
      enc_ = nullptr;
    }
    if (mux_) {
      AMediaMuxer_delete(mux_);
      mux_ = nullptr;
    }
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }
  void feed(bool eos) {
    size_t pos = 0;
    while (pos < pending_.size() || eos) {
      ssize_t ii = AMediaCodec_dequeueInputBuffer(enc_, 5000);
      if (ii >= 0) {
        size_t cap = 0;
        uint8_t* b = AMediaCodec_getInputBuffer(enc_, static_cast<size_t>(ii), &cap);
        const size_t ns = std::min(cap / 2, pending_.size() - pos);
        if (ns) memcpy(b, pending_.data() + pos, ns * 2);
        const bool last = eos && pos + ns == pending_.size();
        const uint64_t pts = static_cast<uint64_t>(frames_ * 1000000 / sr_);
        AMediaCodec_queueInputBuffer(enc_, static_cast<size_t>(ii), 0, ns * 2, pts, last ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);
        frames_ += static_cast<int64_t>(ns);
        pos += ns;
        if (last) break;
      } else if (ii != AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
        throw Error("AAC 编码出错（" + std::to_string(ii) + "）");
      }
      drain(false);
    }
    pending_.clear();
    if (eos) drain(true);
  }
  void drain(bool until_eos) {
    int idle = 0;
    for (;;) {
      AMediaCodecBufferInfo info{};
      ssize_t oi = AMediaCodec_dequeueOutputBuffer(enc_, &info, until_eos ? 10000 : 0);
      if (oi == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
        if (!until_eos || ++idle > 300) return;
        continue;
      }
      if (oi == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
        AMediaFormat* of = AMediaCodec_getOutputFormat(enc_);
        ssize_t t = AMediaMuxer_addTrack(mux_, of);
        AMediaFormat_delete(of);
        if (t < 0 || AMediaMuxer_start(mux_) != AMEDIA_OK) throw Error("写入 m4a 失败");
        track_ = static_cast<size_t>(t);
        muxing_ = true;
        continue;
      }
      if (oi == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) continue;
      if (oi < 0) throw Error("AAC 编码出错（" + std::to_string(oi) + "）");
      idle = 0;
      size_t cap = 0;
      uint8_t* b = AMediaCodec_getOutputBuffer(enc_, static_cast<size_t>(oi), &cap);
      const bool config = info.flags & AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG;  // carried by the track format
      if (b && !config && info.size > 0 && muxing_) AMediaMuxer_writeSampleData(mux_, track_, b, &info);
      AMediaCodec_releaseOutputBuffer(enc_, static_cast<size_t>(oi), false);
      if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) return;
    }
  }

  int fd_ = -1, sr_;
  AMediaMuxer* mux_ = nullptr;
  AMediaCodec* enc_ = nullptr;
  bool started_codec_ = false, muxing_ = false;
  size_t track_ = 0;
  int64_t frames_ = 0;
  std::vector<int16_t> pending_;
};

}  // namespace

// ------------------------------------------------------------------ whole file
namespace {

using FilePtr = std::unique_ptr<FILE, int (*)(FILE*)>;

FilePtr open_file(const std::string& path, const char* mode) {
  FilePtr f(fopen(path.c_str(), mode), fclose);
  if (!f) throw Error("cannot open " + path);
  return f;
}

void put(FILE* f, const float* x, size_t n) {
  if (n && fwrite(x, sizeof(float), n, f) != n) throw Error("写入临时文件失败（存储空间不足？）");
}

bool aac_rate(int sr) {
  for (int r : {48000, 44100, 32000, 24000, 22050, 16000})
    if (sr == r) return true;
  return false;
}

}  // namespace

Json convert_file(OfflinePipeline& pipe, const OfflineParams& prm, const std::string& in, const std::string& out,
                  const std::string& format, OfflineProgress* prog) {
  const bool aac = format == "m4a";
  if (!aac && format != "wav") throw Error("unknown output format: " + format);
  const auto t0 = Clock::now();
  const std::string tmp1 = out + ".raw.f32", tmp2 = out + ".mix.f32";
  struct Cleanup {
    std::string a, b, out;
    bool ok = false;
    ~Cleanup() {
      remove(a.c_str());
      remove(b.c_str());
      if (!ok) remove(out.c_str());
    }
  } clean{tmp1, tmp2, out};
  auto check = [&] {
    if (prog) prog->check();
  };

  // load_audio(file, 16000): decoded, mixed to mono, resampled
  MediaDecoder dec(in);
  std::vector<float> x16, raw;
  {
    std::unique_ptr<StreamResampler> rs;
    int rs_rate = 0;
    const double expect = dec.duration() * 16000;
    for (bool more = true; more;) {
      check();
      raw.clear();
      more = dec.read(raw);
      if (!raw.empty()) {
        if (dec.rate() <= 0) throw Error("unknown sample rate");
        if (dec.rate() != rs_rate) {  // only at the start, unless the stream changes format
          if (rs) rs->finish(x16);
          rs.reset(new StreamResampler(dec.rate(), 16000));
          rs_rate = dec.rate();
        }
        rs->push(raw.data(), raw.size(), x16);
      }
      if (!more && rs) rs->finish(x16);
      if (prog && expect > 0) prog->fraction = 0.05f * static_cast<float>(std::min(1.0, x16.size() / expect));
    }
  }
  if (x16.empty()) throw Error("没有解码出音频");
  const double in_seconds = x16.size() / 16000.0;

  // pass 1: the pipeline
  OfflineResult r;
  {
    FilePtr f = open_file(tmp1, "wb");
    r = pipe.run(std::move(x16), prm, [&](const float* y, size_t n) { put(f.get(), y, n); }, prog, 0.05f, 0.95f);
    if (fclose(f.release()) != 0) throw Error("写入临时文件失败（存储空间不足？）");
  }

  // pass 2: change_rms, resampling (only for AAC from a 40 kHz model), peak
  const int sr = pipe.tgt_sr(), out_sr = aac && !aac_rate(sr) ? 48000 : sr;
  float peak = 0;
  size_t out_samples = 0;
  {
    const RmsMix mix(r, prm.rms_mix_rate);
    StreamResampler rs(sr, out_sr);
    FilePtr fi = open_file(tmp1, "rb"), fo = open_file(tmp2, "wb");
    std::vector<float> buf(static_cast<size_t>(sr)), y;
    size_t pos = 0;
    auto emit = [&] {
      for (float v : y) peak = std::max(peak, std::fabs(v));
      put(fo.get(), y.data(), y.size());
      out_samples += y.size();
      y.clear();
    };
    for (size_t n; (n = fread(buf.data(), sizeof(float), buf.size(), fi.get())) > 0;) {
      check();
      mix.apply(buf.data(), n, pos);
      pos += n;
      rs.push(buf.data(), n, y);
      emit();
      if (prog) prog->fraction = 0.95f + 0.02f * static_cast<float>(pos) / static_cast<float>(std::max<size_t>(r.samples, 1));
    }
    rs.finish(y);
    emit();
    if (fclose(fo.release()) != 0) throw Error("写入临时文件失败（存储空间不足？）");
  }
  remove(tmp1.c_str());

  // pass 3: (audio * 32768 / max(1, peak / 0.99)).astype(int16), encoded
  const double audio_max = peak / 0.99;
  const float scale = static_cast<float>(32768.0 / (audio_max > 1 ? audio_max : 1.0));
  {
    std::unique_ptr<AudioWriter> w;
    if (aac) w.reset(new AacWriter(out, out_sr, 128000));
    else w.reset(new WavWriter(out, out_sr));
    FilePtr fi = open_file(tmp2, "rb");
    std::vector<float> buf(static_cast<size_t>(out_sr));
    std::vector<int16_t> pcm(buf.size());
    size_t done = 0;
    for (size_t n; (n = fread(buf.data(), sizeof(float), buf.size(), fi.get())) > 0;) {
      check();
      for (size_t i = 0; i < n; ++i) pcm[i] = static_cast<int16_t>(std::max(-32768.f, std::min(32767.f, buf[i] * scale)));
      w->write(pcm.data(), n);
      done += n;
      if (prog) prog->fraction = 0.97f + 0.03f * static_cast<float>(done) / static_cast<float>(std::max<size_t>(out_samples, 1));
    }
    w->finish();
  }
  clean.ok = true;

  const double dur = static_cast<double>(out_samples) / out_sr, secs = ms_since(t0) / 1000.0;
  Json j = Json::object();
  j.set("duration_s", dur);
  j.set("input_s", in_seconds);
  j.set("sample_rate", out_sr);
  j.set("format", format);
  j.set("peak", peak);
  j.set("scale", static_cast<double>(scale) / 32768.0);
  j.set("segments", r.segments);
  j.set("pipeline_s", r.seconds);
  j.set("total_s", secs);
  j.set("rtf", in_seconds > 0 ? secs / in_seconds : 0.0);
  j.set("src_mime", dec.mime());
  j.set("src_rate", dec.rate());
  j.set("src_channels", dec.channels());
  RVC_LOGI("offline: %s -> %s, %.1f s of audio in %.1f s (%d segments), peak %.3f", in.c_str(), out.c_str(), in_seconds, secs,
           r.segments, peak);
  return j;
}

}  // namespace rvc
