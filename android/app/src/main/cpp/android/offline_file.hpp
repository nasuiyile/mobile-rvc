// Offline conversion of media files: decoded with the platform's codecs (NDK MediaExtractor /
// MediaCodec: mp3, aac/m4a, flac, ogg/opus, wav, amr, the audio track of mp4/mkv/webm videos...),
// converted by engine/offline, written as WAV (PCM16) or AAC in an .m4a container.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../engine/json.hpp"
#include "../engine/offline.hpp"

struct AMediaExtractor;
struct AMediaCodec;

namespace rvc {

// The first audio track of a media file, mixed down to mono.
class MediaDecoder : public AudioSource {
 public:
  explicit MediaDecoder(const std::string& path);
  ~MediaDecoder() override;
  MediaDecoder(const MediaDecoder&) = delete;
  MediaDecoder& operator=(const MediaDecoder&) = delete;

  bool read(std::vector<float>& out) override;
  int rate() const override { return rate_; }
  double duration() const override { return duration_; }
  int channels() const { return channels_; }
  const std::string& mime() const { return mime_; }

 private:
  void close_all();
  void update_format();
  void append_mono(const uint8_t* p, size_t bytes, std::vector<float>& out);

  int fd_ = -1;
  AMediaExtractor* ex_ = nullptr;
  AMediaCodec* codec_ = nullptr;
  std::string mime_;
  int rate_ = 0, channels_ = 1, encoding_ = 2;  // android.media.AudioFormat ENCODING_*
  double duration_ = 0;
  bool input_eos_ = false, output_eos_ = false;
  int idle_ = 0;
};

// {"duration_s", "sample_rate", "channels", "mime"}. Throws (with a message for the user) when
// the file has no audio track this device can decode.
Json probe_media(const std::string& path);

// RVC's vc_single on a media file: load_audio (decode, mono, 16 kHz), the offline pipeline,
// change_rms, then int16 scaled down when it would clip. format "wav": PCM16 at the model's
// rate, as RVC writes it; "m4a": AAC-LC, at 48 kHz when the model's rate is not an AAC rate.
// `out` is left behind only on success. Returns statistics as JSON.
Json convert_file(OfflinePipeline& pipe, const OfflineParams& p, const std::string& in, const std::string& out,
                  const std::string& format, OfflineProgress* prog);

}  // namespace rvc
