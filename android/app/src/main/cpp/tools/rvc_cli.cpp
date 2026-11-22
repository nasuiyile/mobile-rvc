// Command line front-end for the engine (host tests and `adb shell` benchmarks).
//
//   rvc_cli import  <assets> <model.pth> <index|-> <out_dir>
//   rvc_cli process <assets> <model_dir> <in.wav> <out.wav> [config.json] [k=v ...]
//   rvc_cli bench   <assets> <model_dir> [config.json] [k=v ...]
//
// k=v pairs override config keys or live params (pitch, formant, index_rate, rms_mix_rate,
// threshold); e.g. block_time=0.3 dec.backend=webgpu threads=4.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../engine/common.hpp"
#include "../engine/json.hpp"
#include "../engine/model_import.hpp"
#include "../engine/native_decoder.hpp"
#include "../engine/noise_gate.hpp"
#include "../engine/ort_model.hpp"
#include "../engine/rvc_engine.hpp"
#include "../engine/wav.hpp"
#ifdef __ANDROID__
#include <fcntl.h>
#include <unistd.h>

#include <thread>

#include "../android/rt_driver.hpp"
#endif

using namespace rvc;

static Json parse_overrides(Json cfg, int argc, char** argv, int start, Json* live) {
  for (int i = start; i < argc; ++i) {
    std::string a = argv[i];
    size_t eq = a.find('=');
    if (eq == std::string::npos) {
      cfg = Json::parse(read_file(a));
      continue;
    }
    std::string k = a.substr(0, eq), v = a.substr(eq + 1);
    Json val;
    if (v == "true" || v == "false") val = Json(v == "true");
    else {
      char* end = nullptr;
      double d = strtod(v.c_str(), &end);
      val = (end && *end == 0) ? Json(d) : Json(v);
    }
    if (k == "pitch" || k == "formant" || k == "index_rate" || k == "rms_mix_rate" || k == "threshold" ||
        k == "input_nr" || k == "output_nr") {
      live->set(k, val);
      continue;
    }
    size_t dot = k.find('.');
    if (dot != std::string::npos) {
      std::string sec = k.substr(0, dot), key = k.substr(dot + 1);
      Json sub = cfg.get(sec).is_object() ? cfg[sec] : Json::object();
      sub.set(key, val);
      cfg.set(sec, sub);
    } else {
      cfg.set(k, val);
    }
  }
  return cfg;
}

static void apply_live(RvcEngine& e, const Json& live) {
  e.params.pitch = static_cast<float>(live.num_or("pitch", 0));
  e.params.formant = static_cast<float>(live.num_or("formant", 0));
  e.params.index_rate = static_cast<float>(live.num_or("index_rate", 0));
  e.params.rms_mix_rate = static_cast<float>(live.num_or("rms_mix_rate", 0));
  e.params.threshold = static_cast<float>(live.num_or("threshold", -60));
  e.params.input_nr = live.bool_or("input_nr", false);
  e.params.output_nr = live.bool_or("output_nr", false);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: rvc_cli import|process|bench|providers ...\n");
    return 2;
  }
  std::string cmd = argv[1];
  try {
    if (cmd == "providers") {
      for (auto& p : available_providers()) printf("%s\n", p.c_str());
      return 0;
    }
    if (cmd == "import" && argc >= 6) {
      std::string idx = argv[4];
      auto r = import_rvc_model(argv[2], argv[3], idx == "-" ? "" : idx, argv[5], "cli");
      printf("%s\n", r.meta.dump().c_str());
      return 0;
    }
    if (cmd == "process" && argc >= 6) {
      Json live = Json::object();
      Json cj = parse_overrides(Json::object(), argc, argv, 6, &live);
      int sr = 0;
      std::vector<float> x = read_wav_mono(argv[4], &sr);
      cj.set("sample_rate", sr);
      RvcEngine e(argv[2]);
      e.load(argv[3], EngineConfig::from_json(cj));
      apply_live(e, live);
      fprintf(stderr, "%s\n", e.info().dump().c_str());
      int bf = e.block_frame();
      std::vector<float> y;
      double tot = 0, worst = 0;
      int nb = 0;
      for (size_t i = 0; i + bf <= x.size(); i += bf) {
        std::vector<float> o(bf);
        auto t = Clock::now();
        e.process_block(x.data() + i, o.data());
        double ms = ms_since(t);
        if (nb > 0) { tot += ms; worst = std::max(worst, ms); }
        ++nb;
        y.insert(y.end(), o.begin(), o.end());
      }
      write_wav_f32(argv[5], y, sr);
      double bms = 1000.0 * bf / sr;
      fprintf(stderr, "blocks=%d block=%.0fms avg=%.1fms worst=%.1fms rtf=%.3f\n", nb, bms, tot / std::max(1, nb - 1), worst,
              tot / std::max(1, nb - 1) / bms);
      return 0;
    }
#ifdef __ANDROID__
    if (cmd == "pipetest" && argc >= 6) {
      // realtime driver in virtual-mic (pipe) mode, input paced in real time:
      // pipetest <assets> <model_dir> <in.wav> <out.wav> [k=v ...]
      Json live = Json::object();
      Json cj = parse_overrides(Json::object(), argc, argv, 6, &live);
      int sr = 0;
      std::vector<float> x = read_wav_mono(argv[4], &sr);
      cj.set("sample_rate", sr);
      RvcEngine e(argv[2]);
      e.load(argv[3], EngineConfig::from_json(cj));
      apply_live(e, live);
      int cap[2], play[2];
      if (pipe(cap) || pipe(play)) throw Error("pipe");
      fcntl(play[1], F_SETPIPE_SZ, 1 << 20);
      RealtimeDriver d(&e);
      DriverOptions o;
      o.local_input = false;
      o.local_output = false;
      o.capture_fd = cap[0];
      o.playback_fd = play[1];
      d.start(o);
      std::vector<float> y;
      std::thread reader([&] {
        std::vector<float> buf(4096);
        for (;;) {
          ssize_t n = read(play[0], buf.data(), buf.size() * 4);
          if (n <= 0) break;
          y.insert(y.end(), buf.begin(), buf.begin() + n / 4);
        }
      });
      const int chunk = sr / 100;
      const bool toggle = cj.bool_or("toggle_input", false);
      auto t0 = Clock::now();
      for (size_t i = 0; i + chunk <= x.size(); i += chunk) {
        if (toggle && i % (sr * 2) == 0) {  // exercise runtime open/close of the local mic
          bool on = (i / (sr * 2)) % 2 == 0;
          d.set_capture_mode(on ? 1 : 0);
          printf("t=%.1fs local mic %s\n", static_cast<double>(i) / sr, on ? "on" : "off");
        }
        if (write(cap[1], x.data() + i, chunk * 4) < 0) break;
        auto due = t0 + std::chrono::microseconds(static_cast<int64_t>(1e6 * (i + chunk) / sr));
        std::this_thread::sleep_until(due);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(800));
      Json st = d.status();
      d.stop();  // closes cap[0] / play[1]
      close(cap[1]);
      reader.join();
      close(play[0]);
      write_wav_f32(argv[5], y, sr);
      printf("%s\nout_samples=%zu in_samples=%zu\n", st.dump().c_str(), y.size(), x.size());
      return 0;
    }
#endif
    if (cmd == "hubtest" && argc >= 4) {
      // HuBERT accuracy/speed of an accelerator vs ORT CPU fp32: hubtest <assets> <v1|v2> <backend[32]> [N]
      std::string hd = join_path(argv[2], "hubert");
      std::string name = std::string(argv[3]) == "v1" ? "hubert_v1" : "hubert_v2";
      std::string accel = argc > 4 ? argv[4] : "cpu";
      int N = argc > 5 ? atoi(argv[5]) : 13440;
      std::vector<float> x(N);
      for (int i = 0; i < N; ++i) x[i] = 0.3f * std::sin(2 * M_PI * 180.0 * i / 16000) * std::sin(2 * M_PI * 3.0 * i / 16000) + 0.01f * std::sin(i * 0.37f);
      BackendConfig cpu;
      cpu.threads = 4;
      OnnxModel ref(join_path(hd, name + ".onnx"), cpu, {{"N", N}}, name);
      std::vector<Ort::Value> in;
      in.push_back(make_tensor<float>(x.data(), x.size(), {1, N}));
      auto r = ref.run({"audio"}, in, {"feats"});
      size_t n = r[0].GetTensorTypeAndShapeInfo().GetElementCount();
      const float* y = r[0].GetTensorData<float>();
      bool f32 = accel.size() > 2 && accel.substr(accel.size() - 2) == "32";
      BackendConfig ac;
      ac.backend = backend_from_name(f32 ? accel.substr(0, accel.size() - 2) : accel);
      ac.fp16 = !f32;
      ac.threads = 4;
      OnnxModel m(join_path(hd, name + ".onnx"), ac, {{"N", N}}, name);
      auto o = m.run({"audio"}, in, {"feats"});
      auto t = Clock::now();
      for (int i = 0; i < 5; ++i) o = m.run({"audio"}, in, {"feats"});
      double ms = ms_since(t) / 5;
      const float* a = o[0].GetTensorData<float>();
      double num = 0, den = 0;
      for (size_t i = 0; i < n; ++i) { double d = a[i] - y[i]; num += d * d; den += double(y[i]) * y[i]; }
      printf("%s %s (%s): snr %.1f dB vs CPU fp32, %.1f ms\n", name.c_str(), accel.c_str(), m.backend_used().c_str(),
             10 * log10(den / std::max(num, 1e-30)), ms);
      return 0;
    }
    if (cmd == "gatetest" && argc >= 6) {
      // gatetest <x.f32> <noise.f32> <out.f32> <sr> : TorchGate port on raw float32 files
      std::string xs = read_file(argv[2]), ns = read_file(argv[3]);
      int sr = atoi(argv[5]);
      SpectralGate g(sr, 4 * (sr / 100));
      auto y = g.process(reinterpret_cast<const float*>(xs.data()), static_cast<int>(xs.size() / 4),
                         reinterpret_cast<const float*>(ns.data()), static_cast<int>(ns.size() / 4));
      write_file(argv[4], y.data(), y.size() * 4);
      return 0;
    }
    if (cmd == "dectest" && argc >= 3) {
      // decoder-only comparison: ORT fp32 graph vs native decoder (fp32 / fp16)
      std::string dir = argv[2];
      int T = argc > 3 ? atoi(argv[3]) : 30;
      Json meta = Json::parse(read_file(join_path(dir, "model.json")));
      int upp = meta["upp"].i32();
      std::string eg = read_file(join_path(dir, "emb_g.bin"));
      std::vector<float> g(reinterpret_cast<const float*>(eg.data()), reinterpret_cast<const float*>(eg.data()) + 256);
      std::mt19937 rng(7);
      std::normal_distribution<float> nd(0.f, 1.f);
      std::vector<float> z(static_cast<size_t>(192) * T), har(static_cast<size_t>(T) * upp);
      for (auto& v : z) v = nd(rng) * 0.6f;
      for (size_t i = 0; i < har.size(); ++i) har[i] = std::tanh(-0.83f * 0.1f * std::sin(2 * M_PI * 180.0 * i / (upp * 100.0)));
      BackendConfig bc;
      bc.threads = 4;
      OnnxModel ref(join_path(dir, "dec.onnx"), bc, {}, "dec");
      std::string accel = argc > 4 ? argv[4] : "";
      std::vector<Ort::Value> in;
      in.push_back(make_tensor<float>(z.data(), z.size(), {1, 192, T}));
      in.push_back(make_tensor<float>(har.data(), har.size(), {1, 1, static_cast<int64_t>(har.size())}));
      in.push_back(make_tensor<float>(g.data(), g.size(), {1, 256, 1}));
      auto out = ref.run({"z", "har", "g"}, in, {"audio"});
      const float* y = out[0].GetTensorData<float>();
      if (!accel.empty()) {
        // same graph on an accelerator EP (static shapes)
        BackendConfig ac;
        bool f32 = accel.size() > 2 && accel.substr(accel.size() - 2) == "32";
        ac.backend = backend_from_name(f32 ? accel.substr(0, accel.size() - 2) : accel);
        ac.fp16 = !f32;
        ac.threads = 4;
        OnnxModel m(join_path(dir, "dec.onnx"), ac, {{"T", T}, {"S", static_cast<int64_t>(T) * upp}}, "dec");
        auto o2 = m.run({"z", "har", "g"}, in, {"audio"});
        auto t = Clock::now();
        for (int i = 0; i < 5; ++i) o2 = m.run({"z", "har", "g"}, in, {"audio"});
        double ms = ms_since(t) / 5;
        const float* a = o2[0].GetTensorData<float>();
        double num = 0, den = 0;
        for (size_t i = 0; i < har.size(); ++i) { double d = a[i] - y[i]; num += d * d; den += double(y[i]) * y[i]; }
        printf("%s (%s): snr %.1f dB vs ORT fp32, %.1f ms\n", accel.c_str(), m.backend_used().c_str(), 10 * log10(den / std::max(num, 1e-30)), ms);
      }
      for (int f16 = 0; f16 <= 1; ++f16) {
        NativeDecoder nd2(dir, g, 4, f16 != 0, {});
        std::vector<float> o(har.size());
        nd2.run(z.data(), T, har.data(), o.data());
        auto t = Clock::now();
        for (int i = 0; i < 5; ++i) nd2.run(z.data(), T, har.data(), o.data());
        double ms = ms_since(t) / 5;
        double num = 0, den = 0;
        for (size_t i = 0; i < o.size(); ++i) { double d = o[i] - y[i]; num += d * d; den += double(y[i]) * y[i]; }
        printf("native %s: snr %.1f dB vs ORT fp32, %.1f ms\n", f16 ? "fp16" : "fp32", 10 * log10(den / std::max(num, 1e-30)), ms);
      }
      return 0;
    }
    if (cmd == "failtest" && argc >= 4) {
      // failtest <model_dir> <backend>: a session that cannot be built (inconsistent static
      // shapes), to look at what the wrapper reports for an accelerator failure
      BackendConfig ac;
      ac.backend = backend_from_name(argv[3]);
      ac.threads = 2;
      try {
        OnnxModel m(join_path(argv[2], "dec.onnx"), ac, {{"T", 35}, {"S", 13001}}, "failtest_dec", "", false);
        printf("unexpectedly built on %s\n", m.backend_used().c_str());
      } catch (const std::exception& e) {
        printf("reported failure:\n%s\n", e.what());
      }
      return 0;
    }
    if (cmd == "bench" && argc >= 4) {
      Json live = Json::object();
      Json cj = parse_overrides(Json::object(), argc, argv, 4, &live);
      RvcEngine e(argv[2]);
      e.load(argv[3], EngineConfig::from_json(cj));
      apply_live(e, live);
      int bf = e.block_frame(), sr = e.sample_rate();
      std::vector<float> in(bf), out(bf);
      for (int i = 0; i < bf; ++i) in[i] = 0.2f * std::sin(2 * M_PI * 220.0 * i / sr) + 0.05f * std::sin(2 * M_PI * 440.0 * i / sr);
      int warm = 3, n = static_cast<int>(live.num_or("iters", 20));
      for (int i = 0; i < warm; ++i) e.process_block(in.data(), out.data());
      double sum[8] = {0}, worst = 0;
      for (int i = 0; i < n; ++i) {
        e.process_block(in.data(), out.data());
        BlockStats s = e.stats();
        double v[8] = {s.total_ms, s.hubert_ms, s.f0_ms, s.index_ms, s.enc_ms, s.flow_ms, s.dec_ms, s.post_ms};
        for (int k = 0; k < 8; ++k) sum[k] += v[k];
        worst = std::max(worst, s.total_ms);
      }
      double bms = 1000.0 * bf / sr;
      Json r = e.info();
      Json t = Json::object();
      const char* names[8] = {"total", "hubert", "f0", "index", "enc", "flow", "dec", "post"};
      for (int k = 0; k < 8; ++k) t.set(names[k], sum[k] / n);
      r.set("avg_ms", t);
      r.set("worst_ms", worst);
      r.set("block_ms", bms);
      r.set("rtf", sum[0] / n / bms);
      printf("%s\n", r.dump().c_str());
      return 0;
    }
  } catch (const std::exception& ex) {
    fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }
  fprintf(stderr, "bad arguments\n");
  return 2;
}
