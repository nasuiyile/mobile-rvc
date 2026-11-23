// JNI surface for com.mobilervc.engine.Native (all results are JSON strings).
#include <fcntl.h>
#include <jni.h>
#include <unistd.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "../engine/common.hpp"
#include "../engine/conv_kernels.hpp"
#include "../engine/model_import.hpp"
#include "../engine/ort_model.hpp"
#include "../engine/offline.hpp"
#include "../engine/rvc_engine.hpp"
#include "offline_file.hpp"
#include "rt_driver.hpp"

using namespace rvc;

namespace {

struct Handle {
  std::mutex mu;
  std::unique_ptr<RvcEngine> engine;
  std::unique_ptr<RealtimeDriver> driver;
  // An offline conversion uses the engine outside `mu` (so that status / stop never wait for
  // it); while it runs, everything else that would touch the engine refuses.
  std::atomic<bool> converting{false};
  OfflineProgress offline;
  // The offline pipeline has its own sessions (any length, CPU); only one of it and the
  // realtime engine's models is loaded at a time.
  std::string assets;
  std::unique_ptr<OfflinePipeline> pipe;
  std::string pipe_key;
};

void check_not_converting(Handle* h) {
  if (h->converting) throw Error("正在离线转换，请等它完成或取消");
}

std::string str(JNIEnv* env, jstring s) {
  if (!s) return "";
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string r(c ? c : "");
  if (c) env->ReleaseStringUTFChars(s, c);
  return r;
}

jstring jstr(JNIEnv* env, const std::string& s) { return env->NewStringUTF(s.c_str()); }

jstring error(JNIEnv* env, const std::exception& e) {
  Json j = Json::object();
  j.set("error", e.what());
  return jstr(env, j.dump());
}

Handle* H(jlong h) { return reinterpret_cast<Handle*>(h); }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_mobilervc_engine_Native_create(JNIEnv* env, jclass, jstring assets) {
  auto* h = new Handle();
  h->assets = str(env, assets);
  h->engine.reset(new RvcEngine(h->assets));
  h->driver.reset(new RealtimeDriver(h->engine.get()));
  return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL Java_com_mobilervc_engine_Native_destroy(JNIEnv*, jclass, jlong h) {
  if (!h) return;
  H(h)->driver->stop();
  delete H(h);
}

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_importModel(JNIEnv* env, jclass, jstring assets, jstring pth,
                                                                       jstring index, jstring out_dir, jstring name) {
  try {
    auto r = import_rvc_model(str(env, assets), str(env, pth), str(env, index), str(env, out_dir), str(env, name));
    return jstr(env, r.meta.dump());
  } catch (const std::exception& e) {
    return error(env, e);
  }
}

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_attachIndex(JNIEnv* env, jclass, jstring dir, jstring index) {
  try {
    return jstr(env, attach_index(str(env, dir), str(env, index)).dump());
  } catch (const std::exception& e) {
    return error(env, e);
  }
}

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_inspectModel(JNIEnv* env, jclass, jstring assets, jstring pth) {
  try {
    return jstr(env, inspect_rvc_model(str(env, assets), str(env, pth)).dump());
  } catch (const std::exception& e) {
    return error(env, e);
  }
}

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_load(JNIEnv* env, jclass, jlong h, jstring model_dir, jstring cfg) {
  std::lock_guard<std::mutex> lk(H(h)->mu);
  try {
    check_not_converting(H(h));
    H(h)->driver->stop();
    H(h)->pipe.reset();
    H(h)->pipe_key.clear();
    H(h)->engine->load(str(env, model_dir), EngineConfig::from_json(Json::parse(str(env, cfg))));
    return jstr(env, H(h)->engine->info().dump());
  } catch (const std::exception& e) {
    return error(env, e);
  }
}

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_start(JNIEnv* env, jclass, jlong h, jstring opts, jint capture_fd,
                                                                 jint playback_fd) {
  std::lock_guard<std::mutex> lk(H(h)->mu);
  try {
    check_not_converting(H(h));
    DriverOptions o = DriverOptions::from_json(Json::parse(str(env, opts)));
    o.capture_fd = capture_fd;
    o.playback_fd = playback_fd;
    H(h)->driver->start(o);
    return jstr(env, H(h)->driver->status().dump());
  } catch (const std::exception& e) {
    return error(env, e);
  }
}

JNIEXPORT void JNICALL Java_com_mobilervc_engine_Native_setCaptureMode(JNIEnv*, jclass, jlong h, jint mode) {
  H(h)->driver->set_capture_mode(mode);
}

JNIEXPORT void JNICALL Java_com_mobilervc_engine_Native_stop(JNIEnv*, jclass, jlong h) {
  std::lock_guard<std::mutex> lk(H(h)->mu);
  H(h)->driver->stop();
}

JNIEXPORT void JNICALL Java_com_mobilervc_engine_Native_setParams(JNIEnv*, jclass, jlong h, jfloat pitch, jfloat formant,
                                                                  jfloat index_rate, jfloat rms_mix, jfloat threshold,
                                                                  jboolean input_nr, jboolean output_nr) {
  LiveParams& p = H(h)->engine->params;
  p.pitch = pitch;
  p.formant = formant;
  p.index_rate = index_rate;
  p.rms_mix_rate = rms_mix;
  p.threshold = threshold;
  p.input_nr = input_nr == JNI_TRUE;
  p.output_nr = output_nr == JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_status(JNIEnv* env, jclass, jlong h) {
  Json j = H(h)->driver->status();
  j.set("engine", H(h)->engine->info());
  return jstr(env, j.dump());
}

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_bench(JNIEnv* env, jclass, jlong h, jint iters) {
  std::lock_guard<std::mutex> lk(H(h)->mu);
  try {
    check_not_converting(H(h));
    RvcEngine& e = *H(h)->engine;
    if (!e.loaded()) throw Error("no model loaded");
    if (H(h)->driver->running()) throw Error("stop realtime conversion first");
    int bf = e.block_frame(), sr = e.sample_rate();
    std::vector<float> in(bf), out(bf);
    for (int i = 0; i < bf; ++i) in[i] = 0.2f * std::sin(2 * M_PI * 220.0 * i / sr);
    for (int i = 0; i < 3; ++i) e.process_block(in.data(), out.data());
    double tot = 0, worst = 0, dec = 0, hub = 0;
    for (int i = 0; i < iters; ++i) {
      e.process_block(in.data(), out.data());
      BlockStats s = e.stats();
      tot += s.total_ms;
      dec += s.dec_ms;
      hub += s.hubert_ms;
      worst = std::max(worst, s.total_ms);
    }
    e.reset_state();
    Json r = Json::object();
    double bms = 1000.0 * bf / sr;
    r.set("block_ms", bms);
    r.set("avg_ms", tot / iters);
    r.set("worst_ms", worst);
    r.set("hubert_ms", hub / iters);
    r.set("dec_ms", dec / iters);
    r.set("rtf", tot / iters / bms);
    r.set("engine", e.info());
    return jstr(env, r.dump());
  } catch (const std::exception& ex) {
    return error(env, ex);
  }
}

// ---- offline conversion of media files
JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_probeAudio(JNIEnv* env, jclass, jstring path) {
  try {
    return jstr(env, probe_media(str(env, path)).dump());
  } catch (const std::exception& e) {
    return error(env, e);
  }
}

// RVC's offline inference of a media file. Blocks for the whole conversion; progress and
// cancel come from other threads. cfg: {"threads", "affinity", "fp16", "sid"}; params: OfflineParams.
JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_convertFile(JNIEnv* env, jclass, jlong h, jstring model_dir, jstring cfg,
                                                                       jstring params, jstring in, jstring out, jstring format) {
  Handle* hd = H(h);
  try {
    {
      std::lock_guard<std::mutex> lk(hd->mu);
      check_not_converting(hd);
      if (hd->driver->running()) throw Error("请先停止实时变声");
      hd->offline.fraction = 0.f;
      hd->offline.cancel = false;
      hd->converting = true;
    }
    struct Done {
      Handle* h;
      ~Done() { h->converting = false; }
    } done{hd};
    const std::string dir = str(env, model_dir);
    const Json c = Json::parse(str(env, cfg));
    const std::string key = dir + "|" + c.dump();
    if (!hd->pipe || hd->pipe_key != key) {
      hd->pipe.reset();
      hd->engine->unload();  // the realtime models: load() brings them back when needed
      BackendConfig cpu;
      cpu.threads = static_cast<int>(c.num_or("threads", 4));
      if (c.get("affinity").is_array())
        for (auto& v : c["affinity"].items()) cpu.affinity.push_back(v.i32());
      hd->pipe.reset(new OfflinePipeline(hd->assets, dir, cpu, c.bool_or("fp16", true), static_cast<int>(c.num_or("sid", 0))));
      hd->pipe_key = key;
    }
    Json r = convert_file(*hd->pipe, OfflineParams::from_json(Json::parse(str(env, params))), str(env, in), str(env, out),
                          str(env, format), &hd->offline);
    r.set("pipeline", hd->pipe->info());
    return jstr(env, r.dump());
  } catch (const std::exception& e) {
    return error(env, e);
  }
}

// Frees the offline pipeline's models (leaving the conversion screen).
JNIEXPORT void JNICALL Java_com_mobilervc_engine_Native_releaseOffline(JNIEnv*, jclass, jlong h) {
  std::lock_guard<std::mutex> lk(H(h)->mu);
  if (H(h)->converting) return;
  H(h)->pipe.reset();
  H(h)->pipe_key.clear();
}

JNIEXPORT jfloat JNICALL Java_com_mobilervc_engine_Native_offlineProgress(JNIEnv*, jclass, jlong h) {
  return H(h)->offline.fraction.load();
}

JNIEXPORT void JNICALL Java_com_mobilervc_engine_Native_cancelOffline(JNIEnv*, jclass, jlong h) { H(h)->offline.cancel = true; }

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_deviceInfo(JNIEnv* env, jclass) {
  Json j = cpu_info();
  Json p = Json::array();
  try {
    for (auto& s : available_providers()) p.push(s);
  } catch (...) {
  }
  j.set("providers", p);
  j.set("fp16", fp16_supported());
  j.set("soc", soc_name());
  RVC_LOGI("device: %s", j.dump().c_str());
  return jstr(env, j.dump());
}

// Pipe with a large kernel buffer (block-sized writes must not block / tear).
JNIEXPORT jintArray JNICALL Java_com_mobilervc_engine_Native_createPipe(JNIEnv* env, jclass) {
  int fds[2] = {-1, -1};
  if (pipe2(fds, O_CLOEXEC) != 0) return nullptr;
  fcntl(fds[1], F_SETPIPE_SZ, 1 << 20);
  jintArray a = env->NewIntArray(2);
  env->SetIntArrayRegion(a, 0, 2, fds);
  return a;
}

}  // extern "C"
