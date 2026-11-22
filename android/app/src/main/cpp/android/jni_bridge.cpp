// JNI surface for com.mobilervc.engine.Native (all results are JSON strings).
#include <fcntl.h>
#include <jni.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <memory>
#include <mutex>
#include <string>

#include "../engine/common.hpp"
#include "../engine/conv_kernels.hpp"
#include "../engine/model_import.hpp"
#include "../engine/ort_model.hpp"
#include "../engine/rvc_engine.hpp"
#include "rt_driver.hpp"

using namespace rvc;

namespace {

struct Handle {
  std::mutex mu;
  std::unique_ptr<RvcEngine> engine;
  std::unique_ptr<RealtimeDriver> driver;
};

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
  h->engine.reset(new RvcEngine(str(env, assets)));
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
    H(h)->driver->stop();
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
    RvcEngine& e = *H(h)->engine;
    if (!e.loaded()) throw Error("no model loaded");
    if (H(h)->driver->running()) throw Error("stop realtime conversion first");
    int bf = e.block_frame(), sr = e.sample_rate();
    std::vector<float> in(bf), out(bf);
    for (int i = 0; i < bf; ++i) in[i] = 0.2f * std::sin(2 * M_PI * 220.0 * i / sr);
    for (int i = 0; i < 3; ++i) e.process_block(in.data(), out.data());
    double tot = 0, worst = 0, dec = 0, hub = 0, ana = 0, syn = 0;
    for (int i = 0; i < iters; ++i) {
      e.process_block(in.data(), out.data());
      BlockStats s = e.stats();
      tot += s.total_ms;
      dec += s.dec_ms;
      hub += s.hubert_ms;
      ana += s.analyze_ms;
      syn += s.synth_ms;
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
    r.set("analyze_ms", ana / iters);
    r.set("synth_ms", syn / iters);
    // What has to fit into one block period: both halves, or only the slower one when they run
    // on different processors and overlap. A shorter block is also cheaper, so a block of
    // (that time / 0.65) is safe to try.
    const bool indep = e.stages_independent();
    const double need = indep ? std::max(ana, syn) / iters : tot / iters;
    r.set("independent_stages", indep);
    r.set("suggest_block_ms", std::max(100.0, std::ceil(need / 0.65 / 10.0) * 10.0));  // 0.10 s: the UI's minimum
    r.set("engine", e.info());
    return jstr(env, r.dump());
  } catch (const std::exception& ex) {
    return error(env, ex);
  }
}

JNIEXPORT jstring JNICALL Java_com_mobilervc_engine_Native_deviceInfo(JNIEnv* env, jclass) {
  Json j = cpu_info();
  Json p = Json::array();
  try {
    for (auto& s : available_providers()) p.push(s);
  } catch (...) {
  }
  j.set("providers", p);
  j.set("fp16", fp16_supported());
  // The driver sizes the virtual-mic output delay from the unread bytes in its pipe (FIONREAD);
  // record whether this process may ask (ioctls on pipes are filtered by SELinux).
  int fds[2] = {-1, -1}, pending = -1;
  bool fionread = false;
  if (pipe2(fds, O_CLOEXEC) == 0) {
    char probe[8] = {0};
    fionread = write(fds[1], probe, sizeof(probe)) == sizeof(probe) && ioctl(fds[1], FIONREAD, &pending) == 0 && pending == sizeof(probe);
    close(fds[0]);
    close(fds[1]);
  }
  j.set("pipe_fionread", fionread);
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
