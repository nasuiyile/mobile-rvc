#include "ort_model.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "common.hpp"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#if defined(__ANDROID__) && __has_include(<nnapi_provider_factory.h>)
#include <nnapi_provider_factory.h>
#define RVC_HAVE_NNAPI 1
#else
#define RVC_HAVE_NNAPI 0
#endif

namespace rvc {

const char* backend_name(Backend b) {
  switch (b) {
    case Backend::CPU: return "cpu";
    case Backend::XNNPACK: return "xnnpack";
    case Backend::NNAPI: return "nnapi";
    case Backend::WEBGPU: return "webgpu";
    case Backend::QNN_HTP: return "qnn_htp";
    case Backend::QNN_GPU: return "qnn_gpu";
    case Backend::NATIVE: return "native";
  }
  return "cpu";
}

Backend backend_from_name(const std::string& s) {
  if (s == "xnnpack") return Backend::XNNPACK;
  if (s == "nnapi") return Backend::NNAPI;
  if (s == "webgpu" || s == "gpu") return Backend::WEBGPU;
  if (s == "qnn_htp" || s == "npu" || s == "qnn") return Backend::QNN_HTP;
  if (s == "qnn_gpu") return Backend::QNN_GPU;
  if (s == "native") return Backend::NATIVE;
  return Backend::CPU;
}

std::vector<std::string> available_providers() { return Ort::GetAvailableProviders(); }

// ---------------------------------------------------------------- logging
// ONNX Runtime's messages (and, through it, the accelerator back end's own: QNN reports why
// it refuses an operator this way) go to the system log as before. While a session is being
// created they are also collected, so that a failure can be shown to the user with its cause
// instead of only the last, generic error.
namespace {
std::mutex g_log_mu;
std::string* g_log_capture = nullptr;
bool g_log_capture_all = false;  // diagnostic pass: keep informational lines that mention the accelerator

void ORT_API_CALL ort_log(void*, OrtLoggingLevel severity, const char*, const char*, const char* where, const char* message) {
  const bool important = severity >= ORT_LOGGING_LEVEL_WARNING;
  const char* msg = message ? message : "";
#ifdef __ANDROID__
  if (important) {
    __android_log_print(severity >= ORT_LOGGING_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_WARN, "onnxruntime", "%s [%s]", msg, where ? where : "");
  }
#else
  if (important) fprintf(stderr, "[ort] %s\n", msg);
#endif
  std::lock_guard<std::mutex> lk(g_log_mu);
  if (!g_log_capture) return;
  if (!important) {
    if (!g_log_capture_all) return;
    // from the flood of an informational pass keep what says which operators were refused
    static const char* const keys[] = {"not supported", "failed", "Failed", "Unsupported", "unsupported", "supported by QNN", "Validation", "rror"};
    bool hit = false;
    for (const char* k : keys) hit = hit || strstr(msg, k) != nullptr;
    if (!hit) return;
  }
  // session-independent chatter
  if (strstr(msg, "telemetry") || strstr(msg, "memory pattern")) return;
  std::string line(msg);
  if (line.size() > 400) line.resize(400);
  if (g_log_capture->size() + line.size() > 6000) return;
  if (g_log_capture->find(line) != std::string::npos) return;  // the same complaint once
  *g_log_capture += line;
  *g_log_capture += '\n';
}

struct LogCapture {
  LogCapture(std::string* into, bool all) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log_capture = into;
    g_log_capture_all = all;
  }
  ~LogCapture() {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log_capture = nullptr;
    g_log_capture_all = false;
  }
};
}  // namespace

Ort::Env& ort_env() {
  // intentionally leaked: EP libraries (WebGPU/Dawn, QNN) must not be torn down by static destructors
  static Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "rvc", ort_log, nullptr);
  return *env;
}

std::string soc_name() {
#ifdef __ANDROID__
  char v[PROP_VALUE_MAX] = {0};
  for (const char* key : {"ro.soc.model", "ro.board.platform", "ro.hardware"})
    if (__system_property_get(key, v) > 0) return v;
#endif
  return "";
}

static uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

std::string file_fingerprint(const std::string& path) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) return "";
  uint64_t h = fnv1a(&st.st_size, sizeof(st.st_size));
  FILE* f = fopen(path.c_str(), "rb");
  if (f) {
    std::vector<char> buf(256 * 1024);
    size_t n = fread(buf.data(), 1, buf.size(), f);
    h = fnv1a(buf.data(), n, h);
    if (static_cast<size_t>(st.st_size) > 2 * buf.size() && fseek(f, -static_cast<long>(buf.size()), SEEK_END) == 0) {
      n = fread(buf.data(), 1, buf.size(), f);
      h = fnv1a(buf.data(), n, h);
    }
    fclose(f);
  }
  char out[40];
  snprintf(out, sizeof(out), "%llx", static_cast<unsigned long long>(h));
  return out;
}

Ort::MemoryInfo& cpu_mem() {
  static Ort::MemoryInfo* mi = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  return *mi;
}

namespace {

bool has_provider(const char* name) {
  for (auto& p : Ort::GetAvailableProviders())
    if (p == name) return true;
  return false;
}

std::string sanitize(std::string s) {
  for (auto& c : s)
    if (!isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') c = '_';
  return s;
}

}  // namespace

// Compiled accelerator graphs are large (the weights are inside): keep the newest ones only.
static void prune_cache(const std::string& dir, const std::string& keep, uint64_t limit_bytes) {
  struct Item {
    std::string path;
    time_t mtime;
    uint64_t size;
  };
  std::vector<Item> items;
  uint64_t total = 0;
  DIR* d = opendir(dir.c_str());
  if (!d) return;
  while (dirent* e = readdir(d)) {
    std::string name = e->d_name;
    if (name.size() < 9 || name.compare(name.size() - 9, 9, "_ctx.onnx") != 0) continue;
    std::string p = join_path(dir, name);
    struct stat st {};
    if (stat(p.c_str(), &st) != 0) continue;
    total += static_cast<uint64_t>(st.st_size);
    if (p != keep) items.push_back({p, st.st_mtime, static_cast<uint64_t>(st.st_size)});
  }
  closedir(d);
  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.mtime < b.mtime; });
  for (const Item& it : items) {
    if (total <= limit_bytes) break;
    if (remove(it.path.c_str()) == 0) total -= it.size;
  }
}

OnnxModel::OnnxModel(const std::string& path, const BackendConfig& cfg,
                     const std::vector<std::pair<std::string, int64_t>>& free_dims, const std::string& tag,
                     const std::string& cache_salt, bool cpu_fallback) {
  auto t0 = Clock::now();
  bool accel = cfg.backend != Backend::CPU && cfg.backend != Backend::XNNPACK && cfg.backend != Backend::NATIVE;
  // Graphs are fp32-compute: on ORT's WebGPU EP, fp16-compute graphs measured 7.6 dB (HuBERT) /
  // 13.5 dB (decoder) SNR, fp32 is exact. Reduced precision is requested from the EP instead
  // (QNN enable_htp_fp16_precision, NNAPI_FLAG_USE_FP16) or done by the native decoder.
  std::string log;
  try {
    LogCapture cap(&log, false);
    create(path, cfg, free_dims, tag, cache_salt, false);
  } catch (const std::exception& e) {
    if (!accel && cfg.backend != Backend::XNNPACK) throw;
    warning_ = std::string(backend_name(cfg.backend)) + " unavailable for " + tag + ": " + e.what();
    if (accel) {
      // Once more, listening to the informational messages too: the error above is only the
      // last symptom, the accelerator says per operator what it refused and why.
      std::string detail;
      try {
        LogCapture cap(&detail, true);
        create(path, cfg, free_dims, tag, cache_salt, true);
        session_.reset();
      } catch (...) {
      }
      for (const std::string* part : {&log, &detail}) {
        size_t pos = 0;
        while (pos < part->size()) {  // line by line, each complaint once
          size_t nl = part->find('\n', pos);
          if (nl == std::string::npos) nl = part->size();
          std::string line = part->substr(pos, nl - pos);
          pos = nl + 1;
          if (!line.empty() && warning_.find(line) == std::string::npos && warning_.size() < 5000) warning_ += "\n· " + line;
        }
      }
      std::string soc = soc_name();
      if (!soc.empty()) warning_ += "\n[" + soc + "]";
    }
    RVC_LOGW("%s", warning_.c_str());
    if (!cpu_fallback) throw Error(warning_);
    RVC_LOGW("%s -> falling back to CPU", tag.c_str());
    BackendConfig cpu = cfg;
    cpu.backend = Backend::CPU;
    create(path, cpu, free_dims, tag, cache_salt, false);
  }
  load_ms_ = ms_since(t0);
  RVC_LOGI("session %s on %s (%.0f ms)", tag.c_str(), used_.c_str(), load_ms_);
}

void OnnxModel::create(const std::string& path, const BackendConfig& cfg,
                      const std::vector<std::pair<std::string, int64_t>>& free_dims, const std::string& tag,
                      const std::string& cache_salt, bool diagnose) {
  Ort::Env& env = ort_env();  // must exist before execution providers are appended (logger)
  Ort::SessionOptions so;
  if (diagnose) so.SetLogSeverityLevel(1);  // informational: partitioning and validation results
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  so.SetInterOpNumThreads(1);
  so.SetIntraOpNumThreads(std::max(1, cfg.threads));
  if (!cfg.mem_arena) so.DisableCpuMemArena();
  so.AddConfigEntry("session.intra_op.allow_spinning", cfg.spinning ? "1" : "0");
  if (!cfg.affinity.empty() && cfg.threads > 1 && cfg.backend != Backend::XNNPACK) {
    // thread 0 is the caller; every pool thread may run on any of the chosen (big) cores
    std::string cores;
    for (int c : cfg.affinity) {
      if (!cores.empty()) cores += ",";
      cores += std::to_string(c + 1);  // ORT processor ids are 1-based
    }
    std::string aff;
    for (int i = 1; i < cfg.threads; ++i) {
      if (!aff.empty()) aff += ";";
      aff += cores;
    }
    so.AddConfigEntry("session.intra_op_thread_affinities", aff.c_str());
  }
  for (auto& fd : free_dims) so.AddFreeDimensionOverrideByName(fd.first.c_str(), fd.second);

  std::string load_path = path;
  switch (cfg.backend) {
    case Backend::CPU:
    case Backend::NATIVE:
      used_ = "cpu";
      break;
    case Backend::XNNPACK: {
      if (!has_provider("XnnpackExecutionProvider")) throw Error("XNNPACK EP not built in");
      so.SetIntraOpNumThreads(1);  // XNNPACK has its own pool
      std::unordered_map<std::string, std::string> opt{{"intra_op_num_threads", std::to_string(std::max(1, cfg.threads))}};
      so.AppendExecutionProvider("XNNPACK", opt);
      used_ = "xnnpack";
      break;
    }
    case Backend::NNAPI: {
#if RVC_HAVE_NNAPI
      uint32_t flags = NNAPI_FLAG_CPU_DISABLED;
      if (cfg.fp16) flags |= NNAPI_FLAG_USE_FP16;
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, flags));
      used_ = "nnapi";
#else
      throw Error("NNAPI EP not built in");
#endif
      break;
    }
    case Backend::WEBGPU: {
      if (!has_provider("WebGpuExecutionProvider")) throw Error("WebGPU EP not built in");
      std::unordered_map<std::string, std::string> opt;
      so.AppendExecutionProvider("WebGPU", opt);
      used_ = "webgpu";
      break;
    }
    case Backend::QNN_HTP:
    case Backend::QNN_GPU: {
      if (!has_provider("QNNExecutionProvider")) throw Error("QNN EP not built in (use the qnn app flavor)");
      bool htp = cfg.backend == Backend::QNN_HTP;
      std::unordered_map<std::string, std::string> opt{
          {"backend_path", htp ? "libQnnHtp.so" : "libQnnGpu.so"},
          {"htp_performance_mode", "burst"},
          {"htp_graph_finalization_optimization_mode", "3"},
          {"enable_htp_fp16_precision", cfg.fp16 ? "1" : "0"},
          {"qnn_context_priority", "high"},
      };
      if (!cfg.cache_dir.empty() && !diagnose) {
        mkdir(cfg.cache_dir.c_str(), 0755);
        std::string key = sanitize(tag);
        for (auto& fd : free_dims) key += "_" + std::to_string(fd.second);
        const std::string suffix = std::string(htp ? "_htp" : "_gpu") + "_ctx.onnx";
        // Earlier versions cached by architecture and shape only. A compiled context contains
        // the weights, so two voice models of the same architecture would have shared one, and
        // a corrected graph in an app update would never have been compiled.
        remove(join_path(cfg.cache_dir, key + suffix).c_str());
        key += "_x" + file_fingerprint(path) + cache_salt + (cfg.fp16 ? "h" : "f");
        std::string ctx = join_path(cfg.cache_dir, key + suffix);
        if (file_exists(ctx)) {
          load_path = ctx;  // pre-compiled QNN context: skips minutes of graph finalization
        } else {
          so.AddConfigEntry("ep.context_enable", "1");
          so.AddConfigEntry("ep.context_file_path", ctx.c_str());
          so.AddConfigEntry("ep.context_embed_mode", "1");
        }
        prune_cache(cfg.cache_dir, ctx, 1200ull << 20);
      }
      so.AppendExecutionProvider("QNN", opt);
      used_ = htp ? "qnn_htp" : "qnn_gpu";
      break;
    }
  }
  session_.reset(new Ort::Session(env, load_path.c_str(), so));
}

std::vector<Ort::Value> OnnxModel::run(const std::vector<const char*>& in_names, const std::vector<Ort::Value>& inputs,
                                      const std::vector<const char*>& out_names) {
  return session_->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), out_names.data(),
                       out_names.size());
}

}  // namespace rvc
