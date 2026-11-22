#include "ort_model.hpp"

#include <sys/stat.h>

#include <unordered_map>

#include "common.hpp"

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

Ort::Env& ort_env() {
  // intentionally leaked: EP libraries (WebGPU/Dawn, QNN) must not be torn down by static destructors
  static Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "rvc");
  return *env;
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

OnnxModel::OnnxModel(const std::string& path, const BackendConfig& cfg,
                     const std::vector<std::pair<std::string, int64_t>>& free_dims, const std::string& tag) {
  auto t0 = Clock::now();
  bool accel = cfg.backend != Backend::CPU && cfg.backend != Backend::XNNPACK && cfg.backend != Backend::NATIVE;
  // Graphs are fp32-compute: on ORT's WebGPU EP, fp16-compute graphs measured 7.6 dB (HuBERT) /
  // 13.5 dB (decoder) SNR, fp32 is exact. Reduced precision is requested from the EP instead
  // (QNN enable_htp_fp16_precision, NNAPI_FLAG_USE_FP16) or done by the native decoder.
  try {
    create(path, cfg, free_dims, tag);
  } catch (const std::exception& e) {
    if (!accel && cfg.backend != Backend::XNNPACK) throw;
    warning_ = std::string(backend_name(cfg.backend)) + " unavailable for " + tag + ": " + e.what();
    RVC_LOGW("%s -> falling back to CPU", warning_.c_str());
    BackendConfig cpu = cfg;
    cpu.backend = Backend::CPU;
    create(path, cpu, free_dims, tag);
  }
  load_ms_ = ms_since(t0);
  RVC_LOGI("session %s on %s (%.0f ms)", tag.c_str(), used_.c_str(), load_ms_);
}

void OnnxModel::create(const std::string& path, const BackendConfig& cfg,
                      const std::vector<std::pair<std::string, int64_t>>& free_dims, const std::string& tag) {
  Ort::Env& env = ort_env();  // must exist before execution providers are appended (logger)
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  so.SetInterOpNumThreads(1);
  so.SetIntraOpNumThreads(std::max(1, cfg.threads));
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
      if (!cfg.cache_dir.empty()) {
        mkdir(cfg.cache_dir.c_str(), 0755);
        std::string key = sanitize(tag);
        for (auto& fd : free_dims) key += "_" + std::to_string(fd.second);
        std::string ctx = join_path(cfg.cache_dir, key + (htp ? "_htp" : "_gpu") + "_ctx.onnx");
        if (file_exists(ctx)) {
          load_path = ctx;  // pre-compiled QNN context: skips minutes of graph finalization
        } else {
          so.AddConfigEntry("ep.context_enable", "1");
          so.AddConfigEntry("ep.context_file_path", ctx.c_str());
          so.AddConfigEntry("ep.context_embed_mode", "1");
        }
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
