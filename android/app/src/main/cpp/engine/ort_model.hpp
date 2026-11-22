// ONNX Runtime session wrapper with mobile execution-provider selection.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rvc {

// NATIVE: built-in NEON decoder (decoder only; other stages treat it as CPU)
enum class Backend { CPU = 0, XNNPACK = 1, NNAPI = 2, WEBGPU = 3, QNN_HTP = 4, QNN_GPU = 5, NATIVE = 6 };

const char* backend_name(Backend b);
Backend backend_from_name(const std::string& s);
// Execution providers compiled into the linked onnxruntime library.
std::vector<std::string> available_providers();

struct BackendConfig {
  Backend backend = Backend::CPU;
  int threads = 4;            // CPU intra-op threads (also XNNPACK pool size)
  bool fp16 = true;           // accelerators: fp16 graph / relaxed fp16 arithmetic
  bool spinning = true;       // ORT intra-op spin-wait (lower latency, more power)
  std::vector<int> affinity;  // optional logical CPU ids for intra-op threads
  std::string cache_dir;      // QNN context binaries / compiled-graph cache
};

Ort::Env& ort_env();

class OnnxModel {
 public:
  // free_dims: symbolic input dimensions fixed for this session (static shapes for accelerators).
  OnnxModel(const std::string& path, const BackendConfig& cfg,
            const std::vector<std::pair<std::string, int64_t>>& free_dims, const std::string& tag);

  std::vector<Ort::Value> run(const std::vector<const char*>& in_names, const std::vector<Ort::Value>& inputs,
                              const std::vector<const char*>& out_names);
  const std::string& backend_used() const { return used_; }
  const std::string& warning() const { return warning_; }
  double load_ms() const { return load_ms_; }

 private:
  std::unique_ptr<Ort::Session> session_;
  std::string used_, warning_;
  double load_ms_ = 0;
  void create(const std::string& path, const BackendConfig& cfg,
              const std::vector<std::pair<std::string, int64_t>>& free_dims, const std::string& tag);
};

Ort::MemoryInfo& cpu_mem();

template <typename T>
Ort::Value make_tensor(T* data, size_t count, const std::vector<int64_t>& shape) {
  return Ort::Value::CreateTensor<T>(cpu_mem(), data, count, shape.data(), shape.size());
}

}  // namespace rvc
