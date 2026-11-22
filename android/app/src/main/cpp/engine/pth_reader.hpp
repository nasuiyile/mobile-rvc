// Reader for PyTorch zip checkpoints (torch.save, as used by RVC for weights/*.pth).
// Implements just enough of ZIP (stored entries, ZIP64) and the pickle VM to recover
// the checkpoint dictionary, its plain values and tensor descriptors - without Python.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"

namespace rvc {

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

enum class DType { F16, F32, BF16, F64, I64, Other };

struct TensorRef {
  DType dtype = DType::Other;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  int64_t storage_offset = 0;  // in elements
  std::string storage_key;
  int64_t numel() const;
  bool contiguous() const;
};

// Python-ish value produced by the pickle VM.
struct PyObj;
using PyRef = std::shared_ptr<PyObj>;
struct PyObj {
  enum Kind { None, Bool, Int, Float, Str, Bytes, List, Tuple, Dict, Global, Tensor, Storage, Opaque } kind = None;
  bool b = false;
  int64_t i = 0;
  double f = 0;
  std::string s;                            // Str / Bytes / Global name / Opaque name
  std::vector<PyRef> items;                 // List / Tuple
  std::vector<std::pair<PyRef, PyRef>> kv;  // Dict
  TensorRef tensor;                         // Tensor / Storage(key,dtype)
  const PyObj* get(const std::string& key) const;  // dict lookup by str key
  Json to_json() const;                     // plain values only (tensors -> null)
};

class PthFile {
 public:
  explicit PthFile(const std::string& path);
  const PyObj& root() const { return *root_; }
  // Copy a tensor as IEEE fp16, row-major contiguous. Throws on unsupported dtype.
  std::vector<uint16_t> tensor_f16(const TensorRef& t) const;
  std::vector<float> tensor_f32(const TensorRef& t) const;

 private:
  struct Entry { uint64_t offset = 0, size = 0; };
  MappedFile file_;
  std::map<std::string, Entry> entries_;  // name without archive prefix
  PyRef root_;
  void read_zip();
  const Entry& entry(const std::string& name) const;
  template <typename F> void for_each_element(const TensorRef& t, F&& f) const;
};

}  // namespace rvc
