// Feature retrieval index (RVC "added_IVF*_Flat*.index").
//
// faiss is not shipped: the importer converts the faiss IndexIVFFlat file into a compact
// "RVCIDX1" file (fp32 centroids + fp16 vectors grouped by inverted list), and the
// search below reproduces faiss IVFFlat L2 search + RVC's 1/d^2 weighted blending.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rvc {

class MappedFile;

// Parse a faiss IndexIVFFlat file and write it as index.bin. Returns the vector count.
int64_t convert_faiss_index(const std::string& faiss_path, const std::string& out_path, int* dim_out = nullptr);

class IvfIndex {
 public:
  explicit IvfIndex(const std::string& path);
  ~IvfIndex();
  int dim() const { return dim_; }
  int64_t size() const { return ntotal_; }
  int nprobe() const { return nprobe_; }
  // For each of n query rows (dim floats each) compute the k-NN blend into out (n x dim).
  void search_blend(const float* q, int n, float* out, int k = 8, int nprobe = 0) const;

 private:
  std::unique_ptr<MappedFile> file_;
  int dim_ = 0, nlist_ = 0, nprobe_ = 1;
  bool f16_ = true;
  int64_t ntotal_ = 0;
  const float* centroids_ = nullptr;
  const uint64_t* offsets_ = nullptr;
  const uint16_t* vec16_ = nullptr;
  const float* vec32_ = nullptr;
  std::vector<float> cnorm_;
};

}  // namespace rvc
