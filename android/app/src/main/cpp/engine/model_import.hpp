// On-device import of an original RVC voice model (.pth [+ .index]).
#pragma once

#include <string>

#include "json.hpp"

namespace rvc {

struct ImportResult {
  std::string model_dir;
  Json meta;  // contents of model.json
};

// assets_dir: directory holding synth/<key>/manifest.json templates.
// Writes into out_dir: weights.bin, the template graphs, emb_g.bin, model.json, index.bin.
ImportResult import_rvc_model(const std::string& assets_dir, const std::string& pth_path,
                              const std::string& index_path, const std::string& out_dir,
                              const std::string& display_name);

// Convert a faiss index for an already imported model and update its model.json.
Json attach_index(const std::string& model_dir, const std::string& index_path);

// Lightweight inspection (no files written): version, sr, f0, config, matching template.
Json inspect_rvc_model(const std::string& assets_dir, const std::string& pth_path);

}  // namespace rvc
