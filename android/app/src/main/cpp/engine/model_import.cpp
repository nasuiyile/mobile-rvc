#include "model_import.hpp"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <memory>

#include "common.hpp"
#include "ivf_index.hpp"
#include "pth_reader.hpp"

namespace rvc {

namespace {

std::vector<std::string> list_dirs(const std::string& path) {
  std::vector<std::string> out;
  DIR* d = opendir(path.c_str());
  if (!d) return out;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    out.emplace_back(e->d_name);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

void mkdirs(const std::string& path) {
  std::string cur;
  for (size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      if (!cur.empty()) mkdir(cur.c_str(), 0755);
    }
    if (i < path.size()) cur += path[i];
  }
}

int sr_from(const PyObj* v) {
  if (!v) return 0;
  if (v->kind == PyObj::Int) return static_cast<int>(v->i);
  if (v->kind == PyObj::Str) {
    if (v->s == "32k") return 32000;
    if (v->s == "40k") return 40000;
    if (v->s == "48k") return 48000;
  }
  return 0;
}

struct Checkpoint {
  std::unique_ptr<PthFile> pth;
  const PyObj* weight = nullptr;
  Json config;
  std::string version;
  int f0 = 1;
  int sr = 0;
  int phone_dim = 0;
  Json signature;

  const TensorRef& tensor(const std::string& name) const {
    const PyObj* t = weight->get(name);
    if (!t || t->kind != PyObj::Tensor) throw Error("checkpoint lacks tensor " + name);
    return t->tensor;
  }
};

Checkpoint load_checkpoint(const std::string& pth_path) {
  Checkpoint c;
  c.pth.reset(new PthFile(pth_path));
  const PyObj& root = c.pth->root();
  c.weight = root.get("weight");
  if (!c.weight || c.weight->kind != PyObj::Dict) {
    if (root.get("model") || root.get("state_dict"))
      throw Error("this looks like a training checkpoint (G_*.pth); export the small inference model in RVC first");
    throw Error("not an RVC voice model (no 'weight' dict)");
  }
  const PyObj* cfg = root.get("config");
  if (!cfg || (cfg->kind != PyObj::List && cfg->kind != PyObj::Tuple)) throw Error("model has no config");
  c.config = cfg->to_json();
  if (c.config.size() < 18) throw Error("unexpected model config length");
  const PyObj* ver = root.get("version");
  c.version = ver && ver->kind == PyObj::Str ? ver->s : "v1";
  const PyObj* f0 = root.get("f0");
  c.f0 = f0 && f0->kind == PyObj::Int ? static_cast<int>(f0->i) : 1;
  c.sr = sr_from(root.get("sr"));
  if (!c.sr) {
    const Json& last = c.config[c.config.size() - 1];
    c.sr = last.is_number() ? last.i32() : last.is_string() ? (last.str() == "32k" ? 32000 : last.str() == "48k" ? 48000 : 40000) : 0;
  }
  const TensorRef& emb = c.tensor("enc_p.emb_phone.weight");
  c.phone_dim = static_cast<int>(emb.shape.size() == 2 ? emb.shape[1] : 0);

  // must mirror tools/rvc_mobile/export_synth.py signature()
  const Json& g = c.config;
  Json s = Json::object();
  s.set("phone_dim", c.phone_dim);
  s.set("inter_channels", g[2]);
  s.set("hidden_channels", g[3]);
  s.set("filter_channels", g[4]);
  s.set("n_heads", g[5]);
  s.set("n_layers", g[6]);
  s.set("kernel_size", g[7]);
  s.set("resblock", g[9].is_string() ? g[9] : Json(std::to_string(g[9].i64())));
  s.set("resblock_kernel_sizes", g[10]);
  s.set("resblock_dilation_sizes", g[11]);
  s.set("upsample_rates", g[12]);
  s.set("upsample_initial_channel", g[13]);
  s.set("upsample_kernel_sizes", g[14]);
  s.set("gin_channels", g[16]);
  c.signature = s;
  return c;
}

std::string find_template(const std::string& assets_dir, const Json& sig, Json* manifest) {
  std::string root = join_path(assets_dir, "synth");
  for (auto& key : list_dirs(root)) {
    std::string mp = join_path(join_path(root, key), "manifest.json");
    if (!file_exists(mp)) continue;
    Json m = Json::parse(read_file(mp));
    if (m["signature"] == sig) {
      if (manifest) *manifest = m;
      return join_path(root, key);
    }
  }
  return "";
}

}  // namespace

Json inspect_rvc_model(const std::string& assets_dir, const std::string& pth_path) {
  Checkpoint c = load_checkpoint(pth_path);
  Json out = Json::object();
  out.set("version", c.version);
  out.set("sr", c.sr);
  out.set("f0", c.f0);
  out.set("phone_dim", c.phone_dim);
  out.set("signature", c.signature);
  Json man;
  std::string tpl = find_template(assets_dir, c.signature, &man);
  out.set("template", tpl.empty() ? Json() : Json(man["key"].str()));
  return out;
}

ImportResult import_rvc_model(const std::string& assets_dir, const std::string& pth_path,
                              const std::string& index_path, const std::string& out_dir,
                              const std::string& display_name) {
  Checkpoint c = load_checkpoint(pth_path);
  if (c.f0 != 1) throw Error("only pitch (f0) models are supported for realtime conversion");
  Json man;
  std::string tpl = find_template(assets_dir, c.signature, &man);
  if (tpl.empty()) {
    throw Error("unsupported model architecture (" + c.signature.dump() +
                "); generate a template with tools/rvc_mobile/export_synth.py --pth");
  }
  mkdirs(out_dir);

  // weights.bin: raw fp16 tensors at the template's offsets
  size_t data_size = static_cast<size_t>(man["data_size"].i64());
  std::vector<uint8_t> buf(data_size, 0);
  for (auto& t : man["tensors"].items()) {
    const std::string& name = t["name"].str();
    const TensorRef& tr = c.tensor(name);
    const Json& shp = t["shape"];
    bool ok = shp.size() == tr.shape.size();
    for (size_t i = 0; ok && i < shp.size(); ++i) ok = shp[i].i64() == tr.shape[i];
    if (!ok) throw Error("tensor shape mismatch: " + name);
    std::vector<uint16_t> h = c.pth->tensor_f16(tr);
    size_t off = static_cast<size_t>(t["offset"].i64()), nb = static_cast<size_t>(t["nbytes"].i64());
    if (nb != h.size() * 2 || off + nb > data_size) throw Error("bad manifest entry " + name);
    memcpy(buf.data() + off, h.data(), nb);
  }
  write_file(join_path(out_dir, man["data_file"].str()), buf.data(), buf.size());
  buf.clear();
  buf.shrink_to_fit();
  for (auto& g : man["graphs"].entries()) copy_file(join_path(tpl, g.second.str()), join_path(out_dir, g.second.str()));
  copy_file(join_path(tpl, "manifest.json"), join_path(out_dir, "manifest.json"));  // native decoder

  // speaker embeddings + NSF source merge layer (computed natively, outside the graphs)
  const TensorRef& emb_g = c.tensor("emb_g.weight");
  std::vector<float> eg = c.pth->tensor_f32(emb_g);
  write_file(join_path(out_dir, "emb_g.bin"), eg.data(), eg.size() * 4);
  std::vector<float> lw = c.pth->tensor_f32(c.tensor("dec.m_source.l_linear.weight"));
  std::vector<float> lb = c.pth->tensor_f32(c.tensor("dec.m_source.l_linear.bias"));

  Json meta = Json::object();
  meta.set("format", 1);
  meta.set("name", display_name);
  meta.set("version", c.version);
  meta.set("tgt_sr", c.sr);
  meta.set("upp", man["upp"]);
  meta.set("template", man["key"]);
  meta.set("phone_dim", c.phone_dim);
  meta.set("graphs", man["graphs"]);
  meta.set("n_spk", emb_g.shape.empty() ? 1 : emb_g.shape[0]);
  meta.set("gin_channels", emb_g.shape.size() > 1 ? emb_g.shape[1] : 256);
  meta.set("l_linear_w", lw.empty() ? 1.0 : lw[0]);
  meta.set("l_linear_b", lb.empty() ? 0.0 : lb[0]);
  meta.set("has_index", false);
  if (!index_path.empty()) {
    int dim = 0;
    int64_t n = convert_faiss_index(index_path, join_path(out_dir, "index.bin"), &dim);
    if (dim != c.phone_dim) {
      remove(join_path(out_dir, "index.bin").c_str());
      throw Error("index dimension " + std::to_string(dim) + " does not match model (" + std::to_string(c.phone_dim) + ")");
    }
    meta.set("has_index", true);
    meta.set("index_size", n);
  }
  write_file(join_path(out_dir, "model.json"), meta.dump());
  return ImportResult{out_dir, meta};
}

Json attach_index(const std::string& model_dir, const std::string& index_path) {
  std::string mp = join_path(model_dir, "model.json");
  Json meta = Json::parse(read_file(mp));
  std::string tmp = join_path(model_dir, "index.bin.new");
  int dim = 0;
  int64_t n = convert_faiss_index(index_path, tmp, &dim);
  if (dim != meta["phone_dim"].i32()) {
    remove(tmp.c_str());
    throw Error("index dimension " + std::to_string(dim) + " does not match model (" +
                std::to_string(meta["phone_dim"].i32()) + ")");
  }
  if (rename(tmp.c_str(), join_path(model_dir, "index.bin").c_str()) != 0) throw Error("cannot replace index.bin");
  meta.set("has_index", true);
  meta.set("index_size", n);
  write_file(mp, meta.dump());
  return meta;
}

}  // namespace rvc
