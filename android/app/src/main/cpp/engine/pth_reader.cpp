#include "pth_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <functional>

#include "common.hpp"

namespace rvc {

// ------------------------------------------------------------------ MappedFile
MappedFile::MappedFile(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw Error("cannot open " + path);
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    throw Error("cannot stat " + path);
  }
  size_ = static_cast<size_t>(st.st_size);
  void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) throw Error("mmap failed " + path);
  data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
  if (data_) munmap(const_cast<uint8_t*>(data_), size_);
}

// ------------------------------------------------------------------ TensorRef
int64_t TensorRef::numel() const {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

bool TensorRef::contiguous() const {
  int64_t s = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] != 1 && stride[i] != s) return false;
    s *= shape[i];
  }
  return true;
}

const PyObj* PyObj::get(const std::string& key) const {
  if (kind != Dict) return nullptr;
  for (auto& p : kv)
    if (p.first && p.first->kind == Str && p.first->s == key) return p.second.get();
  return nullptr;
}

Json PyObj::to_json() const {
  switch (kind) {
    case Bool: return Json(b);
    case Int: return Json(i);
    case Float: return Json(f);
    case Str: return Json(s);
    case List:
    case Tuple: {
      Json a = Json::array();
      for (auto& it : items) a.push(it ? it->to_json() : Json());
      return a;
    }
    case Dict: {
      Json o = Json::object();
      for (auto& p : kv)
        if (p.first && p.first->kind == Str) o.set(p.first->s, p.second ? p.second->to_json() : Json());
      return o;
    }
    default: return Json();
  }
}

// ------------------------------------------------------------------ ZIP
static uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24); }
static uint64_t rd64(const uint8_t* p) { return rd32(p) | (static_cast<uint64_t>(rd32(p + 4)) << 32); }

void PthFile::read_zip() {
  const uint8_t* d = file_.data();
  size_t n = file_.size();
  if (n < 22) throw Error("not a zip file");
  if (!(d[0] == 'P' && d[1] == 'K')) {
    throw Error("unsupported checkpoint format (legacy torch.save); re-save it with a recent PyTorch");
  }
  // End of central directory
  size_t eocd = SIZE_MAX;
  size_t lo = n > 65557 ? n - 65557 : 0;
  for (size_t i = n - 22 + 1; i-- > lo;) {
    if (rd32(d + i) == 0x06054b50) { eocd = i; break; }
  }
  if (eocd == SIZE_MAX) throw Error("zip: no end of central directory");
  uint64_t count = rd16(d + eocd + 10);
  uint64_t cd_size = rd32(d + eocd + 12);
  uint64_t cd_off = rd32(d + eocd + 16);
  if (count == 0xFFFF || cd_off == 0xFFFFFFFFu || cd_size == 0xFFFFFFFFu) {
    if (eocd < 20 || rd32(d + eocd - 20) != 0x07064b50) throw Error("zip64: missing locator");
    uint64_t z64 = rd64(d + eocd - 20 + 8);
    if (z64 + 56 > n || rd32(d + z64) != 0x06064b50) throw Error("zip64: bad record");
    count = rd64(d + z64 + 32);
    cd_size = rd64(d + z64 + 40);
    cd_off = rd64(d + z64 + 48);
  }
  if (cd_off + cd_size > n) throw Error("zip: central directory out of range");
  size_t p = cd_off;
  std::string prefix;
  for (uint64_t k = 0; k < count; ++k) {
    if (p + 46 > n || rd32(d + p) != 0x02014b50) throw Error("zip: bad central header");
    uint16_t method = rd16(d + p + 10);
    uint64_t csize = rd32(d + p + 20), usize = rd32(d + p + 24);
    uint16_t nlen = rd16(d + p + 28), xlen = rd16(d + p + 30), clen = rd16(d + p + 32);
    uint64_t loff = rd32(d + p + 42);
    std::string name(reinterpret_cast<const char*>(d + p + 46), nlen);
    // ZIP64 extra field
    const uint8_t* x = d + p + 46 + nlen;
    for (size_t q = 0; q + 4 <= xlen;) {
      uint16_t id = rd16(x + q), sz = rd16(x + q + 2);
      if (id == 0x0001) {
        size_t r = q + 4;
        if (usize == 0xFFFFFFFFu && r + 8 <= q + 4 + sz) { usize = rd64(x + r); r += 8; }
        if (csize == 0xFFFFFFFFu && r + 8 <= q + 4 + sz) { csize = rd64(x + r); r += 8; }
        if (loff == 0xFFFFFFFFu && r + 8 <= q + 4 + sz) { loff = rd64(x + r); r += 8; }
      }
      q += 4 + sz;
    }
    p += 46 + nlen + xlen + clen;
    if (loff + 30 > n || rd32(d + loff) != 0x04034b50) throw Error("zip: bad local header");
    uint64_t data_off = loff + 30 + rd16(d + loff + 26) + rd16(d + loff + 28);
    if (method != 0) {
      if (name.find("data/") != std::string::npos || name.find("data.pkl") != std::string::npos)
        throw Error("zip: compressed entries are not supported (" + name + ")");
      continue;
    }
    if (data_off + usize > n) throw Error("zip: entry out of range");
    size_t slash = name.find('/');
    std::string rel = slash == std::string::npos ? name : name.substr(slash + 1);
    entries_[rel] = Entry{data_off, usize};
    (void)csize;
  }
}

const PthFile::Entry& PthFile::entry(const std::string& name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) throw Error("checkpoint is missing entry " + name);
  return it->second;
}

// ------------------------------------------------------------------ pickle VM
namespace {

PyRef mk(PyObj::Kind k) { auto o = std::make_shared<PyObj>(); o->kind = k; return o; }
PyRef mk_str(PyObj::Kind k, std::string s) { auto o = mk(k); o->s = std::move(s); return o; }
PyRef mk_int(int64_t v) { auto o = mk(PyObj::Int); o->i = v; return o; }

DType storage_dtype(const std::string& g) {
  if (g == "torch.HalfStorage") return DType::F16;
  if (g == "torch.FloatStorage") return DType::F32;
  if (g == "torch.BFloat16Storage") return DType::BF16;
  if (g == "torch.DoubleStorage") return DType::F64;
  if (g == "torch.LongStorage") return DType::I64;
  return DType::Other;
}

std::vector<int64_t> int_tuple(const PyRef& t) {
  std::vector<int64_t> v;
  if (!t || (t->kind != PyObj::Tuple && t->kind != PyObj::List)) return v;
  for (auto& it : t->items) v.push_back(it && it->kind == PyObj::Int ? it->i : 0);
  return v;
}

class Unpickler {
 public:
  Unpickler(const uint8_t* d, size_t n) : d_(d), n_(n) {}

  PyRef run() {
    for (;;) {
      uint8_t op = u8();
      switch (op) {
        case 0x80: u8(); break;                            // PROTO
        case 0x95: skip(8); break;                         // FRAME
        case '.': return pop();                            // STOP
        case '(': marks_.push_back(stack_.size()); break;  // MARK
        case '}': stack_.push_back(mk(PyObj::Dict)); break;
        case ']': stack_.push_back(mk(PyObj::List)); break;
        case ')': stack_.push_back(mk(PyObj::Tuple)); break;
        case 'N': stack_.push_back(mk(PyObj::None)); break;
        case 0x88: { auto o = mk(PyObj::Bool); o->b = true; stack_.push_back(o); break; }
        case 0x89: stack_.push_back(mk(PyObj::Bool)); break;
        case 'J': stack_.push_back(mk_int(static_cast<int32_t>(u32()))); break;
        case 'K': stack_.push_back(mk_int(u8())); break;
        case 'M': stack_.push_back(mk_int(u16())); break;
        case 0x8a: { size_t k = u8(); stack_.push_back(mk_int(read_long(k))); break; }
        case 0x8b: { size_t k = u32(); stack_.push_back(mk_int(read_long(k))); break; }
        case 'G': {
          uint64_t b = 0;
          for (int i = 0; i < 8; ++i) b = (b << 8) | u8();
          auto o = mk(PyObj::Float);
          memcpy(&o->f, &b, 8);
          stack_.push_back(o);
          break;
        }
        case 'X': { size_t k = u32(); stack_.push_back(mk_str(PyObj::Str, bytes(k))); break; }
        case 0x8c: { size_t k = u8(); stack_.push_back(mk_str(PyObj::Str, bytes(k))); break; }
        case 0x8d: { size_t k = static_cast<size_t>(u64()); stack_.push_back(mk_str(PyObj::Str, bytes(k))); break; }
        case 'T': { size_t k = u32(); stack_.push_back(mk_str(PyObj::Str, bytes(k))); break; }
        case 'U': { size_t k = u8(); stack_.push_back(mk_str(PyObj::Str, bytes(k))); break; }
        case 'B': { size_t k = u32(); stack_.push_back(mk_str(PyObj::Bytes, bytes(k))); break; }
        case 'C': { size_t k = u8(); stack_.push_back(mk_str(PyObj::Bytes, bytes(k))); break; }
        case 'c': {  // GLOBAL module\nname\n
          std::string mod = line(), name = line();
          stack_.push_back(mk_str(PyObj::Global, mod + "." + name));
          break;
        }
        case 0x93: {  // STACK_GLOBAL
          PyRef name = pop(), mod = pop();
          stack_.push_back(mk_str(PyObj::Global, mod->s + "." + name->s));
          break;
        }
        case 'q': memo_[u8()] = top(); break;
        case 'r': memo_[u32()] = top(); break;
        case 0x94: memo_[memo_.size()] = top(); break;
        case 'h': stack_.push_back(memo_get(u8())); break;
        case 'j': stack_.push_back(memo_get(u32())); break;
        case 't': { auto t = mk(PyObj::Tuple); t->items = pop_mark(); stack_.push_back(t); break; }
        case 0x85: { auto t = mk(PyObj::Tuple); t->items = {pop()}; stack_.push_back(t); break; }
        case 0x86: { auto t = mk(PyObj::Tuple); PyRef b = pop(), a = pop(); t->items = {a, b}; stack_.push_back(t); break; }
        case 0x87: {
          auto t = mk(PyObj::Tuple);
          PyRef c = pop(), b = pop(), a = pop();
          t->items = {a, b, c};
          stack_.push_back(t);
          break;
        }
        case 'l': { auto l = mk(PyObj::List); l->items = pop_mark(); stack_.push_back(l); break; }
        case 'd': {
          auto dct = mk(PyObj::Dict);
          auto it = pop_mark();
          for (size_t i = 0; i + 1 < it.size(); i += 2) dct->kv.emplace_back(it[i], it[i + 1]);
          stack_.push_back(dct);
          break;
        }
        case 'a': { PyRef v = pop(); top()->items.push_back(v); break; }
        case 'e': { auto it = pop_mark(); auto& l = top()->items; l.insert(l.end(), it.begin(), it.end()); break; }
        case 's': { PyRef v = pop(), k = pop(); top()->kv.emplace_back(k, v); break; }
        case 'u': {
          auto it = pop_mark();
          auto& kv = top()->kv;
          for (size_t i = 0; i + 1 < it.size(); i += 2) kv.emplace_back(it[i], it[i + 1]);
          break;
        }
        case 'Q': {  // BINPERSID: ('storage', storage_type, key, location, numel)
          PyRef pid = pop();
          auto st = mk(PyObj::Storage);
          if (pid->kind == PyObj::Tuple && pid->items.size() >= 3) {
            st->tensor.dtype = storage_dtype(pid->items[1] ? pid->items[1]->s : "");
            st->tensor.storage_key = pid->items[2] ? pid->items[2]->s : "";
          }
          stack_.push_back(st);
          break;
        }
        case 'R': {  // REDUCE
          PyRef args = pop(), fn = pop();
          stack_.push_back(reduce(fn, args));
          break;
        }
        case 0x81: {  // NEWOBJ
          PyRef args = pop(), cls = pop();
          stack_.push_back(mk_str(PyObj::Opaque, cls->s));
          (void)args;
          break;
        }
        case 'b': pop(); break;  // BUILD: state ignored (OrderedDict / parameter metadata)
        case '0': pop(); break;  // POP
        case '1': pop_mark(); break;
        case '2': stack_.push_back(top()); break;  // DUP
        default: {
          char buf[64];
          snprintf(buf, sizeof buf, "pickle: unsupported opcode 0x%02x", op);
          throw Error(buf);
        }
      }
    }
  }

 private:
  const uint8_t* d_;
  size_t n_, p_ = 0;
  std::vector<PyRef> stack_;
  std::vector<size_t> marks_;
  std::map<size_t, PyRef> memo_;

  void need(size_t k) { if (p_ + k > n_) throw Error("pickle: truncated"); }
  uint8_t u8() { need(1); return d_[p_++]; }
  uint16_t u16() { need(2); uint16_t v = rd16(d_ + p_); p_ += 2; return v; }
  uint32_t u32() { need(4); uint32_t v = rd32(d_ + p_); p_ += 4; return v; }
  uint64_t u64() { need(8); uint64_t v = rd64(d_ + p_); p_ += 8; return v; }
  void skip(size_t k) { need(k); p_ += k; }
  std::string bytes(size_t k) { need(k); std::string s(reinterpret_cast<const char*>(d_ + p_), k); p_ += k; return s; }
  std::string line() {
    size_t q = p_;
    while (q < n_ && d_[q] != '\n') ++q;
    std::string s(reinterpret_cast<const char*>(d_ + p_), q - p_);
    p_ = q + 1;
    return s;
  }
  int64_t read_long(size_t k) {
    need(k);
    if (k == 0) return 0;
    uint64_t v = 0;
    for (size_t i = 0; i < k && i < 8; ++i) v |= static_cast<uint64_t>(d_[p_ + i]) << (8 * i);
    if (k < 8 && (d_[p_ + k - 1] & 0x80)) v |= ~0ULL << (8 * k);
    p_ += k;
    return static_cast<int64_t>(v);
  }
  PyRef pop() {
    if (stack_.empty()) throw Error("pickle: stack underflow");
    PyRef v = stack_.back();
    stack_.pop_back();
    return v;
  }
  PyRef& top() {
    if (stack_.empty()) throw Error("pickle: empty stack");
    return stack_.back();
  }
  std::vector<PyRef> pop_mark() {
    if (marks_.empty()) throw Error("pickle: no mark");
    size_t m = marks_.back();
    marks_.pop_back();
    std::vector<PyRef> v(stack_.begin() + static_cast<long>(m), stack_.end());
    stack_.resize(m);
    return v;
  }
  PyRef memo_get(size_t k) {
    auto it = memo_.find(k);
    if (it == memo_.end()) throw Error("pickle: bad memo");
    return it->second;
  }

  PyRef reduce(const PyRef& fn, const PyRef& args) {
    const std::string& name = fn->s;
    auto& a = args->items;
    if (name == "collections.OrderedDict" || name == "builtins.dict") return mk(PyObj::Dict);
    if ((name == "torch._utils._rebuild_tensor_v2" || name == "torch._utils._rebuild_tensor") && a.size() >= 4) {
      auto t = mk(PyObj::Tensor);
      t->tensor = a[0]->tensor;  // storage: dtype + key
      t->tensor.storage_offset = a[1] && a[1]->kind == PyObj::Int ? a[1]->i : 0;
      t->tensor.shape = int_tuple(a[2]);
      t->tensor.stride = int_tuple(a[3]);
      return t;
    }
    if (name == "torch._utils._rebuild_parameter" && !a.empty()) return a[0];
    if (name == "builtins.set" || name == "builtins.frozenset" || name == "builtins.list") {
      auto l = mk(PyObj::List);
      if (!a.empty() && a[0]) l->items = a[0]->items;
      return l;
    }
    return mk_str(PyObj::Opaque, name);
  }
};

}  // namespace

PthFile::PthFile(const std::string& path) : file_(path) {
  read_zip();
  const Entry& pk = entry("data.pkl");
  Unpickler up(file_.data() + pk.offset, pk.size);
  root_ = up.run();
  if (!root_ || root_->kind != PyObj::Dict) throw Error("checkpoint root is not a dict");
}

template <typename F>
void PthFile::for_each_element(const TensorRef& t, F&& f) const {
  size_t esz = t.dtype == DType::F32 ? 4 : t.dtype == DType::F64 || t.dtype == DType::I64 ? 8 : 2;
  const Entry& e = entry("data/" + t.storage_key);
  const uint8_t* base = file_.data() + e.offset;
  int64_t n = t.numel();
  if (t.contiguous()) {
    if (static_cast<uint64_t>((t.storage_offset + n) * static_cast<int64_t>(esz)) > e.size) throw Error("tensor out of storage range");
    for (int64_t i = 0; i < n; ++i) f(i, base + (t.storage_offset + i) * esz);
    return;
  }
  // generic strided walk
  size_t nd = t.shape.size();
  std::vector<int64_t> idx(nd, 0);
  for (int64_t i = 0; i < n; ++i) {
    int64_t off = t.storage_offset;
    for (size_t k = 0; k < nd; ++k) off += idx[k] * t.stride[k];
    if (static_cast<uint64_t>((off + 1) * static_cast<int64_t>(esz)) > e.size) throw Error("tensor out of storage range");
    f(i, base + off * esz);
    for (size_t k = nd; k-- > 0;) {
      if (++idx[k] < t.shape[k]) break;
      idx[k] = 0;
    }
  }
}

std::vector<uint16_t> PthFile::tensor_f16(const TensorRef& t) const {
  std::vector<uint16_t> out(static_cast<size_t>(t.numel()));
  switch (t.dtype) {
    case DType::F16: for_each_element(t, [&](int64_t i, const uint8_t* p) { memcpy(&out[i], p, 2); }); break;
    case DType::BF16: for_each_element(t, [&](int64_t i, const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); out[i] = bf16_to_half(v); }); break;
    case DType::F32: for_each_element(t, [&](int64_t i, const uint8_t* p) { float v; memcpy(&v, p, 4); out[i] = float_to_half(v); }); break;
    case DType::F64: for_each_element(t, [&](int64_t i, const uint8_t* p) { double v; memcpy(&v, p, 8); out[i] = float_to_half(static_cast<float>(v)); }); break;
    default: throw Error("unsupported tensor dtype");
  }
  return out;
}

std::vector<float> PthFile::tensor_f32(const TensorRef& t) const {
  std::vector<float> out(static_cast<size_t>(t.numel()));
  switch (t.dtype) {
    case DType::F16: for_each_element(t, [&](int64_t i, const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); out[i] = half_to_float(v); }); break;
    case DType::BF16: for_each_element(t, [&](int64_t i, const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); uint32_t x = static_cast<uint32_t>(v) << 16; memcpy(&out[i], &x, 4); }); break;
    case DType::F32: for_each_element(t, [&](int64_t i, const uint8_t* p) { memcpy(&out[i], p, 4); }); break;
    case DType::F64: for_each_element(t, [&](int64_t i, const uint8_t* p) { double v; memcpy(&v, p, 8); out[i] = static_cast<float>(v); }); break;
    default: throw Error("unsupported tensor dtype");
  }
  return out;
}

}  // namespace rvc
