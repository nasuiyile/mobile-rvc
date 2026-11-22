// Minimal JSON value + parser/serializer (manifests, model.json, settings, stats).
#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rvc {

class Json {
 public:
  enum Type { Null, Bool, Number, String, Array, Object };

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : type_(Bool), b_(b) {}
  Json(int v) : type_(Number), n_(v) {}
  Json(int64_t v) : type_(Number), n_(static_cast<double>(v)) {}
  Json(size_t v) : type_(Number), n_(static_cast<double>(v)) {}
  Json(double v) : type_(Number), n_(v) {}
  Json(float v) : type_(Number), n_(v) {}
  Json(const char* s) : type_(String), s_(s) {}
  Json(std::string s) : type_(String), s_(std::move(s)) {}

  static Json array() { Json j; j.type_ = Array; return j; }
  static Json object() { Json j; j.type_ = Object; return j; }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Null; }
  bool is_number() const { return type_ == Number; }
  bool is_string() const { return type_ == String; }
  bool is_array() const { return type_ == Array; }
  bool is_object() const { return type_ == Object; }

  double num() const { expect(Number); return n_; }
  int64_t i64() const { return static_cast<int64_t>(std::llround(num())); }
  int i32() const { return static_cast<int>(i64()); }
  bool boolean() const { expect(Bool); return b_; }
  const std::string& str() const { expect(String); return s_; }

  size_t size() const { return type_ == Array ? arr_.size() : type_ == Object ? obj_.size() : 0; }
  const Json& operator[](size_t i) const { expect(Array); return arr_.at(i); }
  Json& push(Json v) { if (type_ == Null) type_ = Array; expect(Array); arr_.push_back(std::move(v)); return arr_.back(); }
  const std::vector<Json>& items() const { expect(Array); return arr_; }

  bool has(const std::string& k) const {
    if (type_ != Object) return false;
    for (auto& kv : obj_) if (kv.first == k) return true;
    return false;
  }
  const Json& operator[](const std::string& k) const {
    expect(Object);
    for (auto& kv : obj_) if (kv.first == k) return kv.second;
    throw std::runtime_error("json: missing key '" + k + "'");
  }
  const Json& get(const std::string& k) const {
    static const Json null;
    if (type_ != Object) return null;
    for (auto& kv : obj_) if (kv.first == k) return kv.second;
    return null;
  }
  Json& set(const std::string& k, Json v) {
    if (type_ == Null) type_ = Object;
    expect(Object);
    for (auto& kv : obj_) if (kv.first == k) { kv.second = std::move(v); return kv.second; }
    obj_.emplace_back(k, std::move(v));
    return obj_.back().second;
  }
  const std::vector<std::pair<std::string, Json>>& entries() const { expect(Object); return obj_; }

  double num_or(const std::string& k, double d) const { const Json& v = get(k); return v.is_number() ? v.n_ : d; }
  bool bool_or(const std::string& k, bool d) const { const Json& v = get(k); return v.type_ == Bool ? v.b_ : d; }
  std::string str_or(const std::string& k, const std::string& d) const { const Json& v = get(k); return v.is_string() ? v.s_ : d; }

  bool operator==(const Json& o) const {
    if (type_ != o.type_) return false;
    switch (type_) {
      case Null: return true;
      case Bool: return b_ == o.b_;
      case Number: return n_ == o.n_;
      case String: return s_ == o.s_;
      case Array: return arr_ == o.arr_;
      case Object: {
        if (obj_.size() != o.obj_.size()) return false;
        for (auto& kv : obj_) if (!o.has(kv.first) || !(o[kv.first] == kv.second)) return false;
        return true;
      }
    }
    return false;
  }
  bool operator!=(const Json& o) const { return !(*this == o); }

  static Json parse(const std::string& text) {
    size_t p = 0;
    Json v = parse_value(text, p);
    skip_ws(text, p);
    if (p != text.size()) throw std::runtime_error("json: trailing characters");
    return v;
  }

  std::string dump() const { std::string out; dump_to(out); return out; }

 private:
  Type type_ = Null;
  bool b_ = false;
  double n_ = 0;
  std::string s_;
  std::vector<Json> arr_;
  std::vector<std::pair<std::string, Json>> obj_;

  void expect(Type t) const {
    if (type_ != t) throw std::runtime_error("json: unexpected value type");
  }

  static void skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p;
  }

  static void put_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  static std::string parse_string(const std::string& s, size_t& p) {
    if (s[p] != '"') throw std::runtime_error("json: expected string");
    ++p;
    std::string out;
    while (p < s.size() && s[p] != '"') {
      char c = s[p++];
      if (c != '\\') { out += c; continue; }
      if (p >= s.size()) break;
      char e = s[p++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': {
          uint32_t cp = std::stoul(s.substr(p, 4), nullptr, 16);
          p += 4;
          if (cp >= 0xD800 && cp < 0xDC00 && p + 6 <= s.size() && s[p] == '\\' && s[p + 1] == 'u') {
            uint32_t lo = std::stoul(s.substr(p + 2, 4), nullptr, 16);
            p += 6;
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          put_utf8(out, cp);
          break;
        }
        default: out += e;
      }
    }
    if (p >= s.size()) throw std::runtime_error("json: unterminated string");
    ++p;
    return out;
  }

  static Json parse_value(const std::string& s, size_t& p) {
    skip_ws(s, p);
    if (p >= s.size()) throw std::runtime_error("json: unexpected end");
    char c = s[p];
    if (c == '{') {
      Json o = object();
      ++p; skip_ws(s, p);
      if (s[p] == '}') { ++p; return o; }
      for (;;) {
        skip_ws(s, p);
        std::string k = parse_string(s, p);
        skip_ws(s, p);
        if (s[p++] != ':') throw std::runtime_error("json: expected ':'");
        o.obj_.emplace_back(std::move(k), parse_value(s, p));
        skip_ws(s, p);
        if (s[p] == ',') { ++p; continue; }
        if (s[p] == '}') { ++p; return o; }
        throw std::runtime_error("json: expected ',' or '}'");
      }
    }
    if (c == '[') {
      Json a = array();
      ++p; skip_ws(s, p);
      if (s[p] == ']') { ++p; return a; }
      for (;;) {
        a.arr_.push_back(parse_value(s, p));
        skip_ws(s, p);
        if (s[p] == ',') { ++p; continue; }
        if (s[p] == ']') { ++p; return a; }
        throw std::runtime_error("json: expected ',' or ']'");
      }
    }
    if (c == '"') return Json(parse_string(s, p));
    if (s.compare(p, 4, "true") == 0) { p += 4; return Json(true); }
    if (s.compare(p, 5, "false") == 0) { p += 5; return Json(false); }
    if (s.compare(p, 4, "null") == 0) { p += 4; return Json(); }
    size_t q = p;
    while (q < s.size() && (isdigit(static_cast<unsigned char>(s[q])) || s[q] == '-' || s[q] == '+' || s[q] == '.' || s[q] == 'e' || s[q] == 'E')) ++q;
    if (q == p) throw std::runtime_error("json: bad token");
    double v = std::strtod(s.c_str() + p, nullptr);
    p = q;
    return Json(v);
  }

  void dump_to(std::string& out) const {
    switch (type_) {
      case Null: out += "null"; break;
      case Bool: out += b_ ? "true" : "false"; break;
      case Number: {
        if (std::isfinite(n_) && n_ == std::floor(n_) && std::fabs(n_) < 1e15) {
          out += std::to_string(static_cast<int64_t>(n_));
        } else if (std::isfinite(n_)) {
          char buf[32];
          snprintf(buf, sizeof buf, "%.9g", n_);
          out += buf;
        } else {
          out += "null";
        }
        break;
      }
      case String: {
        out += '"';
        for (char c : s_) {
          switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
              if (static_cast<unsigned char>(c) < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
              else out += c;
          }
        }
        out += '"';
        break;
      }
      case Array: {
        out += '[';
        for (size_t i = 0; i < arr_.size(); ++i) { if (i) out += ','; arr_[i].dump_to(out); }
        out += ']';
        break;
      }
      case Object: {
        out += '{';
        for (size_t i = 0; i < obj_.size(); ++i) {
          if (i) out += ',';
          Json(obj_[i].first).dump_to(out);
          out += ':';
          obj_[i].second.dump_to(out);
        }
        out += '}';
        break;
      }
    }
  }
};

}  // namespace rvc
