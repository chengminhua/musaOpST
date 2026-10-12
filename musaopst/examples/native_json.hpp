// SPDX-License-Identifier: MulanPSL-2.0
// Minimal original JSON value/parser for the explicit musaopst host protocol.
#pragma once
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace opst {
struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    std::string scalar;
    std::vector<Json> items;
    std::map<std::string, Json> members;
    Json() = default;
    Json(const char* s): kind(String), scalar(s) {}
    Json(const std::string& s): kind(String), scalar(s) {}
    Json(bool value): kind(Bool), scalar(value ? "true" : "false") {}
    Json(int value): kind(Number), scalar(std::to_string(value)) {}
    Json(int64_t value): kind(Number), scalar(std::to_string(value)) {}
    Json(size_t value): kind(Number), scalar(std::to_string(value)) {}
    Json(double value): kind(Number) {
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite JSON number");
        std::ostringstream out; out << std::setprecision(17) << value; scalar = out.str();
    }
    static Json array() { Json x; x.kind = Array; return x; }
    static Json object() { Json x; x.kind = Object; return x; }
    bool has(const std::string& key) const { return kind == Object && members.count(key); }
    const Json& at(const std::string& key) const {
        auto found = members.find(key);
        if (kind != Object || found == members.end()) throw std::runtime_error("Missing JSON key " + key);
        return found->second;
    }
    Json& operator[](const std::string& key) { if (kind == Null) kind = Object; if (kind != Object) throw std::runtime_error("Expected object"); return members[key]; }
    std::string str() const { if (kind != String) throw std::runtime_error("Expected JSON string"); return scalar; }
    int64_t integer() const {
        if (kind != Number || scalar.find_first_of(".eE") != std::string::npos) throw std::runtime_error("Expected integer");
        size_t end; auto result = std::stoll(scalar, &end); if (end != scalar.size()) throw std::runtime_error("Invalid integer"); return result;
    }
    double number() const { if (kind != Number) throw std::runtime_error("Expected number"); return std::stod(scalar); }
    bool boolean() const { if (kind != Bool) throw std::runtime_error("Expected bool"); return scalar == "true"; }
    std::string text(const std::string& key, const std::string& fallback) const { return has(key) ? at(key).str() : fallback; }
    int64_t integer(const std::string& key, int64_t fallback) const { return has(key) ? at(key).integer() : fallback; }
    bool boolean(const std::string& key, bool fallback) const { return has(key) ? at(key).boolean() : fallback; }
    static std::string quote(const std::string& text) {
        std::ostringstream out; out << '"';
        for (unsigned char c : text) {
            if (c == '"' || c == '\\') out << '\\' << c;
            else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
            else out << c;
        }
        out << '"'; return out.str();
    }
    std::string dump() const {
        if (kind == Null) return "null";
        if (kind == Bool || kind == Number) return scalar;
        if (kind == String) return quote(scalar);
        std::string result = kind == Array ? "[" : "{";
        bool first = true;
        if (kind == Array) for (const auto& item : items) { if (!first) result += ','; first = false; result += item.dump(); }
        else for (const auto& pair : members) { if (!first) result += ','; first = false; result += quote(pair.first) + ':' + pair.second.dump(); }
        return result + (kind == Array ? "]" : "}");
    }
};
struct JsonParser {
    const std::string& source; size_t position = 0;
    explicit JsonParser(const std::string& s): source(s) {}
    void ws() { while (position < source.size() && std::isspace(static_cast<unsigned char>(source[position]))) ++position; }
    bool take(char c) { ws(); if (position < source.size() && source[position] == c) { ++position; return true; } return false; }
    [[noreturn]] void fail() { throw std::runtime_error("Malformed JSON at byte " + std::to_string(position)); }
    unsigned hex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) { if (position >= source.size()) fail(); char c = source[position++]; value <<= 4;
            if (c >= '0' && c <= '9') value += c-'0'; else if (c >= 'a' && c <= 'f') value += c-'a'+10; else if (c >= 'A' && c <= 'F') value += c-'A'+10; else fail(); }
        return value;
    }
    static void utf8(std::string& out, unsigned c) {
        if (c < 0x80) out += char(c);
        else if (c < 0x800) { out += char(0xC0 | c>>6); out += char(0x80 | (c&63)); }
        else if (c < 0x10000) { out += char(0xE0 | c>>12); out += char(0x80 | ((c>>6)&63)); out += char(0x80 | (c&63)); }
        else { out += char(0xF0 | c>>18); out += char(0x80 | ((c>>12)&63)); out += char(0x80 | ((c>>6)&63)); out += char(0x80 | (c&63)); }
    }
    std::string string() {
        if (!take('"')) fail(); std::string out;
        while (position < source.size()) {
            unsigned char c = source[position++]; if (c == '"') return out; if (c < 32) fail();
            if (c != '\\') { out += char(c); continue; }
            if (position >= source.size()) fail(); c = source[position++];
            switch (c) {
                case '"': case '\\': case '/': out += char(c); break;
                case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
                case 'r': out += '\r'; break; case 't': out += '\t'; break;
                case 'u': { unsigned code = hex4(); if (code >= 0xD800 && code <= 0xDBFF) {
                    if (position+2 > source.size() || source[position++] != '\\' || source[position++] != 'u') fail();
                    unsigned low = hex4(); if (low < 0xDC00 || low > 0xDFFF) fail(); code = 0x10000 + ((code-0xD800)<<10) + low-0xDC00;
                } else if (code >= 0xDC00 && code <= 0xDFFF) fail(); utf8(out, code); break; }
                default: fail();
            }
        }
        fail();
    }
    Json value(unsigned depth = 0) {
        if (depth > 64) throw std::runtime_error("JSON nesting limit exceeded"); ws(); if (position >= source.size()) fail();
        if (source[position] == '"') return Json(string());
        if (take('{')) { Json out = Json::object(); if (take('}')) return out; do { auto key = string(); if (!take(':') || out.has(key)) fail(); out[key] = value(depth+1); } while (take(',')); if (!take('}')) fail(); return out; }
        if (take('[')) { Json out = Json::array(); if (take(']')) return out; do { out.items.push_back(value(depth+1)); } while (take(',')); if (!take(']')) fail(); return out; }
        for (const auto& token : {std::string("null"), std::string("true"), std::string("false")}) {
            if (source.compare(position, token.size(), token) == 0) { position += token.size(); return token == "null" ? Json() : Json(token == "true"); }
        }
        size_t start = position; if (source[position] == '-') ++position;
        if (position >= source.size() || !std::isdigit(source[position])) fail();
        if (source[position] == '0') ++position; else while (position < source.size() && std::isdigit(source[position])) ++position;
        if (position < source.size() && source[position] == '.') { ++position; if (position >= source.size() || !std::isdigit(source[position])) fail(); while (position < source.size() && std::isdigit(source[position])) ++position; }
        if (position < source.size() && (source[position] == 'e' || source[position] == 'E')) { ++position; if (position < source.size() && (source[position] == '+' || source[position] == '-')) ++position; if (position >= source.size() || !std::isdigit(source[position])) fail(); while (position < source.size() && std::isdigit(source[position])) ++position; }
        Json out; out.kind = Json::Number; out.scalar = source.substr(start, position-start);
        if (!std::isfinite(out.number())) fail(); return out;
    }
    Json parse() { Json result = value(); ws(); if (position != source.size()) fail(); return result; }
};
inline Json parse_json(const std::string& source) { return JsonParser(source).parse(); }
inline Json integer_array(const std::vector<int64_t>& values) { Json out = Json::array(); for (auto v: values) out.items.emplace_back(v); return out; }
} // namespace opst
