#pragma once

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

namespace welder {

struct TraceConfig {
  // 向 stderr 输出可读的 trace。
  bool text = false;
  // 若为 true，在文本输出中包含按候选/按 pass 的详细事件。
  bool verbose = false;
  // JSONL trace（每行一个 JSON 对象），写入 `jsonlPath`。
  bool jsonl = false;
  std::string jsonlPath;
  bool jsonlAppend = false;
};

class Tracer {
public:
  explicit Tracer(TraceConfig cfg) : cfg_(std::move(cfg)) {
    start_ = std::chrono::steady_clock::now();
    if (cfg_.jsonl && !cfg_.jsonlPath.empty()) {
      std::ios::openmode mode = std::ios::out;
      if (cfg_.jsonlAppend)
        mode |= std::ios::app;
      else
        mode |= std::ios::trunc;
      json_.emplace(cfg_.jsonlPath, mode);
      if (!*json_) {
      // 尽力而为：若启用则继续向 stderr 记录，但不因此失败。
        cfg_.jsonl = false;
        if (cfg_.text) {
          llvm::errs() << "[trace] warning: cannot open trace file: "
                       << cfg_.jsonlPath << "\n";
          llvm::errs().flush();
        }
      }
    }
  }

  Tracer(const Tracer &) = delete;
  Tracer &operator=(const Tracer &) = delete;

  bool enabled() const { return cfg_.text || cfg_.jsonl; }
  bool textEnabled() const { return cfg_.text; }
  bool jsonEnabled() const { return cfg_.jsonl && json_.has_value(); }
  bool verbose() const { return cfg_.verbose; }

  double nowMs() const {
    auto now = std::chrono::steady_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            now - start_);
    return dur.count();
  }

  static std::string formatMs(double ms) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", ms);
    return std::string(buf);
  }

  void log(llvm::StringRef msg, bool isVerbose = false) {
    if (!enabled())
      return;
    if (cfg_.text && (!isVerbose || cfg_.verbose)) {
      std::lock_guard<std::mutex> lock(mu_);
      double t = nowMs();
      llvm::errs() << "[+" << formatMs(t) << "ms] " << msg << "\n";
      llvm::errs().flush();
    }
    if (jsonEnabled()) {
      llvm::json::Object obj;
      obj["t_ms"] = nowMs();
      obj["event"] = "log";
      obj["msg"] = msg.str();
      writeJsonLocked(std::move(obj));
    }
  }

  void event(llvm::StringRef name, llvm::json::Object fields = {},
             bool isVerbose = false) {
    if (!enabled())
      return;

    if (cfg_.text && (!isVerbose || cfg_.verbose)) {
      std::string jsonFields;
      {
        llvm::raw_string_ostream os(jsonFields);
        llvm::json::Value(llvm::json::Object(fields)).print(os);
      }
      std::lock_guard<std::mutex> lock(mu_);
      double t = nowMs();
      llvm::errs() << "[+" << formatMs(t) << "ms] " << name;
      if (!fields.empty())
        llvm::errs() << " " << jsonFields;
      llvm::errs() << "\n";
      llvm::errs().flush();
    }

    if (jsonEnabled()) {
      llvm::json::Object obj;
      obj["t_ms"] = nowMs();
      obj["event"] = name.str();
    // 包含进程内稳定的线程标识符。
      {
        std::string tid;
        {
          std::ostringstream ss;
          ss << std::this_thread::get_id();
          tid = ss.str();
        }
        obj["tid"] = tid;
      }
      for (auto &kv : fields) {
        obj[kv.first] = std::move(kv.second);
      }
      writeJsonLocked(std::move(obj));
    }
  }

  class Span {
  public:
    Span() = default;

    Span(Tracer *tracer, std::string name, llvm::json::Object fields,
         bool isVerbose)
        : tracer_(tracer), name_(std::move(name)),
          fields_(std::move(fields)), isVerbose_(isVerbose),
          start_(std::chrono::steady_clock::now()) {
      if (tracer_)
        tracer_->event(name_ + ".start", fields_, isVerbose_);
    }

    Span(const Span &) = delete;
    Span &operator=(const Span &) = delete;

    Span(Span &&other) noexcept { *this = std::move(other); }
    Span &operator=(Span &&other) noexcept {
      if (this == &other)
        return *this;
    // 覆盖前先结束该 span。
      end();
      tracer_ = std::exchange(other.tracer_, nullptr);
      name_ = std::move(other.name_);
      fields_ = std::move(other.fields_);
      isVerbose_ = other.isVerbose_;
      start_ = other.start_;
      return *this;
    }

    ~Span() { end(); }

    void end() {
      if (!tracer_)
        return;
      auto endT = std::chrono::steady_clock::now();
      auto dur =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
              endT - start_);
      fields_["dur_ms"] = dur.count();
      tracer_->event(name_ + ".end", std::move(fields_), isVerbose_);
      tracer_ = nullptr;
    }

  private:
    Tracer *tracer_ = nullptr;
    std::string name_;
    llvm::json::Object fields_;
    bool isVerbose_ = false;
    std::chrono::steady_clock::time_point start_{};
  };

  Span span(llvm::StringRef name, llvm::json::Object fields = {},
            bool isVerbose = false) {
    if (!enabled())
      return Span();
    return Span(this, name.str(), std::move(fields), isVerbose);
  }

private:
  void writeJsonLocked(llvm::json::Object obj) {
    if (!jsonEnabled())
      return;
    std::lock_guard<std::mutex> lock(mu_);
    writeJsonLockedNoLock(llvm::json::Value(std::move(obj)));
  }

  void writeJsonLockedNoLock(const llvm::json::Value &v) {
    if (!jsonEnabled())
      return;
    std::string s;
    {
      llvm::raw_string_ostream os(s);
      v.print(os);
    }
    (*json_) << s << "\n";
    json_->flush();
  }

  std::chrono::steady_clock::time_point start_;
  TraceConfig cfg_;
  mutable std::mutex mu_;
  std::optional<std::ofstream> json_;
};

} // 命名空间 welder
