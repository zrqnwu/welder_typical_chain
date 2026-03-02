static int64_t evalI64Token(
    const std::string &tok, const ConstTables &tbl,
    const std::unordered_map<std::string, int64_t> &overrides) {
  std::string t = trim(tok);
  if (t.empty())
    return 0;
  if (t[0] == '%') {
    auto it = tbl.i64.find(t);
    if (it != tbl.i64.end())
      return it->second;
    if (auto it2 = overrides.find(t); it2 != overrides.end())
      return it2->second;
    llvm::errs() << "error: missing i64 value for token: " << t << "\n";
    llvm::errs() << "hint: pass overrides like: --i64 " << t << "=0\n";
    std::exit(2);
  }
  return std::stoll(t);
}

static float evalF32Token(const std::string &tok, const ConstTables &tbl) {
  std::string t = trim(tok);
  if (t.empty())
    return 0.0f;
  if (t[0] == '%') {
    auto it = tbl.f32.find(t);
    if (it == tbl.f32.end()) {
      // 某些 nvvm runner 模块会传入 llvm.mlir.poison；按 0 处理。
      return 0.0f;
    }
    return it->second;
  }
  return std::stof(t);
}

static float evalF16Token(const std::string &tok, const ConstTables &tbl) {
  std::string t = trim(tok);
  if (t.empty())
    return 0.0f;
  if (t[0] == '%') {
    auto it = tbl.f16.find(t);
    if (it == tbl.f16.end()) {
      // 某些 nvvm runner 模块会传入 llvm.mlir.poison；按 0 处理。
      return 0.0f;
    }
    return it->second;
  }
  return std::stof(t);
}

static uint16_t f32ToF16Bits(float x) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(x), "f32 must be 32-bit");
  std::memcpy(&bits, &x, sizeof(bits));

  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff);
  uint32_t mant = bits & 0x7fffff;

  uint16_t outSign = static_cast<uint16_t>(sign << 15);

  // NaN/Inf 处理
  if (exp == 0xff) {
    if (mant == 0)
      return static_cast<uint16_t>(outSign | 0x7c00); // 无穷大（Inf）
    return static_cast<uint16_t>(outSign | 0x7e00);   // 静默 NaN（qNaN）
  }

  // 指数偏置调整。
  int32_t halfExp = exp - 127 + 15;
  if (halfExp >= 31) {
    // 上溢 -> Inf。
    return static_cast<uint16_t>(outSign | 0x7c00);
  }

  if (halfExp <= 0) {
    // 非规格化/下溢。
    if (halfExp < -10)
      return outSign; // 过小 -> 0

    // 补上隐式前导 1。
    mant |= 0x800000;
    int32_t shift = 14 - halfExp; // 24-10 = 14
    uint32_t mantRounded = mant + (1u << (shift - 1));
    uint16_t outMant = static_cast<uint16_t>(mantRounded >> shift);
    return static_cast<uint16_t>(outSign | outMant);
  }

  // 规格化数。
  uint32_t mantRounded = mant + 0x1000; // 四舍五入到最近值（近似 ties-to-even）
  if (mantRounded & 0x800000) {
    // 尾数上溢 -> 指数加 1。
    mantRounded = 0;
    ++halfExp;
    if (halfExp >= 31)
      return static_cast<uint16_t>(outSign | 0x7c00);
  }
  uint16_t outExp = static_cast<uint16_t>(halfExp << 10);
  uint16_t outMant = static_cast<uint16_t>((mantRounded >> 13) & 0x3ff);
  return static_cast<uint16_t>(outSign | outExp | outMant);
}
