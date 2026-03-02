static std::vector<MemrefDesc>
detectMemrefDescriptors(
    const LaunchSpec &launch, const ConstTables &consts,
    const std::unordered_map<std::string, int64_t> &i64OverrideMap,
    int64_t elemBytes) {
  std::vector<MemrefDesc> out;
  const auto &args = launch.args;
  size_t i = 0;
  while (i < args.size()) {
    // Rank-2 memref 描述符：ptr, ptr, i64, i64, i64, i64, i64。
    if (i + 6 < args.size()) {
      const LaunchArg &a0 = args[i + 0];
      const LaunchArg &a1 = args[i + 1];
      const LaunchArg &a2 = args[i + 2];
      const LaunchArg &a3 = args[i + 3];
      const LaunchArg &a4 = args[i + 4];
      const LaunchArg &a5 = args[i + 5];
      const LaunchArg &a6 = args[i + 6];
      if (a0.type == "!llvm.ptr" && a1.type == "!llvm.ptr" && a2.type == "i64" &&
          a3.type == "i64" && a4.type == "i64" && a5.type == "i64" &&
          a6.type == "i64") {
        int64_t offset = evalI64Token(a2.value, consts, i64OverrideMap);
        int64_t size0 = evalI64Token(a3.value, consts, i64OverrideMap);
        int64_t size1 = evalI64Token(a4.value, consts, i64OverrideMap);
        int64_t stride0 = evalI64Token(a5.value, consts, i64OverrideMap);
        int64_t stride1 = evalI64Token(a6.value, consts, i64OverrideMap);

        if (size0 > 0 && size1 > 0 && stride0 > 0 && stride1 > 0) {
          __int128 maxIdx = static_cast<__int128>(offset);
          maxIdx += static_cast<__int128>(size0 - 1) *
                    static_cast<__int128>(stride0);
          maxIdx += static_cast<__int128>(size1 - 1) *
                    static_cast<__int128>(stride1);
          maxIdx += 1;
          if (maxIdx > 0) {
            __int128 bytes128 = maxIdx * static_cast<__int128>(elemBytes);
            if (bytes128 > static_cast<__int128>(1ULL << 34)) {
              llvm::errs() << "error: inferred memref allocation is too large ("
                           << (long long)bytes128 << " bytes)\n";
              std::exit(2);
            }
            MemrefDesc d;
            d.rank = 2;
            d.baseSym = a0.value;
            d.alignedSym = a1.value;
            d.offset = offset;
            d.sizes = {size0, size1};
            d.strides = {stride0, stride1};
            d.bytes = static_cast<int64_t>(bytes128);
            out.push_back(std::move(d));
            i += 7;
            continue;
          }
        }
      }
    }

    // Rank-1 memref 描述符：ptr, ptr, i64, i64, i64。
    if (i + 4 < args.size()) {
      const LaunchArg &a0 = args[i + 0];
      const LaunchArg &a1 = args[i + 1];
      const LaunchArg &a2 = args[i + 2];
      const LaunchArg &a3 = args[i + 3];
      const LaunchArg &a4 = args[i + 4];
      if (a0.type == "!llvm.ptr" && a1.type == "!llvm.ptr" && a2.type == "i64" &&
          a3.type == "i64" && a4.type == "i64") {
        int64_t offset = evalI64Token(a2.value, consts, i64OverrideMap);
        int64_t size0 = evalI64Token(a3.value, consts, i64OverrideMap);
        int64_t stride0 = evalI64Token(a4.value, consts, i64OverrideMap);

        if (size0 > 0 && stride0 > 0) {
          __int128 maxIdx = static_cast<__int128>(offset);
          maxIdx += static_cast<__int128>(size0 - 1) *
                    static_cast<__int128>(stride0);
          maxIdx += 1;
          if (maxIdx > 0) {
            __int128 bytes128 = maxIdx * static_cast<__int128>(elemBytes);
            if (bytes128 > static_cast<__int128>(1ULL << 34)) {
              llvm::errs() << "error: inferred memref allocation is too large ("
                           << (long long)bytes128 << " bytes)\n";
              std::exit(2);
            }
            MemrefDesc d;
            d.rank = 1;
            d.baseSym = a0.value;
            d.alignedSym = a1.value;
            d.offset = offset;
            d.sizes = {size0, 0};
            d.strides = {stride0, 0};
            d.bytes = static_cast<int64_t>(bytes128);
            out.push_back(std::move(d));
            i += 5;
            continue;
          }
        }
      }
    }

    ++i;
  }
  return out;
}

static bool writeJsonMetaForDump(const std::string &metaPath,
                                 const MemrefDesc &d, int64_t elemBytes) {
  std::ofstream out(metaPath, std::ios::out);
  if (!out)
    return false;

  out << "{\n";
  out << "  \"sym\": \"" << d.baseSym << "\",\n";
  out << "  \"rank\": " << d.rank << ",\n";
  out << "  \"elem_bytes\": " << elemBytes << ",\n";
  out << "  \"offset\": " << d.offset << ",\n";
  out << "  \"sizes\": [" << d.sizes[0];
  if (d.rank >= 2)
    out << ", " << d.sizes[1];
  out << "],\n";
  out << "  \"strides\": [" << d.strides[0];
  if (d.rank >= 2)
    out << ", " << d.strides[1];
  out << "],\n";
  out << "  \"bytes\": " << d.bytes << "\n";
  out << "}\n";
  return true;
}

static void initMemrefF32(const MemrefDesc &d, float *buf,
                          const std::string &mode, uint64_t seed) {
  const int64_t n0 = d.sizes[0];
  const int64_t n1 = (d.rank >= 2) ? d.sizes[1] : 1;
  const int64_t s0 = d.strides[0];
  const int64_t s1 = (d.rank >= 2) ? d.strides[1] : 0;
  const int64_t off = d.offset;

  auto linearVal = [&](int64_t flat) -> float {
    int64_t t = flat % 1024;
    return static_cast<float>(t) * 0.001f - 0.5f;
  };

  uint32_t rng = static_cast<uint32_t>((seed == 0) ? 1 : seed);
  auto randVal = [&]() -> float {
    // xorshift32 随机数生成。
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    float u = static_cast<float>(rng) * (1.0f / 4294967296.0f);
    return u * 2.0f - 1.0f;
  };

  if (mode == "zero")
    return;

  for (int64_t i = 0; i < n0; ++i) {
    if (d.rank == 1) {
      int64_t idx = off + i * s0;
      float v = (mode == "random") ? randVal() : linearVal(i);
      buf[idx] = v;
      continue;
    }
    for (int64_t j = 0; j < n1; ++j) {
      int64_t idx = off + i * s0 + j * s1;
      int64_t flat = i * n1 + j;
      float v = (mode == "random") ? randVal() : linearVal(flat);
      buf[idx] = v;
    }
  }
}

static std::optional<int64_t> inferMemrefElementCount(const MemrefDesc &d) {
  if (d.rank <= 0)
    return std::nullopt;
  if (d.offset < 0)
    return std::nullopt;
  __int128 maxIdx = static_cast<__int128>(d.offset);
  if (d.sizes[0] <= 0 || d.strides[0] <= 0)
    return std::nullopt;
  maxIdx += static_cast<__int128>(d.sizes[0] - 1) *
            static_cast<__int128>(d.strides[0]);
  if (d.rank >= 2) {
    if (d.sizes[1] <= 0 || d.strides[1] <= 0)
      return std::nullopt;
    maxIdx += static_cast<__int128>(d.sizes[1] - 1) *
              static_cast<__int128>(d.strides[1]);
  }
  maxIdx += 1;
  if (maxIdx <= 0 ||
      maxIdx > static_cast<__int128>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  return static_cast<int64_t>(maxIdx);
}

static void initMemrefF16(const MemrefDesc &d, uint16_t *buf,
                          const std::string &mode, uint64_t seed) {
  const int64_t n0 = d.sizes[0];
  const int64_t n1 = (d.rank >= 2) ? d.sizes[1] : 1;
  const int64_t s0 = d.strides[0];
  const int64_t s1 = (d.rank >= 2) ? d.strides[1] : 0;
  const int64_t off = d.offset;

  auto linearVal = [&](int64_t flat) -> float {
    int64_t t = flat % 1024;
    return static_cast<float>(t) * 0.001f - 0.5f;
  };

  uint32_t rng = static_cast<uint32_t>((seed == 0) ? 1 : seed);
  auto randVal = [&]() -> float {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    float u = static_cast<float>(rng) * (1.0f / 4294967296.0f);
    return u * 2.0f - 1.0f;
  };

  if (mode == "zero")
    return;

  for (int64_t i = 0; i < n0; ++i) {
    if (d.rank == 1) {
      int64_t idx = off + i * s0;
      float v = (mode == "random") ? randVal() : linearVal(i);
      buf[idx] = f32ToF16Bits(v);
      continue;
    }
    for (int64_t j = 0; j < n1; ++j) {
      int64_t idx = off + i * s0 + j * s1;
      int64_t flat = i * n1 + j;
      float v = (mode == "random") ? randVal() : linearVal(flat);
      buf[idx] = f32ToF16Bits(v);
    }
  }
}
