#include "AffineIntervalEvaluator.h"

using namespace mlir;

namespace welder {

static std::optional<int64_t> getConstant(AffineExpr expr) {
  if (auto c = dyn_cast<AffineConstantExpr>(expr))
    return c.getValue();
  return std::nullopt;
}

static Interval scaleInterval(const Interval &in, int64_t k) {
  if (!in.isValid())
    return in;
  // 注意：k 允许为负数，需要翻转 min/max。
  int64_t a = in.min * k;
  int64_t b = in.max * k;
  Interval out;
  out.min = std::min(a, b);
  out.max = std::max(a, b);
  return out;
}

static int64_t floorDivSigned(int64_t a, int64_t b) {
  // 整数下取整除法（向 -inf 方向），与 MLIR affine 语义一致
  // （针对正除数）。
  // 前置条件：b != 0。
  if (b < 0) {
    a = -a;
    b = -b;
  }
  if (a >= 0)
    return a / b;
  // 当 a < 0 且 b > 0。
  // 公式：floor(a/b) = -ceil((-a)/b)
  return -(((-a) + b - 1) / b);
}

static int64_t ceilDivSigned(int64_t a, int64_t b) {
  // 整数上取整除法（向 +inf 方向）。前置条件：b != 0。
  // 公式：ceil(a/b) = -floor((-a)/b)
  return -floorDivSigned(-a, b);
}

static int64_t modFloorSigned(int64_t a, int64_t b) {
  // 正除数 b 的 floormod：结果位于 [0, b-1]。
  // 前置条件：b > 0。
  int64_t q = floorDivSigned(a, b);
  return a - q * b;
}

std::optional<Interval>
evalAffineExprInterval(AffineExpr expr, llvm::ArrayRef<Interval> dims,
                       llvm::ArrayRef<Interval> symbols) {
  if (!expr)
    return std::nullopt;

  switch (expr.getKind()) {
  case AffineExprKind::Constant: {
    auto c = cast<AffineConstantExpr>(expr).getValue();
    return Interval{c, c};
  }
  case AffineExprKind::DimId: {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    if (pos >= dims.size())
      return std::nullopt;
    return dims[pos];
  }
  case AffineExprKind::SymbolId: {
    unsigned pos = cast<AffineSymbolExpr>(expr).getPosition();
    if (pos >= symbols.size())
      return std::nullopt;
    return symbols[pos];
  }
  case AffineExprKind::Add: {
    auto bin = cast<AffineBinaryOpExpr>(expr);
    auto lhs = evalAffineExprInterval(bin.getLHS(), dims, symbols);
    auto rhs = evalAffineExprInterval(bin.getRHS(), dims, symbols);
    if (!lhs || !rhs)
      return std::nullopt;
    return Interval{lhs->min + rhs->min, lhs->max + rhs->max};
  }
  case AffineExprKind::Mul: {
    // 约定：AffineExpr 的 Mul 其中一侧应该是常数或符号（但这里我们先只支持常数）。
    auto bin = cast<AffineBinaryOpExpr>(expr);
    AffineExpr a = bin.getLHS();
    AffineExpr b = bin.getRHS();

    if (auto kc = getConstant(a)) {
      auto rhs = evalAffineExprInterval(b, dims, symbols);
      if (!rhs)
        return std::nullopt;
      return scaleInterval(*rhs, *kc);
    }
    if (auto kc = getConstant(b)) {
      auto lhs = evalAffineExprInterval(a, dims, symbols);
      if (!lhs)
        return std::nullopt;
      return scaleInterval(*lhs, *kc);
    }
    return std::nullopt;
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
  // 仅支持常量除数（论文对齐：允许保守过近似）。
    auto bin = cast<AffineBinaryOpExpr>(expr);
    auto lhs = evalAffineExprInterval(bin.getLHS(), dims, symbols);
    if (!lhs)
      return std::nullopt;
    auto divisorOpt = getConstant(bin.getRHS());
    if (!divisorOpt || *divisorOpt == 0)
      return std::nullopt;
    int64_t d = *divisorOpt;
    Interval out;
    if (expr.getKind() == AffineExprKind::FloorDiv) {
      out.min = floorDivSigned(lhs->min, d);
      out.max = floorDivSigned(lhs->max, d);
    } else {
      out.min = ceilDivSigned(lhs->min, d);
      out.max = ceilDivSigned(lhs->max, d);
    }
    if (out.min > out.max)
      std::swap(out.min, out.max);
    return out;
  }
  case AffineExprKind::Mod: {
    auto bin = cast<AffineBinaryOpExpr>(expr);
    auto lhs = evalAffineExprInterval(bin.getLHS(), dims, symbols);
    if (!lhs)
      return std::nullopt;
    auto divisorOpt = getConstant(bin.getRHS());
    if (!divisorOpt || *divisorOpt <= 0)
      return std::nullopt;
    int64_t d = *divisorOpt;

  // 保守过近似：若区间跨越多个商桶，
  // 余数可取 [0, d-1] 中任意值。
    int64_t qMin = floorDivSigned(lhs->min, d);
    int64_t qMax = floorDivSigned(lhs->max, d);
    if (qMin != qMax || (lhs->max - lhs->min) >= d) {
      return Interval{0, d - 1};
    }

  // 商相同时，余数随 a 单调变化。
    int64_t rMin = modFloorSigned(lhs->min, d);
    int64_t rMax = modFloorSigned(lhs->max, d);
    if (rMin > rMax)
      std::swap(rMin, rMax);
    return Interval{rMin, rMax};
  }
  }
  return std::nullopt;
}

std::optional<std::vector<Interval>>
evalAffineMapIntervals(AffineMap map, llvm::ArrayRef<Interval> dims,
                       llvm::ArrayRef<Interval> symbols) {
  if (!map)
    return std::nullopt;
  if (map.getNumDims() != dims.size() || map.getNumSymbols() != symbols.size())
    return std::nullopt;

  std::vector<Interval> out;
  out.reserve(map.getNumResults());
  for (AffineExpr e : map.getResults()) {
    auto interval = evalAffineExprInterval(e, dims, symbols);
    if (!interval)
      return std::nullopt;
    out.push_back(*interval);
  }
  return out;
}

} // 命名空间 welder
