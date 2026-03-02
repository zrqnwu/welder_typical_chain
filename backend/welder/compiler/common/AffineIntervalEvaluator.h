#pragma once

#include "WelderSolverLib.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <vector>

namespace welder {

// 一个极简的 AffineExpr 区间求值器：
// - 输入：AffineExpr + 每个 dim/symbol 的取值区间 [min,max]
// - 输出：该表达式的区间上界 [min,max]（闭区间）
//
// 目标：用于 Welder 的 Footprint Inference（从输出 tile 推导输入 footprint）。
//
// 当前版本（第一步）刻意只支持最常见的仿射形式：
// - Add（加法）
// - Mul（乘法，其中一侧必须是常数）
// - Constant / DimId / SymbolId（常量 / 维度ID / 符号ID）
//
// 同时支持：
// - FloorDiv/CeilDiv/Mod（仅当除数为常数；必要时做保守过近似）
//
// 对于更复杂情况（非常量除数 / 非线性组合），返回 std::nullopt，上层可以选择
// 保守回退（整维全取）。
std::optional<Interval>
evalAffineExprInterval(mlir::AffineExpr expr, llvm::ArrayRef<Interval> dims,
                       llvm::ArrayRef<Interval> symbols);

// 对一个 AffineMap 的每个结果分别求区间。
std::optional<std::vector<Interval>>
evalAffineMapIntervals(mlir::AffineMap map, llvm::ArrayRef<Interval> dims,
                       llvm::ArrayRef<Interval> symbols);

} // 命名空间 welder
