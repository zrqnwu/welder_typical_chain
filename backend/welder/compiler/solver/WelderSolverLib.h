#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace welder {

class Tracer;

struct ProblemSize {
  int64_t m = -1;
  int64_t n = -1;
  int64_t k = -1;
};

// Welder 论文里的 loop-based abstraction（Phase 9 起步）：用 loop 列表描述问题，而不是
// “m/n/k” 三元组。
struct LoopDim {
  int64_t size = 1;
  mlir::utils::IteratorType type = mlir::utils::IteratorType::parallel;
};

struct GenericProblem {
  std::vector<LoopDim> loops;
  mlir::Operation *targetOp = nullptr; // 非拥有指针

  std::string getOpName() const {
    return targetOp ? targetOp->getName().getStringRef().str() : "None";
  }
};

struct ArchConfig {
  int64_t smemBytes = 48 * 1024;
  int64_t numSM = 80;
  int64_t maxBlocksPerSM = 4;
  int64_t elementBytes = 4; // f32
  // 论文/Welder 对齐（CUDA 风格）：warp 大小与 SM 分区启发式参数。
  int64_t warpSize = 32;
  int64_t smPartition = 4;
  // 论文/Welder 对齐：每个 SM 的 shared 使用预算（字节）。参考实现在 CUDA 上使用
  // 公式：`max_smem_usage = 2 * smem_cap`。
  // 若为 0/负数，solver 回退到 `2 * smemBytes`。
  int64_t maxSmemUsageBytes = 0;
  // 论文对齐的内存事务宽度（字节）。
  // Welder 参考实现在 CUDA 上使用 [store=32B, load=128B]。
  // 对非合并访问会计入额外事务，从而影响 MemTraffic。
  int64_t globalReadTransactionBytes = 128;
  int64_t globalWriteTransactionBytes = 32;
  // 向后兼容的单事务宽度（旧代码路径）；当 read/write <= 0 时，
  // 使用该值作为回退。
  int64_t globalTransactionBytes = 128;
  // 论文/Welder 对齐：粗略带宽（仅用于平分时的启发式决策）。
  double bandwidthStore = 750.0;
  double bandwidthLoad = 12080.0;
  // 简单 occupancy 启发式（默认近似 Ampere）。
  int64_t maxThreadsPerSM = 2048;
  // 论文对齐（较粗略）的寄存器文件容量模型（默认近似 Ampere）。
  // 注意：这些上限仅用于搜索剪枝；真实性能仍以 profiling 为准。
  int64_t maxRegistersPerSM = 65536;   // 每个 SM 的 32-bit 寄存器数（近似）。
  int64_t maxRegistersPerThread = 255; // 架构上限（近似）。
};

struct Traffic {
  double bytesA = 0;
  double bytesB = 0;
  double bytesC = 0;
  // Phase 13A: cut-edge 产生的额外 global traffic（中间 tensor 落地）。
  double bytesCut = 0;
  double totalBytes() const { return bytesA + bytesB + bytesC + bytesCut; }
};

// 论文对齐的代价模型拆解（可调试、公式优先）。
//
// Welder 参考实现会计算 MemTraffic/MemFootprint/Waves，并可选使用
// d.Profile(configs) 做测量。这里两套信息都保留：
// - `rawTrafficBytes`：由 footprint 体积直接得到的纯字节量。
// - `memTrafficBytes`：启用合并访问惩罚后，按内存事务计费得到的字节量
//   （参数：arch.globalTransactionBytes）。
// - `estimatedLatency`：solver 的公式估计值（非 profiling）。
// - `profiledMs`：可选的硬件实测结果。
struct CostBreakdown {
  Traffic rawTraffic;
  Traffic memTraffic;
  // 论文对齐：用于寄存器层调度决策（connectLevel>1）的
  // shared<->register 流量估计（字节）。
  //
  // 这是基于当前每个算子传播后 tile 的 operand footprint 推导出的启发式模型。
  double sharedToRegBytes = 0.0;
  int64_t waves = 1;
  int64_t blocksTotal = 1;
  int64_t blocksPerSM = 1;
  int64_t sharedFootprintBytes = 0;
  double underutilPenalty = 1.0;
  double bankConflictFactor = 1.0;
  double regPenalty = 1.0;
  double estimatedLatency = 0.0;
  std::optional<double> profiledMs;

  std::string toString() const;
};

struct Candidate {
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  // 寄存器层（每线程）tile 大小，作用于前两个并行维。
  // 若设置（>0），会影响推导出的 blockDim = (tileN/threadTileN, tileM/threadTileM)。
  int64_t threadTileM = 0;
  int64_t threadTileN = 0;
  // 通用 tile：每个 loop 的 tile extent（Phase 9）。
  // 为空表示仍使用旧的 MatMul 专用三元组（tileM/tileN/tileK）。
  std::vector<int64_t> loopTileExtents;
  // 传统含义：kernel 级别的 shared memory 占用（MatMul 专用模型）。
  int64_t smemBytes = 0;
  // 论文对齐（2-level skeleton）：tile-graph 在 shared 层的 footprint 估算（字节）。
  // 注意：这不是 MatMul 的 A/B promotion smemBytes；它用于 MemFootprint 过滤/调试。
  int64_t estFootprintBytes = 0;
  int64_t blocksM = 0;
  int64_t blocksN = 0;
  int64_t blocksTotal = 0;
  int64_t blocksPerSM = 0;
  int64_t numWave = 0;
  // 论文对齐调试/启发式：每线程寄存器压力估计。
  int64_t estRegsPerThread = 0;
  // 论文对齐调试/启发式：shared-memory bank conflict 因子估计（>=1.0）。
  // 在未启用 profiling 时可作为可选惩罚项。
  double estSharedBankConflict = 1.0;
  Traffic traffic;
  double score = 0;
  CostBreakdown cost;

  //===--------------------------------------------------------------------===//
  // 论文对齐的 codegen 旋钮（由 solver 搜索；由 compiler/profiler 消费）
  //===--------------------------------------------------------------------===//
  bool enableAsyncCopy = false;
  bool enableSoftwarePipelining = false;
  int64_t pipelineDepth = 2;
  bool pipelinePeelEpilogue = true;
  // 启用 software pipelining 时，生成非零 cp.async wait_group 计数
  // （in-flight 组数），而非保守的 wait_group 0。
  bool pipelineSetAsyncWaitGroups = false;
  bool asyncBypassL1 = true;
  int64_t workgroupMultiBufferDepth = 1;
  int64_t workgroupPadLastDim = 0;
  // 若为 true，仅对喂给 linalg.matmul 输入（A/B）的 workgroup buffer 做 padding。
  // 这更贴近参考 TCPolicy 的 stride_map 行为，并避免影响无关的 workgroup buffer
  // （例如逐元素临时张量）。
  bool workgroupPadLastDimMatmulOnly = false;
  // shared-memory 布局：最后一维做 XOR swizzle（0=禁用）。
  int64_t workgroupSwizzleXor = 0;
  // Block 光栅化（近似）：用 block_id.y 对 block_id.x 做 XOR 混排（0=禁用）。
  // 在 WorkgroupAllocToLaunch pass 中实现为：
  //   bx' = bx ^ (by & (N-1))   （要求常量且为 2 的幂的 grid_size_x）。
  int64_t blockRasterizeXor = 0;
  // 论文/Welder 对齐：可调 panel 宽度的 2D 光栅化（Row/Column）。
  // 对应参考实现的 Rasterization2DRow/Column（num_wave 较大时可提升 L2 局部性）。
  // - 0：禁用
  // - 1：行光栅化
  // - 2：列光栅化
  int blockRasterizeMode = 0;
  int blockRasterizePanelWidth = 0;
  // Block/thread 映射顺序（论文中每个 op 的 block 顺序可能不同）。
  // false：(M, N) -> (block<y>, block<x>) 且 (thread<y>, thread<x>) [默认]
  // true ：(M, N) -> (block<x>, block<y>) 且 (thread<x>, thread<y>)
  bool swapBlockDims = false;

  // Row-reduction 链融合/codegen 旋钮（通用图）：
  // - reuse-fusion：将 1D 归约中间值保留在融合后的 gpu.launch 内
  //   （正确的分阶段 barrier + shared 广播）。
  // - input-promotion：将 2D 输入 tile 一次性搬到 workgroup 内存，
  //   供多个归约/逐元素消费者复用（论文风格的流量降低）。
  // - input-promotion-vectorize：对协作式 global->shared 搬运做向量化。
  bool enableRowReductionChainReuseFusion = false;
  bool enableRowReductionInputPromotion = false;
  bool enableRowReductionInputPromotionVectorize = false;
  // Row-reduction kernel 形状：优先 warp 级归约（每行一个 warp）。
  bool enableRowReductionWarp = false;
  // Row-reduction 逐元素向量化（例如 2D tile 上的 exp/div）。
  bool enableRowReductionVectorize = false;
  // Row-reduction 逐元素向量宽度（0 = 自动）。
  int64_t rowReductionVectorWidth = 0;
  // Row-reduction 在 X 方向的线程数（0 = 自动）。
  int64_t rowReductionThreadsX = 0;
  // Row-reduction：允许清理冗余 barrier。
  bool enableRowReductionRelaxBarriers = false;
  // Row-reduction：在合并归约后跳过 barrier（不安全；仅用于 profiling）。
  bool enableRowReductionSkipCombineBarrier = false;
  // Row-reduction 输入 staging 向量宽度（0 = 自动）。
  int64_t rowReductionInputVectorWidth = 0;
  // Row-reduction 合并阶段向量化（作用于 combine reduce op）。
  bool enableRowReductionCombineVectorize = false;

  // Matmul->Softmax 融合规范化（通用图）：
  // 将 matmul 输出 tile 保留在 shared memory，并在后续 row-reduction 链
  // （max/exp/sum/div）中复用，避免过大的每线程本地缓冲和重复 producer 链。
  bool enableMatmulSoftmaxSharedReuseFusion = false;

  // matmul 的 TensorCore（TF32）路径（最小支持，见
  // 参考文件：mlir_pipeline/transform_k_tiled_tensorcore_tf32.mlir）。
  bool enableTensorCoreTf32 = false;
  // matmul 的 TensorCore（F16）路径：通过
  // 参考 pass：transform.nvgpu.rewrite_matmul_as_mma_sync。
  bool enableTensorCoreF16 = false;
  // 论文/Welder 对齐（TCPolicy）：为 TensorCore 选择 MMA 指令形状。
  // - Cutlass 风格（Ampere）：m16n8k16
  // - 非 cutlass 回退：m16n16k16、m32n8k16、m8n32k16
  //
  // 这些配置由 solver 选择，并在启用 TensorCore lowering 时由 MLIR compiler 消费。
  bool useCutlassMma = false;
  int64_t mmaM = 0;
  int64_t mmaN = 0;
  int64_t mmaK = 0;

  //===--------------------------------------------------------------------===//
  // 论文对齐的可行性诊断（约束驱动剪枝）
  //===--------------------------------------------------------------------===//
  // 0 = OK。非零表示候选（或请求的后端模式）被拒绝或降级的原因
  // （例如 TensorCore -> SIMT 回退）。
  //
  // 这里有意保持粗粒度，便于写入 TSV 日志并与参考实现对比 solver 行为。
  int32_t feasibilityCode = 0;
};

struct SolveOptions {
  // 可选：实时 trace（进度 + 计时）。非空时，solver 会输出阶段摘要，
  // 并在 verbose 模式下输出按候选/按 pass 的事件。
  // 输出由 Tracer 配置控制（stderr 文本和/或 JSONL）。
  Tracer *tracer = nullptr;

  ArchConfig arch;
  std::vector<int64_t> candidatesMN{32, 64, 128};
  std::vector<int64_t> candidatesK{8, 16, 32};
  // 论文对齐：根据问题规模/硬件自动生成候选 tile，而不是依赖上面的固定列表。
  bool autoCandidates = false;
  // 论文对齐：在搜索空间中纳入每线程 tile（寄存器层）。
  bool enableRegisterLevelSchedule = false;
  // 每线程 tile 候选大小（csv）。常见值：1,2,4,8。
  std::vector<int64_t> candidatesThreadMN{1, 2, 4, 8};
  bool requirePerfectTiling = true;
  bool assumeFusedRelu = true;
  // 实验开关：尝试用 indexing_maps + footprint inference 来计算 traffic（更通用）。
  // 默认关闭，先保留旧的 MatMul 专用硬编码 traffic 公式作为 baseline。
  bool enableFootprintInference = false;
  // 实验开关：构建 TileGraph，并做 consumer-driven 的 tile 反向传播（Welder 的 Tile
  // Propagation）。默认关闭，先作为调试/对照。
  bool enableTilePropagation = false;
  // 实验开关：Phase A（全融合假设）的全图 traffic 记账。
  // - 依赖 TileGraph + tile propagation，给每个 node 得到 requiredTile。
  // - 只统计“图外输入的 global read”和“最终 sink 的 global write”。
  // - 目标：在 matmul->relu 这种链条上，与旧的 MatMul 专用公式严格对齐（至少 total bytes 对齐）。
  bool enableGlobalTraffic = false;
  // 实验开关：Phase 13A（cut-edge）。
  // - 当 tile propagation 遇到冲突时，不再直接判失败，而是把冲突边标记为 cut（走 global memory）。
  // - 打分时会把 cut 的读写流量计入 bytesCut。
  bool enableCutEdges = false;

  // 论文对齐（2-level skeleton）：global <-> shared 的子图调度入口。
  //
  // 当前最小版本做两件事：
  // 1) 用 tile propagation + cut-edge 得到一套“全图一致的”（或被 cut 打断的）tile 约束；
  // 2) 估算 shared 层的 tile-graph footprint，并用 arch.smemBytes 做过滤（MemFootprint）。
  //
  // 后续会在这个开关下逐步补齐论文 Figure 7 的 GraphConnecting/SubGraphTiling 递归流程。
  bool enableTwoLevelSchedule = false;

  // Welder 论文 Figure 7：Two-step tile-graph scheduling（GraphConnecting + SubGraphTiling）。
  //
  // 目标：
  // - 完整复现论文中的“先连边再调度”流程：
  //   1) GraphConnecting：对每条 edge 试探不同 connect level，并用 SubGraphTiling 评估；
  //   2) SubGraphTiling：执行 EnumerateSubtiles -> Propagate -> MemFootprint/MemTraffic -> TopK，
  //      然后递归调度更高层（当前 repo 先从 global<->shared 开始，后续扩展到更多层）。
  //
  // 
  // - 由于当前工程还没有真实的 end-to-end profiling runner，这里用 traffic-based latency
  //   estimate 来替代论文中的 d.Profile(configs)。接口留好后可再接入真实 profiling。
  bool enablePaperSchedule = false;
  // 打印 solver 侧的代价/约束拆解信息（针对选中配置）。
  bool verboseCostModel = false;
  // 论文对齐（更贴近参考实现）：将 shared->register 视为显式的内层
  // SubGraphTiling 步骤。solver 会对每个 shared 层 tile 搜索寄存器层
  // thread tile（及 codegen 旋钮），并用最佳内层配置作为 GraphConnecting
  // 与最终排序的评分依据。
  bool paperRecursiveRegisterLevel = true;
  // 递归 SubGraphTiling 的内层阶段边界（minLevelExclusive）。
  // - <=0：自动（旧行为保持一次递归跳转；可由 paperRecursiveMaxStages 放宽）。
  // -  >0：显式覆盖内层阶段边界。
  // 该设置在保持当前 2-level 行为不变（maxConnectLevel=2 => 1）的同时，
  // 允许逐步扩展到更深的 connect level。
  int paperRecursiveInnerMinLevelExclusive = 0;
  // 为 maxConnectLevel>2 时的递归 SubGraphTiling 阶段深度设置上限。
  // - <=0：保持旧行为（自动模式下仅一个递归阶段）。
  // -  >0：最多允许这么多个递归阶段窗口。
  //         示例：maxConnectLevel=4、cap=2 => 使用窗口 [2->3, 3->4]。
  int paperRecursiveMaxStages = 0;
  // 更严格的论文对齐行为：
  // - 在 SubGraphTiling 中仅使用 MemTraffic（纯字节）作为排序键；
  // - 启用 profiling 时，编译/测量失败的配置直接丢弃，
  //   不再回退到启发式估计。
  bool paperStrict = false;
  // 论文对齐：选定 shared 层 tile 后，在 shared-memory footprint 约束下
  // 贪心放大 reduction tile（例如 K），类似原实现的 reduce-step 扩展。
  bool paperExpandReductionTile = false;
  bool pruneOnProfileFailure = false;
  // 论文对齐：对非合并 global memory 访问计入额外内存事务
  // （事务宽度来自 ArchConfig）。
  bool enableCoalescingPenalty = true;
  // SubGraphTiling 的 Top-K（论文中为 k，控制组合爆炸）。
  int64_t scheduleTopK = 8;
  // GraphConnecting 允许尝试的最高 connect level（0 表示完全不连接/落地到 global）。
  // 当前 repo 的 codegen 只区分 cut(0) vs fuse(>0)，因此默认 1（global<->shared）。
  int maxConnectLevel = 1;

  struct ProfileOptions {
    // Welder 论文 Figure 7 的 d.Profile(configs)：对候选 config 做真实硬件测量。
    //
    // 当前实现依赖：
    // - `compiler/run_welder_to_nvvm_isa.sh`：把输入 MLIR 编译到 `out.nvvm.runnable.mlir`
    // - `welder-profiler`：从该 MLIR 中抽取 PTX，并用 CUDA Event 计时单次 kernel launch。
    bool enable = false;
    // 这两个路径若为空，会走默认发现逻辑（由 front-end 填充；库内不做假设）。
    std::string compilerToNvvmScript;
    std::string profilerBin;
    int warmup = 10;
    int iters = 100;
    // 为 true 时，按出现顺序执行整个模块内所有 gpu.launch_func kernel，
    // 并测量端到端时间（welder-profiler --run-all-kernels）。
    // 这对 cut-edge 调度和多 kernel 融合 A/B（如 LayerNorm/Softmax 归约链）很重要。
    bool runAllKernels = false;
    // 可选：把 key->avg_ms 写到磁盘，避免重复 profile（简单文本格式）。
    std::string cachePath;
    bool verbose = false;
    // 论文对齐的编译加速：允许多个独立配置并行编译/测量（尽力而为）。
    // 默认值保持行为确定性并避免 GPU 争用。
    int maxParallelJobs = 1;
    // 论文对齐的鲁棒性：将超慢的编译/测量视为失败。
    // 当前在可用时通过 `timeout` 命令做尽力实现。
    int timeoutSec = 0;

    // Phase 14 旋钮（codegen/profiling）：
    bool enableAsyncCopy = false;
    bool enableSoftwarePipelining = false;
    int64_t pipelineDepth = 2;
    bool pipelinePeelEpilogue = true;
    bool pipelineSetAsyncWaitGroups = false;
    bool asyncBypassL1 = true;
    int64_t workgroupMultiBufferDepth = 1;
    // 论文/Welder TCPolicy 对齐：通过给 shared 布局增加偏移（元素数）缓解
    // bank conflict。参考实现中 tensorcore 调度通常为 8（见 policy/tc.py）。
    int64_t workgroupPadLastDim = 0;
    bool workgroupPadLastDimMatmulOnly = false;
    int64_t workgroupSwizzleXor = 0;
    bool swapBlockDims = false;
    bool enableTensorCoreF16 = false;
    int blockRasterizeMode = 0;
    int blockRasterizePanelWidth = 0;

    // Row-reduction 链融合/codegen 旋钮（通用图）。
    bool enableRowReductionChainReuseFusion = false;
    bool enableRowReductionInputPromotion = false;
    bool enableRowReductionInputPromotionVectorize = false;
    bool enableRowReductionWarp = false;
    bool enableRowReductionVectorize = false;
    int64_t rowReductionVectorWidth = 0;
    int64_t rowReductionThreadsX = 0;
    bool enableRowReductionRelaxBarriers = false;
    bool enableRowReductionSkipCombineBarrier = false;
    int64_t rowReductionInputVectorWidth = 0;
    bool enableRowReductionCombineVectorize = false;

    // Matmul->Softmax shared tile 复用融合（通用图）。
    bool enableMatmulSoftmaxSharedReuseFusion = false;
  };
  ProfileOptions profile;

  // 论文对齐的 codegen 旋钮搜索空间。启用后，solver 会按候选枚举这些旋钮，
  // 并可选对其进行 profiling。
  struct CodegenSearchSpace {
    bool enable = false;
    // shared-memory 布局填充（缓解 bank conflict）。
    // 包含 8 以匹配参考 TCPolicy 的 stride 偏移。
    std::vector<int64_t> workgroupPadLastDim{0, 1, 8};
    // 仅对 matmul 输入 workgroup buffer（A/B）应用 padding。
    std::vector<bool> workgroupPadLastDimMatmulOnly{false, true};
    // Async copy（cp.async）选项。
    std::vector<bool> enableAsyncCopy{false, true};
    std::vector<bool> asyncBypassL1{true, false};
    // Software pipelining 选项。
    std::vector<bool> enableSoftwarePipelining{false, true};
    std::vector<int64_t> pipelineDepth{2, 3, 4};
    std::vector<bool> pipelinePeelEpilogue{true};
    // 启用 pipelining 时，选择是否设置 wait_group in-flight 计数
    // （0 保持保守默认值）。
    std::vector<bool> pipelineSetAsyncWaitGroups{false, true};
    std::vector<int64_t> workgroupMultiBufferDepth{1, 2, 3, 4};
    std::vector<int64_t> workgroupSwizzleXor{0, 4, 8};
    // Block 光栅化选项（0=禁用）。
    std::vector<int64_t> blockRasterizeXor{0, 4, 8, 16};
    // 论文/Welder 对齐的光栅化模式（0=关，1=行，2=列）。
    std::vector<int> blockRasterizeMode{0, 1, 2};
    // 启用光栅化时尝试的 panel 宽度。
    std::vector<int> blockRasterizePanelWidth{1, 2, 4, 8, 16};
    // 备选的 block/thread 映射顺序。
    std::vector<bool> swapBlockDims{false, true};
    // TensorCore 选项（仅 matmul）。
    std::vector<bool> enableTensorCoreTf32{false, true};
    std::vector<bool> enableTensorCoreF16{false, true};

    // Row-reduction 链融合/codegen 旋钮（通用图）。
    std::vector<bool> enableRowReductionChainReuseFusion{false, true};
    std::vector<bool> enableRowReductionInputPromotion{false, true};
    std::vector<bool> enableRowReductionInputPromotionVectorize{false, true};
    std::vector<bool> enableRowReductionWarp{false, true};
    std::vector<bool> enableRowReductionVectorize{false, true};
    std::vector<int64_t> rowReductionVectorWidth{0, 2, 4, 8};
    std::vector<int64_t> rowReductionThreadsX{0, 16, 32};
    std::vector<bool> enableRowReductionRelaxBarriers{false, true};
    std::vector<bool> enableRowReductionSkipCombineBarrier{false, true};
    std::vector<int64_t> rowReductionInputVectorWidth{0, 2, 4, 8};
    std::vector<bool> enableRowReductionCombineVectorize{false, true};

    // Matmul->Softmax shared tile 复用融合（通用图）。
    std::vector<bool> enableMatmulSoftmaxSharedReuseFusion{false, true};
  };
  CodegenSearchSpace codegenSearch;
};

struct SolveResult {
  ProblemSize problem;
  bool detectedConsumerChain = false;
  std::vector<Candidate> sortedCandidates; // 按 score 升序
};

//===----------------------------------------------------------------------===//
// Phase 6 (Welder): Tile-graph + Footprint Inference 基础设施
//
// 目标：
// - 用通用方式（indexing maps / AffineMap）推导一个 op-tile 的输入依赖范围
//   （Welder 论文里称为 Footprint Inference / Tile Propagation 的底层算子）。
// - 目前这里只搭接口和数据结构；具体算法实现放到 *.cpp 里逐步补齐。
// - 注意：先不要删除现有 MatMul 专用的硬编码逻辑；新逻辑会先作为可选路径/对照。
//===----------------------------------------------------------------------===//

// 一个整数区间 [min, max]（闭区间）。
struct Interval {
  int64_t min = 0;
  int64_t max = -1;

  bool isValid() const { return min <= max; }
  int64_t size() const { return isValid() ? (max - min + 1) : 0; }
};

// 对一个 tile 的描述：当前阶段先把 tile 定义在 “op 的 loop 迭代空间” 上。
// 对 linalg 来说：loop 维度 = iterator_types 的维度（parallel + reduction）。
struct OpTile {
  // 每个 loop 的 tile extent（长度 = op.getNumLoops()）。
  // 约定：extent > 0；如果你想表达“不切该维”，请在实现里把它展开成 full size。
  std::vector<int64_t> loopExtents;

  // reduction 维的 rstep（可选）。最小实现可以先不使用；后续做 K 分块时会用到。
  // 约定：如果为空，表示 reduction 维默认取 full range。
  std::vector<int64_t> reductionSteps;
};

// 对一个 operand 的 footprint：每个维度的索引区间 + 推导出来的 footprint shape。
struct OperandFootprint {
  std::vector<Interval> indexBounds;  // 每个 operand 维度
  std::vector<int64_t> shape;         // 每个 operand 维度（== bounds.size）
};

// 一次 footprint 推导的结果：对 op 的每个 operand（inputs + outputs）给出 footprint。
struct FootprintResult {
  std::vector<OperandFootprint> perOperand;
};

// Footprint 推导接口：输入 (op, 输出 tile/loop tile) -> 输出 operand footprints。
// 这是后续 Tile Propagation/Traffic Model 的基础。
class FootprintInference {
public:
  virtual ~FootprintInference() = default;

  virtual std::optional<FootprintResult> infer(mlir::Operation *op,
                                               const OpTile &tile) const = 0;
};

// 基于 Linalg indexing_maps 的最小可行 Footprint 推导器。
// - 通过分析 op 的 AffineMap（indexing maps）推导每个 operand 的 index bounds。
// - 当前版本先只支持常见的 affine 形式（Add/Mul(const)/DimId/SymbolId/Constant）。
// - 推导失败时返回 std::nullopt（由上层选择 fallback 策略）。
class LinalgIndexingMapsFootprintInference final : public FootprintInference {
public:
  std::optional<FootprintResult> infer(mlir::Operation *op,
                                       const OpTile &tile) const override;
};

// Tile-graph：Welder 的 tile-level data-flow 抽象。
// 这里先定义最小结构（Node/Edge），后续再把 propagation / connect 等逻辑补上。
//
// memory/connect 层级语义（论文对齐）：
// - Global   (0)：切边，值会物化到 global memory（分离 kernels）
// - Shared   (1)：融合边，值在 shared memory 中复用（单 kernel）
// - Register (2)：值进一步在寄存器中复用（单 kernel，内层调度）
//
// 注意：在当前原型中，codegen 仅区分：
//   connectLevel==0（cut/global）与 connectLevel>0（fused/shared+）。
enum class MemoryLevel : int {
  Global = 0,
  Shared = 1,
  Register = 2,
};
constexpr int kConnectLevelGlobal = static_cast<int>(MemoryLevel::Global);
constexpr int kConnectLevelShared = static_cast<int>(MemoryLevel::Shared);
constexpr int kConnectLevelRegister = static_cast<int>(MemoryLevel::Register);

// 论文/Welder 对齐（stride_map/layout，最小 v1）：
// 描述由 solver 推断出的 shared/workgroup memory 布局调整。
//
// 该原型目前只建模 MLIR codegen 暴露出的子集：
// - workgroup-pad-last-dim（TCPolicy stride 偏移，通常为 8）
// - workgroup-swizzle-xor （缓解 bank conflict）
struct SharedLayoutInfo {
  // 最后一维填充 N 个元素（>=0）。
  int64_t padLastDim = 0;
  // 最后一维的 XOR swizzle 因子（0/1 表示禁用）。
  int64_t swizzleXor = 0;

  bool hasPadding() const { return padLastDim > 0; }
  bool hasSwizzle() const { return swizzleXor > 1; }
};

struct TileGraphNode {
  // 非 owning：指向 payload op（Module 生命周期内有效）。
  mlir::Operation *op = nullptr;

  // 图拓扑。
  std::vector<int> inEdges;
  std::vector<int> outEdges;

  // 论文/Welder 对齐（block 顺序提示）：
  // - 参考实现可为每个节点分配 block_id 表达式（block_order），以在图变换
  //   （如 transpose）后保持光栅化一致性。
  // - 本 MLIR 原型目前仅暴露最小子集：用布尔提示表示前两个并行维是否映射到
  //   (block<x>, block<y>)，而非 (block<y>, block<x>)。
  // - 若缺失该提示，codegen 回退到全局默认值。
  std::optional<bool> swapXYHint;

  // tile 传播状态。
  bool hasRequiredTile = false;
  OpTile requiredTile;
};

struct TileGraphEdge {
  // 连接：src -> dst
  int src = -1;
  int dst = -1;

  // （可选）指明 edge 对应 src 的哪个输出 / dst 的哪个输入。
  int srcResult = -1;
  int dstOperand = -1;

  // 这个 edge 代表的 SSA value（通常是 producer 的一个 tensor result）。
  // 注意：Value 是轻量句柄，随 module 生命周期有效。
  mlir::Value value;

  // 该 SSA 边上 producer->consumer 之间可选的一串 view-like op
  // （例如 tensor.expand_shape/collapse_shape）。
  // 这样可在仅 reshape 的边上完成 tile 传播，而无需物化中间张量。
  //
  // 存储顺序为 consumer->producer（首个 op 定义 consumer operand）。
  std::vector<mlir::Operation *> viewOps;

  // 当前边上（reuse-tile / data tile）的 footprint（后续由 propagation 决定）。
  OperandFootprint footprint;
  // Welder 论文 SetConnect：该 edge 的复用层级（memory level）。
  // - 0：Global/cut（分离 kernels，global memory 物化）
  // - 1：Shared（融合 kernel，shared 复用）
  // - 2：Register（融合 kernel，寄存器复用）
  //
  // 默认值为 Shared，用于建模常见的“全融合”假设；论文风格 GraphConnecting
  // 会按需显式重置各层级。
  int connectLevel = kConnectLevelShared;
  // Phase-13A“切边”行为的向后兼容缓存。
  // 不变量：isCut == (connectLevel <= kConnectLevelGlobal)。
  bool isCut = false;
};

struct TileGraph {
  std::vector<TileGraphNode> nodes;
  std::vector<TileGraphEdge> edges;
};

inline void syncCutFlagFromConnectLevel(TileGraphEdge &e) {
  e.isCut = (e.connectLevel <= kConnectLevelGlobal);
}

inline void setEdgeConnectLevel(TileGraphEdge &e, int level) {
  e.connectLevel = level;
  syncCutFlagFromConnectLevel(e);
}

struct TilePropagationOptions {
  // reduction loop 的默认 tile extent（用于构造 OpTile.loopExtents）。
  // <= 0 表示使用 full range（要求静态 loop range）。
  int64_t defaultReductionTile = 0;
  // 论文/Welder 对齐：允许按节点设置 reduction-loop tile（rstep）。
  //
  // - 若设置，则按 TileGraph 节点索引。
  // - 每个条目是该 op 各 reduction loop 的 extent 列表，顺序与 iterator_types 一致。
  // - 空列表表示“使用 defaultReductionTile/full range”。
  //
  // 该语义与参考实现的 rstep_map 一致。
  const std::vector<std::vector<int64_t>> *reductionTilesByNode = nullptr;
  // Phase 13A：允许在冲突时切断边而不是失败。
  bool enableCutEdges = false;
  // 是否在传播开始时重置 global-level cut。
  // - true（默认）：先将 connectLevel<=Global 的边提升到 Shared 以清空 cut 决策
  //   （先按“全融合”建模），再重新推导 cut。
  // - false：跨迭代保留现有 connectLevel/cut 决策
  //   （用于 GraphConnecting/TwoLevelSchedule 多轮流程）。
  bool resetCutEdges = true;
  // 是否在传播开始时重置 TileGraphNode.requiredTile。
  //
  // 默认 true（单 sink 行为）。对于多 sink / 多输出图，
  // 可能会多次运行传播以累积约束；此时后续 sink 可设为 false，
  // 使先前 requiredTile + cut 决策得以保留。
  bool resetGraphState = true;
  bool verbose = false;
};

struct TilePropagationResult {
  bool success = false;
  std::string error;
};

// 论文/Welder 对齐：从 linalg op 的 indexing maps 推断最小 block/thread remap 提示。
//
// 当前仅返回常见 2D 情况（swap/no-swap），由输出 indexing map 推导：
// - false：前两个并行 loop 映射为 (x,y) 时使用 (p1->x, p0->y)
// - true ：映射为 (p0->x, p1->y)
//
// 当无法推断该提示时返回 std::nullopt。
std::optional<bool> inferSwapXYHintForLinalgOp(mlir::Operation *op);

// 构建一个基于 LinalgOp 的 TileGraph：nodes = LinalgOp，edges 通过 SSA use-def
// 链接 producer->consumer（仅覆盖 tensor SSA 的路径；memref 的就地写入后续再补）。
std::optional<TileGraph> buildLinalgTileGraph(mlir::ModuleOp module);

// Welder-style 反向传播：从 root（通常是一个 sink consumer）开始，给定 rootTile，
// 通过 footprint inference 推导其输入 operand 的 footprint，并把约束传播到上游 producer。
TilePropagationResult propagateTilesBackward(TileGraph &graph, int rootNode,
                                            const OpTile &rootTile,
                                            const FootprintInference &inference,
                                            const TilePropagationOptions &opts);

// Phase 14（论文对齐，骨架）：
// 2-level (global <-> shared) GraphConnecting + Propagate 的最小公共入口，供 solver/compiler 复用。
//
// - 从 rootOp（sink）出发，用 cand.tileM/tileN 映射到前两维 parallel loops，
//   并用 cand.tileK 作为 reduction 默认 tile（若存在 reduction loops）。
// - 先做一次 tile propagation；若 enableTwoLevelSchedule 且 footprint 超过 arch.smemBytes，
//   则在 enableCutEdges 下迭代 cut-edge + 复跑 propagation，直到满足 capacity 或无边可 cut。
//
// 返回：TilePropagationResult（成功/失败 + 错误信息）。
// 若 outEstFootprintBytes 非空，返回最终（cut 后）的 shared footprint 估算（bytes）。
TilePropagationResult propagateTilesBackwardTwoLevel(
    TileGraph &graph, int rootNode, mlir::Operation *rootOp,
    const Candidate &cand, const SolveOptions &opts,
    const FootprintInference &inference, int64_t *outEstFootprintBytes);

std::vector<int64_t> parseCsvIntList(const std::string &s);

std::optional<ProblemSize> analyzeMatmulProblem(mlir::ModuleOp module);

// 通用分析器：选择一个“最有价值”的 linalg op 作为锚点，并提取其 loop 列表（静态大小 +
// iterator 类型）。
std::optional<GenericProblem> analyzeGenericProblem(mlir::ModuleOp module);

bool detectMatmulConsumerChain(mlir::ModuleOp module);

SolveResult solve(mlir::ModuleOp module, const SolveOptions &opts);

// 将原始 Linalg TileGraph（nodes/edges）导出为 JSON 文件。
//
// 用途是调试与构建论文对齐的“可观测性”产物。
// 该导出不执行调度或传播，只反映 linalg op 之间的 SSA producer->consumer 结构。
bool dumpTileGraphJson(mlir::ModuleOp module, const SolveOptions &opts,
                       const std::string &path);

// 论文对齐（Figure 8）：导出所选 tile-graph 调度的层级执行计划
// （connect levels + 每 op 传播后的 tile + shared/register footprint 分配）。
//
// 这是“仅计划”的导出，用于论文级可追踪性；不会改变 codegen 行为。
bool dumpPaperExecutionPlan(mlir::ModuleOp module, const SolveOptions &opts,
                            const std::string &path);

// 通用求解：不要求存在 linalg.matmul，基于 GenericProblem 的 loop 列表做最小枚举。
// 目前用于让 solver 能在 Conv2D 等算子上跑起来，先不替换旧 MatMul 路径。
SolveResult solveGeneric(mlir::ModuleOp module, const SolveOptions &opts);

} // 命名空间 welder
