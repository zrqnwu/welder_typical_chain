Title: [MLIR][MemRefToLLVM] memref.dealloc lowering emits invalid free() call for non-default addrspace pointers (e.g. workgroup/shared)

### Summary

Lowering `memref.dealloc` to LLVM/NVVM currently unconditionally emits `llvm.call @free(...)`. When the memref is in a non-default address space (e.g. `#gpu.address_space<workgroup>` which becomes `addrspace(3)`), the generated call is ill-typed: it passes `!llvm.ptr<3>` to `@free`, whose argument type is `!llvm.ptr` (addrspace(0)). This causes the NVVM pipeline to fail with an operand type mismatch.

### Steps to reproduce

1. Build `mlir-opt` with GPU/NVVM support.
2. Run:

```bash
BIN=llvm-project/build/bin
$BIN/mlir-opt mlir_pipeline/llvm_bug_report/repro.mlir \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_86 cubin-format=isa" \
  -o /tmp/out.nvvm.mlir
```

### Actual result

```
error: 'llvm.call' op operand type mismatch for operand 0: '!llvm.ptr<3>' != '!llvm.ptr'
note: see current operation: "llvm.call" ... callee = @free ... : (!llvm.ptr<3>) -> ()
```

### Expected result

The pipeline should not produce invalid LLVM dialect IR. Possible fixes:

* Insert an `llvm.addrspacecast` to `!llvm.ptr` (addrspace(0)) before calling `@free`, so `memref.alloc`/`memref.dealloc` remain symmetric for non-zero memory spaces.
* Alternatively, emit a clear diagnostic that `memref.dealloc` for non-default address spaces is unsupported (instead of generating an invalid `llvm.call`).
* For GPU workgroup/shared memory specifically, it may be preferable to erase `memref.dealloc` (no-op semantics) rather than calling `free()`, but the type mismatch is the immediate correctness issue.

### Minimal reproducer

See: `mlir_pipeline/llvm_bug_report/repro.mlir`

### Root cause analysis

`mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp` has:

* `AllocOpLowering` uses `castAllocFuncResult` to `addrspacecast` the `malloc` result to the memref address space.
* `DeallocOpLowering` computes `allocatedPtr` from the memref descriptor and calls `@free(allocatedPtr)` directly, without casting back to addrspace(0).

This results in `llvm.call @free(!llvm.ptr<3>)`.

### Environment

* llvm-project commit: `95fdfcca9bcd4b1883e52e2aad32234729ffdaee` (describe: `llvmorg-23-init-1701-g95fdfcca9bcd`)
* `mlir-opt --version`: LLVM 23.0.0git (Optimized build with assertions)
* OS: Linux 6.14.0-37-generic (Ubuntu 24.04.1)

