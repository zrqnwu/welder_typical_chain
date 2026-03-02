cmake_minimum_required(VERSION 3.20)

project(WelderSolver LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CUDAToolkit)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(HandleLLVMOptions)
include(AddLLVM)
include(AddMLIR)

set(WELDER_COMPILER_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/driver
  ${CMAKE_CURRENT_SOURCE_DIR}/dispatch
  ${CMAKE_CURRENT_SOURCE_DIR}/transforms
  ${CMAKE_CURRENT_SOURCE_DIR}/postprocess
  ${CMAKE_CURRENT_SOURCE_DIR}/solver
)

set(WELDER_SOLVER_LIB_SRCS
  common/AffineIntervalEvaluator.cpp
  solver/WelderSolverLib.cpp
)

set(WELDER_SOLVER_SRCS
  solver/WelderSolverCandidateGenerator.cpp
  solver/WelderSolverCostModel.cpp
  solver/WelderSolverProfilerRunner.cpp
  solver/WelderSolverReportWriter.cpp
  solver/WelderSolver.cpp
)

set(WELDER_COMPILER_CORE_SRCS
  dispatch/WelderCompilerAnchorDiscovery.cpp
  transforms/WelderCompilerAsyncCopyRewrite.cpp
  transforms/WelderCompilerAsyncPipelineHelpers.cpp
  dispatch/WelderCompilerDispatchPlan.cpp
  dispatch/WelderCompilerFusionAnchors.cpp
  dispatch/WelderCompilerFusionPairBuild.cpp
  transforms/WelderCompilerGenericCutEdgesTransformLibrary.cpp
  transforms/WelderCompilerGenericTransformAndFusion.cpp
  driver/WelderCompilerDriverHelpers.cpp
  driver/WelderCompilerRunConfig.cpp
  transforms/WelderCompilerMatmulTransformLibrary.cpp
  dispatch/WelderCompilerModeDispatch.cpp
  dispatch/WelderCompilerModeDispatchKernelAttrsBranch.cpp
  dispatch/WelderCompilerModeDispatchGenericBranch.cpp
  dispatch/WelderCompilerModeDispatchMatmulBranch.cpp
  postprocess/WelderCompilerMemoryReuseTransforms.cpp
  postprocess/WelderCompilerPostprocessCleanup.cpp
  postprocess/WelderCompilerMatmulSoftmaxCanonicalize.cpp
  driver/WelderCompilerPassTraceAndEnv.cpp
  transforms/WelderCompilerRowReductionInputPromotion.cpp
  dispatch/WelderCompilerTileDecision.cpp
  driver/WelderCompiler.cpp
  postprocess/WelderCompilerPostbufferizeFixups.cpp
)

add_library(welder-solver-lib ${WELDER_SOLVER_LIB_SRCS})

target_include_directories(welder-solver-lib PUBLIC
  ${WELDER_COMPILER_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(welder-solver-lib PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(welder-solver-lib PUBLIC
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
)

add_library(welder-solver-capi SHARED
  solver/WelderSolverCAPI.cpp
)

target_include_directories(welder-solver-capi PUBLIC
  ${WELDER_COMPILER_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(welder-solver-capi PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(welder-solver-capi PUBLIC
  welder-solver-lib
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
)
llvm_update_compile_flags(welder-solver-capi)

add_executable(welder-solver ${WELDER_SOLVER_SRCS})

target_include_directories(welder-solver PRIVATE
  ${WELDER_COMPILER_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(welder-solver PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(welder-solver PRIVATE
  welder-solver-lib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
)

llvm_update_compile_flags(welder-solver)

add_executable(welder-compiler ${WELDER_COMPILER_CORE_SRCS})

target_include_directories(welder-compiler PRIVATE
  ${WELDER_COMPILER_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(welder-compiler PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(welder-compiler PRIVATE
  welder-solver-lib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRLinalgTransformOps
  MLIRGPUTransformOps
  MLIRNVGPUDialect
  MLIRNVGPUTransformOps
  MLIRNVGPUTransforms
  MLIRBufferizationTransforms
  MLIRArithTransforms
  MLIRControlFlowTransforms
  MLIRLinalgTransforms
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRVectorTransforms
  MLIRVectorTransformOps
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
)

llvm_update_compile_flags(welder-compiler)

add_library(welder-compile-capi SHARED
  ${WELDER_COMPILER_CORE_SRCS}
  driver/WelderCompileSession.cpp
  driver/WelderCompileCAPI.cpp
  ../mlir_pipeline/workgroup_alloc_to_launch_pass/WorkgroupAllocToLaunch.cpp
)

target_include_directories(welder-compile-capi PRIVATE
  ${WELDER_COMPILER_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(welder-compile-capi PRIVATE ${LLVM_DEFINITIONS})
target_compile_definitions(welder-compile-capi PRIVATE WELDER_COMPILER_NO_MAIN)

target_link_libraries(welder-compile-capi PRIVATE
  welder-solver-lib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRToLLVMIRTranslationRegistration
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRLinalgTransformOps
  MLIRGPUTransformOps
  MLIRNVGPUDialect
  MLIRNVGPUTransformOps
  MLIRNVGPUTransforms
  MLIRBufferizationTransforms
  MLIRArithTransforms
  MLIRControlFlowTransforms
  MLIRLinalgTransforms
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRVectorTransforms
  MLIRVectorTransformOps
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRLLVMDialect
)

llvm_update_compile_flags(welder-compile-capi)

add_executable(welder-pipeline
  driver/WelderPipeline.cpp
)

target_include_directories(welder-pipeline PRIVATE
  ${WELDER_COMPILER_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(welder-pipeline PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(welder-pipeline PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRSCFTransforms
  MLIRNVGPUDialect
  MLIRNVGPUTransforms
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
)

llvm_update_compile_flags(welder-pipeline)

# Optional: CUDA Driver API profiler (used for 论文对齐 hardware 性能测量).
if (CUDAToolkit_FOUND)
  add_executable(welder-profiler
    solver/WelderProfiler.cpp
  )

  target_include_directories(welder-profiler PRIVATE
    ${WELDER_COMPILER_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_compile_definitions(welder-profiler PRIVATE ${LLVM_DEFINITIONS})

  target_link_libraries(welder-profiler PRIVATE
    LLVMSupport
    CUDA::cuda_driver
  )

  llvm_update_compile_flags(welder-profiler)
else()
  message(STATUS "CUDAToolkit not found: skipping welder-profiler build")
endif()
