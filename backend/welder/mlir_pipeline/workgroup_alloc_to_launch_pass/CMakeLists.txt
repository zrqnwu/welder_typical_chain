cmake_minimum_required(VERSION 3.20)

project(WorkgroupAllocToLaunchPass LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(HandleLLVMOptions)
include(AddLLVM)
include(AddMLIR)

add_llvm_library(WorkgroupAllocToLaunchPass
  MODULE
  WorkgroupAllocToLaunch.cpp

  DEPENDS
  PLUGIN_TOOL
  mlir-opt
)

target_include_directories(WorkgroupAllocToLaunchPass PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(WorkgroupAllocToLaunchPass PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(WorkgroupAllocToLaunchPass PRIVATE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRArithDialect
  MLIRGPUDialect
  MLIRNVGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
