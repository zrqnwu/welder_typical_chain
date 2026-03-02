cmake_minimum_required(VERSION 3.20)

project(WelderTypicalChain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(WTC_BUILD_COMPILER "Build wtc-compiler prototype" ON)
option(WTC_ENABLE_MLIR "Enable MLIR integration (future)" OFF)

if (WTC_BUILD_COMPILER)
  add_subdirectory(compiler)
endif()
