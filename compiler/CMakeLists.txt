set(WTC_COMPILER_SOURCES
  main.cpp
  backend/Toolchain.cpp
  pipeline/PipelineRunner.cpp
  ir/Tagging.cpp
  ir/ChainCanonicalize.cpp
  scheduler/Search.cpp
  scheduler/SearchCommon.cpp
  scheduler/SearchBackendShell.cpp
  scheduler/SearchBackendProcessChain.cpp
  scheduler/SearchBackendApi.cpp
  transform/BuildCutEdgesTransform.cpp
  transform/BuildCutEdgesTransformCommon.cpp
  transform/BuildCutEdgesTransformShell.cpp
  transform/BuildCutEdgesTransformProcessChain.cpp
  transform/BuildCutEdgesTransformApi.cpp
  transform/CompileApiSession.cpp
  runtime/PostbufferizeFixups.cpp
)

add_executable(wtc-compiler ${WTC_COMPILER_SOURCES})

target_include_directories(wtc-compiler PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (UNIX AND NOT APPLE)
  target_link_libraries(wtc-compiler PRIVATE dl)
endif()

if (WTC_ENABLE_MLIR)
  target_compile_definitions(wtc-compiler PRIVATE WTC_ENABLE_MLIR=1)
endif()
