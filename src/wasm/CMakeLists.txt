# UCC WebAssembly target (Emscripten)
#
# Build via: just build-wasm
# Uses emcmake to cross-compile the UCC core + Embind bridge to .wasm/.js.

cmake_minimum_required(VERSION 3.20)
project(ucc_wasm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Strip native arch flags that are incompatible with Wasm.
# These may leak in from the parent CMakeLists or toolchain defaults.
string(REPLACE "-march=native" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-mtune=native" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-mcpu=native" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-mavx512f" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-ffast-math" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Fetch dependencies (same versions as native build)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FetchStim.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FetchFastFloat.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FetchNlohmannJson.cmake)

# Build the core library (reuses the same source list)
add_library(ucc_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/circuit/parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/frontend/frontend.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/optimizer/bytecode_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/optimizer/expand_t_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/optimizer/multi_gate_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/optimizer/noise_block_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/optimizer/hir_pass_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/optimizer/peephole.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/optimizer/swap_meas_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/optimizer/single_axis_fusion_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/backend/backend.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/svm/svm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/svm/svm_scalar.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/svm/svm_state.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/util/introspection.cc
)

target_include_directories(ucc_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(ucc_core PUBLIC libstim)
target_link_libraries(ucc_core PRIVATE fast_float)

# Stim also throws exceptions; ensure it's compiled with exception support.
target_compile_options(libstim PRIVATE -fexceptions)

# Generate version header (minimal, just needs to exist)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/ucc/util)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../pyproject.toml" PYPROJECT_CONTENT)
string(REGEX MATCH "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ "${PYPROJECT_CONTENT}")
set(UCC_VERSION "${CMAKE_MATCH_1}")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../ucc/util/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/ucc/util/version.h
    @ONLY
)

# Wasm executable (produces .js + .wasm)
add_executable(ucc_wasm bindings.cc)
target_link_libraries(ucc_wasm PRIVATE ucc_core nlohmann_json::nlohmann_json)

# Enable C++ exception handling (required for try/catch in bindings.cc)
target_compile_options(ucc_wasm PRIVATE -fexceptions)
target_compile_options(ucc_core PRIVATE -fexceptions)

# Emscripten link flags
# Use -sKEY=VALUE (no space) so CMake doesn't split them into separate args.
target_link_options(ucc_wasm PRIVATE
    -lembind
    -fexceptions
    -sMODULARIZE=1
    -sEXPORT_NAME=createUccModule
    -sALLOW_MEMORY_GROWTH=1
    -sENVIRONMENT=web,worker,node
    -sEXPORTED_RUNTIME_METHODS=[]
    -sTOTAL_STACK=1048576
    -O2
)

set_target_properties(ucc_wasm PROPERTIES
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
