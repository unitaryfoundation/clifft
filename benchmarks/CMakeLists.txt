cmake_minimum_required(VERSION 3.20)

# This directory is also a standalone driver so CI can hold the benchmark
# definition fixed while compiling either the base or proposed source tree.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(clifft_benchmarks LANGUAGES CXX)

    get_filename_component(_CLIFFT_DEFAULT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(CLIFFT_SOURCE_DIR "${_CLIFFT_DEFAULT_SOURCE_DIR}" CACHE PATH
        "Clifft source tree to benchmark")
    set(CLIFFT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(CLIFFT_BUILD_PROFILER OFF CACHE BOOL "" FORCE)
    set(CLIFFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)

    add_subdirectory("${CLIFFT_SOURCE_DIR}" clifft-source)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FetchGoogleBenchmark.cmake")
endif()

add_executable(clifft_benchmarks clifft_benchmarks.cc)
target_link_libraries(clifft_benchmarks PRIVATE clifft_core benchmark::benchmark_main)
target_compile_features(clifft_benchmarks PRIVATE cxx_std_20)
get_filename_component(
    _CLIFFT_BENCHMARK_FIXTURES_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures"
    ABSOLUTE
)
target_compile_definitions(clifft_benchmarks PRIVATE
    CLIFFT_BENCHMARK_FIXTURES_DIR="${_CLIFFT_BENCHMARK_FIXTURES_DIR}"
)
