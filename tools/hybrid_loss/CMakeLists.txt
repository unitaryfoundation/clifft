# Standalone research build. Uses the ordinary core without changing its API.
cmake_minimum_required(VERSION 3.20)
project(clifft_hybrid_experiment LANGUAGES CXX)
set(CLIFFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLIFFT_BUILD_PROFILER OFF CACHE BOOL "" FORCE)
add_subdirectory(../.. clifft)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/instrument_driver.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/clifft/noncomp/trajectory_driver.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/profiled_driver.cc" COMMAND_ERROR_IS_FATAL ANY)
add_executable(hybrid_loss main.cc hybrid.cc "${CMAKE_CURRENT_BINARY_DIR}/profiled_driver.cc")
target_link_libraries(hybrid_loss PRIVATE clifft_core)
target_compile_features(hybrid_loss PRIVATE cxx_std_20)
target_compile_options(hybrid_loss PRIVATE -Wall -Wextra -Wpedantic -march=native)
