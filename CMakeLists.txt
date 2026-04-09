cmake_minimum_required(VERSION 3.20)
project(clifft LANGUAGES CXX)

# Version: derived from git tags (or VERSION file in sdists)
include(cmake/ClifftVersion.cmake)

# Generate version header
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/clifft/util)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clifft/util/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/clifft/util/version.h
    @ONLY
)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type if user doesn't specify one (e.g., cmake -DCMAKE_BUILD_TYPE=Release).
# Python builds default to Release; standalone C++ builds default to Debug.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    if(SKBUILD)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    else()
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    endif()
endif()

# Export compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# SIMD intrinsics: enable AVX2/FMA on x86 for ALL build types so that Debug
# builds exercise the same SIMD code paths as Release. This only adds the
# instruction-set flags; it does NOT enable aggressive optimizations like
# -ffast-math or -march=native (those remain Release-only below).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-mavx2 HAS_AVX2)
        check_cxx_compiler_flag(-mfma HAS_FMA)
        if(HAS_AVX2)
            add_compile_options(-mavx2)
            message(STATUS "Enabled AVX2 instructions for all build types")
        endif()
        if(HAS_FMA)
            add_compile_options(-mfma)
            message(STATUS "Enabled FMA instructions for all build types")
        endif()
    endif()
endif()

# Additional Release-only optimizations.
#
# When building wheels for PyPI (SKBUILD is set), use portable baselines so
# the resulting binary runs on a wide range of hardware.  Local developer
# builds use -march=native for maximum performance on the build machine.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-ffast-math)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
            if(SKBUILD)
                message(STATUS "PyPI wheel build: using portable ARM optimizations")
            else()
                add_compile_options(-mcpu=native)
                message(STATUS "Enabled ARM native CPU optimizations: -mcpu=native -ffast-math")
            endif()
        else()
            if(SKBUILD)
                # x86-64-v3 baseline: AVX2 + FMA (~2015 CPUs and later)
                add_compile_options(-march=x86-64-v3)
                message(STATUS "PyPI wheel build: using x86-64-v3 baseline (AVX2)")
            else()
                add_compile_options(-march=native -mtune=native)
                message(STATUS "Enabled x86 native CPU optimizations: -march=native -mtune=native -ffast-math")
            endif()
        endif()
    endif()
endif()

# Maximum qubit count (compile-time). Determines the width of inline Pauli
# masks (BitMask<N>) used by the HIR and SVM Pauli frame. The VM Instruction
# struct stays 32 bytes regardless -- only compile-time structures grow.
# Default is 64; set to 512 for magic state cultivation circuits.
# Can be set via -DCLIFFT_MAX_QUBITS=N or the CLIFFT_MAX_QUBITS env var.
# The env var is checked every configure, so `export CLIFFT_MAX_QUBITS=512`
# works with scikit-build-core editable installs that reconfigure each build.
if(DEFINED ENV{CLIFFT_MAX_QUBITS})
    set(CLIFFT_MAX_QUBITS "$ENV{CLIFFT_MAX_QUBITS}" CACHE STRING
        "Maximum supported qubits (must be a multiple of 64, >= 64)" FORCE)
else()
    set(CLIFFT_MAX_QUBITS "64" CACHE STRING
        "Maximum supported qubits (must be a multiple of 64, >= 64)")
endif()
if(CLIFFT_MAX_QUBITS LESS 64)
    message(FATAL_ERROR "CLIFFT_MAX_QUBITS must be >= 64, got ${CLIFFT_MAX_QUBITS}")
endif()
math(EXPR _CLIFFT_MOD "${CLIFFT_MAX_QUBITS} % 64")
if(NOT _CLIFFT_MOD EQUAL 0)
    message(FATAL_ERROR "CLIFFT_MAX_QUBITS must be a multiple of 64, got ${CLIFFT_MAX_QUBITS}")
endif()
message(STATUS "CLIFFT_MAX_QUBITS = ${CLIFFT_MAX_QUBITS}")

# Code coverage option (for use with gcov/lcov)
# Note: Coverage flags are applied per-target in src/clifft/CMakeLists.txt to avoid
# instrumenting external dependencies (Stim, Catch2) which would slow builds.
option(CLIFFT_COVERAGE "Enable code coverage instrumentation" OFF)
if(CLIFFT_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(WARNING "Code coverage requires GCC or Clang")
        set(CLIFFT_COVERAGE OFF)
    else()
        message(STATUS "Code coverage enabled (Clifft targets only)")
    endif()
endif()

# Fetch external dependencies
include(cmake/FetchStim.cmake)
include(cmake/FetchFastFloat.cmake)

# Build the core library
add_subdirectory(src/clifft)

# Python bindings (only when building via scikit-build-core)
if(SKBUILD)
    add_subdirectory(src/python)
else()
    # Standalone C++ build: include Catch2 and tests
    include(cmake/FetchCatch2.cmake)
    enable_testing()
    add_subdirectory(tests)

    # Profiling tool (opt-in, not part of default or coverage builds)
    option(CLIFFT_BUILD_PROFILER "Build the SVM profiling harness" OFF)
    if(CLIFFT_BUILD_PROFILER)
        add_executable(profile_svm tools/profile/profile_svm.cpp)
        target_link_libraries(profile_svm PRIVATE clifft_core)
        target_include_directories(profile_svm PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
        )
    endif()
endif()
