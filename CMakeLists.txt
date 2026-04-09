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

# CPU baseline policy for optimized builds.
#
# Local source builds default to native tuning. Release wheel CI should set an
# explicit portable baseline via CLIFFT_CPU_BASELINE.
set(CLIFFT_CPU_BASELINE "native" CACHE STRING
    "CPU baseline for optimized builds: native, generic, or x86-64-v3")
set_property(CACHE CLIFFT_CPU_BASELINE PROPERTY STRINGS native generic x86-64-v3)

if(NOT CLIFFT_CPU_BASELINE STREQUAL "native" AND
   NOT CLIFFT_CPU_BASELINE STREQUAL "generic" AND
   NOT CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
    message(FATAL_ERROR
        "CLIFFT_CPU_BASELINE must be one of: native, generic, x86-64-v3. "
        "Got '${CLIFFT_CPU_BASELINE}'.")
endif()

message(STATUS "CLIFFT_CPU_BASELINE = ${CLIFFT_CPU_BASELINE}")

# If the selected build baseline explicitly guarantees AVX2/FMA, enable those
# instructions globally so the generic code can also use the same floor.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)" AND
       CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-mavx2 HAS_AVX2)
        check_cxx_compiler_flag(-mfma HAS_FMA)
        set(_CLIFFT_X86_64_V3_FEATURES "")
        if(HAS_AVX2)
            add_compile_options(-mavx2)
            list(APPEND _CLIFFT_X86_64_V3_FEATURES "AVX2")
        endif()
        if(HAS_FMA)
            add_compile_options(-mfma)
            list(APPEND _CLIFFT_X86_64_V3_FEATURES "FMA")
        endif()
        if(_CLIFFT_X86_64_V3_FEATURES)
            list(JOIN _CLIFFT_X86_64_V3_FEATURES "/" _CLIFFT_X86_64_V3_FEATURES_STR)
            message(STATUS "Enabled x86-64-v3 SIMD baseline (${_CLIFFT_X86_64_V3_FEATURES_STR})")
        else()
            message(STATUS "Compiler did not accept extra AVX2/FMA flags for x86-64-v3 baseline")
        endif()
    endif()
endif()

# Additional Release-only optimizations.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-ffast-math)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
            if(CLIFFT_CPU_BASELINE STREQUAL "native")
                add_compile_options(-mcpu=native)
                message(STATUS "Enabled ARM native CPU optimizations: -mcpu=native -ffast-math")
            elseif(CLIFFT_CPU_BASELINE STREQUAL "generic")
                message(STATUS "Using portable ARM baseline with -ffast-math")
            else()
                message(FATAL_ERROR
                    "CLIFFT_CPU_BASELINE=x86-64-v3 is only valid on x86-64 builds")
            endif()
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
            if(CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
                # x86-64-v3 baseline: AVX2 + BMI2 + FMA (~2015 CPUs and later)
                add_compile_options(-march=x86-64-v3)
                message(STATUS "Using x86-64-v3 CPU baseline: -march=x86-64-v3 -ffast-math")
            elseif(CLIFFT_CPU_BASELINE STREQUAL "native")
                add_compile_options(-march=native -mtune=native)
                message(STATUS "Enabled x86 native CPU optimizations: -march=native -mtune=native -ffast-math")
            else()
                message(STATUS "Using generic x86-64 baseline with -ffast-math")
            endif()
        elseif(CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
            message(FATAL_ERROR
                "CLIFFT_CPU_BASELINE=x86-64-v3 is only valid on x86-64 builds")
        elseif(CLIFFT_CPU_BASELINE STREQUAL "native")
            message(STATUS
                "No architecture-specific native tuning configured for "
                "${CMAKE_SYSTEM_PROCESSOR}; using generic optimized flags")
        endif()
    endif()
endif()

# Maximum qubit count (compile-time). Determines the width of inline Pauli
# masks (BitMask<N>) used by the HIR and SVM Pauli frame. The VM Instruction
# struct stays 32 bytes regardless -- only compile-time structures grow.
# Default 128 covers the paper-facing install experience (distance-7 surface
# code and 85-qubit distillation). Rebuild with 64 for best small/QV
# performance or 512 for larger cultivation circuits.
# Can be set via -DCLIFFT_MAX_QUBITS=N or the CLIFFT_MAX_QUBITS env var.
# The env var is checked every configure, so `export CLIFFT_MAX_QUBITS=512`
# works with scikit-build-core editable installs that reconfigure each build.
if(DEFINED ENV{CLIFFT_MAX_QUBITS})
    set(CLIFFT_MAX_QUBITS "$ENV{CLIFFT_MAX_QUBITS}" CACHE STRING
        "Maximum supported qubits (must be a multiple of 64, >= 64)" FORCE)
else()
    set(CLIFFT_MAX_QUBITS "128" CACHE STRING
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
