cmake_minimum_required(VERSION 3.20)

# Make MSVC Debug builds compatible with ccache. The default ProgramDatabase
# format uses /Zi, which ccache treats as uncacheable; Embedded uses /Z7.
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif()

project(clifft LANGUAGES CXX)

option(CLIFFT_ENABLE_HIP "Build the experimental AMD HIP sampling backend" OFF)
if(CLIFFT_ENABLE_HIP)
    if(CMAKE_VERSION VERSION_LESS 3.21)
        message(FATAL_ERROR "CLIFFT_ENABLE_HIP requires CMake 3.21 or newer")
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES gfx942 CACHE STRING
            "AMD GPU architectures for the experimental HIP backend")
    endif()
    enable_language(HIP)
    message(STATUS "CLIFFT_ENABLE_HIP = ON (${CMAKE_HIP_ARCHITECTURES})")
endif()

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

# Keep assert() active in optimized configurations. Assertion coverage is this
# project's primary internal-invariant check, so CI can pair it with optimized
# code where a full Debug build is too slow.
option(CLIFFT_FORCE_ASSERTS "Keep assert() active regardless of build type" OFF)
if(CLIFFT_FORCE_ASSERTS)
    foreach(lang C CXX)
        foreach(config RELEASE RELWITHDEBINFO MINSIZEREL)
            string(REPLACE "/DNDEBUG" "" CMAKE_${lang}_FLAGS_${config}
                   "${CMAKE_${lang}_FLAGS_${config}}")
            string(REPLACE "-DNDEBUG" "" CMAKE_${lang}_FLAGS_${config}
                   "${CMAKE_${lang}_FLAGS_${config}}")
        endforeach()
    endforeach()
    message(STATUS "CLIFFT_FORCE_ASSERTS: assert() remains active in optimized configurations")
endif()

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
    "CPU baseline for optimized builds: native, generic, x86-64-v2, or x86-64-v3")
set_property(CACHE CLIFFT_CPU_BASELINE PROPERTY STRINGS native generic x86-64-v2 x86-64-v3)

if(NOT CLIFFT_CPU_BASELINE STREQUAL "native" AND
   NOT CLIFFT_CPU_BASELINE STREQUAL "generic" AND
   NOT CLIFFT_CPU_BASELINE STREQUAL "x86-64-v2" AND
   NOT CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
    message(FATAL_ERROR
        "CLIFFT_CPU_BASELINE must be one of: native, generic, x86-64-v2, x86-64-v3. "
        "Got '${CLIFFT_CPU_BASELINE}'.")
endif()

message(STATUS "CLIFFT_CPU_BASELINE = ${CLIFFT_CPU_BASELINE}")

set(CLIFFT_OPENMP "AUTO" CACHE STRING
    "OpenMP intra-shot kernels: OFF, AUTO, or ON")
set_property(CACHE CLIFFT_OPENMP PROPERTY STRINGS OFF AUTO ON)
if(NOT CLIFFT_OPENMP STREQUAL "OFF" AND
   NOT CLIFFT_OPENMP STREQUAL "AUTO" AND
   NOT CLIFFT_OPENMP STREQUAL "ON")
    message(FATAL_ERROR "CLIFFT_OPENMP must be one of: OFF, AUTO, ON")
endif()

set(CLIFFT_OPENMP_ENABLED OFF)
if(NOT CLIFFT_OPENMP STREQUAL "OFF")
    if(EMSCRIPTEN)
        if(CLIFFT_OPENMP STREQUAL "ON")
            message(FATAL_ERROR "CLIFFT_OPENMP=ON is not supported under Emscripten")
        endif()
    else()
        # Apple Clang does not search Homebrew's libomp prefix by default.
        if(APPLE AND NOT DEFINED OpenMP_ROOT)
            execute_process(
                COMMAND brew --prefix libomp
                OUTPUT_VARIABLE _CLIFFT_BREW_LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _CLIFFT_BREW_LIBOMP_RESULT
            )
            if(_CLIFFT_BREW_LIBOMP_RESULT EQUAL 0 AND _CLIFFT_BREW_LIBOMP_PREFIX)
                set(OpenMP_ROOT "${_CLIFFT_BREW_LIBOMP_PREFIX}")
            endif()
        endif()
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            set(CLIFFT_OPENMP_ENABLED ON)
        elseif(CLIFFT_OPENMP STREQUAL "ON")
            message(FATAL_ERROR "CLIFFT_OPENMP=ON requested, but OpenMP C++ support was not found")
        endif()
    endif()
endif()
message(STATUS "CLIFFT_OPENMP = ${CLIFFT_OPENMP} (enabled: ${CLIFFT_OPENMP_ENABLED})")

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
                    "CLIFFT_CPU_BASELINE=${CLIFFT_CPU_BASELINE} is only valid on x86-64 builds")
            endif()
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
            if(CLIFFT_CPU_BASELINE STREQUAL "x86-64-v2")
                add_compile_options(-march=x86-64-v2)
                message(STATUS "Using x86-64-v2 CPU baseline: -march=x86-64-v2 -ffast-math")
            elseif(CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
                # x86-64-v3 baseline: AVX2 + BMI2 + FMA (~2015 CPUs and later)
                add_compile_options(-march=x86-64-v3)
                message(STATUS "Using x86-64-v3 CPU baseline: -march=x86-64-v3 -ffast-math")
            elseif(CLIFFT_CPU_BASELINE STREQUAL "native")
                add_compile_options(-march=native -mtune=native)
                message(STATUS "Enabled x86 native CPU optimizations: -march=native -mtune=native -ffast-math")
            else()
                message(STATUS "Using generic x86-64 baseline with -ffast-math")
            endif()
        elseif(CLIFFT_CPU_BASELINE STREQUAL "x86-64-v2" OR
               CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
            message(FATAL_ERROR
                "CLIFFT_CPU_BASELINE=${CLIFFT_CPU_BASELINE} is only valid on x86-64 builds")
        elseif(CLIFFT_CPU_BASELINE STREQUAL "native")
            message(STATUS
                "No architecture-specific native tuning configured for "
                "${CMAKE_SYSTEM_PROCESSOR}; using generic optimized flags")
        endif()
    endif()
endif()

# Code coverage option (for use with gcov/lcov)
# Note: Coverage flags are applied per-target in src/clifft/CMakeLists.txt to avoid
# instrumenting Catch2, which would slow builds.
option(CLIFFT_COVERAGE "Enable code coverage instrumentation" OFF)
if(CLIFFT_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(WARNING "Code coverage requires GCC or Clang")
        set(CLIFFT_COVERAGE OFF)
    else()
        message(STATUS "Code coverage enabled (Clifft targets only)")
    endif()
endif()

# x86-64 runtime dispatch compiles the AVX2/AVX-512 kernel translation units.
# The core library and the tests that call those kernels directly must agree
# on this one predicate, so it is decided once here.
if((CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)") AND NOT MSVC)
    set(CLIFFT_X86_RUNTIME_DISPATCH ON)
else()
    set(CLIFFT_X86_RUNTIME_DISPATCH OFF)
endif()

# Fetch production dependencies
include(cmake/FetchFastFloat.cmake)

# Build the core library
add_subdirectory(src/clifft)

# Python bindings (only when building via scikit-build-core)
if(SKBUILD)
    add_subdirectory(src/python)
else()
    option(CLIFFT_BUILD_TESTS "Build the C++ test suite" ON)
    if(CLIFFT_BUILD_TESTS)
        include(cmake/FetchCatch2.cmake)
        enable_testing()
        add_subdirectory(tests)
    endif()

    # Profiling tool (opt-in, not part of default or coverage builds)
    option(CLIFFT_BUILD_PROFILER "Build profiling harnesses" OFF)
    if(CLIFFT_BUILD_PROFILER)
        add_executable(profile_compile tools/profile/profile_compile.cpp)
        target_link_libraries(profile_compile PRIVATE clifft_core)
        target_include_directories(profile_compile PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
        )
        add_executable(profile_probability tools/profile/profile_probability.cpp)
        target_link_libraries(profile_probability PRIVATE clifft_core)
        target_include_directories(profile_probability PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
        )
        add_executable(profile_sample tools/profile/profile_sample.cpp)
        target_link_libraries(profile_sample PRIVATE clifft_core)
        target_include_directories(profile_sample PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
        )
    endif()
endif()
