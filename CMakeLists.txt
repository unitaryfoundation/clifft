cmake_minimum_required(VERSION 3.20)
project(ucc LANGUAGES CXX)

# Version: parse from pyproject.toml (single source of truth)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENT)
string(REGEX MATCH "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ "${PYPROJECT_CONTENT}")
set(UCC_VERSION "${CMAKE_MATCH_1}")

# Generate version header
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/ucc/util)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ucc/util/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/ucc/util/version.h
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
endif()

# Performance optimizations for Release builds
# Use native CPU features (AVX2, AVX-512 on x86; NEON on ARM) when available.
#
# TODO(PyPI): When building wheels for PyPI distribution, replace -march=native
# with a baseline like -march=x86-64-v3 (AVX2, ~2015 CPUs) to ensure compatibility.
# For now, -march=native is fine since users build from source on their own machine.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Detect architecture for correct native tuning flags
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
            # Apple Silicon and other ARM: use -mcpu=native
            add_compile_options(-mcpu=native -ffast-math)
            message(STATUS "Enabled ARM native CPU optimizations: -mcpu=native -ffast-math")
        else()
            # x86/x86_64: use -march=native -mtune=native
            add_compile_options(-march=native -mtune=native -ffast-math)
            message(STATUS "Enabled x86 native CPU optimizations: -march=native -mtune=native -ffast-math")
        endif()
    endif()
endif()

# Code coverage option (for use with gcov/lcov)
# Note: Coverage flags are applied per-target in src/ucc/CMakeLists.txt to avoid
# instrumenting external dependencies (Stim, Catch2) which would slow builds.
option(UCC_COVERAGE "Enable code coverage instrumentation" OFF)
if(UCC_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(WARNING "Code coverage requires GCC or Clang")
        set(UCC_COVERAGE OFF)
    else()
        message(STATUS "Code coverage enabled (UCC targets only)")
    endif()
endif()

# Fetch external dependencies
include(cmake/FetchStim.cmake)
include(cmake/FetchFastFloat.cmake)

# Build the core library
add_subdirectory(src/ucc)

# Python bindings (only when building via scikit-build-core)
if(SKBUILD)
    add_subdirectory(src/python)
else()
    # Standalone C++ build: include Catch2 and tests
    include(cmake/FetchCatch2.cmake)
    enable_testing()
    add_subdirectory(tests)

    # Profiling tool (opt-in, not part of default or coverage builds)
    option(UCC_BUILD_PROFILER "Build the SVM profiling harness" OFF)
    if(UCC_BUILD_PROFILER)
        add_executable(profile_svm tools/profile/profile_svm.cpp)
        target_link_libraries(profile_svm PRIVATE ucc_core)
        target_include_directories(profile_svm PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
        )
    endif()
endif()
