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

# Fetch external dependencies
include(cmake/FetchStim.cmake)

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
endif()
