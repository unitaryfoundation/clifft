# Fetch Stim v1.15.0 only for the independent C++ test oracle.

include(FetchContent)

FetchContent_Declare(
    stim
    GIT_REPOSITORY https://github.com/quantumlib/Stim.git
    GIT_TAG        v1.15.0
    GIT_SHALLOW    TRUE
)

set(STIM_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(STIM_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Match the test oracle's machine flags to Clifft's selected baseline. This
# prevents ccache entries built on an AVX-512 host from failing on a narrower
# x86 runner while still respecting native host tuning when explicitly chosen.
if(NOT DEFINED CACHE{SIMD_WIDTH})
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
        if(CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
            set(SIMD_WIDTH 256 CACHE STRING
                "Pinned by Clifft tests for x86-64-v3: Stim uses -mavx2 -msse2.")
        elseif(CLIFFT_CPU_BASELINE STREQUAL "x86-64-v2" OR
               CLIFFT_CPU_BASELINE STREQUAL "generic")
            set(SIMD_WIDTH 128 CACHE STRING
                "Pinned by Clifft tests for a portable x86 baseline: Stim uses -msse2 only.")
        endif()
    endif()
endif()

# Excluding the upstream directory from the default install prevents test-only
# Stim targets from entering Clifft packages.
FetchContent_GetProperties(stim)
if(NOT stim_POPULATED)
    FetchContent_Populate(stim)
    add_subdirectory(${stim_SOURCE_DIR} ${stim_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
