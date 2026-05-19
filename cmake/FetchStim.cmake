# Fetch Stim v1.15.0 from upstream
# Stim is used as an unmodified tableau math library

include(FetchContent)

FetchContent_Declare(
    stim
    GIT_REPOSITORY https://github.com/quantumlib/Stim.git
    GIT_TAG        v1.15.0
    GIT_SHALLOW    TRUE
)

# Stim build options - we only need the core library
set(STIM_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(STIM_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Pin stim's SIMD width so libstim's machine flag matches Clifft's
# selected baseline, instead of stim's default -march=native. The default
# would otherwise emit whatever ISA the build host supports (AVX-512 on
# Skylake-SP / Ice Lake-SP), and ccache happily returns those .o files to
# a later AMD Zen 3 runner with the same -march=native command line --
# the freshly linked binary then SIGILLs at catch_discover_tests on the
# AMD side. The same bug would ship to PyPI users on sub-AVX-512 CPUs if
# the release-builder runner happened to have AVX-512.
#
# Stim's CMakeLists maps SIMD_WIDTH:
#   256 -> -mavx2 -msse2    (matches x86-64-v3 baseline)
#   128 -> -mno-avx2 -msse2 (matches SSE2-only "generic" baseline)
#    64 -> -mno-avx2 -mno-sse2
# Unset on x86 -> -march=native (the bug we are avoiding).
#
# We only override on x86_64 -- on ARM64 etc. stim leaves MACHINE_FLAG empty.
# `native` is intentionally left unset: the user opted into host-tuned
# code, so stim probing the build host matches user intent.
# Respect a user-supplied SIMD_WIDTH (no FORCE) so callers can override.
if(NOT DEFINED CACHE{SIMD_WIDTH})
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
        if(CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
            set(SIMD_WIDTH 256 CACHE STRING
                "Pinned by Clifft for x86-64-v3: libstim uses -mavx2 -msse2.")
        elseif(CLIFFT_CPU_BASELINE STREQUAL "generic")
            set(SIMD_WIDTH 128 CACHE STRING
                "Pinned by Clifft for generic baseline: libstim uses -msse2 only.")
        endif()
        # CLIFFT_CPU_BASELINE=native: leave SIMD_WIDTH unset so stim's
        # libstim also tunes to the build host, matching Clifft's intent.
    endif()
endif()

# Use FetchContent_GetProperties + add_subdirectory(EXCLUDE_FROM_ALL)
# instead of FetchContent_MakeAvailable so install rules are also excluded.
# This prevents CMake from trying to install stim executables we don't build.
FetchContent_GetProperties(stim)
if(NOT stim_POPULATED)
    FetchContent_Populate(stim)
    add_subdirectory(${stim_SOURCE_DIR} ${stim_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
