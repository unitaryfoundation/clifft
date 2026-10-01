# Fetch Catch2 v3 for unit testing

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Catch2)

# Keep Catch2's compiled sources outside Clifft's CI warning-as-error policy.
set_target_properties(Catch2 Catch2WithMain PROPERTIES COMPILE_WARNING_AS_ERROR OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-Wc2y-extensions CLIFFT_HAS_C2Y_EXTENSION_WARNING)
    if(CLIFFT_HAS_C2Y_EXTENSION_WARNING)
        # Catch2 3.5 expands __COUNTER__ in consumer translation units. Keep
        # this exception after directory warning flags, which can re-enable it.
        target_compile_options(Catch2 PUBLIC -Wno-c2y-extensions)
    endif()
endif()

# Add Catch2's CMake helpers for test discovery
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
