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

# Use FetchContent_GetProperties + add_subdirectory(EXCLUDE_FROM_ALL)
# instead of FetchContent_MakeAvailable so install rules are also excluded.
# This prevents CMake from trying to install stim executables we don't build.
FetchContent_GetProperties(stim)
if(NOT stim_POPULATED)
    FetchContent_Populate(stim)
    add_subdirectory(${stim_SOURCE_DIR} ${stim_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
