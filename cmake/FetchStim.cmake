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

FetchContent_MakeAvailable(stim)
