# ClifftVersion.cmake -- resolve the project version.
#
# Sets CLIFFT_VERSION in the caller's scope.
#
# Set CLIFFT_PROJECT_ROOT before including this file. Defaults to
# CMAKE_CURRENT_SOURCE_DIR if not set.
#
# Priority:
#   1. SKBUILD_PROJECT_VERSION_FULL  (set by scikit-build-core during pip install)
#   2. git describe --tags            (standalone dev builds and CI)
#   3. "0.0.0"                        (absolute fallback)

if(NOT DEFINED CLIFFT_PROJECT_ROOT)
    set(CLIFFT_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# When building via scikit-build-core (pip install / uv build), the version
# is passed from setuptools-scm through scikit-build-core as a CMake cache
# variable. Use the full version string so dev builds include the suffix.
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
    set(CLIFFT_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
    message(STATUS "Clifft version from scikit-build-core: ${CLIFFT_VERSION}")
    return()
endif()

# Standalone C++ builds: read the version from git describe.
# Tagged commits:  v0.2.0        -> "0.2.0"
# Dev commits:     v0.2.0-3-gabc -> "0.2.0.dev3+gabc"
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --tags --match "v[0-9]*"
        WORKING_DIRECTORY "${CLIFFT_PROJECT_ROOT}"
        OUTPUT_VARIABLE _git_describe
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_rc
    )
    if(_git_rc EQUAL 0)
        if(_git_describe MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)-([0-9]+)-(g[0-9a-f]+)$")
            # Dev build: v0.2.0-3-gabc -> 0.2.0.dev3+gabc
            set(CLIFFT_VERSION "${CMAKE_MATCH_1}.dev${CMAKE_MATCH_2}+${CMAKE_MATCH_3}")
        elseif(_git_describe MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)$")
            # Exact tag: v0.2.0 -> 0.2.0
            set(CLIFFT_VERSION "${CMAKE_MATCH_1}")
        endif()
        if(DEFINED CLIFFT_VERSION)
            message(STATUS "Clifft version from git: ${CLIFFT_VERSION}")
            return()
        endif()
    endif()
endif()

set(CLIFFT_VERSION "0.0.0")
message(WARNING "Could not determine Clifft version from git, using ${CLIFFT_VERSION}")
