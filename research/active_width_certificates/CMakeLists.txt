cmake_minimum_required(VERSION 3.20)
project(active_width_certificates LANGUAGES CXX)

# This tool certifies properties of a clifft checkout; two levels up from
# research/active_width_certificates/ is the repository root, matching
# benchmarks/CMakeLists.txt's own standalone-driver convention.
get_filename_component(_CLIFFT_DEFAULT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CLIFFT_SOURCE_DIR "${_CLIFFT_DEFAULT_SOURCE_DIR}" CACHE PATH
    "Clifft source tree this tool searches over")

set(CLIFFT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CLIFFT_BUILD_PROFILER OFF CACHE BOOL "" FORCE)
set(CLIFFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory("${CLIFFT_SOURCE_DIR}" clifft-source)
include("${CLIFFT_SOURCE_DIR}/cmake/FetchCatch2.cmake")

# The search itself, reusable by both the driver and the tests, so it is
# compiled once here rather than duplicated into each target's source list.
add_library(active_width_search STATIC
    active_width_search.cc
)
target_include_directories(active_width_search PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(active_width_search PUBLIC clifft_core)
target_compile_features(active_width_search PUBLIC cxx_std_20)

add_executable(width_certificate width_certificate.cc)
target_link_libraries(width_certificate PRIVATE active_width_search)
target_compile_features(width_certificate PRIVATE cxx_std_20)

enable_testing()

# The test file reaches into the main tree's tests/ for
# sampling_equivalence_helpers.h, test_helpers.h, and the fixtures directory,
# rather than duplicating them: those helpers are shared with the library's
# own schedule-dependence tests and drifting a second copy would be worse
# than the cross-tree include.
get_filename_component(_CLIFFT_TEST_DIR "${CLIFFT_SOURCE_DIR}/tests" ABSOLUTE)

add_executable(width_certificate_tests test_active_width_search.cc)
target_link_libraries(width_certificate_tests PRIVATE
    active_width_search
    Catch2::Catch2WithMain
)
target_include_directories(width_certificate_tests PRIVATE "${_CLIFFT_TEST_DIR}")
target_compile_definitions(width_certificate_tests PRIVATE
    CLIFFT_FIXTURES_DIR="${_CLIFFT_TEST_DIR}/fixtures"
)
target_compile_features(width_certificate_tests PRIVATE cxx_std_20)

# One CTest entry running the whole Catch2 binary, rather than
# catch_discover_tests's per-case registration: this project has no CI job of
# its own to benefit from per-case granularity.
add_test(NAME width_certificate_tests COMMAND width_certificate_tests)
