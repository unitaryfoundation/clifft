# Clifft Test Suite
# Uses Catch2 v3 for unit testing

include(CTest)
include(Catch)

add_executable(clifft_tests
    test_benchmarks.cc
    test_bitmask.cc
    test_parser.cc
    test_stim_contract.cc
    test_hir.cc
    test_frontend.cc
    test_backend.cc
    test_svm_risc.cc
    test_svm_avx512_axis01.cc
    test_statevector.cc
    test_optimizer.cc
    test_bytecode_passes.cc
    test_single_axis_fusion.cc
    test_tile_axis_fusion.cc
    test_source_map.cc
    test_importance_sampling.cc
    test_exp_value.cc
    test_openmp.cc
)

target_link_libraries(clifft_tests PRIVATE
    clifft_core
    Catch2::Catch2WithMain
)

# Provide absolute path to test fixture files so benchmarks work from any cwd.
target_compile_definitions(clifft_tests PRIVATE
    CLIFFT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

# Code coverage instrumentation for test executable
if(CLIFFT_COVERAGE)
    target_compile_options(clifft_tests PRIVATE --coverage)
    target_link_options(clifft_tests PRIVATE --coverage)
endif()

# Auto-discover Catch2 tests for CTest integration
catch_discover_tests(clifft_tests)
