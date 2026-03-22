# UCC Test Suite
# Uses Catch2 v3 for unit testing

include(CTest)
include(Catch)

add_executable(ucc_tests
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
)

target_link_libraries(ucc_tests PRIVATE
    ucc_core
    Catch2::Catch2WithMain
)

# Provide absolute path to test fixture files so benchmarks work from any cwd.
target_compile_definitions(ucc_tests PRIVATE
    UCC_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

# Code coverage instrumentation for test executable
if(UCC_COVERAGE)
    target_compile_options(ucc_tests PRIVATE --coverage)
    target_link_options(ucc_tests PRIVATE --coverage)
endif()

# Auto-discover Catch2 tests for CTest integration
catch_discover_tests(ucc_tests)
