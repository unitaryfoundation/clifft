# UCC Test Suite
# Uses Catch2 v3 for unit testing

include(CTest)
include(Catch)

add_executable(ucc_tests
    test_bitmask.cc
    test_parser.cc
    test_stim_contract.cc
    test_hir.cc
    test_frontend.cc
    test_backend.cc
    test_svm_risc.cc
    test_statevector.cc
    test_optimizer.cc
    test_bytecode_passes.cc
    test_source_map.cc
)

target_link_libraries(ucc_tests PRIVATE
    ucc_core
    Catch2::Catch2WithMain
)

# Code coverage instrumentation for test executable
if(UCC_COVERAGE)
    target_compile_options(ucc_tests PRIVATE --coverage)
    target_link_options(ucc_tests PRIVATE --coverage)
endif()

# Auto-discover Catch2 tests for CTest integration
catch_discover_tests(ucc_tests)
