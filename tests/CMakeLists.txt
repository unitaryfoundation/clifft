# Clifft Test Suite
# Uses Catch2 v3 for unit testing

include(CTest)
include(Catch)

add_executable(clifft_tests
    test_benchmarks.cc
    test_bitmask.cc
    test_mask_view.cc
    test_page_allocation.cc
    test_parser.cc
    test_stim_contract.cc
    test_hir.cc
    test_frontend.cc
    test_frontend_instruments.cc
    test_sampling_plan.cc
    test_inspection_format.cc
    test_sampling_planner.cc
    test_sampling_planner_frame.cc
    test_sampling_executor.cc
    test_fused_rotation.cc
    test_sampling_kernels.cc
    test_optimizer.cc
    test_source_map.cc
    test_importance_sampling.cc
    test_runtime_isa.cc
    test_shot_parallel.cc
    test_fuzz.cc
    test_noncomp_value_types.cc
    test_noncomp_transition_instrument.cc
    test_noncomp_classifier.cc
    test_noncomp_model.cc
    test_noncomp_status_walk.cc
    test_noncomp_rewriter.cc
    test_noncomp_continuation.cc
    test_trajectory_driver.cc
    test_noncomp_validation.cc
)

target_link_libraries(clifft_tests PRIVATE
    clifft_core
    Catch2::Catch2WithMain
)

# Provide absolute path to test fixture files so benchmarks work from any cwd.
target_compile_definitions(clifft_tests PRIVATE
    CLIFFT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    CLIFFT_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../docs"
)

# Tests that call the x86 kernel entry points directly can only link on
# platforms whose core library compiles the runtime-dispatch kernels.
if(CLIFFT_X86_RUNTIME_DISPATCH)
    target_compile_definitions(clifft_tests PRIVATE CLIFFT_TESTS_HAVE_X86_KERNELS)
endif()

# Code coverage instrumentation for test executable
if(CLIFFT_COVERAGE)
    target_compile_options(clifft_tests PRIVATE --coverage)
    target_link_options(clifft_tests PRIVATE --coverage)
endif()

# Auto-discover Catch2 tests for CTest integration
catch_discover_tests(clifft_tests)
