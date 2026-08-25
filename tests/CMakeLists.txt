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
    test_tableau.cc
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
    test_hip_executable_plan.cc
)

target_link_libraries(clifft_tests PRIVATE clifft_hip_plan)

target_link_libraries(clifft_tests PRIVATE
    clifft_core
    Catch2::Catch2WithMain
    libstim
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

if(CLIFFT_OPENMP_ENABLED)
    target_link_libraries(clifft_tests PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(clifft_tests PRIVATE
        CLIFFT_TESTS_HAVE_OPENMP
        CLIFFT_USE_OPENMP
    )
endif()

# Code coverage instrumentation for test executable
if(CLIFFT_COVERAGE)
    target_compile_options(clifft_tests PRIVATE --coverage)
    target_link_options(clifft_tests PRIVATE --coverage)
endif()

catch_discover_tests(clifft_tests)

if(CLIFFT_ENABLE_HIP)
    # Keep hardware-dependent skips scoped to the HIP executable so exit code
    # 4 cannot hide a failure in the ordinary CPU suite.
    add_executable(clifft_hip_tests test_hip_sampler.cc)
    target_link_libraries(clifft_hip_tests PRIVATE
        clifft_core
        clifft_hip_backend
        Catch2::Catch2WithMain
    )
    catch_discover_tests(clifft_hip_tests PROPERTIES SKIP_RETURN_CODE 4)
endif()

if(CLIFFT_OPENMP_ENABLED)
    # These cases need an explicit environment because processor binding and a
    # runtime-limited team are otherwise host-dependent and easy to miss in CI.
    add_test(NAME clifft_openmp_bound_layout_validation
        COMMAND clifft_tests "Sampling thread layouts validate explicit worker counts")
    set_tests_properties(clifft_openmp_bound_layout_validation PROPERTIES
        ENVIRONMENT "OMP_PROC_BIND=spread;OMP_PLACES=cores")

    add_test(NAME clifft_openmp_limited_fixed_rows
        COMMAND clifft_tests "Threaded fixed-row sampling preserves seeded shot order")
    set_tests_properties(clifft_openmp_limited_fixed_rows PROPERTIES
        ENVIRONMENT "OMP_PROC_BIND=spread;OMP_PLACES=cores;OMP_THREAD_LIMIT=1")

    add_test(NAME clifft_openmp_bound_trajectory
        COMMAND clifft_tests "trajectory: worker counts preserve seeded rows and sidecars")
    set_tests_properties(clifft_openmp_bound_trajectory PROPERTIES
        ENVIRONMENT "OMP_PROC_BIND=spread;OMP_PLACES=cores")
endif()
