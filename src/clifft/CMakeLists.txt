# Clifft Core Library
# This library contains the compiler pipeline and sampling runtime.

add_library(clifft_core STATIC
    circuit/parser.cc
    circuit/qasm2_parser.cc
    tableau/pauli_string.cc
    tableau/tableau.cc
    frontend/frontend.cc
    optimizer/hir_pass_manager.cc
    optimizer/active_width_analysis.cc
    optimizer/active_width_closure.cc
    optimizer/drop_non_unitary_pass.cc
    optimizer/commutation.cc
    optimizer/peephole.cc
    optimizer/pass_factory.cc
    optimizer/remove_noise_pass.cc
    optimizer/schedule_dependence.cc
    optimizer/active_width_search.cc
    optimizer/active_width_schedule_pass.cc
    optimizer/statevector_squeeze_pass.cc
    sampling/plan.cc
    sampling/plan_inspection.cc
    sampling/inspection_format.cc
    sampling/planner.cc
    sampling/planner_frame.cc
    sampling/pauli_preparation.cc
    sampling/batch/bits.cc
    sampling/batch/executor.cc
    sampling/batch/policy.cc
    sampling/batch/presampled_program.cc
    sampling/kernel_dispatch.cc
    sampling/executable_plan.cc
    sampling/executable_plan_inspection.cc
    sampling/executable_plan_builder.cc
    sampling/executor.cc
    sampling/fused_rotation.cc
    sampling/batch/interleaved_kernels.cc
    sampling/batch/interleaved_state.cc
    sampling/kernels.cc
    sampling/sampler.cc
    sampling/state.cc
    api/basis_probabilities.cc
    api/reference_syndrome.cc
    api/statevector.cc
    util/fault_sampling.cc
    util/hir_introspection.cc
    util/page_allocation.cc
    util/runtime_isa.cc
    util/xoshiro.cc
    noncomp/transition_instrument.cc
    noncomp/transition_hooks.cc
    noncomp/instrument_options.cc
    noncomp/classifier.cc
    noncomp/model.cc
    noncomp/status_walk.cc
    noncomp/rewriter.cc
    noncomp/trajectory_driver.cc
    noncomp/sample.cc
)

# x86-64 runtime dispatch: compile AVX2 and AVX-512 TUs with SIMD flags.
if(CLIFFT_X86_RUNTIME_DISPATCH)
    target_sources(clifft_core PRIVATE
        sampling/kernels_avx2.cc
        sampling/kernels_avx512.cc
    )
    set_source_files_properties(sampling/kernels_avx2.cc
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mbmi2;-mfma"
    )
    set_source_files_properties(sampling/kernels_avx512.cc
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mbmi2;-mfma;-mavx512f;-mavx512dq"
    )
    target_compile_definitions(clifft_core PRIVATE CLIFFT_ENABLE_X86_RUNTIME_DISPATCH)
endif()

# Apple arm64 has a process-wide NEON baseline. Keep its implementation in a
# separate TU so portable code does not expose architecture-specific types.
if(CLIFFT_APPLE_NEON_DISPATCH)
    target_sources(clifft_core PRIVATE sampling/kernels_neon.cc)
    target_compile_definitions(clifft_core PRIVATE CLIFFT_ENABLE_APPLE_NEON)
endif()

# Enable position-independent code for use in shared libraries (Python bindings).
set_target_properties(clifft_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(clifft_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries(clifft_core PRIVATE fast_float)

find_package(Threads REQUIRED)
target_link_libraries(clifft_core PUBLIC Threads::Threads)

if(CLIFFT_OPENMP_ENABLED)
    target_link_libraries(clifft_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(clifft_core PRIVATE CLIFFT_USE_OPENMP)
endif()

target_compile_features(clifft_core PUBLIC cxx_std_20)

add_library(clifft_hip_plan STATIC
    sampling/hip/executable_plan.cc
)
set_target_properties(clifft_hip_plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(clifft_hip_plan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_BINARY_DIR}/generated
)
target_link_libraries(clifft_hip_plan PUBLIC clifft_core)
target_compile_features(clifft_hip_plan PUBLIC cxx_std_20)

if(CLIFFT_ENABLE_HIP)
    add_library(clifft_hip_backend STATIC sampling/hip/sampler.hip)
    set_target_properties(clifft_hip_backend PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED ON
    )
    target_include_directories(clifft_hip_backend PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_BINARY_DIR}/generated
    )
    target_link_libraries(clifft_hip_backend PUBLIC clifft_hip_plan)
    target_compile_features(clifft_hip_backend PUBLIC cxx_std_20)
    target_compile_options(clifft_hip_backend PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math;-ffp-contract=off>
    )
endif()

# Code coverage instrumentation (only for Clifft code, not dependencies)
if(CLIFFT_COVERAGE)
    target_compile_options(clifft_core PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_options(clifft_core PRIVATE --coverage)
endif()
