# UCC Core Library
# This library contains the compiler pipeline and VM runtime.

add_library(ucc_core STATIC
    circuit/parser.cc
    frontend/frontend.cc
    optimizer/bytecode_pass.cc
    optimizer/noise_block_pass.cc
    optimizer/multi_gate_pass.cc
    optimizer/expand_t_pass.cc
    optimizer/swap_meas_pass.cc
    optimizer/single_axis_fusion_pass.cc
    optimizer/hir_pass_manager.cc
    optimizer/peephole.cc
    optimizer/remove_noise_pass.cc
    backend/backend.cc
    backend/reference_syndrome.cc
    svm/svm.cc
    svm/svm_scalar.cc
    svm/svm_state.cc
    util/introspection.cc
)

# x86-64 runtime dispatch: compile AVX2 and AVX-512 TUs with SIMD flags.
if((CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)") AND NOT MSVC)
    target_sources(ucc_core PRIVATE svm/svm_avx2.cc svm/svm_avx512.cc)
    set_source_files_properties(svm/svm_avx2.cc
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mbmi2;-mfma"
    )
    set_source_files_properties(svm/svm_avx512.cc
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mbmi2;-mfma;-mavx512f;-mavx512dq"
    )
    target_compile_definitions(ucc_core PRIVATE UCC_ENABLE_RUNTIME_DISPATCH)
endif()

# Enable position-independent code for use in shared libraries (Python bindings).
set_target_properties(ucc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ucc_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries(ucc_core PUBLIC
    libstim
)

target_link_libraries(ucc_core PRIVATE fast_float)

if(DEFINED UCC_MAX_QUBITS)
    target_compile_definitions(ucc_core PUBLIC UCC_MAX_QUBITS=${UCC_MAX_QUBITS})
endif()

target_compile_features(ucc_core PUBLIC cxx_std_20)

# Code coverage instrumentation (only for UCC code, not dependencies)
if(UCC_COVERAGE)
    target_compile_options(ucc_core PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_options(ucc_core PRIVATE --coverage)
endif()
