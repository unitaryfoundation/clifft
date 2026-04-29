# Clifft Core Library
# This library contains the compiler pipeline and VM runtime.

add_library(clifft_core STATIC
    circuit/parser.cc
    frontend/frontend.cc
    optimizer/bytecode_pass.cc
    optimizer/noise_block_pass.cc
    optimizer/multi_gate_pass.cc
    optimizer/expand_t_pass.cc
    optimizer/swap_meas_pass.cc
    optimizer/tile_axis_fusion_pass.cc
    optimizer/single_axis_fusion_pass.cc
    optimizer/hir_pass_manager.cc
    optimizer/commutation.cc
    optimizer/peephole.cc
    optimizer/pass_factory.cc
    optimizer/remove_noise_pass.cc
    optimizer/statevector_squeeze_pass.cc
    backend/backend.cc
    api/reference_syndrome.cc
    svm/svm.cc
    svm/svm_scalar.cc
    svm/svm_state.cc
    util/introspection.cc
)

# x86-64 runtime dispatch: compile AVX2 and AVX-512 TUs with SIMD flags.
if((CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)") AND NOT MSVC)
    target_sources(clifft_core PRIVATE svm/svm_avx2.cc svm/svm_avx512.cc)
    set_source_files_properties(svm/svm_avx2.cc
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mbmi2;-mfma"
    )
    set_source_files_properties(svm/svm_avx512.cc
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mbmi2;-mfma;-mavx512f;-mavx512dq"
    )
    target_compile_definitions(clifft_core PRIVATE CLIFFT_ENABLE_RUNTIME_DISPATCH)
endif()

# Enable position-independent code for use in shared libraries (Python bindings).
set_target_properties(clifft_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(clifft_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries(clifft_core PUBLIC
    libstim
)

target_link_libraries(clifft_core PRIVATE fast_float)

# Optional OpenMP for multi-core array parallelism (large statevectors).
# Not available under Emscripten (WASM). When absent, OMP pragmas are
# silently ignored and all paths remain single-threaded.
if(NOT EMSCRIPTEN)
    # On macOS with Apple clang, Homebrew's libomp is not on the default
    # search path. Point CMake at it so find_package(OpenMP) succeeds.
    if(APPLE AND NOT DEFINED OpenMP_ROOT)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE _BREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _BREW_LIBOMP_RESULT
        )
        if(_BREW_LIBOMP_RESULT EQUAL 0 AND _BREW_LIBOMP_PREFIX)
            set(OpenMP_ROOT "${_BREW_LIBOMP_PREFIX}")
        endif()
    endif()

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(clifft_core PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(DEFINED CLIFFT_MAX_QUBITS)
    target_compile_definitions(clifft_core PUBLIC CLIFFT_MAX_QUBITS=${CLIFFT_MAX_QUBITS})
endif()

target_compile_features(clifft_core PUBLIC cxx_std_20)

# Code coverage instrumentation (only for Clifft code, not dependencies)
if(CLIFFT_COVERAGE)
    target_compile_options(clifft_core PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_options(clifft_core PRIVATE --coverage)
endif()
