// Clifft Performance Benchmarks
//
// Catch2 benchmark tests for regression tracking. Run with:
//   ctest --test-dir build -R Bench
// Or for detailed output:
//   ./build/tests/clifft_tests "[bench]" --benchmark-samples 10
//
// Each benchmark targets ~100ms per iteration to keep CI fast while
// providing meaningful measurements.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/expand_t_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/multi_gate_pass.h"
#include "clifft/optimizer/noise_block_pass.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/single_axis_fusion_pass.h"
#include "clifft/optimizer/swap_meas_pass.h"
#include "clifft/svm/svm.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace clifft;

// Resolved at build time by CMake so tests work from any working directory.
#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

static std::string fixture(const char* name) {
    return std::string(CLIFFT_FIXTURES_DIR) + "/" + name;
}

// Compile a circuit file through the full optimizer pipeline.
static CompiledModule compile_circuit(const std::string& path) {
    auto circuit = parse_file(path);
    auto hir = trace(circuit);
    HirPassManager pm;
    pm.add_pass(std::make_unique<PeepholeFusionPass>());
    pm.run(hir);
    auto mod = lower(hir);
    BytecodePassManager bpm;
    bpm.add_pass(std::make_unique<NoiseBlockPass>());
    bpm.add_pass(std::make_unique<MultiGatePass>());
    bpm.add_pass(std::make_unique<ExpandTPass>());
    bpm.add_pass(std::make_unique<ExpandRotPass>());
    bpm.add_pass(std::make_unique<SwapMeasPass>());
    bpm.add_pass(std::make_unique<SingleAxisFusionPass>());
    bpm.run(mod);
    return mod;
}

// ---------------------------------------------------------------------------
// QV-10: 10 qubits, peak_rank=10, dense SU(4) layers with measurements.
// ~1ms/shot baseline -> 100 shots ~= 100ms.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: QV-10 sampling 100 shots", "[bench]") {
    auto mod = compile_circuit(fixture("qv10.stim"));
    REQUIRE(mod.peak_rank == 10);

    BENCHMARK("QV-10 x100 shots") {
        return sample(mod, 100, 0);
    };
}

// ---------------------------------------------------------------------------
// Magic state cultivation d=5: 42 physical qubits, peak_rank=10,
// sparse QEC with noise, T gates, postselection.
// ~0.09ms/shot baseline -> 1000 shots ~= 90ms.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: cultivation d5 sampling 1000 shots", "[bench]") {
    auto mod = compile_circuit(fixture("cultivation_d5.stim"));
    REQUIRE(mod.peak_rank == 10);

    BENCHMARK("cultivation-d5 x1000 shots") {
        return sample_survivors(mod, 1000, 0, false);
    };
}
