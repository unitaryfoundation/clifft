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

#include "stim.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <string>

using namespace clifft;

// Resolved at build time by CMake so tests work from any working directory.
#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

static std::string fixture(const char* name) {
    return std::string(CLIFFT_FIXTURES_DIR) + "/" + name;
}

// Compile a parsed Circuit through the full optimizer pipeline.
static CompiledModule compile_parsed(Circuit circuit) {
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

static CompiledModule compile_circuit(const std::string& path) {
    return compile_parsed(parse_file(path));
}

static CompiledModule compile_text(const std::string& text) {
    return compile_parsed(parse(text));
}

// Generate a rotated-Z-memory surface code circuit with uniform noise via Stim.
static std::string surface_code_text(uint32_t distance, uint64_t rounds, double p) {
    stim::CircuitGenParameters params(rounds, distance, "rotated_memory_z");
    params.before_round_data_depolarization = p;
    params.before_measure_flip_probability = p;
    params.after_clifford_depolarization = p;
    params.after_reset_flip_probability = p;
    return stim::generate_surface_code_circuit(params).circuit.str();
}

// EXP_VAL-heavy synthetic circuit: prepares a Clifford state on n qubits,
// then evaluates `num_probes` weight-3 multi-Pauli expectation values per shot.
// Stays at peak_rank=0 (fully Clifford prep) so the cost is dominated by the
// EXP_VAL frame-conjugation path.
static std::string exp_val_heavy_text(uint32_t num_qubits, uint32_t num_probes) {
    std::ostringstream s;
    for (uint32_t q = 0; q < num_qubits; ++q)
        s << "H " << q << "\n";
    for (uint32_t i = 0; i + 1 < num_qubits; ++i)
        s << "CX " << i << " " << (i + 1) << "\n";
    static constexpr const char* kBasis[3] = {"X", "Y", "Z"};
    for (uint32_t i = 0; i < num_probes; ++i) {
        uint32_t q1 = i % num_qubits;
        uint32_t q2 = (i * 7 + 3) % num_qubits;
        uint32_t q3 = (i * 11 + 5) % num_qubits;
        if (q2 == q1)
            q2 = (q2 + 1) % num_qubits;
        if (q3 == q1 || q3 == q2)
            q3 = (q3 + 2) % num_qubits;
        s << "EXP_VAL " << kBasis[i % 3] << q1 << "*" << kBasis[(i / 3) % 3] << q2 << "*"
          << kBasis[(i / 9) % 3] << q3 << "\n";
    }
    return s.str();
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

// ---------------------------------------------------------------------------
// Surface code d=7 r=7 p=1e-3: paper QEC throughput benchmark.
// ~118 qubits, fully Clifford (peak_rank=0), low noise so most NOISE sites
// stay silent. Throughput dominated by frame opcodes and the gap-sampler.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: surface d7 r7 p1e-3 sampling 10000 shots", "[bench]") {
    auto mod = compile_text(surface_code_text(7, 7, 1e-3));
    REQUIRE(mod.peak_rank == 0);
    REQUIRE(mod.num_qubits <= 128);

    BENCHMARK("surface-d7-r7 p=1e-3 x10000 shots") {
        return sample(mod, 10000, 0);
    };
}

// ---------------------------------------------------------------------------
// Surface code d=5 r=5 with high physical noise (p=0.05): forces most NOISE
// sites to fire, exercising the APPLY_PAULI / NOISE full-mask XOR + popcount
// path. Throughput is dominated by the per-fire mask composition.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: surface d5 r5 high-noise APPLY_PAULI heavy", "[bench]") {
    auto mod = compile_text(surface_code_text(5, 5, 0.05));

    BENCHMARK("surface-d5-r5 p=0.05 x10000 shots") {
        return sample(mod, 10000, 0);
    };
}

// ---------------------------------------------------------------------------
// EXP_VAL heavy: 20 qubits, 200 weight-3 multi-Pauli probes per shot.
// Exercises exec_exp_val (frame conjugation + dormant/active split). Each
// probe walks the full mask twice (popcount of x & p_z, z & p_x).
// ---------------------------------------------------------------------------
TEST_CASE("Bench: EXP_VAL 20q 200 probes", "[bench]") {
    auto mod = compile_text(exp_val_heavy_text(20, 200));
    REQUIRE(mod.num_exp_vals == 200);

    BENCHMARK("exp-val 20q 200 probes x100000 shots") {
        return sample(mod, 100000, 0);
    };
}
