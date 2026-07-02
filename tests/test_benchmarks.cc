// Clifft Performance Benchmarks
//
// Opt-in performance regression checks. Excluded from the default
// correctness/coverage CI; the cases below run Catch2's default sample
// count (100) at multi-hundred-millisecond per-iteration cost, which
// adds tens of minutes of wall-clock without contributing coverage
// signal that the unit and integration suites do not already provide.
//
// Run explicitly via ctest:
//   ctest --test-dir build -R Bench
//
// Or directly through the test binary for detailed Catch2 output:
//   ./build/tests/clifft_tests "[bench]" --benchmark-samples 10

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/expand_t_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/multi_gate_pass.h"
#include "clifft/optimizer/noise_block_pass.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/single_axis_fusion_pass.h"
#include "clifft/optimizer/swap_meas_pass.h"
#include "clifft/svm/svm.h"

#include "stim.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
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
// Surface code d=11 r=11 p=1e-3: ~242 qubits, two 64-bit Pauli mask
// words. Sits above the single-word frame regime to give a regression
// baseline for the runtime-width path.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: surface d11 r11 p1e-3 sampling 1000 shots", "[bench]") {
    auto mod = compile_text(surface_code_text(11, 11, 1e-3));
    REQUIRE(mod.peak_rank == 0);
    REQUIRE(mod.num_qubits > 128);

    BENCHMARK("surface-d11-r11 p=1e-3 x1000 shots") {
        return sample(mod, 1000, 0);
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

    // BENCHMARK names must fit in Catch2's console-reporter name column
    // (~35 chars). Longer names wrap onto two lines and break the
    // bench-history workflow's parser (.github/workflows/bench.yml).
    BENCHMARK("exp-val 20q 200 probes x100k") {
        return sample(mod, 100000, 0);
    };
}

// ---------------------------------------------------------------------------
// Fence spike: cost of segmenting the default pass pipelines at every noise
// site -- the realistic fence density for per-site instrument fences, where
// no pass may observe, fuse, or move operations across a site. Each workload
// compares compile time and sampling throughput, fenced vs unfenced, and
// prints the module-shape deltas (instruction count, noise coalescing,
// peak rank).
// ---------------------------------------------------------------------------

// Fence density proxies for the spike. Instruments attach per (annotated
// operation, operand), so the truth for a layered circuit lies between:
//   EveryNoise    -- a fence at every noise site (atomized; hard upper bound:
//                    even layer-internal noise runs stop coalescing).
//   NoiseRunStarts -- a fence at the first site of each contiguous noise run
//                    (per-layer; run-internal coalescing survives, matching
//                    instruments clustered at the gates before the layer).
enum class FenceDensity { Unfenced, EveryNoise, NoiseRunStarts };

static CompiledModule compile_default_pipeline(const Circuit& circuit, FenceDensity density) {
    HirModule hir = trace(circuit);
    HirPassManager pm = default_hir_pass_manager();
    switch (density) {
        case FenceDensity::Unfenced:
            pm.run(hir);
            break;
        case FenceDensity::EveryNoise:
            pm.run_segmented(hir,
                             [](const HeisenbergOp& op) { return op.op_type() == OpType::NOISE; });
            break;
        case FenceDensity::NoiseRunStarts:
            pm.run_segmented(hir, [prev = false](const HeisenbergOp& op) mutable {
                const bool is_noise = op.op_type() == OpType::NOISE;
                const bool fence = is_noise && !prev;
                prev = is_noise;
                return fence;
            });
            break;
    }
    CompiledModule mod = lower(hir);
    BytecodePassManager bpm = default_bytecode_pass_manager();
    switch (density) {
        case FenceDensity::Unfenced:
            bpm.run(mod);
            break;
        case FenceDensity::EveryNoise:
            bpm.run_segmented(mod,
                              [](const Instruction& in) { return in.opcode == Opcode::OP_NOISE; });
            break;
        case FenceDensity::NoiseRunStarts:
            bpm.run_segmented(mod, [prev = false](const Instruction& in) mutable {
                const bool is_noise = in.opcode == Opcode::OP_NOISE;
                const bool fence = is_noise && !prev;
                prev = is_noise;
                return fence;
            });
            break;
    }
    return mod;
}

static size_t count_opcode(const CompiledModule& m, Opcode op) {
    size_t n = 0;
    for (const Instruction& in : m.bytecode) {
        if (in.opcode == op)
            ++n;
    }
    return n;
}

static void print_fence_shape(const char* name, const CompiledModule& plain,
                              const CompiledModule& fenced) {
    std::printf(
        "fence-spike %s: bytecode %zu -> %zu, noise %zu -> %zu, "
        "noise-block %zu -> %zu, peak_rank %u -> %u\n",
        name, plain.bytecode.size(), fenced.bytecode.size(), count_opcode(plain, Opcode::OP_NOISE),
        count_opcode(fenced, Opcode::OP_NOISE), count_opcode(plain, Opcode::OP_NOISE_BLOCK),
        count_opcode(fenced, Opcode::OP_NOISE_BLOCK), plain.peak_rank, fenced.peak_rank);
}

TEST_CASE("Bench: fence spike surface d7 r7", "[bench]") {
    Circuit c = parse(surface_code_text(7, 7, 1e-3));
    auto plain = compile_default_pipeline(c, FenceDensity::Unfenced);
    auto fenced = compile_default_pipeline(c, FenceDensity::EveryNoise);
    auto layered = compile_default_pipeline(c, FenceDensity::NoiseRunStarts);
    print_fence_shape("surface-d7-r7 p=1e-3 atomized", plain, fenced);
    print_fence_shape("surface-d7-r7 p=1e-3 run-start", plain, layered);

    BENCHMARK("d7r7 compile unfenced") {
        return compile_default_pipeline(c, FenceDensity::Unfenced);
    };
    BENCHMARK("d7r7 compile fenced") {
        return compile_default_pipeline(c, FenceDensity::EveryNoise);
    };
    BENCHMARK("d7r7 x2000 shots unfenced") {
        return sample(plain, 2000, 0);
    };
    BENCHMARK("d7r7 x2000 shots fenced") {
        return sample(fenced, 2000, 0);
    };
    BENCHMARK("d7r7 x2000 shots run-fenced") {
        return sample(layered, 2000, 0);
    };
}

TEST_CASE("Bench: fence spike surface d5 r5 high noise", "[bench]") {
    Circuit c = parse(surface_code_text(5, 5, 0.05));
    auto plain = compile_default_pipeline(c, FenceDensity::Unfenced);
    auto fenced = compile_default_pipeline(c, FenceDensity::EveryNoise);
    auto layered = compile_default_pipeline(c, FenceDensity::NoiseRunStarts);
    print_fence_shape("surface-d5-r5 p=0.05 atomized", plain, fenced);
    print_fence_shape("surface-d5-r5 p=0.05 run-start", plain, layered);

    BENCHMARK("d5r5 hi-noise x5000 unfenced") {
        return sample(plain, 5000, 0);
    };
    BENCHMARK("d5r5 hi-noise x5000 fenced") {
        return sample(fenced, 5000, 0);
    };
    BENCHMARK("d5r5 hi-noise x5000 run-fenced") {
        return sample(layered, 5000, 0);
    };
}

TEST_CASE("Bench: fence spike cultivation d5", "[bench]") {
    Circuit c = parse_file(fixture("cultivation_d5.stim"));
    auto plain = compile_default_pipeline(c, FenceDensity::Unfenced);
    auto fenced = compile_default_pipeline(c, FenceDensity::EveryNoise);
    auto layered = compile_default_pipeline(c, FenceDensity::NoiseRunStarts);
    print_fence_shape("cultivation-d5 atomized", plain, fenced);
    print_fence_shape("cultivation-d5 run-start", plain, layered);

    BENCHMARK("cultivation compile unfenced") {
        return compile_default_pipeline(c, FenceDensity::Unfenced);
    };
    BENCHMARK("cultivation compile fenced") {
        return compile_default_pipeline(c, FenceDensity::EveryNoise);
    };
    BENCHMARK("cultivation x1000 unfenced") {
        return sample_survivors(plain, 1000, 0, false);
    };
    BENCHMARK("cultivation x1000 fenced") {
        return sample_survivors(fenced, 1000, 0, false);
    };
    BENCHMARK("cultivation x1000 run-fenced") {
        return sample_survivors(layered, 1000, 0, false);
    };
}
