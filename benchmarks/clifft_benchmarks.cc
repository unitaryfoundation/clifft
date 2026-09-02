#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_schedule_pass.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

#include <benchmark/benchmark.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace clifft;

namespace {

std::string fixture(const char* name) {
    const auto path = std::filesystem::path(CLIFFT_BENCHMARK_FIXTURES_DIR) / name;
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("benchmark fixture is missing: " + path.string());
    }
    return path.string();
}

std::string read_fixture(const char* name) {
    std::ifstream input(fixture(name));
    if (!input) {
        throw std::runtime_error("could not read benchmark fixture");
    }
    std::ostringstream source;
    source << input.rdbuf();
    return source.str();
}

sampling::ExecutablePlan compile_parsed(Circuit circuit) {
    auto hir = trace(circuit);
    default_hir_pass_manager().run(hir);
    return sampling::ExecutablePlan(sampling::plan_sampling(hir));
}

sampling::ExecutablePlan compile_circuit(const std::string& path) {
    return compile_parsed(parse_file(path));
}

sampling::ExecutablePlan compile_text(const std::string& text) {
    return compile_parsed(parse(text));
}

HirModule parallel_t_hir(uint32_t num_qubits) {
    std::ostringstream source;
    source << "T";
    for (uint32_t qubit = 0; qubit < num_qubits; ++qubit) {
        source << " " << qubit;
    }
    return trace(parse(source.str()));
}

std::string exp_val_heavy_text(uint32_t num_qubits, uint32_t num_probes) {
    std::ostringstream source;
    for (uint32_t qubit = 0; qubit < num_qubits; ++qubit) {
        source << "H " << qubit << "\n";
    }
    for (uint32_t qubit = 0; qubit + 1 < num_qubits; ++qubit) {
        source << "CX " << qubit << " " << (qubit + 1) << "\n";
    }
    static constexpr const char* basis[3] = {"X", "Y", "Z"};
    for (uint32_t probe = 0; probe < num_probes; ++probe) {
        uint32_t q1 = probe % num_qubits;
        uint32_t q2 = (probe * 7 + 3) % num_qubits;
        uint32_t q3 = (probe * 11 + 5) % num_qubits;
        if (q2 == q1) {
            q2 = (q2 + 1) % num_qubits;
        }
        if (q3 == q1 || q3 == q2) {
            q3 = (q3 + 2) % num_qubits;
        }
        source << "EXP_VAL " << basis[probe % 3] << q1 << "*" << basis[(probe / 3) % 3] << q2 << "*"
               << basis[(probe / 9) % 3] << q3 << "\n";
    }
    return source.str();
}

// Expansion-only suffixes have no useful bypass destination. This wide convoy
// catches accidental per-T full-mask commutation scans in squeeze planning.
void squeeze_parallel_t(benchmark::State& state) {
    auto hir = parallel_t_hir(8192);
    if (hir.ops.size() != 8192) {
        state.SkipWithError("unexpected parallel-T HIR size");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        StatevectorSqueezePass{}.run(hir);
        benchmark::DoNotOptimize(hir.ops.size());
    }
}

// Cultivation combines parsing, optimization, postselection planning, and
// active-state lowering, protecting the complete compiler pipeline's latency.
void compile_plan_cultivation_d5(benchmark::State& state) {
    // Keep filesystem behavior outside the compiler pipeline being measured.
    const auto source = read_fixture("cultivation_d5.stim");
    const auto validation = compile_text(source);
    if (validation.peak_active_width() != 10) {
        state.SkipWithError("unexpected cultivation active width");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        auto plan = compile_text(source);
        benchmark::DoNotOptimize(plan);
    }
}

// Prepares the incumbent ActiveWidthSchedulePass::run() times below: parsed
// and traced, then PeepholeFusionPass and StatevectorSqueezePass, matching
// the pipeline the pass is documented to follow and never the full default
// manager, so a later default-pipeline change cannot fold the scheduling
// pass into this preparation step and time it twice.
HirModule prepare_for_schedule_pass(const char* fixture_name) {
    auto hir = trace(parse_file(fixture(fixture_name)));
    PeepholeFusionPass{}.run(hir);
    StatevectorSqueezePass{}.run(hir);
    return hir;
}

// coherent_d5_r5 is the corpus's slow fixture for this pass: its beam
// search branches on tens of simultaneously-ready, mutually independent
// expanding rotations at many steps, so this benchmark protects the
// two-phase scoring cost that keeps that branching affordable. Each
// iteration copies the prepared HIR fresh so it always times a first-time
// schedule rather than a no-op repeat on an already-scheduled HIR.
void schedule_pass_coherent_d5_r5(benchmark::State& state) {
    const HirModule prepared = prepare_for_schedule_pass("coherent_d5_r5.stim");
    HirModule validation = prepared;
    ActiveWidthSchedulePass validation_pass;
    validation_pass.run(validation);
    if (validation_pass.result_peak() != 13) {
        state.SkipWithError("unexpected coherent d5 r5 scheduled peak");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        HirModule hir = prepared;
        benchmark::DoNotOptimize(hir);
        ActiveWidthSchedulePass{}.run(hir);
        benchmark::DoNotOptimize(hir.ops.size());
    }
}

// cultivation_d5 is far cheaper than coherent_d5_r5 for this pass, so this
// benchmark protects the pass's typical cost separately from its
// worst-case cost above.
void schedule_pass_cultivation_d5(benchmark::State& state) {
    const HirModule prepared = prepare_for_schedule_pass("cultivation_d5.stim");
    HirModule validation = prepared;
    ActiveWidthSchedulePass validation_pass;
    validation_pass.run(validation);
    if (validation_pass.result_peak() != 10) {
        state.SkipWithError("unexpected cultivation d5 scheduled peak");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        HirModule hir = prepared;
        benchmark::DoNotOptimize(hir);
        ActiveWidthSchedulePass{}.run(hir);
        benchmark::DoNotOptimize(hir.ops.size());
    }
}

// Dense arbitrary two-qubit layers drive the width-10 coefficient-state path
// instead of the mostly Clifford and noise-oriented execution paths below.
void sample_qv10(benchmark::State& state) {
    const auto plan = compile_circuit(fixture("qv10.stim"));
    if (plan.peak_active_width() != 10) {
        state.SkipWithError("unexpected QV-10 active width");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        auto result = sampling::sample(plan, 100, 0);
        benchmark::DoNotOptimize(result);
    }
}

// Sparse noisy QEC with T gates and postselection protects survivor sampling
// at moderate active width.
void sample_cultivation_d5(benchmark::State& state) {
    const auto plan = compile_circuit(fixture("cultivation_d5.stim"));
    if (plan.peak_active_width() != 10) {
        state.SkipWithError("unexpected cultivation active width");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        auto result = sampling::sample_survivors(plan, 1000, 0, false);
        benchmark::DoNotOptimize(result);
    }
}

// Coherent QEC protects production squeeze decisions and coefficient-state
// execution at width 13, beyond the width-10 workloads above.
void sample_coherent_d5(benchmark::State& state) {
    const auto plan = compile_circuit(fixture("coherent_d5_r5.stim"));
    if (plan.peak_active_width() != 13) {
        state.SkipWithError("unexpected coherent d5 active width");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        auto result = sampling::sample(plan, 100, 0);
        benchmark::DoNotOptimize(result);
    }
}

// This low-noise Clifford workload keeps most noise sites dormant, making
// symbolic dispatch and gap sampling dominant in the single-word frame regime.
void sample_surface_d7(benchmark::State& state) {
    const auto plan = compile_circuit(fixture("surface_d7_r7_p001.stim"));
    if (plan.peak_active_width() != 0 || plan.num_qubits() > 128) {
        state.SkipWithError("unexpected surface d7 plan shape");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        auto result = sampling::sample(plan, 10000, 0);
        benchmark::DoNotOptimize(result);
    }
}

// High physical noise forces most noise sites to fire, protecting the
// full-mask XOR and popcount path used to compose Pauli errors.
void sample_surface_d5_high_noise(benchmark::State& state) {
    const auto plan = compile_circuit(fixture("surface_d5_r5_p05.stim"));
    if (plan.peak_active_width() != 0 || plan.num_qubits() > 128) {
        state.SkipWithError("unexpected high-noise surface d5 plan shape");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        auto result = sampling::sample(plan, 10000, 0);
        benchmark::DoNotOptimize(result);
    }
}

// The larger Clifford surface workload crosses the single-word frame boundary,
// protecting runtime-width mask handling without active coefficient state.
void sample_surface_d11(benchmark::State& state) {
    const auto plan = compile_circuit(fixture("surface_d11_r11_p001.stim"));
    if (plan.peak_active_width() != 0 || plan.num_qubits() <= 128) {
        state.SkipWithError("unexpected surface d11 plan shape");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        auto result = sampling::sample(plan, 1000, 0);
        benchmark::DoNotOptimize(result);
    }
}

// Weight-three probes protect EXP_VAL frame conjugation and the dormant/active
// split; each probe must walk both halves of the full Pauli mask.
void sample_exp_val(benchmark::State& state) {
    const auto plan = compile_text(exp_val_heavy_text(20, 200));
    if (plan.num_exp_vals() != 200) {
        state.SkipWithError("unexpected EXP_VAL count");
        return;
    }
    for ([[maybe_unused]] auto _ : state) {
        auto result = sampling::sample(plan, 100000, 0);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(squeeze_parallel_t)->Name("squeeze_parallel_t_8192");
BENCHMARK(compile_plan_cultivation_d5)->Name("compile_plan_cultivation_d5");
BENCHMARK(schedule_pass_coherent_d5_r5)->Name("schedule_pass_coherent_d5_r5");
BENCHMARK(schedule_pass_cultivation_d5)->Name("schedule_pass_cultivation_d5");
BENCHMARK(sample_qv10)->Name("sample_qv10_100_shots");
BENCHMARK(sample_cultivation_d5)->Name("sample_cultivation_d5_1000_shots");
BENCHMARK(sample_coherent_d5)->Name("sample_coherent_d5_r5_100_shots");
BENCHMARK(sample_surface_d7)->Name("sample_surface_d7_r7_10000_shots");
BENCHMARK(sample_surface_d5_high_noise)->Name("sample_surface_d5_r5_high_noise_10000_shots");
BENCHMARK(sample_surface_d11)->Name("sample_surface_d11_r11_1000_shots");
BENCHMARK(sample_exp_val)->Name("sample_exp_val_20q_200_probes_100000_shots");

}  // namespace
