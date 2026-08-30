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

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"

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
static sampling::ExecutablePlan compile_parsed(Circuit circuit) {
    auto hir = trace(circuit);
    default_hir_pass_manager().run(hir);
    return sampling::ExecutablePlan(sampling::plan_sampling(hir));
}

static sampling::ExecutablePlan compile_circuit(const std::string& path) {
    return compile_parsed(parse_file(path));
}

static sampling::ExecutablePlan compile_text(const std::string& text) {
    return compile_parsed(parse(text));
}

static HirModule parallel_t_hir(uint32_t num_qubits) {
    std::ostringstream s;
    s << "T";
    for (uint32_t q = 0; q < num_qubits; ++q) {
        s << " " << q;
    }
    return trace(parse(s.str()));
}

// A suffix containing only expansions has no useful bypass destination. This
// wide case guards against searching that suffix with full-mask commutation
// checks once for every T gate.
TEST_CASE("Bench: squeeze parallel T convoy", "[bench]") {
    auto hir = parallel_t_hir(8192);
    REQUIRE(hir.ops.size() == 8192);

    BENCHMARK("squeeze 8192 parallel T gates") {
        StatevectorSqueezePass{}.run(hir);
        return hir.ops.size();
    };
}

// EXP_VAL-heavy synthetic circuit: prepares a Clifford state on n qubits,
// then evaluates `num_probes` weight-3 multi-Pauli expectation values per shot.
// Stays at zero peak active width (fully Clifford prep) so the cost is dominated by the
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
// QV-10: 10 qubits, peak active width 10, dense SU(4) layers with measurements.
// ~1ms/shot baseline -> 100 shots ~= 100ms.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: QV-10 sampling 100 shots", "[bench]") {
    auto mod = compile_circuit(fixture("qv10.stim"));
    REQUIRE(mod.peak_active_width() == 10);

    BENCHMARK("QV-10 x100 shots") {
        return sampling::sample(mod, 100, 0);
    };
}

// ---------------------------------------------------------------------------
// Magic state cultivation d=5: 42 physical qubits, peak active width 10,
// sparse QEC with noise, T gates, postselection.
// ~0.09ms/shot baseline -> 1000 shots ~= 90ms.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: cultivation d5 sampling 1000 shots", "[bench]") {
    auto mod = compile_circuit(fixture("cultivation_d5.stim"));
    REQUIRE(mod.peak_active_width() == 10);

    BENCHMARK("cultivation-d5 x1000 shots") {
        return sampling::sample_survivors(mod, 1000, 0, false);
    };
}

// ---------------------------------------------------------------------------
// Surface code d=7 r=7 p=1e-3: paper QEC throughput benchmark.
// ~118 qubits, fully Clifford (zero peak active width), low noise so most NOISE sites
// stay silent. Throughput is dominated by symbolic actions and the gap-sampler.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: surface d7 r7 p1e-3 sampling 10000 shots", "[bench]") {
    auto mod = compile_circuit(fixture("surface_d7_r7_p001.stim"));
    REQUIRE(mod.peak_active_width() == 0);
    REQUIRE(mod.num_qubits() <= 128);

    BENCHMARK("surface-d7-r7 p=1e-3 x10000 shots") {
        return sampling::sample(mod, 10000, 0);
    };
}

// ---------------------------------------------------------------------------
// Surface code d=5 r=5 with high physical noise (p=0.05): forces most NOISE
// sites to fire, exercising the APPLY_PAULI / NOISE full-mask XOR + popcount
// path. Throughput is dominated by the per-fire mask composition.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: surface d5 r5 high-noise APPLY_PAULI heavy", "[bench]") {
    auto mod = compile_circuit(fixture("surface_d5_r5_p05.stim"));

    BENCHMARK("surface-d5-r5 p=0.05 x10000 shots") {
        return sampling::sample(mod, 10000, 0);
    };
}

// ---------------------------------------------------------------------------
// Surface code d=11 r=11 p=1e-3: ~242 qubits, two 64-bit Pauli mask
// words. Sits above the single-word frame regime to give a regression
// baseline for the runtime-width path.
// ---------------------------------------------------------------------------
TEST_CASE("Bench: surface d11 r11 p1e-3 sampling 1000 shots", "[bench]") {
    auto mod = compile_circuit(fixture("surface_d11_r11_p001.stim"));
    REQUIRE(mod.peak_active_width() == 0);
    REQUIRE(mod.num_qubits() > 128);

    BENCHMARK("surface-d11-r11 p=1e-3 x1000 shots") {
        return sampling::sample(mod, 1000, 0);
    };
}

// ---------------------------------------------------------------------------
// EXP_VAL heavy: 20 qubits, 200 weight-3 multi-Pauli probes per shot.
// Exercises exec_exp_val (frame conjugation + dormant/active split). Each
// probe walks the full mask twice (popcount of x & p_z, z & p_x).
// ---------------------------------------------------------------------------
TEST_CASE("Bench: EXP_VAL 20q 200 probes", "[bench]") {
    auto mod = compile_text(exp_val_heavy_text(20, 200));
    REQUIRE(mod.num_exp_vals() == 200);

    BENCHMARK("exp-val 20q 200 probes x100k") {
        return sampling::sample(mod, 100000, 0);
    };
}
