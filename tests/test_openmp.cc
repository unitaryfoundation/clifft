// OpenMP multi-threading correctness tests.
//
// These tests build circuits that reach peak_rank=18 (the kMinRankForThreads
// threshold) so the parallel branches in parallel_for, parallel_reduce,
// parallel_stride_loop, etc. are actually exercised.
//
// The key correctness check is determinism: the same circuit + seed
// must produce identical measurement results regardless of thread count.
// Thread Sanitizer (TSan) can then detect data races in these paths.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/svm/svm.h"
#include "clifft/svm/svm_internal.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace clifft;

// Build a circuit with `n` qubits that reaches peak_rank=n.
// Uses H+T on each qubit (non-Clifford) to force statevector simulation,
// then measures all qubits. Compiled without optimizer passes to preserve
// peak_rank (the optimizer can fuse rotations and lower it).
static CompiledModule build_high_rank_circuit(int n) {
    std::string src;
    for (int q = 0; q < n; ++q)
        src += "H " + std::to_string(q) + "\n";
    for (int q = 0; q < n; ++q)
        src += "T " + std::to_string(q) + "\n";
    for (int q = 0; q < n; ++q)
        src += "M " + std::to_string(q) + "\n";
    auto circuit = parse(src);
    auto hir = trace(circuit);
    return lower(hir);
}

// Build a circuit with EXP_VAL probes at high rank, exercising the
// parallel_reduce path in the exp_val kernel.
static CompiledModule build_exp_val_high_rank() {
    std::string src;
    for (int q = 0; q < 18; ++q)
        src += "H " + std::to_string(q) + "\n";
    for (int q = 0; q < 18; ++q)
        src += "T " + std::to_string(q) + "\n";
    src += "EXP_VAL Z0\nEXP_VAL X1\nEXP_VAL Y2\n";
    for (int q = 0; q < 18; ++q)
        src += "M " + std::to_string(q) + "\n";
    auto circuit = parse(src);
    auto hir = trace(circuit);
    return lower(hir);
}

TEST_CASE("OpenMP: peak_rank >= kMinRankForThreads", "[openmp]") {
    auto mod = build_high_rank_circuit(18);
    REQUIRE(mod.peak_rank >= kMinRankForThreads);
}

TEST_CASE("OpenMP: single-shot determinism across thread counts", "[openmp]") {
    auto mod = build_high_rank_circuit(18);
    REQUIRE(mod.peak_rank >= kMinRankForThreads);

    // Sample with 1 thread
    SampleResult r1;
#ifdef _OPENMP
    int original = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    r1 = sample(mod, 1, uint64_t{42});
#ifdef _OPENMP
    omp_set_num_threads(original);
#endif

    // Sample with default (all) threads
    auto r2 = sample(mod, 1, uint64_t{42});

    REQUIRE(r1.measurements.size() == r2.measurements.size());
    for (size_t i = 0; i < r1.measurements.size(); ++i) {
        CHECK(r1.measurements[i] == r2.measurements[i]);
    }
}

TEST_CASE("OpenMP: multi-shot determinism across thread counts", "[openmp]") {
    auto mod = build_high_rank_circuit(18);
    REQUIRE(mod.peak_rank >= kMinRankForThreads);

    constexpr uint32_t shots = 10;

    SampleResult r1;
#ifdef _OPENMP
    int original = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    r1 = sample(mod, shots, uint64_t{123});
#ifdef _OPENMP
    omp_set_num_threads(original);
#endif

    auto r2 = sample(mod, shots, uint64_t{123});

    REQUIRE(r1.measurements.size() == r2.measurements.size());
    for (size_t i = 0; i < r1.measurements.size(); ++i) {
        CHECK(r1.measurements[i] == r2.measurements[i]);
    }
}

TEST_CASE("OpenMP: high-rank execution does not crash", "[openmp]") {
    auto mod = build_high_rank_circuit(18);
    REQUIRE(mod.peak_rank >= kMinRankForThreads);

    auto result = sample(mod, 5, uint64_t{7});
    REQUIRE(result.measurements.size() == 5 * mod.num_measurements);
    for (auto m : result.measurements) {
        CHECK((m == 0 || m == 1));
    }
}

TEST_CASE("OpenMP: EXP_VAL determinism across thread counts at high rank", "[openmp]") {
    auto mod = build_exp_val_high_rank();
    REQUIRE(mod.peak_rank >= kMinRankForThreads);
    REQUIRE(mod.num_exp_vals == 3);

    SampleResult r1;
#ifdef _OPENMP
    int original = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    r1 = sample(mod, 1, uint64_t{77});
#ifdef _OPENMP
    omp_set_num_threads(original);
#endif

    auto r2 = sample(mod, 1, uint64_t{77});

    REQUIRE(r1.exp_vals.size() == r2.exp_vals.size());
    // Tolerance, not bit-equality: OpenMP reductions can sum partial results
    // in different orders across thread counts, so the last bits of a float
    // expectation value can differ even when the algorithm is deterministic.
    for (size_t i = 0; i < r1.exp_vals.size(); ++i) {
        CHECK_THAT(r1.exp_vals[i], Catch::Matchers::WithinAbs(r2.exp_vals[i], 1e-12));
    }
}
