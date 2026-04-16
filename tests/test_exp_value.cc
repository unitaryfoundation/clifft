// End-to-end tests for EXP_VAL: parse -> trace -> lower -> execute.
// Uses the compile_circuit pattern to test the full pipeline.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <string>

using namespace clifft;
using Catch::Matchers::WithinAbs;

// Helper: compile a circuit string through the full pipeline and return the module.
static CompiledModule compile_circuit(const std::string& text) {
    auto circuit = clifft::parse(text);
    auto hir = clifft::trace(circuit);
    return clifft::lower(hir);
}

// Helper: compile and execute one shot, return the exp_vals vector.
static std::vector<double> run_exp_vals(const std::string& text, uint64_t seed = 42) {
    auto mod = compile_circuit(text);
    SchrodingerState state({.peak_rank = mod.peak_rank,
                            .num_measurements = mod.total_meas_slots,
                            .num_detectors = mod.num_detectors,
                            .num_observables = mod.num_observables,
                            .num_exp_vals = mod.num_exp_vals,
                            .seed = seed});
    execute(mod, state);
    return state.exp_vals;
}

// =============================================================================
// Single-qubit Pauli expectations on |0>
// =============================================================================

TEST_CASE("EXP_VAL: <Z> on |0> = +1") {
    auto ev = run_exp_vals("EXP_VAL Z0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));
}

TEST_CASE("EXP_VAL: <X> on |0> = 0") {
    auto ev = run_exp_vals("EXP_VAL X0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(0.0, 1e-12));
}

TEST_CASE("EXP_VAL: <Y> on |0> = 0") {
    auto ev = run_exp_vals("EXP_VAL Y0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(0.0, 1e-12));
}

// =============================================================================
// Single-qubit on |1> (X|0>)
// =============================================================================

TEST_CASE("EXP_VAL: <Z> on |1> = -1") {
    auto ev = run_exp_vals("X 0\nEXP_VAL Z0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(-1.0, 1e-12));
}

// =============================================================================
// Single-qubit on |+> (H|0>)
// =============================================================================

TEST_CASE("EXP_VAL: <X> on |+> = +1") {
    auto ev = run_exp_vals("H 0\nEXP_VAL X0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));
}

TEST_CASE("EXP_VAL: <Z> on |+> = 0") {
    auto ev = run_exp_vals("H 0\nEXP_VAL Z0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(0.0, 1e-12));
}

// =============================================================================
// Single-qubit on |+i> (S H |0>)
// =============================================================================

TEST_CASE("EXP_VAL: <Y> on |+i> = +1") {
    auto ev = run_exp_vals("H 0\nS 0\nEXP_VAL Y0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));
}

// =============================================================================
// Bell state (|00> + |11>)/sqrt(2)
// =============================================================================

TEST_CASE("EXP_VAL: <Z0*Z1> on Bell = +1") {
    auto ev = run_exp_vals("H 0\nCX 0 1\nEXP_VAL Z0*Z1");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));
}

TEST_CASE("EXP_VAL: <X0*X1> on Bell = +1") {
    auto ev = run_exp_vals("H 0\nCX 0 1\nEXP_VAL X0*X1");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));
}

TEST_CASE("EXP_VAL: <Z0> on Bell = 0") {
    auto ev = run_exp_vals("H 0\nCX 0 1\nEXP_VAL Z0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(0.0, 1e-12));
}

// =============================================================================
// Multiple EXP_VAL probes
// =============================================================================

TEST_CASE("EXP_VAL: multiple probes get consecutive indices") {
    auto ev = run_exp_vals("H 0\nEXP_VAL X0\nEXP_VAL Z0");
    REQUIRE(ev.size() == 2);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));  // <X> on |+>
    CHECK_THAT(ev[1], WithinAbs(0.0, 1e-12));  // <Z> on |+>
}

TEST_CASE("EXP_VAL: multi-product single instruction") {
    auto ev = run_exp_vals("H 0\nEXP_VAL X0 Z0");
    REQUIRE(ev.size() == 2);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));  // <X> on |+>
    CHECK_THAT(ev[1], WithinAbs(0.0, 1e-12));  // <Z> on |+>
}

// =============================================================================
// EXP_VAL does not affect measurements
// =============================================================================

TEST_CASE("EXP_VAL: does not disturb measurement outcomes") {
    // |0> -> EXP_VAL Z0 -> M 0: measurement should still give 0
    auto mod = compile_circuit("EXP_VAL Z0\nM 0");
    auto result = sample(mod, 100, 42);
    // All measurements should be 0 (deterministic |0> state)
    for (size_t i = 0; i < result.measurements.size(); ++i) {
        CHECK(result.measurements[i] == 0);
    }
}

// =============================================================================
// Non-Clifford: T gate
// =============================================================================

TEST_CASE("EXP_VAL: <Z> after T on |0> = +1") {
    // T|0> = |0> (T is diagonal), so <Z> = +1
    auto ev = run_exp_vals("T 0\nEXP_VAL Z0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));
}

TEST_CASE("EXP_VAL: <X> after T on |+>") {
    // H|0> = |+>, then T|+> = (|0> + e^{i*pi/4}|1>)/sqrt(2)
    // <X> = Re(e^{-i*pi/4}) = cos(pi/4) = 1/sqrt(2)
    auto ev = run_exp_vals("H 0\nT 0\nEXP_VAL X0");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(1.0 / std::sqrt(2.0), 1e-10));
}

// =============================================================================
// Sampling API: exp_vals in SampleResult
// =============================================================================

TEST_CASE("EXP_VAL: sample() returns exp_vals") {
    auto mod = compile_circuit("EXP_VAL Z0");
    auto result = sample(mod, 10, 42);
    REQUIRE(result.exp_vals.size() == 10);
    for (size_t i = 0; i < 10; ++i) {
        CHECK_THAT(result.exp_vals[i], WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("EXP_VAL: no EXP_VAL circuit has empty exp_vals") {
    auto mod = compile_circuit("H 0\nM 0");
    auto result = sample(mod, 5, 42);
    CHECK(result.exp_vals.empty());
    CHECK(mod.num_exp_vals == 0);
}

// =============================================================================
// Dormant qubit with X support -> 0
// =============================================================================

TEST_CASE("EXP_VAL: dormant qubit X support gives 0") {
    // qubit 1 is dormant (never activated), probe X1
    auto ev = run_exp_vals("EXP_VAL X1");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(0.0, 1e-12));
}

TEST_CASE("EXP_VAL: dormant qubit Y support gives 0") {
    auto ev = run_exp_vals("EXP_VAL Y1");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(0.0, 1e-12));
}

TEST_CASE("EXP_VAL: dormant qubit Z support gives +1") {
    auto ev = run_exp_vals("EXP_VAL Z1");
    REQUIRE(ev.size() == 1);
    CHECK_THAT(ev[0], WithinAbs(1.0, 1e-12));
}
