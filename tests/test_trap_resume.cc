// Trap and resume: the runtime half of the exact-jump protocol.
//
// execute() halts at a leaked/lost instrument fire with state.pending_trap
// set and the carrier collapsed onto the drawn source wherever the form
// allows; resume() continues the state in a (possibly different) module at
// a bytecode offset, growing the amplitude array and record buffer when
// the continuation needs more, and re-anchoring the noise-gap cursor at
// the entry offset. The orchestrator driver and continuation cache are a
// separate step; these tests hand-build their continuations, leaning on
// the prefix-identity contract that makes cross-module re-entry sound.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <complex>
#include <cstring>
#include <string>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;

namespace {

// A transition whose entire fire mass is leaked/lost, with per-source
// rates (p_g, p_e).
InstrumentSpec leak(double p_g, double p_e) {
    InstrumentSpec spec;
    spec.p_total[0] = p_g;
    spec.p_total[1] = p_e;
    return spec;
}

CompiledModule compile_raw(const char* text, const InstrumentTraceOptions& options) {
    auto hir = trace(parse(text), &options);
    return lower(hir);
}

SchrodingerState make_shot_state(const CompiledModule& module, uint64_t seed) {
    return SchrodingerState(StateConfig{.peak_rank = module.peak_rank,
                                        .num_measurements = module.total_meas_slots,
                                        .num_qubits = module.num_qubits,
                                        .num_detectors = module.num_detectors,
                                        .num_observables = module.num_observables,
                                        .num_exp_vals = module.num_exp_vals,
                                        .seed = seed});
}

// Resume a trapped state past its trap site in `continuation`.
void resume_past_trap(const CompiledModule& continuation, SchrodingerState& state) {
    REQUIRE(state.pending_trap.has_value());
    const uint32_t offset = continuation.instrument_offsets.at(state.pending_trap->site_id);
    resume(continuation, state, offset + 1);
}

}  // namespace

TEST_CASE("trap+resume: a spectator loss resumes in the same module and completes") {
    // The lost qubit has no downstream operations, so the original module
    // is its own valid continuation. H;T ... T_DAG;H on qubit 0 composes
    // to the identity: the record is deterministically 0, trap or no trap.
    InstrumentTraceOptions options;
    auto module = compile_raw("H 0\nT 0\nLOSS(1.0) 1\nT_DAG 0\nH 0\nM 0", options);

    for (uint64_t seed = 1; seed <= 20; ++seed) {
        auto state = make_shot_state(module, seed);
        execute(module, state);

        REQUIRE(state.pending_trap.has_value());
        REQUIRE(state.pending_trap->site_id == 0);
        // The destination class is already drawn (leaked/lost); only the
        // level within the trap remainder is the host's to pick.
        REQUIRE(!state.pending_trap->destination_pending);

        resume_past_trap(module, state);
        REQUIRE(!state.pending_trap.has_value());
        REQUIRE(state.meas_record == std::vector<uint8_t>{0});
    }
}

TEST_CASE("trap+resume: a neglect-form trap reports its destination as pending") {
    InstrumentTraceOptions options;
    options.transitions.emplace("jump", [] {
        InstrumentSpec spec;
        spec.p_total[0] = 1.0;
        spec.p_total[1] = 1.0;
        spec.p_dest[0][1] = 0.5;  // half the column is computational:
        spec.p_dest[1][0] = 0.5;  // the host must draw over all of it
        return spec;
    }());
    options.neglect_damping = true;

    auto module = compile_raw("H 0\nLEVEL_TRANSITION[jump] 0\nM 0", options);
    auto state = make_shot_state(module, /*seed=*/6);
    execute(module, state);

    REQUIRE(state.pending_trap.has_value());
    REQUIRE(state.pending_trap->destination_pending);
    REQUIRE(state.active_k == 0);  // no expansion, carrier untouched
}

TEST_CASE("trap+resume: resume refuses a state with no pending trap") {
    InstrumentTraceOptions options;
    auto module = compile_raw("LOSS(1.0) 0\nM 0", options);
    auto state = make_shot_state(module, /*seed=*/8);
    REQUIRE_THROWS_WITH(resume(module, state, 1), ContainsSubstring("no pending trap"));
}

TEST_CASE("trap+resume: resume rejects an offset that does not follow the trapped site") {
    // A stale or miscomputed driver offset must not silently skip or
    // re-run bytecode; only the instruction after the trapped site is a
    // valid entry. Rejected attempts leave the trap pending, so the
    // correct offset still works afterward.
    InstrumentTraceOptions options;
    auto module = compile_raw("LOSS(1.0) 0\nX 0\nM 0", options);
    auto state = make_shot_state(module, /*seed=*/12);
    execute(module, state);
    REQUIRE(state.pending_trap.has_value());

    const uint32_t good = module.instrument_offsets.at(0) + 1;
    REQUIRE_THROWS_WITH(resume(module, state, good + 1), ContainsSubstring("does not follow"));
    REQUIRE_THROWS_WITH(resume(module, state, good - 1), ContainsSubstring("does not follow"));
    REQUIRE(state.pending_trap.has_value());

    resume(module, state, good);
    REQUIRE(!state.pending_trap.has_value());
    REQUIRE(state.meas_record == std::vector<uint8_t>{1});
}

TEST_CASE("trap+resume: the continuation's suffix governs the outcome") {
    // Two modules share the trapping prefix bit-for-bit (the barrier
    // contract); their suffixes differ. Resuming the same trapped state
    // into each must produce that module's suffix behavior: the original
    // undoes the H (measures 0), the continuation adds an X (measures 1).
    InstrumentTraceOptions options;
    auto original = compile_raw("H 0\nT 0\nLOSS(1.0) 1\nT_DAG 0\nH 0\nM 0", options);
    auto continuation = compile_raw("H 0\nT 0\nLOSS(1.0) 1\nT_DAG 0\nH 0\nX 0\nM 0", options);

    const uint32_t offset = original.instrument_offsets.at(0);
    REQUIRE(continuation.instrument_offsets.at(0) == offset);
    for (uint32_t i = 0; i <= offset; ++i) {
        REQUIRE(std::memcmp(&original.bytecode[i], &continuation.bytecode[i],
                            sizeof(Instruction)) == 0);
    }

    for (uint64_t seed = 1; seed <= 10; ++seed) {
        auto state = make_shot_state(original, seed);
        execute(original, state);
        REQUIRE(state.pending_trap.has_value());
        resume_past_trap(continuation, state);
        REQUIRE(state.meas_record == std::vector<uint8_t>{1});
    }
}

TEST_CASE("trap+resume: an active-site trap hands over a collapsed carrier") {
    // A certain leak on an active superposed qubit: the fire draws a
    // source from the populations, and the carrier must arrive at the
    // trap collapsed onto exactly that source -- the discarded half is
    // zero and the surviving half matches the reported source.
    InstrumentTraceOptions options;
    options.transitions.emplace("leak", leak(1.0, 1.0));
    auto module = compile_raw("H 0\nT 0\nLEVEL_TRANSITION[leak] 0\nM 0", options);

    bool saw_g = false;
    bool saw_e = false;
    for (uint64_t seed = 1; seed <= 30; ++seed) {
        auto state = make_shot_state(module, seed);
        execute(module, state);

        REQUIRE(state.pending_trap.has_value());
        const uint8_t source = state.pending_trap->source;
        (source == 0 ? saw_g : saw_e) = true;

        // Axis 0 is the active axis; with no localization sign or frame
        // bit in play here, array half `source` survives.
        REQUIRE(state.active_k == 1);
        const std::complex<double> discarded = state.v()[1 - source];
        REQUIRE(discarded == std::complex<double>{0.0, 0.0});
        REQUIRE(std::abs(state.v()[source]) > 0.5);
    }
    REQUIRE(saw_g);
    REQUIRE(saw_e);
}

TEST_CASE("trap+resume: the record buffer grows for a continuation with more hidden slots") {
    // The continuation replaces the suffix with one containing a reset,
    // whose lowering adds a hidden measurement slot beyond what the
    // original module allocated. Visible slots stay layout-stable.
    InstrumentTraceOptions options;
    auto original = compile_raw("H 0\nT 0\nLOSS(1.0) 1\nM 0", options);
    auto continuation = compile_raw("H 0\nT 0\nLOSS(1.0) 1\nT_DAG 0\nH 0\nR 1\nM 0", options);
    REQUIRE(continuation.total_meas_slots > original.total_meas_slots);

    auto state = make_shot_state(original, /*seed=*/5);
    execute(original, state);
    REQUIRE(state.pending_trap.has_value());
    REQUIRE(state.meas_record.size() == original.total_meas_slots);

    resume_past_trap(continuation, state);
    REQUIRE(state.meas_record.size() == continuation.total_meas_slots);
    REQUIRE(state.meas_record.at(0) == 0);  // H;T;T_DAG;H = identity
}

TEST_CASE("trap+resume: the amplitude array grows for a higher-rank continuation") {
    // The original needs k = 0; the continuation's suffix activates two
    // qubits (k = 2), exceeding the state's original allocation.
    InstrumentTraceOptions options;
    auto original = compile_raw("LOSS(1.0) 2\nM 0", options);
    auto continuation = compile_raw("LOSS(1.0) 2\nH 0\nT 0\nH 1\nT 1\nCX 0 1\nM 0\nM 1", options);
    REQUIRE(original.peak_rank == 0);
    REQUIRE(continuation.peak_rank == 2);
    REQUIRE(continuation.num_measurements == original.num_measurements + 1);

    auto state = make_shot_state(original, /*seed=*/3);
    execute(original, state);
    REQUIRE(state.pending_trap.has_value());
    REQUIRE(state.array_size() == 1);

    // The continuation also carries one more visible measurement, so
    // grow the record buffer expectation accordingly.
    resume_past_trap(continuation, state);
    REQUIRE(state.array_size() >= 4);
    REQUIRE(!state.pending_trap.has_value());
    REQUIRE(state.meas_record.size() == continuation.total_meas_slots);
}

TEST_CASE("trap+resume: suffix noise fires after re-anchoring, prefix noise does not refire") {
    // Both errors are certain (p = 1). The prefix error flips qubit 0
    // before the trap; the suffix error flips it again after resume. A
    // broken cursor either refires the prefix site (record 0) or skips
    // the suffix site (record 0); only fire-each-exactly-once gives 0
    // flipped twice -- M reads 0 -- wait: two X flips restore |0>, so the
    // deterministic record is 0, and a single miss or double-fire yields
    // 1 instead.
    InstrumentTraceOptions options;
    auto module = compile_raw("X_ERROR(1) 0\nLOSS(1.0) 1\nX_ERROR(1) 0\nM 0", options);

    for (uint64_t seed = 1; seed <= 10; ++seed) {
        auto state = make_shot_state(module, seed);
        // Mirror sample()'s per-shot noise-cursor setup.
        state.next_noise_idx = 0;
        state.draw_next_noise(module.constant_pool.noise_hazards);

        execute(module, state);
        REQUIRE(state.pending_trap.has_value());

        resume_past_trap(module, state);
        REQUIRE(state.meas_record == std::vector<uint8_t>{0});
    }
}

TEST_CASE("trap+resume: a chain of traps resumes one site at a time") {
    InstrumentTraceOptions options;
    auto module = compile_raw("LOSS(1.0) 1\nX 0\nLOSS(1.0) 2\nX 0\nM 0", options);

    auto state = make_shot_state(module, /*seed=*/11);
    execute(module, state);
    REQUIRE(state.pending_trap.has_value());
    REQUIRE(state.pending_trap->site_id == 0);

    resume_past_trap(module, state);
    REQUIRE(state.pending_trap.has_value());
    REQUIRE(state.pending_trap->site_id == 1);

    resume_past_trap(module, state);
    REQUIRE(!state.pending_trap.has_value());
    REQUIRE(state.meas_record == std::vector<uint8_t>{0});  // X twice
}

TEST_CASE("trap+resume: reset clears a pending trap and its partial records") {
    InstrumentTraceOptions options;
    auto module = compile_raw("X 0\nM 0\nLOSS(1.0) 1\nM 0", options);

    auto state = make_shot_state(module, /*seed=*/2);
    execute(module, state);
    REQUIRE(state.pending_trap.has_value());
    REQUIRE(state.meas_record.at(0) == 1);  // written before the trap

    state.reset();
    REQUIRE(!state.pending_trap.has_value());
    REQUIRE(state.meas_record.at(0) == 0);
}

TEST_CASE("trap+resume: plain sampling rejects a shot that traps") {
    InstrumentTraceOptions options;
    auto module = compile_raw("LOSS(1.0) 0\nM 0", options);
    REQUIRE_THROWS_WITH(sample(module, /*shots=*/4, /*seed=*/9),
                        ContainsSubstring("exact-mode driver"));
}

TEST_CASE("trap+resume: resume validates the handoff") {
    InstrumentTraceOptions options;
    auto module = compile_raw("LOSS(1.0) 0\nM 0", options);
    auto other_qubits = compile_raw("LOSS(1.0) 0\nM 0\nM 1", options);

    auto state = make_shot_state(module, /*seed=*/4);
    execute(module, state);
    REQUIRE(state.pending_trap.has_value());

    REQUIRE_THROWS_WITH(resume(other_qubits, state, 1), ContainsSubstring("qubits"));
}
