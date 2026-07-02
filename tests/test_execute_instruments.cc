// Execution tests for the OP_INSTRUMENT_* dispatch.
//
// The dispatch composes the draw-free instrument kernels with the fire
// draws. Fires are stochastic, so end-to-end determinism comes from
// certain-fire channels (p = 1, which also exercises the eval-only
// route the fused form cannot take), source-dependence contrasts on
// definite preparations, and same-seed reproducibility. Distributional
// validation belongs to the exact-mode validation campaign.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;

namespace {

// A transition spec with total rate p_g/p_e from each source and every
// fire landing on computational level `dest`.
InstrumentSpec to_level(double p_g, double p_e, int dest) {
    InstrumentSpec spec;
    spec.p_total[0] = p_g;
    spec.p_total[1] = p_e;
    spec.p_dest[0][dest] = p_g;
    spec.p_dest[1][dest] = p_e;
    return spec;
}

CompiledModule compile_raw(const char* text, const InstrumentTraceOptions& options) {
    auto hir = trace(parse(text), &options);
    return lower(hir);
}

// Execute one shot with a fixed seed and return the measurement record.
std::vector<uint8_t> run_shot(const CompiledModule& module, uint64_t seed) {
    SchrodingerState state(StateConfig{.peak_rank = module.peak_rank,
                                       .num_measurements = module.total_meas_slots,
                                       .num_qubits = module.num_qubits,
                                       .seed = seed});
    execute(module, state);
    return state.meas_record;
}

}  // namespace

TEST_CASE("execute: a certain relaxation fires in-line and flips the record") {
    // From |1>, a p = 1 transition to g fires every shot; the fixup turns
    // the subsequent measurement into 0. The same site on |0> has rate 0
    // from g and never fires.
    InstrumentTraceOptions options;
    options.transitions.emplace("relax", to_level(/*p_g=*/0.0, /*p_e=*/1.0, /*dest=*/0));

    auto relaxed = compile_raw("X 0\nLEVEL_TRANSITION[relax] 0\nM 0", options);
    auto untouched = compile_raw("LEVEL_TRANSITION[relax] 0\nM 0", options);

    for (uint64_t seed = 1; seed <= 20; ++seed) {
        REQUIRE(run_shot(relaxed, seed) == std::vector<uint8_t>{0});
        REQUIRE(run_shot(untouched, seed) == std::vector<uint8_t>{0});
    }
}

TEST_CASE("execute: a certain excitation from a definite g fires to e") {
    InstrumentTraceOptions options;
    options.transitions.emplace("excite", to_level(/*p_g=*/1.0, /*p_e=*/0.0, /*dest=*/1));

    auto module = compile_raw("LEVEL_TRANSITION[excite] 0\nM 0", options);
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        REQUIRE(run_shot(module, seed) == std::vector<uint8_t>{1});
    }
}

TEST_CASE("execute: an active-axis certain reset collapses the carrier to g") {
    // H; T activates the qubit in superposition. A source-independent
    // p = 1 jump to g takes the eval-only route (r = 0 forbids the fused
    // form): whichever source the populations select, the destination is
    // g, so the measurement is deterministically 0.
    InstrumentTraceOptions options;
    options.transitions.emplace("reset_g", to_level(/*p_g=*/1.0, /*p_e=*/1.0, /*dest=*/0));

    auto module = compile_raw("H 0\nT 0\nLEVEL_TRANSITION[reset_g] 0\nM 0", options);
    REQUIRE(module.peak_rank == 1);
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        REQUIRE(run_shot(module, seed) == std::vector<uint8_t>{0});
    }
}

TEST_CASE("execute: a dormant-random certain reset expands, collapses, and measures g") {
    // H leaves the qubit dormant-random; under exact damping the site
    // expands (k -> k+1), fires with certainty, and collapses to g.
    InstrumentTraceOptions options;
    options.transitions.emplace("reset_g", to_level(1.0, 1.0, /*dest=*/0));

    auto module = compile_raw("H 0\nLEVEL_TRANSITION[reset_g] 0\nM 0", options);
    REQUIRE(module.peak_rank == 1);  // the site's own expansion
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        REQUIRE(run_shot(module, seed) == std::vector<uint8_t>{0});
    }
}

TEST_CASE("execute: the no-fire path of a weak site leaves a definite carrier alone") {
    // From |1> with a small e-rate, most seeds do not fire; whenever the
    // shot does not fire the record must be 1. Verify per-seed against
    // the fire draw's determinism: identical seeds reproduce identical
    // records, and across seeds only {0, 1} appear with 1 dominating.
    InstrumentTraceOptions options;
    options.transitions.emplace("relax", to_level(0.0, 0.05, /*dest=*/0));

    auto module = compile_raw("X 0\nLEVEL_TRANSITION[relax] 0\nM 0", options);
    int ones = 0;
    for (uint64_t seed = 1; seed <= 50; ++seed) {
        auto first = run_shot(module, seed);
        REQUIRE(run_shot(module, seed) == first);  // same-seed reproducibility
        ones += first[0];
    }
    REQUIRE(ones > 30);  // p = 0.05: overwhelmingly no-fire
}

TEST_CASE("execute: a leaked/lost fire names the site and line in the trap error") {
    InstrumentTraceOptions options;  // LOSS needs no spec
    auto module = compile_raw("H 1\nLOSS(1.0) 0\nM 0", options);

    SchrodingerState state(StateConfig{.peak_rank = module.peak_rank,
                                       .num_measurements = module.total_meas_slots,
                                       .num_qubits = module.num_qubits,
                                       .seed = 7});
    REQUIRE_THROWS_WITH(execute(module, state),
                        ContainsSubstring("instrument site 0") &&
                            ContainsSubstring("circuit line 2") &&
                            ContainsSubstring("resumable traps are not implemented yet"));
}

TEST_CASE("execute: a neglect-mode dormant-random site is silent until it fires") {
    InstrumentTraceOptions options;
    options.transitions.emplace("leak", InstrumentSpec{{0.3, 0.3}, {{0.0, 0.0}, {0.0, 0.0}}});
    options.neglect_damping = true;

    // Fire probability is 0.3 per shot; k stays 0 either way. Fired
    // shots trap (all destinations are leaked/lost here); silent shots
    // measure the untouched |+> fairly.
    auto module = compile_raw("H 0\nLEVEL_TRANSITION[leak] 0\nM 0", options);
    REQUIRE(module.peak_rank == 0);

    int traps = 0;
    int completed = 0;
    for (uint64_t seed = 1; seed <= 40; ++seed) {
        SchrodingerState state(StateConfig{.peak_rank = module.peak_rank,
                                           .num_measurements = module.total_meas_slots,
                                           .num_qubits = module.num_qubits,
                                           .seed = seed});
        try {
            execute(module, state);
            ++completed;
        } catch (const std::runtime_error&) {
            ++traps;
        }
    }
    REQUIRE(traps > 0);
    REQUIRE(completed > 0);
    REQUIRE(traps + completed == 40);
}
