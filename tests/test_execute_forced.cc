// Tests for execute()'s forced-mode dispatch labels.
//
// The forced measurement opcodes (OP_MEAS_*_FORCED, OP_SWAP_MEAS_INTERFERE_FORCED)
// are wired into the same dispatcher that handles the sampling kernels. Each
// label calls the corresponding forced kernel from svm_forced_kernels.h and
// returns early from execute() when the kernel reports the record is
// unreachable.
//
// These tests verify dispatcher-level behavior end-to-end: a hand-built
// CompiledModule with forced opcodes runs through execute() and produces the
// same state changes that the standalone kernel tests already cover.

#include "clifft/backend/backend.h"
#include "clifft/svm/svm.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <span>
#include <vector>

using namespace clifft;
using Catch::Matchers::WithinAbs;

namespace {

CompiledModule make_module(std::vector<Instruction> bytecode, uint32_t peak_rank,
                           uint32_t num_meas) {
    CompiledModule mod;
    mod.bytecode = std::move(bytecode);
    mod.peak_rank = peak_rank;
    mod.num_measurements = num_meas;
    mod.total_meas_slots = num_meas;
    return mod;
}

// Set up state.forced_record on a SchrodingerState. Caller owns the storage.
void install_forced_record(SchrodingerState& state, std::span<const uint8_t> record) {
    state.forced_record = record;
}

}  // namespace

TEST_CASE("execute: dispatches OP_MEAS_DORMANT_STATIC_FORCED for matching outcome") {
    auto mod = make_module(
        {
            make_meas(Opcode::OP_MEAS_DORMANT_STATIC_FORCED, /*axis=*/0,
                      /*classical_idx=*/0, /*sign=*/false),
        },
        /*peak_rank=*/2, /*num_meas=*/1);
    SchrodingerState state(2, 1);
    std::vector<uint8_t> record{0};
    install_forced_record(state, record);

    execute(mod, state);

    REQUIRE(state.forced_reachable == true);
    REQUIRE(state.forced_log_probability == 0.0);
    REQUIRE(state.meas_record[0] == 0);
}

TEST_CASE("execute: dispatches OP_MEAS_DORMANT_STATIC_FORCED for mismatched outcome") {
    auto mod = make_module(
        {
            make_meas(Opcode::OP_MEAS_DORMANT_STATIC_FORCED, 0, 0, false),
        },
        /*peak_rank=*/2, /*num_meas=*/1);
    SchrodingerState state(2, 1);
    std::vector<uint8_t> record{1};  // p_x[0]=0 -> deterministic 0; record says 1.
    install_forced_record(state, record);

    execute(mod, state);

    REQUIRE(state.forced_reachable == false);
}

TEST_CASE("execute: dispatches OP_MEAS_DORMANT_STATIC_FORCED with FLAG_IDENTITY") {
    // FLAG_IDENTITY measurements skip p_x and produce a deterministic outcome
    // equal to the SIGN bit. Forcing a matching outcome is reachable;
    // forcing the opposite is unreachable.
    auto identity_meas = make_meas(Opcode::OP_MEAS_DORMANT_STATIC_FORCED, 0, 0, /*sign=*/true);
    identity_meas.flags |= Instruction::FLAG_IDENTITY;

    {
        auto mod = make_module({identity_meas}, /*peak_rank=*/2, /*num_meas=*/1);
        SchrodingerState state(2, 1);
        std::vector<uint8_t> record{1};  // FLAG_IDENTITY + SIGN -> expected outcome 1.
        install_forced_record(state, record);
        execute(mod, state);
        REQUIRE(state.forced_reachable == true);
        REQUIRE(state.forced_log_probability == 0.0);
        REQUIRE(state.meas_record[0] == 1);
    }
    {
        auto mod = make_module({identity_meas}, /*peak_rank=*/2, /*num_meas=*/1);
        SchrodingerState state(2, 1);
        std::vector<uint8_t> record{0};  // mismatched.
        install_forced_record(state, record);
        execute(mod, state);
        REQUIRE(state.forced_reachable == false);
    }
}

TEST_CASE("execute: dispatches OP_MEAS_DORMANT_RANDOM_FORCED for either outcome") {
    for (uint8_t forced : {uint8_t{0}, uint8_t{1}}) {
        auto mod = make_module({make_meas(Opcode::OP_MEAS_DORMANT_RANDOM_FORCED, 0, 0, false)},
                               /*peak_rank=*/2, /*num_meas=*/1);
        SchrodingerState state(2, 1);
        std::vector<uint8_t> record{forced};
        install_forced_record(state, record);

        execute(mod, state);

        REQUIRE(state.forced_reachable == true);
        REQUIRE_THAT(state.forced_log_probability, WithinAbs(std::log(0.5), 1e-15));
        REQUIRE(state.meas_record[0] == forced);
    }
}

TEST_CASE("execute: dispatches OP_MEAS_ACTIVE_DIAGONAL_FORCED") {
    // EXPAND lifts k=0 -> k=1 with the active state in |+>. ARRAY_H then
    // takes |+> -> |0>, so the diagonal (Z-basis) measurement is
    // deterministic and the forced outcome 0 contributes log(1) = 0.
    auto mod = make_module(
        {
            make_expand(0),
            make_array_h(0),
            make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL_FORCED, 0, 0, false),
        },
        /*peak_rank=*/1, /*num_meas=*/1);
    SchrodingerState state(1, 1);
    std::vector<uint8_t> record{0};
    install_forced_record(state, record);

    execute(mod, state);

    REQUIRE(state.forced_reachable == true);
    REQUIRE_THAT(state.forced_log_probability, WithinAbs(0.0, 1e-15));
    REQUIRE(state.active_k == 0);
    REQUIRE(state.meas_record[0] == 0);
}

TEST_CASE("execute: dispatches OP_MEAS_ACTIVE_INTERFERE_FORCED") {
    // Start in |+>_active by going through EXPAND + virtual frame to put
    // the active qubit in the |+> eigenstate of X. The simplest setup: do
    // EXPAND, manually set v[] to |+>, then measure on X.
    auto mod = make_module(
        {
            make_expand(0),
            make_meas(Opcode::OP_MEAS_ACTIVE_INTERFERE_FORCED, 0, 0, false),
        },
        /*peak_rank=*/1, /*num_meas=*/1);
    SchrodingerState state(1, 1);
    // EXPAND on rank-0 state yields |+>_active (v[0]=v[1]=1/sqrt(2)).
    std::vector<uint8_t> record{0};  // m_phys = 0 == plus branch.
    install_forced_record(state, record);

    execute(mod, state);

    REQUIRE(state.forced_reachable == true);
    REQUIRE_THAT(state.forced_log_probability, WithinAbs(0.0, 1e-15));
    REQUIRE(state.active_k == 0);
}

TEST_CASE("execute: dispatches OP_SWAP_MEAS_INTERFERE_FORCED") {
    // Same idea as above; just exercise the swap-fused opcode with f==t so
    // the kernel forwards to active_interfere internally.
    auto mod = make_module(
        {
            make_expand(0),
            make_swap_meas_interfere(/*swap_from=*/0, /*swap_to=*/0, /*classical_idx=*/0,
                                     /*sign=*/false),
        },
        /*peak_rank=*/1, /*num_meas=*/1);
    // Patch the opcode to its FORCED variant (the make_ helper sets the
    // sampling opcode; the dispatcher only sees what's in the bytecode).
    mod.bytecode[1].opcode = Opcode::OP_SWAP_MEAS_INTERFERE_FORCED;

    SchrodingerState state(1, 1);
    std::vector<uint8_t> record{0};
    install_forced_record(state, record);

    execute(mod, state);

    REQUIRE(state.forced_reachable == true);
    REQUIRE_THAT(state.forced_log_probability, WithinAbs(0.0, 1e-15));
    REQUIRE(state.active_k == 0);
}

TEST_CASE("execute: unreachable forced record short-circuits the remaining bytecode") {
    // Two measurements. Force the first to an outcome it can't satisfy
    // (p_x[0]=0 deterministic 0, record says 1). The dispatcher must
    // return immediately, leaving the second measurement's slot untouched.
    auto mod = make_module(
        {
            make_meas(Opcode::OP_MEAS_DORMANT_STATIC_FORCED, 0, 0, false),
            make_meas(Opcode::OP_MEAS_DORMANT_STATIC_FORCED, 1, 1, false),
        },
        /*peak_rank=*/2, /*num_meas=*/2);

    SchrodingerState state(2, 2);
    state.meas_record[1] = 99;          // sentinel
    std::vector<uint8_t> record{1, 0};  // first record entry is unsatisfiable
    install_forced_record(state, record);

    execute(mod, state);

    REQUIRE(state.forced_reachable == false);
    // Second measurement's slot must not have been written (sentinel preserved).
    REQUIRE(state.meas_record[1] == 99);
}

TEST_CASE("execute: forced log-probabilities accumulate across multiple measurements") {
    // Two random-dormant measurements. Each contributes log(1/2), so the
    // running total should be 2*log(1/2) = log(1/4).
    auto mod = make_module(
        {
            make_meas(Opcode::OP_MEAS_DORMANT_RANDOM_FORCED, 0, 0, false),
            make_meas(Opcode::OP_MEAS_DORMANT_RANDOM_FORCED, 1, 1, false),
        },
        /*peak_rank=*/2, /*num_meas=*/2);

    SchrodingerState state(2, 2);
    std::vector<uint8_t> record{0, 1};
    install_forced_record(state, record);

    execute(mod, state);

    REQUIRE(state.forced_reachable == true);
    REQUIRE_THAT(state.forced_log_probability, WithinAbs(std::log(0.25), 1e-15));
    REQUIRE(state.meas_record[0] == 0);
    REQUIRE(state.meas_record[1] == 1);
}

TEST_CASE("execute: sampling-mode measurement opcodes are not disturbed by step 5") {
    // Regression check that wiring forced opcodes in did not touch the
    // sampling dispatch. Sampling a dormant_static still works.
    auto mod = make_module({make_meas(Opcode::OP_MEAS_DORMANT_STATIC, 0, 0, /*sign=*/false)},
                           /*peak_rank=*/2, /*num_meas=*/1);
    SchrodingerState state(2, 1);

    execute(mod, state);

    REQUIRE(state.forced_reachable == true);  // unchanged from default
    REQUIRE(state.forced_log_probability == 0.0);
    // Deterministic outcome from p_x[0] = 0.
    REQUIRE(state.meas_record[0] == 0);
}
