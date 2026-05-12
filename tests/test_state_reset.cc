// Tests for the SchrodingerState::reset() reuse invariant:
// after reset() the state must be safe to feed to execute() again, with
// the result matching what execute() would produce on a freshly
// constructed instance with the same config. The underlying v_, p_x,
// and p_z allocations must be preserved across the call so that reuse
// is allocation-free.
//
// Note: reset() does not unconditionally clear meas_record / det_record.
// Those buffers may carry stale outcome bits from a prior non-discarded
// run; execute() overwrites each slot it touches before any subsequent
// kernel reads it.

#include "clifft/backend/backend.h"
#include "clifft/svm/svm.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <vector>

using namespace clifft;
using clifft::test::test_lcg;

namespace {

CompiledModule make_unitary_program(std::vector<Instruction> bytecode, uint32_t peak_rank) {
    CompiledModule mod;
    mod.bytecode = std::move(bytecode);
    mod.peak_rank = peak_rank;
    mod.num_measurements = 0;
    return mod;
}

CompiledModule make_measured_program(std::vector<Instruction> bytecode, uint32_t peak_rank,
                                     uint32_t num_meas) {
    CompiledModule mod;
    mod.bytecode = std::move(bytecode);
    mod.peak_rank = peak_rank;
    mod.num_measurements = num_meas;
    mod.total_meas_slots = num_meas;
    return mod;
}

// Check the post-execute state of two SchrodingerState instances. Used
// after both have just run the same program from a known-equivalent
// starting state -- i.e. either both fresh, or one fresh and one
// reset-then-fed-the-same-program. meas_record / det_record contents
// are compared element-wise here because both states ran the program,
// so both should have the same outcome bits in those slots.
void require_post_execute_states_match(const SchrodingerState& a, const SchrodingerState& b) {
    REQUIRE(a.active_k == b.active_k);
    REQUIRE(a.v_size() == b.v_size());
    REQUIRE(a.array_size() == b.array_size());
    REQUIRE(a.gamma() == b.gamma());
    REQUIRE(a.discarded == b.discarded);
    REQUIRE(a.has_exp_vals == b.has_exp_vals);

    REQUIRE(a.p_x.size() == b.p_x.size());
    REQUIRE(a.p_z.size() == b.p_z.size());
    for (size_t i = 0; i < a.p_x.size(); ++i) {
        REQUIRE(a.p_x[i] == b.p_x[i]);
        REQUIRE(a.p_z[i] == b.p_z[i]);
    }

    REQUIRE(a.meas_record.size() == b.meas_record.size());
    REQUIRE(a.det_record.size() == b.det_record.size());
    REQUIRE(a.obs_record.size() == b.obs_record.size());
    REQUIRE(a.exp_vals.size() == b.exp_vals.size());

    for (size_t i = 0; i < a.meas_record.size(); ++i) {
        REQUIRE(a.meas_record[i] == b.meas_record[i]);
    }
    for (size_t i = 0; i < a.det_record.size(); ++i) {
        REQUIRE(a.det_record[i] == b.det_record[i]);
    }
    for (size_t i = 0; i < a.obs_record.size(); ++i) {
        REQUIRE(a.obs_record[i] == b.obs_record[i]);
    }
    for (size_t i = 0; i < a.exp_vals.size(); ++i) {
        REQUIRE(a.exp_vals[i] == b.exp_vals[i]);
    }

    for (uint64_t i = 0; i < a.v_size(); ++i) {
        REQUIRE(a.v()[i] == b.v()[i]);
    }
}

}  // namespace

TEST_CASE("reset() on a freshly constructed state leaves it ready for execute()") {
    StateConfig cfg{
        .peak_rank = 4,
        .num_measurements = 2,
        .num_qubits = 4,
        .num_detectors = 1,
        .num_observables = 1,
        .num_exp_vals = 1,
        .seed = 42,
    };
    auto prog = make_unitary_program(
        {
            make_expand(0),
            make_array_h(0),
            make_expand(1),
            make_array_t(1),
            make_frame_cnot(0, 1),
        },
        /*peak_rank=*/2);

    SchrodingerState fresh(cfg);
    SchrodingerState reset_after_construct(cfg);
    reset_after_construct.reset();

    execute(prog, fresh);
    execute(prog, reset_after_construct);

    require_post_execute_states_match(fresh, reset_after_construct);
}

TEST_CASE("execute-reset-execute matches execute on a fresh state (Clifford+T)") {
    // The load-bearing reuse invariant: re-running execute() after
    // reset() must produce the same end-state as running execute() on
    // a freshly constructed instance.
    auto prog = make_unitary_program(
        {
            make_expand(0),
            make_array_h(0),
            make_expand(1),
            make_array_t(1),
            make_frame_cnot(0, 1),
            make_array_t(1),
            make_array_h(0),
        },
        /*peak_rank=*/2);

    StateConfig cfg{.peak_rank = 2, .num_measurements = 0, .num_qubits = 2, .seed = 0};

    SchrodingerState reused(cfg);
    execute(prog, reused);
    reused.reset();
    execute(prog, reused);

    SchrodingerState fresh(cfg);
    execute(prog, fresh);

    require_post_execute_states_match(fresh, reused);
}

TEST_CASE("execute-reset-execute on a measured circuit is reproducible") {
    // Same invariant for a circuit with measurements. Reseed before the
    // second run so the two runs see the same PRNG stream.
    auto prog = make_measured_program(
        {
            make_expand(0),
            make_array_h(0),
            make_expand(1),
            make_array_h(1),
            make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL, 1, 0, false),
            make_meas(Opcode::OP_MEAS_ACTIVE_DIAGONAL, 0, 1, false),
        },
        /*peak_rank=*/2,
        /*num_meas=*/2);

    StateConfig cfg{.peak_rank = 2, .num_measurements = 2, .num_qubits = 2, .seed = 12345};

    SchrodingerState reused(cfg);
    execute(prog, reused);
    auto saved_first_record = reused.meas_record;
    reused.reset();
    reused.reseed(12345);
    execute(prog, reused);

    SchrodingerState fresh(cfg);
    execute(prog, fresh);

    require_post_execute_states_match(fresh, reused);
    // The repeat run reproduces the first run's record because the
    // second execute on `reused` ran with seed 12345, same as `fresh`.
    REQUIRE(saved_first_record[0] == fresh.meas_record[0]);
    REQUIRE(saved_first_record[1] == fresh.meas_record[1]);
}

TEST_CASE("reset() preserves the v_ allocation pointer") {
    // Reuse must be allocation-free: reset() must not realloc v_.
    SchrodingerState state(/*peak_rank=*/8, /*num_measurements=*/0);
    auto* arr_before = state.v();
    state.reset();
    REQUIRE(state.v() == arr_before);
}

TEST_CASE("reset() preserves p_x / p_z buffer sizing") {
    // p_x and p_z are sized by num_qubits at construction. reset()
    // must not resize them, and must clear their contents (the Pauli
    // frame is part of the state that reset must restore).
    StateConfig cfg{.peak_rank = 2, .num_measurements = 0, .num_qubits = 200};
    SchrodingerState state(cfg);
    auto px_size = state.p_x.size();
    auto pz_size = state.p_z.size();
    REQUIRE(px_size > 0);
    REQUIRE(pz_size == px_size);

    state.p_x[0] = 0xDEADBEEF;
    state.p_z[1] = 0xCAFEBABE;
    state.reset();

    REQUIRE(state.p_x.size() == px_size);
    REQUIRE(state.p_z.size() == pz_size);
    REQUIRE(state.p_x[0] == 0);
    REQUIRE(state.p_z[1] == 0);
}
