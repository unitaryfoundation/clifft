#include "clifft/backend/backend.h"
#include "clifft/noncomp/continuation_prefix.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;

namespace {

CompiledModule module_with_prefix_constants() {
    CompiledModule module;
    module.num_qubits = 2;
    ConstantPool& pool = module.constant_pool;

    pool.fused_u2_nodes.emplace_back();
    pool.fused_u2_nodes[0].matrices[0][0] = {1.0, 0.25};
    pool.fused_u2_nodes[0].gamma_multipliers[0] = {0.5, -0.5};
    pool.fused_u2_nodes[0].out_states[0] = 1;

    pool.fused_u4_nodes.emplace_back();
    pool.fused_u4_nodes[0].entries[0].matrix[0][0] = {0.75, 0.125};
    pool.fused_u4_nodes[0].entries[0].gamma_multiplier = {0.0, 1.0};
    pool.fused_u4_nodes[0].entries[0].out_state = 2;

    pool.pauli_masks = PauliMaskArena(2, 1);
    pool.pauli_masks.mut_at(PauliMaskHandle{0}).x().bit_set(0, true);
    pool.exp_val_masks = PauliMaskArena(2, 1);
    pool.exp_val_masks.mut_at(PauliMaskHandle{0}).z().bit_set(1, true);

    pool.noise_channel_masks = PauliMaskArena(2, 2);
    pool.noise_channel_masks.mut_at(PauliMaskHandle{0}).x().bit_set(0, true);
    pool.noise_channel_masks.mut_at(PauliMaskHandle{1}).z().bit_set(1, true);
    pool.noise_sites = {
        NoiseSite{{NoiseChannel{PauliMaskHandle{0}, 0.125}}},
        NoiseSite{{NoiseChannel{PauliMaskHandle{1}, 0.25}}},
    };
    pool.noise_hazards = {0.13353139262452263, 0.42121346507630353};

    pool.readout_noise = {ReadoutNoiseEntry{3, 0.1, 0.2}};
    pool.detector_targets = {{0, 2}};
    pool.observable_targets = {{1, 3}};

    pool.instrument_destination_flip_masks = PauliMaskArena(2, 1);
    pool.instrument_destination_flip_masks.mut_at(PauliMaskHandle{0}).x().bit_set(1, true);
    CompiledInstrumentSite site;
    site.site_id = 4;
    site.destination_flip_mask = PauliMaskHandle{0};
    site.probabilities.p_fire[0] = 0.2;
    site.probabilities.p_fire[1] = 0.3;
    site.probabilities.p_computational_dest[0][0] = 0.05;
    site.probabilities.p_computational_dest[1][1] = 0.1;
    pool.instrument_sites.push_back(site);

    module.bytecode = {
        make_array_u2(0, 0),
        make_array_u4(0, 1, 0),
        make_apply_pauli(0, 0),
        make_noise(0),
        make_noise_block(1, 1),
        make_readout_noise(0),
        make_detector(0, 0, ExpectedParity::Zero),
        make_observable(0, 0),
        make_exp_val(0, 0),
        make_instrument(Opcode::OP_INSTRUMENT_ACTIVE, 0, 0, false, 0.8, 0.7),
    };
    return module;
}

}  // namespace

TEST_CASE("continuation prefix validates bytecode and referenced constants") {
    const CompiledModule executed = module_with_prefix_constants();
    CompiledModule continuation = executed;
    const uint32_t prefix_end = static_cast<uint32_t>(executed.bytecode.size());

    SECTION("matching modules pass") {
        REQUIRE_NOTHROW(validate_continuation_prefix(continuation, executed, prefix_end));
    }
    SECTION("bytecode divergence rejects") {
        continuation.bytecode[0].axis_1 = 1;
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("bytecode prefix diverged at instruction 0"));
    }
    SECTION("fused U2 data divergence rejects") {
        continuation.constant_pool.fused_u2_nodes[0].matrices[0][0] = {0.0, 0.0};
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 0"));
    }
    SECTION("fused U4 data divergence rejects") {
        continuation.constant_pool.fused_u4_nodes[0].entries[0].out_state = 3;
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 1"));
    }
    SECTION("conditional Pauli mask divergence rejects") {
        continuation.constant_pool.pauli_masks.mut_at(PauliMaskHandle{0}).z().bit_set(0, true);
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 2"));
    }
    SECTION("individual noise-site divergence rejects") {
        continuation.constant_pool.noise_sites[0].channels[0].prob = 0.5;
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 3"));
    }
    SECTION("noise-block mask divergence rejects") {
        continuation.constant_pool.noise_channel_masks.mut_at(PauliMaskHandle{1})
            .x()
            .bit_set(1, true);
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 4"));
    }
    SECTION("noise hazard divergence rejects") {
        continuation.constant_pool.noise_hazards[1] = 0.5;
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 4"));
    }
    SECTION("readout data divergence rejects") {
        continuation.constant_pool.readout_noise[0].prob_one_to_zero = 0.4;
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 5"));
    }
    SECTION("detector targets divergence rejects") {
        continuation.constant_pool.detector_targets[0][0] = 1;
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 6"));
    }
    SECTION("observable targets divergence rejects") {
        continuation.constant_pool.observable_targets[0][0] = 0;
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 7"));
    }
    SECTION("expectation mask divergence rejects") {
        continuation.constant_pool.exp_val_masks.mut_at(PauliMaskHandle{0}).x().bit_set(1, true);
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 8"));
    }
    SECTION("instrument data divergence rejects") {
        continuation.constant_pool.instrument_sites[0].probabilities.p_fire[0] = 0.4;
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 9"));
    }
    SECTION("instrument destination mask divergence rejects") {
        continuation.constant_pool.instrument_destination_flip_masks.mut_at(PauliMaskHandle{0})
            .z()
            .bit_set(1, true);
        REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, prefix_end),
                            ContainsSubstring("constant-pool prefix diverged at instruction 9"));
    }
}

TEST_CASE("continuation prefix permits only the expected forced opcode change") {
    CompiledModule continuation;
    continuation.num_qubits = 1;
    continuation.bytecode.push_back(make_meas(Opcode::OP_MEAS_DORMANT_RANDOM, 0, 2, false));
    CompiledModule executed = continuation;
    executed.bytecode[0].opcode = Opcode::OP_MEAS_DORMANT_RANDOM_FORCED;

    REQUIRE_NOTHROW(validate_continuation_prefix(continuation, executed, 1));

    executed.bytecode[0].classical.classical_idx = 3;
    REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, 1),
                        ContainsSubstring("bytecode prefix diverged"));
}

TEST_CASE("continuation prefix ignores constants referenced only by the suffix") {
    CompiledModule executed = module_with_prefix_constants();
    CompiledModule continuation = executed;
    continuation.constant_pool.instrument_sites[0].probabilities.p_fire[0] = 0.9;

    REQUIRE_NOTHROW(validate_continuation_prefix(continuation, executed, 1));
}

TEST_CASE("continuation prefix rejects invalid comparison bounds") {
    CompiledModule executed;
    CompiledModule continuation;
    executed.num_qubits = continuation.num_qubits = 1;

    REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, 1),
                        ContainsSubstring("prefix length 1 exceeds"));

    continuation.num_qubits = 2;
    REQUIRE_THROWS_WITH(validate_continuation_prefix(continuation, executed, 0),
                        ContainsSubstring("changed the module's qubit count"));
}
