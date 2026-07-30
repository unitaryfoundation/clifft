// Backend lowering tests for INSTRUMENT ops: opcode selection by the
// localized-basis classification, the site -> bytecode offset table, the
// destination flip mask, peak_rank accounting, and the fixed
// prefix-identity contract that trap re-entry depends on.

#include "clifft/backend/backend.h"
#include "clifft/frontend/frontend.h"

#include "instrument_test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstring>
#include <string>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::test::compile_instruments_full;
using clifft::test::compile_instruments_raw;
using clifft::test::source_dependent_jump_options;

namespace {

constexpr double kTol = 1e-15;

bool is_instrument_opcode(Opcode op) {
    return op == Opcode::OP_INSTRUMENT_ACTIVE || op == Opcode::OP_INSTRUMENT_DORMANT_STATIC ||
           op == Opcode::OP_INSTRUMENT_EXPAND || op == Opcode::OP_INSTRUMENT_DORMANT_NEGLECT;
}

// Index of the single instrument instruction in the module.
size_t sole_instrument_index(const CompiledModule& module) {
    size_t found = module.bytecode.size();
    size_t count = 0;
    for (size_t i = 0; i < module.bytecode.size(); ++i) {
        if (is_instrument_opcode(module.bytecode[i].opcode)) {
            found = i;
            ++count;
        }
    }
    REQUIRE(count == 1);
    return found;
}

}  // namespace

// Classification

TEST_CASE("lowering: a fresh-qubit site is dormant-static with its spec in the pool") {
    auto module =
        compile_instruments_raw("LEVEL_TRANSITION[jump] 0", source_dependent_jump_options());

    const size_t at = sole_instrument_index(module);
    const Instruction& instr = module.bytecode[at];
    REQUIRE(instr.opcode == Opcode::OP_INSTRUMENT_DORMANT_STATIC);
    REQUIRE(instr.axis_1 == 0);
    REQUIRE((instr.flags & Instruction::FLAG_SIGN) == 0);
    REQUIRE_THAT(instr.instrument.r_g, WithinAbs(std::sqrt(0.9), kTol));
    REQUIRE_THAT(instr.instrument.r_e, WithinAbs(std::sqrt(0.6), kTol));
    REQUIRE(module.peak_rank == 0);

    REQUIRE(module.constant_pool.instrument_sites.size() == 1);
    const CompiledInstrumentSite& site =
        module.constant_pool.instrument_sites[instr.instrument.cp_site_idx];
    const InstrumentProbabilities& probabilities = site.probabilities;
    REQUIRE(site.site_id == 0);
    REQUIRE_THAT(probabilities.p_fire[0], WithinAbs(0.1, kTol));
    REQUIRE_THAT(probabilities.p_fire[1], WithinAbs(0.4, kTol));
    REQUIRE_THAT(probabilities.p_computational_dest[0][0], WithinAbs(0.02, kTol));
    REQUIRE_THAT(probabilities.p_computational_dest[0][1], WithinAbs(0.03, kTol));
    REQUIRE_THAT(probabilities.p_noncomputational_dest(1), WithinAbs(0.4, kTol));

    // Identity frame: the destination flip is exactly X on qubit 0.
    auto flip =
        module.constant_pool.instrument_destination_flip_masks.at(site.destination_flip_mask);
    REQUIRE(flip.x().bit_get(0));
    REQUIRE(flip.x().popcount() == 1);
    REQUIRE(flip.z().popcount() == 0);

    REQUIRE(module.instrument_offsets.size() == 1);
    REQUIRE(module.instrument_offsets[0] == at);
}

TEST_CASE("lowering: a Pauli before the site lands in FLAG_SIGN") {
    auto module =
        compile_instruments_raw("X 0\nLEVEL_TRANSITION[jump] 0", source_dependent_jump_options());
    const Instruction& instr = module.bytecode[sole_instrument_index(module)];
    REQUIRE(instr.opcode == Opcode::OP_INSTRUMENT_DORMANT_STATIC);
    REQUIRE((instr.flags & Instruction::FLAG_SIGN) != 0);
}

TEST_CASE("lowering: an active qubit gets the fused active form on its axis") {
    // H 0; T 0 activates qubit 0 (the T's route expands it); the site's
    // rewound X-projector maps through the route's virtual H back to Z.
    auto module = compile_instruments_raw("H 0\nT 0\nLEVEL_TRANSITION[jump] 0",
                                          source_dependent_jump_options());

    const size_t at = sole_instrument_index(module);
    const Instruction& instr = module.bytecode[at];
    REQUIRE(instr.opcode == Opcode::OP_INSTRUMENT_ACTIVE);
    REQUIRE(instr.axis_1 == 0);
    REQUIRE(module.peak_rank == 1);  // the T's expansion; the instrument adds nothing
    // Z-routed directly: no basis-change Hadamard before the site.
    REQUIRE(module.bytecode[at - 1].opcode != Opcode::OP_ARRAY_H);
}

TEST_CASE("lowering: an active X-basis site gets an absorbed array Hadamard") {
    // The trailing H makes the rewound projector plain Z, which the
    // route's accumulated virtual H maps to X on the active axis.
    auto module = compile_instruments_raw("H 0\nT 0\nH 0\nLEVEL_TRANSITION[jump] 0",
                                          source_dependent_jump_options());

    const size_t at = sole_instrument_index(module);
    REQUIRE(module.bytecode[at].opcode == Opcode::OP_INSTRUMENT_ACTIVE);
    REQUIRE(at > 0);
    REQUIRE(module.bytecode[at - 1].opcode == Opcode::OP_ARRAY_H);
}

TEST_CASE("lowering: source-dependent rates under exact damping expand at the site") {
    auto module =
        compile_instruments_raw("H 0\nLEVEL_TRANSITION[jump] 0", source_dependent_jump_options());

    const size_t at = sole_instrument_index(module);
    REQUIRE(module.bytecode[at].opcode == Opcode::OP_INSTRUMENT_EXPAND);
    REQUIRE(at > 0);
    REQUIRE(module.bytecode[at - 1].opcode == Opcode::OP_FRAME_H);
    REQUIRE(module.peak_rank == 1);  // +1 to k at the site
}

TEST_CASE("lowering: equal per-source rates skip the expansion even under exact damping") {
    // The trap-form lowering is exact when the two computational columns
    // agree (the skipped no-fire back-action is proportional to identity),
    // so exact damping takes it and k stays flat.
    InstrumentTraceOptions options;
    InstrumentProbabilities spec;
    spec.p_fire[0] = 0.1;
    spec.p_computational_dest[0][0] = 0.02;
    spec.p_computational_dest[0][1] = 0.03;
    spec.p_fire[1] = 0.1;
    spec.p_computational_dest[1][0] = 0.02;
    spec.p_computational_dest[1][1] = 0.03;
    options.transitions.emplace("jump", spec);

    auto module = compile_instruments_raw("H 0\nLEVEL_TRANSITION[jump] 0", options);
    const size_t at = sole_instrument_index(module);
    REQUIRE(module.bytecode[at].opcode == Opcode::OP_INSTRUMENT_DORMANT_NEGLECT);
    REQUIRE(module.peak_rank == 0);
}

TEST_CASE("lowering: dormant-random under neglect keeps k and skips the expansion") {
    auto module = compile_instruments_raw("H 0\nLEVEL_TRANSITION[jump] 0",
                                          source_dependent_jump_options(/*neglect_damping=*/true));

    const size_t at = sole_instrument_index(module);
    REQUIRE(module.bytecode[at].opcode == Opcode::OP_INSTRUMENT_DORMANT_NEGLECT);
    REQUIRE(module.peak_rank == 0);
    for (const auto& instr : module.bytecode) {
        REQUIRE(instr.opcode != Opcode::OP_EXPAND);
        REQUIRE(instr.opcode != Opcode::OP_FRAME_H);
    }
}

TEST_CASE("lowering: an entangled site still localizes to one instrument") {
    // The rewound projector of the CNOT target is a two-qubit Pauli; the
    // localization emits reduction ops and one instrument on the pivot.
    auto module = compile_instruments_raw("H 0\nCX 0 1\nT 0\nLEVEL_TRANSITION[jump] 1",
                                          source_dependent_jump_options());
    const size_t at = sole_instrument_index(module);
    const auto& site =
        module.constant_pool.instrument_sites[module.bytecode[at].instrument.cp_site_idx];
    // The destination flip is the virtualized X of the annotated qubit; entanglement
    // makes it a genuine mask, not necessarily single-qubit.
    auto flip =
        module.constant_pool.instrument_destination_flip_masks.at(site.destination_flip_mask);
    REQUIRE(flip.x().popcount() + flip.z().popcount() > 0);
}

// Offset table and prefix identity

TEST_CASE("lowering: offsets stay valid through the default bytecode passes") {
    // Noise before the site coalesces into a block under the default
    // passes, shifting instruction indices; the rebuilt table must still
    // point at the instrument.
    auto module = compile_instruments_full(
        "X_ERROR(0.01) 0\nX_ERROR(0.01) 1\nX_ERROR(0.01) 2\n"
        "LEVEL_TRANSITION[jump] 0\nM 0",
        source_dependent_jump_options());

    REQUIRE(module.instrument_offsets.size() == 1);
    const uint32_t offset = module.instrument_offsets[0];
    REQUIRE(offset < module.bytecode.size());
    REQUIRE(is_instrument_opcode(module.bytecode[offset].opcode));
}

TEST_CASE("fences: noise blocks never coalesce across an instrument") {
    // The adjacency-driven bytecode passes stop at unrecognized opcodes
    // by construction; exercise the pass that matters most (noise-block
    // coalescing drove the atomized-fence cost in the spike): noise on
    // both sides of a site stays on both sides, in separate runs.
    auto module = compile_instruments_full(
        "X_ERROR(0.01) 0\nX_ERROR(0.01) 1\nLEVEL_TRANSITION[jump] 2\n"
        "X_ERROR(0.01) 0\nX_ERROR(0.01) 1\nM 0",
        source_dependent_jump_options());

    const size_t at = sole_instrument_index(module);
    bool noise_before = false;
    bool noise_after = false;
    for (size_t i = 0; i < module.bytecode.size(); ++i) {
        const Instruction& instr = module.bytecode[i];
        if (instr.opcode == Opcode::OP_NOISE || instr.opcode == Opcode::OP_NOISE_BLOCK) {
            (i < at ? noise_before : noise_after) = true;
            if (instr.opcode == Opcode::OP_NOISE_BLOCK) {
                REQUIRE(instr.pauli.condition_idx < 4);  // never one run of all four
            }
        }
    }
    REQUIRE(noise_before);
    REQUIRE(noise_after);
}

TEST_CASE("exact record and basis probabilities reject instrument programs") {
    auto module =
        compile_instruments_raw("LEVEL_TRANSITION[jump] 0\nM 0", source_dependent_jump_options());
    const std::vector<uint8_t> record{0};
    REQUIRE_THROWS_WITH(
        record_probabilities(module, record, 1),
        ContainsSubstring("record_probabilities()") && ContainsSubstring("transition instruments"));

    // basis_probabilities takes measurement-free unitary programs, so its
    // rejection path needs its own instrument program to exercise.
    auto unitary =
        compile_instruments_raw("LEVEL_TRANSITION[jump] 0", source_dependent_jump_options());
    const std::vector<uint64_t> masks{0};
    REQUIRE_THROWS_WITH(
        basis_probabilities(unitary, masks, 1, 1),
        ContainsSubstring("basis_probabilities()") && ContainsSubstring("transition instruments"));
}

TEST_CASE("fences: prefix compilation is bit-identical across different suffixes") {
    // The re-entry contract:
    // everything up to and including the instrument must compile
    // identically no matter what follows the fence, through the full
    // default pipeline. The suffixes differ in ways the peephole would
    // exploit across the fence if it could (a fusable T pair).
    const auto options = source_dependent_jump_options();
    const char* with_fusion_bait = "H 0\nT 0\nLEVEL_TRANSITION[jump] 0\nT 0\nM 0";
    const char* with_other_suffix = "H 0\nT 0\nLEVEL_TRANSITION[jump] 0\nT_DAG 0\nH 0\nM 0\nM 1";

    auto a = compile_instruments_full(with_fusion_bait, options);
    auto b = compile_instruments_full(with_other_suffix, options);

    REQUIRE(a.instrument_offsets.size() == 1);
    REQUIRE(b.instrument_offsets.size() == 1);
    const uint32_t offset = a.instrument_offsets[0];
    REQUIRE(b.instrument_offsets[0] == offset);

    for (uint32_t i = 0; i <= offset; ++i) {
        REQUIRE(std::memcmp(&a.bytecode[i], &b.bytecode[i], sizeof(Instruction)) == 0);
    }
}
