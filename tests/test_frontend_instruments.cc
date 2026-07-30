// Frontend materialization and fence tests for INSTRUMENT ops.
//
// trace() with InstrumentTraceOptions materializes LEVEL_TRANSITION and
// LOSS annotations into OpType::INSTRUMENT ops carrying the rewound
// source projector Z_q, with per-source probabilities in the
// InstrumentSite side-table. Without options the annotations reject as
// before. Instruments are optimization fences: no pass moves anything
// across one, enforced in can_swap() and the peephole's commute check.

#include "clifft/backend/backend.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/instrument_options.h"
#include "clifft/noncomp/model.h"
#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"

#include "instrument_test_helpers.h"
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::test::source_dependent_jump_options;

namespace {

constexpr double kTol = 1e-15;

HirModule hir_with_instruments(const char* text, const InstrumentTraceOptions& options) {
    return trace(parse(text), &options);
}

// True iff the op's mask is exactly the single-qubit Pauli (x_bit, z_bit)
// on `qubit` with the given sign.
bool mask_is(const HirModule& hir, const HeisenbergOp& op, uint32_t qubit, bool x_bit, bool z_bit,
             bool sign) {
    auto destab = hir.destab_mask(op);
    auto stab = hir.stab_mask(op);
    for (uint32_t q = 0; q < hir.num_qubits; ++q) {
        const bool want_x = x_bit && q == qubit;
        const bool want_z = z_bit && q == qubit;
        if (destab.bit_get(q) != want_x || stab.bit_get(q) != want_z) {
            return false;
        }
    }
    return hir.sign(op) == sign;
}

}  // namespace

// Materialization

TEST_CASE("trace: annotations still reject without instrument options") {
    REQUIRE_THROWS_WITH(trace(parse("LEVEL_TRANSITION[jump] 0")),
                        ContainsSubstring("noncomputational annotation"));
    REQUIRE_THROWS_WITH(trace(parse("LOSS(0.1) 0")),
                        ContainsSubstring("noncomputational annotation"));
}

TEST_CASE("trace: LEVEL_TRANSITION materializes an INSTRUMENT with its spec") {
    const auto options = source_dependent_jump_options();
    auto hir = hir_with_instruments("H 1\nLEVEL_TRANSITION[jump] 0", options);

    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::INSTRUMENT);
    REQUIRE(!hir.is_deterministic());

    REQUIRE(hir.instrument_sites.size() == 1);
    const InstrumentSite& site = hir.instrument_sites[0];
    const InstrumentProbabilities& probabilities = site.probabilities;
    REQUIRE(site.qubit == 0);
    REQUIRE_FALSE(hir.neglect_instrument_damping);
    REQUIRE_THAT(probabilities.p_fire[0], WithinAbs(0.1, kTol));
    REQUIRE_THAT(probabilities.p_fire[1], WithinAbs(0.4, kTol));
    REQUIRE_THAT(probabilities.p_computational_dest[0][0], WithinAbs(0.02, kTol));
    REQUIRE_THAT(probabilities.p_computational_dest[0][1], WithinAbs(0.03, kTol));
    REQUIRE_THAT(probabilities.p_computational_dest[1][0], WithinAbs(0.0, kTol));
    REQUIRE_THAT(probabilities.p_computational_dest[1][1], WithinAbs(0.0, kTol));
    REQUIRE_THAT(probabilities.p_noncomputational_dest(0), WithinAbs(0.05, kTol));
    REQUIRE_THAT(probabilities.p_noncomputational_dest(1), WithinAbs(0.4, kTol));

    // Identity tableau at the site: the mask is plain Z on qubit 0.
    REQUIRE(hir.ops[0].instrument_site_idx() == InstrumentSiteIdx{0});
    REQUIRE(mask_is(hir, hir.ops[0], 0, /*x=*/false, /*z=*/true, /*sign=*/false));
}

TEST_CASE("trace: the instrument mask is the rewound source projector") {
    const auto options = source_dependent_jump_options();

    // X before the site: Z -> -Z, so the mask picks up a sign.
    auto flipped = hir_with_instruments("X 0\nLEVEL_TRANSITION[jump] 0", options);
    REQUIRE(flipped.ops.size() == 1);
    REQUIRE(mask_is(flipped, flipped.ops[0], 0, false, true, /*sign=*/true));

    // H before the site: Z -> X, the localized basis rotates.
    auto rotated = hir_with_instruments("H 0\nLEVEL_TRANSITION[jump] 0", options);
    REQUIRE(rotated.ops.size() == 1);
    REQUIRE(mask_is(rotated, rotated.ops[0], 0, /*x=*/true, /*z=*/false, false));
}

TEST_CASE("trace: LOSS materializes a source-independent all-trap site per target") {
    const InstrumentTraceOptions options;  // LOSS needs no named specs
    auto hir = hir_with_instruments("LOSS(0.25) 0 1", options);

    REQUIRE(hir.ops.size() == 2);
    REQUIRE(hir.instrument_sites.size() == 2);
    for (int i = 0; i < 2; ++i) {
        const InstrumentSite& site = hir.instrument_sites[static_cast<size_t>(i)];
        const InstrumentProbabilities& probabilities = site.probabilities;
        REQUIRE(site.qubit == static_cast<uint32_t>(i));
        REQUIRE_THAT(probabilities.p_fire[0], WithinAbs(0.25, kTol));
        REQUIRE_THAT(probabilities.p_fire[1], WithinAbs(0.25, kTol));
        REQUIRE_THAT(probabilities.p_noncomputational_dest(0), WithinAbs(0.25, kTol));
        REQUIRE_THAT(probabilities.p_noncomputational_dest(1), WithinAbs(0.25, kTol));
        REQUIRE(mask_is(hir, hir.ops[static_cast<size_t>(i)], static_cast<uint32_t>(i), false, true,
                        false));
    }
}

TEST_CASE("trace: a zero-rate site is elided") {
    InstrumentTraceOptions options;
    options.transitions.emplace("nothing", InstrumentProbabilities{});

    auto hir = hir_with_instruments("LOSS(0) 0\nLEVEL_TRANSITION[nothing] 0", options);
    REQUIRE(hir.ops.empty());
    REQUIRE(hir.instrument_sites.empty());
    REQUIRE(hir.is_deterministic());
}

TEST_CASE("trace: an unresolved tag names itself in the error") {
    const InstrumentTraceOptions options;
    REQUIRE_THROWS_WITH(
        hir_with_instruments("LEVEL_TRANSITION[ghost] 0", options),
        ContainsSubstring("LEVEL_TRANSITION[ghost]") && ContainsSubstring("instrument options"));
}

TEST_CASE("trace: malformed compressed instrument probabilities reject") {
    InstrumentTraceOptions options;
    InstrumentProbabilities invalid;
    invalid.p_fire[0] = 0.1;
    invalid.p_computational_dest[0][0] = 0.2;
    options.transitions.emplace("invalid", invalid);

    REQUIRE_THROWS_WITH(
        hir_with_instruments("LEVEL_TRANSITION[invalid] 0", options),
        ContainsSubstring("LEVEL_TRANSITION[invalid]") && ContainsSubstring("above p_fire"));
}

TEST_CASE("trace: non-finite compressed instrument probabilities reject") {
    InstrumentTraceOptions options;
    InstrumentProbabilities invalid;
    invalid.p_fire[0] = clifft::test::opaque_nan();
    options.transitions.emplace("invalid", invalid);

    REQUIRE_THROWS_WITH(
        hir_with_instruments("LEVEL_TRANSITION[invalid] 0", options),
        ContainsSubstring("LEVEL_TRANSITION[invalid]") && ContainsSubstring("invalid p_fire"));
}

TEST_CASE("trace: the damping policy is recorded once on the HIR module") {
    auto options = source_dependent_jump_options();
    options.neglect_instrument_damping = true;
    auto hir = hir_with_instruments("LEVEL_TRANSITION[jump] 0\nLOSS(0.1) 1", options);
    REQUIRE(hir.instrument_sites.size() == 2);
    REQUIRE(hir.neglect_instrument_damping);
}

// instrument_trace_options: model -> specs

TEST_CASE("instrument_trace_options: resolves model transitions and policy") {
    // T[to][from]: from e, 0.1 relaxes to g and 0.3 leaks; from g, nothing.
    std::vector<std::vector<double>> matrix(5, std::vector<double>(5, 0.0));
    matrix[0][1] = 0.1;  // e -> g
    matrix[3][1] = 0.3;  // e -> leak_e

    NonComputationalPolicy policy;
    policy.damping = DampingPolicy::Neglect;
    const auto model = NonComputationalModel::from_spec({1.0, 0.0, 0.0, 0.0, 0.0},
                                                        {{"relax", matrix}}, std::nullopt, policy);

    const InstrumentTraceOptions options = instrument_trace_options(model);
    REQUIRE(options.neglect_instrument_damping);
    REQUIRE(options.transitions.size() == 1);
    const InstrumentProbabilities& probabilities = options.transitions.at("relax");
    REQUIRE_THAT(probabilities.p_fire[0], WithinAbs(0.0, kTol));
    REQUIRE_THAT(probabilities.p_fire[1], WithinAbs(0.4, kTol));
    REQUIRE_THAT(probabilities.p_computational_dest[1][0], WithinAbs(0.1, kTol));
    REQUIRE_THAT(probabilities.p_computational_dest[1][1], WithinAbs(0.0, kTol));
    REQUIRE_THAT(probabilities.p_noncomputational_dest(1), WithinAbs(0.3, kTol));
}

// Fences

TEST_CASE("fences: can_swap refuses to move anything across an instrument") {
    const auto options = source_dependent_jump_options();
    // The T and the instrument act on different qubits: their masks
    // commute, so only the positional clause can block the swap.
    auto hir = hir_with_instruments("T 0\nLEVEL_TRANSITION[jump] 1", options);
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(!can_swap(hir.ops[0], hir.ops[1], hir));
    REQUIRE(!can_swap(hir.ops[1], hir.ops[0], hir));
}

TEST_CASE("fences: the peephole does not fuse a T pair across an instrument") {
    const auto options = source_dependent_jump_options();
    auto fenced = hir_with_instruments("T 0\nLEVEL_TRANSITION[jump] 1\nT 0", options);
    REQUIRE(fenced.ops.size() == 3);

    PeepholeFusionPass pass;
    pass.run(fenced);

    REQUIRE(pass.fusions() == 0);
    REQUIRE(fenced.ops.size() == 3);
    REQUIRE(fenced.ops[0].op_type() == OpType::T_GATE);
    REQUIRE(fenced.ops[1].op_type() == OpType::INSTRUMENT);
    REQUIRE(fenced.ops[2].op_type() == OpType::T_GATE);
}

TEST_CASE("fences: the squeeze pass does not bubble a measurement across an instrument") {
    const auto options = source_dependent_jump_options();
    // Without the barrier, M 0 commutes with a site on qubit 1 and would
    // compact leftward past it.
    auto hir = hir_with_instruments("LEVEL_TRANSITION[jump] 1\nM 0", options);
    REQUIRE(hir.ops.size() == 2);

    StatevectorSqueezePass pass;
    pass.run(hir);

    REQUIRE(hir.ops[0].op_type() == OpType::INSTRUMENT);
    REQUIRE(hir.ops[1].op_type() == OpType::MEASURE);
}

TEST_CASE("fences: an absorbed virtual S conjugates the instrument mask like a measurement") {
    const auto options = source_dependent_jump_options();
    // T 0; T 0 fuses to a virtual S along Z(0). The H makes the site's
    // rewound projector X(0), which anti-commutes with the S axis, so the
    // absorption must rotate it to Y(0) -- the same conjugation measures
    // and probes receive. The site's destination flip (rewound X = Z(0) here)
    // commutes with the S axis and must stay put.
    auto hir = hir_with_instruments("T 0\nT 0\nH 0\nLEVEL_TRANSITION[jump] 0", options);
    REQUIRE(hir.ops.size() == 3);
    REQUIRE(mask_is(hir, hir.ops[2], 0, /*x=*/true, /*z=*/false, false));

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(pass.fusions() == 1);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::INSTRUMENT);
    // Y-like: both destab and stab bits set on qubit 0.
    auto destab = hir.destab_mask(hir.ops[0]);
    auto stab = hir.stab_mask(hir.ops[0]);
    REQUIRE(destab.bit_get(0));
    REQUIRE(stab.bit_get(0));

    auto flip = hir.pauli_masks.at(hir.instrument_sites[0].destination_flip_mask);
    REQUIRE(!flip.x().bit_get(0));
    REQUIRE(flip.z().bit_get(0));
}

TEST_CASE("fences: an absorbed virtual S conjugates the side-table destination flip too") {
    const auto options = source_dependent_jump_options();
    // With the T pair after the H, the virtual S runs along X(0): now the
    // site's own mask (X-like) commutes and stays put, while the destination flip
    // (rewound X = Z(0)) anti-commutes and must rotate to Y(0). A sweep
    // that only conjugates op-attached masks misses the side-table
    // destination flip.
    auto hir = hir_with_instruments("H 0\nT 0\nT 0\nLEVEL_TRANSITION[jump] 0", options);
    REQUIRE(hir.ops.size() == 3);

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(pass.fusions() == 1);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::INSTRUMENT);
    // Op mask: still X-like.
    REQUIRE(hir.destab_mask(hir.ops[0]).bit_get(0));
    REQUIRE(!hir.stab_mask(hir.ops[0]).bit_get(0));
    // Destination flip: rotated to Y-like.
    auto flip = hir.pauli_masks.at(hir.instrument_sites[0].destination_flip_mask);
    REQUIRE(flip.x().bit_get(0));
    REQUIRE(flip.z().bit_get(0));
}

TEST_CASE("trace: a hand-built LOSS without its argument rejects") {
    // The parser guarantees LOSS(p); a programmatically built node with
    // no argument must reject here rather than trace as a silent
    // zero-probability site.
    const InstrumentTraceOptions options;
    Circuit c = parse("H 0");
    c.nodes.push_back(AstNode{GateType::LOSS, {Target::qubit(0)}, {}, 0});
    REQUIRE_THROWS_WITH(trace(c, &options), ContainsSubstring("exactly one argument"));
}

// forced_traceout_node / forced_traceout_slot

TEST_CASE("trace: forced_traceout_node reports the hidden slot of the requested reset") {
    // The SINGLE-arity parser emits one node per target, so "R 2 3" becomes
    // two nodes. Parsed circuit: M 0 (n0), M 1 (n1), R 2 (n2), R 3 (n3),
    // R 4 (n4), M 5 (n5). num_visible = 3; hidden_meas_idx starts at 3.
    //   node 2 (R 2): hidden slot 3
    //   node 3 (R 3): hidden slot 4  <- forced_traceout_node = 3
    //   node 4 (R 4): hidden slot 5
    // forced_traceout_slot = 4 (third hidden slot, index 1 in hidden slots)
    InstrumentTraceOptions options;
    options.forced_traceout_node = 3;  // R 3 (second of the two split resets)
    auto hir = trace(parse("M 0\nM 1\nR 2 3\nR 4\nM 5"), &options);
    REQUIRE(hir.forced_traceout_slot == 4);
}

TEST_CASE("trace: forced_traceout_node on the first reset yields slot == num_visible") {
    // Circuit: R 0  M 1  M 2
    //   node 0: R 0 -> hidden slot 2 (num_visible = 2)
    //   node 1: M 1 -> visible slot 0
    //   node 2: M 2 -> visible slot 1
    // hidden before node 0 = 0
    // forced_traceout_slot = 2 + 0 = 2
    InstrumentTraceOptions options;
    options.forced_traceout_node = 0;  // R 0, the first and only reset
    auto hir = trace(parse("R 0\nM 1\nM 2"), &options);
    REQUIRE(hir.forced_traceout_slot == 2);
}

TEST_CASE("trace: forced_traceout_node rejects a multi-target reset") {
    Circuit circuit;
    circuit.num_qubits = 2;
    circuit.nodes.push_back(AstNode{GateType::R, {Target::qubit(0), Target::qubit(1)}, {}, 0});
    InstrumentTraceOptions options;
    options.forced_traceout_node = 0;
    REQUIRE_THROWS_WITH(trace(circuit, &options), ContainsSubstring("single-target reset"));
}

TEST_CASE("trace: forced_traceout_node unset leaves forced_traceout_slot empty") {
    // No forced_traceout_node set: the output slot stays at its default.
    InstrumentTraceOptions options;
    // forced_traceout_node defaults to nullopt -- no request
    auto hir = trace(parse("M 0\nR 1\nM 2"), &options);
    REQUIRE(!hir.forced_traceout_slot.has_value());
}
