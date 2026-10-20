#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/known_stabilizers.h"
#include "clifft/optimizer/phase_polynomial_pass.h"
#include "clifft/optimizer/symbolic_stabilizers.h"
#include "clifft/optimizer/wide_phase_region.h"

#include "instrument_test_helpers.h"
#include "sampling_equivalence_helpers.h"
#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clifft;

namespace {

HirModule parity_block(uint32_t width = 4, const std::vector<uint32_t>& positions = {0, 1, 2, 3}) {
    HirModule hir(width, 16);
    hir.final_tableau.emplace(width);
    for (uint32_t term = 1; term <= 16; ++term) {
        const uint32_t parity = term == 16 ? 15 : term;
        hir.append_tgate(false, [&](MutablePauliMaskView mask) {
            for (uint32_t j = 0; j < 4; ++j) {
                if (parity & (1U << j)) {
                    mask.x().bit_set(positions[j], true);
                }
            }
        });
        hir.source_map.push_back({term});
    }
    return hir;
}

}  // namespace

TEST_CASE("Phase pass reduces collective parity structure and preserves provenance",
          "[optimizer]") {
    auto hir = parity_block();
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(pass.blocks_reduced() == 1);
    REQUIRE(pass.input_t_count() == 16);
    REQUIRE(pass.output_t_count() == 1);
    REQUIRE(analyze_active_width(hir).peak_width == 1);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.destab_mask(hir.ops[0]).words[0] == 15);
    REQUIRE(hir.stab_mask(hir.ops[0]).is_zero());
    REQUIRE(hir.source_map[0].size() == 16);
    REQUIRE(hir.final_tableau->satisfies_invariants());
}

TEST_CASE("Phase pass supports Pauli axes spanning multiple words", "[optimizer]") {
    auto hir = parity_block(130, {0, 63, 64, 129});
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(pass.output_t_count() == 1);
    const auto axis = hir.destab_mask(hir.ops[0]);
    for (uint32_t bit : {0, 63, 64, 129}) {
        REQUIRE(axis.bit_get(bit));
    }
    REQUIRE(axis.popcount() == 4);
}

TEST_CASE("Phase pass preserves an unreduced prefix and its provenance", "[optimizer]") {
    HirModule hir(5, 18);
    hir.final_tableau.emplace(5);
    test::append_tgate(hir, 16, 0, false);
    test::append_measure(hir, 0, 16, false, MeasRecordIdx{0});
    hir.num_measurements = 1;
    hir.source_map = {{1}, {2}};
    for (uint32_t term = 1; term <= 16; ++term) {
        test::append_tgate(hir, term == 16 ? 15 : term, 0, false);
        hir.source_map.push_back({term + 2});
    }
    const auto original = hir;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.blocks_reduced() == 1);
    REQUIRE(hir.ops.size() == 3);
    REQUIRE(hir.source_map.size() == 3);
    for (size_t i = 0; i < 2; ++i) {
        REQUIRE(hir.ops[i].op_type() == original.ops[i].op_type());
        REQUIRE(hir.mask_view(hir.ops[i]) == original.mask_view(original.ops[i]));
        REQUIRE(hir.source_map[i] == original.source_map[i]);
    }
    REQUIRE(hir.source_map.back().size() == 16);
    REQUIRE(hir.source_map.back().front() == original.source_map[2].front());
    REQUIRE(hir.source_map.back().back() == original.source_map.back().back());
}

TEST_CASE("Phase pass removes rotations equivalent on the known input", "[optimizer]") {
    HirModule hir(2, 2);
    hir.final_tableau.emplace(2);
    test::append_tgate(hir, 1, 0, false);
    test::append_tgate(hir, 1, 2, false);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.num_t_gates() == 0);
}

TEST_CASE("Known stabilizers preserve correlations shared with the noiseless reference",
          "[optimizer]") {
    KnownStabilizers known(2);
    auto hir = trace(parse("PAULI_CHANNEL_2(0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0) 0 1\nX_ERROR(1) 0"));
    PauliString z0(2);
    z0.set_pauli(0, false, true);
    PauliString zz = z0;
    zz.set_pauli(1, false, true);
    known.advance(hir, hir.ops[0]);
    REQUIRE_FALSE(known.eigenvalue(z0).has_value());
    REQUIRE(known.eigenvalue(zz) == false);
    known.advance(hir, hir.ops[1]);
    REQUIRE_FALSE(known.eigenvalue(zz).has_value());
}

TEST_CASE("Known stabilizers do not assume a sampled measurement outcome", "[optimizer]") {
    KnownStabilizers known(1);
    auto hir = trace(parse("MX 0"));
    known.advance(hir, hir.ops[0]);
    PauliString x(1);
    x.set_pauli(0, true, false);
    PauliString z(1);
    z.set_pauli(0, false, true);
    REQUIRE_FALSE(known.eigenvalue(x).has_value());
    REQUIRE_FALSE(known.eigenvalue(z).has_value());
}

TEST_CASE("Phase pass can reduce axes modulo known entry constraints", "[optimizer]") {
    for (uint32_t ancilla : {1, 64, 129}) {
        CAPTURE(ancilla);
        const auto cx = "CX " + std::to_string(ancilla) + " 0\n";
        auto hir = trace(parse("H 0\nT 0\n" + cx + "T 0\n" + cx));
        PhasePolynomialPass pass;
        pass.run(hir);
        REQUIRE(pass.applied());
        REQUIRE(hir.num_t_gates() == 0);
    }
}

TEST_CASE("Phase pass respects constraints changed by a crossed boundary", "[optimizer]") {
    for (const auto* boundary : {"X_ERROR(0.25) 1", "MX 1"}) {
        CAPTURE(boundary);
        auto hir = trace(parse(std::string("H 0\nT 0\n") + boundary + "\nCX 1 0\nT 0\nCX 1 0"));
        PhasePolynomialPass pass;
        pass.run(hir);
        REQUIRE_FALSE(pass.applied());
        REQUIRE(hir.num_t_gates() == 2);
    }
}

TEST_CASE("Phase pass composes noncommuting Clifford factors across regions", "[optimizer]") {
    // A mixed input keeps all Clifford factors observable, including ones
    // that would be global on a known eigenstate.
    const std::string prefix = "DEPOLARIZE1(0.1) 0\n";
    auto hir = trace(parse(prefix + "H 0\nT 0\nT 0\nH 0\nT 0\nT 0\nH 0\nT_DAG 0\nT_DAG 0"));
    const auto reference = trace(parse(prefix + "H 0\nS 0\nH 0\nS 0\nH 0\nS_DAG 0"));
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Phase pass pulls back measurements without changing record order", "[optimizer]") {
    HirModule hir(1, 5);
    hir.final_tableau.emplace(1);
    test::append_tgate(hir, 1, 0, false);
    test::append_tgate(hir, 1, 0, false);
    test::append_measure(hir, 0, 1, false, MeasRecordIdx{0});
    test::append_tgate(hir, 1, 0, false);
    test::append_measure(hir, 0, 1, false, MeasRecordIdx{1});
    hir.num_measurements = 2;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(pass.pauli_pullbacks() == 1);
    REQUIRE(hir.num_t_gates() == 1);
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});
    REQUIRE(hir.destab_mask(hir.ops[0]).words[0] == 1);
    REQUIRE(hir.stab_mask(hir.ops[0]).words[0] == 1);
    REQUIRE_FALSE(hir.sign(hir.ops[0]));
    REQUIRE(hir.ops[2].meas_record_idx() == MeasRecordIdx{1});
}

TEST_CASE("Phase pass leaves long sequences of non-Pauli observer barriers intact", "[optimizer]") {
    std::string source = "H 0\n";
    for (unsigned i = 0; i < 1024; ++i) {
        source += "T 0\nMX 0\n";
    }
    auto hir = trace(parse(source));
    const auto original = hir;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.ops.size() == original.ops.size());
    REQUIRE(hir.source_map == original.source_map);
    REQUIRE(hir.logical_noise_prefix == original.logical_noise_prefix);
    REQUIRE(hir.final_tableau == original.final_tableau);
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        REQUIRE(hir.ops[i].op_type() == original.ops[i].op_type());
        REQUIRE(hir.mask_view(hir.ops[i]) == original.mask_view(original.ops[i]));
    }
    REQUIRE(hir.num_measurements == 1024);
}

TEST_CASE("Phase pass retains crossed noise sites and categorical probabilities", "[optimizer]") {
    auto hir = parity_block();
    hir.noise_channel_masks = PauliMaskArena(4, 2);
    const auto first = test::claim_noise_channel_mask(hir, 1, 0);
    const auto second = test::claim_noise_channel_mask(hir, 2, 0);
    hir.noise_sites.push_back({0.3, {{first, 0.1}, {second, 0.2}}});
    hir.ops.insert(hir.ops.begin() + 8, HeisenbergOp::make_noise(NoiseSiteIdx{0}));
    hir.source_map.insert(hir.source_map.begin() + 8, {17});
    const auto sites = hir.noise_sites;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.num_t_gates() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.noise_sites == sites);
    REQUIRE(hir.source_map[0] == std::vector<uint32_t>{17});
}

TEST_CASE("Phase pass bounds regions at noncommuting noise", "[optimizer]") {
    auto hir = trace(parse("H 0\nT 0\nX_ERROR(0.3) 0\nT 0\nMX 0"));
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.num_t_gates() == 2);
}

TEST_CASE("Phase pass keeps noise unsigned while retaining coherent operand signs", "[optimizer]") {
    for (bool collected : {false, true}) {
        CAPTURE(collected);
        std::string source = "H 0\nT 0\nT 0\n";
        if (collected) {
            source += "R_PAULI(0.25) X0\n";
        }
        source += "X_ERROR(0.1) 0\nMX 0";
        auto hir = trace(parse(source));
        PhasePolynomialPass pass;
        pass.run(hir);
        REQUIRE(pass.applied());
        const auto channel = hir.noise_channel_masks.at(hir.noise_sites[0].channels[0].mask);
        REQUIRE(channel.x().words[0] == 1);
        REQUIRE(channel.z().words[0] == 1);
        REQUIRE_FALSE(channel.sign());
        const auto measured = hir.mask_view(hir.ops.back());
        REQUIRE(measured.x().words[0] == 1);
        REQUIRE(measured.z().words[0] == 1);
        REQUIRE_FALSE(measured.sign());
    }
}

TEST_CASE("Phase pass transforms instrument bodies and destination flips", "[optimizer]") {
    const auto options = test::source_dependent_jump_options(false);
    auto hir = trace(parse("H 0\nT 0\nT 0\nH 0\nLEVEL_TRANSITION[jump] 0\nM 0"), &options);
    const auto reference = trace(parse("H 0\nS 0\nH 0\nLEVEL_TRANSITION[jump] 0\nM 0"), &options);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.ops.size() == reference.ops.size());
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        REQUIRE(hir.ops[i].op_type() == reference.ops[i].op_type());
        if (hir.ops[i].has_mask()) {
            REQUIRE(hir.mask_view(hir.ops[i]) == reference.mask_view(reference.ops[i]));
        }
    }
    REQUIRE(hir.pauli_masks.at(hir.instrument_sites[0].destination_flip_mask) ==
            reference.pauli_masks.at(reference.instrument_sites[0].destination_flip_mask));
    REQUIRE(hir.final_tableau == reference.final_tableau);
}

TEST_CASE("Phase pass respects disabling and prior noise scheduling", "[optimizer]") {
    auto hir = parity_block();
    PhasePolynomialPass disabled({0});
    disabled.run(hir);
    REQUIRE_FALSE(disabled.applied());
    hir.logical_noise_prefix.assign(hir.ops.size(), 1);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.num_t_gates() == 16);
    REQUIRE(hir.logical_noise_prefix.size() == hir.ops.size());
    REQUIRE_THROWS_AS(PhasePolynomialPass(PhasePolynomialOptions{129}), std::invalid_argument);
}

TEST_CASE("Phase pass clears statistics between runs", "[optimizer]") {
    PhasePolynomialPass pass;
    auto reducible = parity_block();
    pass.run(reducible);
    REQUIRE(pass.applied());
    auto simple = trace(parse("H 0\nT 0"));
    pass.run(simple);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.pauli_pullbacks() == 0);
    REQUIRE(pass.input_t_count() == 1);
    REQUIRE(pass.output_t_count() == 1);
    REQUIRE(pass.blocks_examined() == 1);
    REQUIRE(pass.blocks_capped() == 0);
    REQUIRE(pass.expansion_attempts() == 0);
    REQUIRE(pass.blocks_expanded() == 0);
}

TEST_CASE("Phase expansion respects explicit limits and the representation ceiling",
          "[optimizer]") {
    for (uint32_t width : {33, 48, 64, 65, 87, 128, 129}) {
        CAPTURE(width);
        HirModule original(width, 2 * width);
        original.final_tableau.emplace(width);
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            for (uint32_t q = 0; q < width; ++q) {
                original.append_tgate(
                    false, [&](MutablePauliMaskView mask) { mask.x().bit_set(q, true); });
            }
        }
        auto bounded = original;
        PhasePolynomialPass limited({32});
        limited.run(bounded);
        REQUIRE(limited.blocks_capped() > 0);
        REQUIRE(limited.expansion_attempts() == 0);
        REQUIRE(bounded.num_t_gates() == original.num_t_gates());
        auto default_result = original;
        PhasePolynomialPass default_pass;
        default_pass.run(default_result);
        REQUIRE(default_pass.expansion_attempts() == 1);
        REQUIRE(default_pass.wide_attempts() == 0);
        REQUIRE(default_pass.wide_work() == 0);
        REQUIRE(default_result.num_t_gates() == (width <= 64 ? 0 : 2 * width));
        PhasePolynomialPass adaptive({128});
        adaptive.run(original);
        REQUIRE(adaptive.expansion_attempts() > 0);
        if (width <= 128) {
            REQUIRE(adaptive.blocks_expanded() == 1);
            REQUIRE(adaptive.blocks_capped() == 0);
            REQUIRE(original.num_t_gates() == 0);
            REQUIRE(analyze_active_width(original).peak_width == 0);
            std::string clifford;
            for (uint32_t q = 0; q < width; ++q) {
                const auto target = std::to_string(q) + "\n";
                clifford += "H " + target + "S " + target + "H " + target;
            }
            REQUIRE(original.final_tableau == trace(parse(clifford)).final_tableau);
        } else {
            REQUIRE(adaptive.blocks_expanded() == 0);
            REQUIRE(adaptive.blocks_capped() > 0);
            REQUIRE(original.num_t_gates() == 2 * width);
        }
        auto empty = trace(parse("M 0"));
        adaptive.run(empty);
        REQUIRE(adaptive.blocks_examined() == 0);
        REQUIRE(adaptive.blocks_capped() == 0);
        REQUIRE(adaptive.expansion_attempts() == 0);
        REQUIRE(adaptive.blocks_expanded() == 0);
    }
}

TEST_CASE("Phase expansion reads pending Clifford corrections only once", "[optimizer]") {
    std::string source = "H 0\nT 0\nT 0\nH 0\nT 0\n";
    std::string reference = "H 0\nS 0\nH 0\nT 0\n";
    for (uint32_t q = 1; q <= 70; ++q) {
        const auto target = std::to_string(q) + "\n";
        source += "H " + target;
        reference += "H " + target + "S " + target + "H " + target;
    }
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        for (uint32_t q = 1; q <= 70; ++q) {
            source += "T " + std::to_string(q) + "\n";
        }
    }
    for (uint32_t q = 1; q <= 70; ++q) {
        source += "H " + std::to_string(q) + "\n";
    }
    auto hir = trace(parse(source));
    const auto expected = trace(parse(reference));
    PhasePolynomialPass pass({128});
    pass.run(hir);
    REQUIRE(pass.blocks_reduced() == 2);
    REQUIRE(pass.blocks_expanded() == 1);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.mask_view(hir.ops[0]) == expected.mask_view(expected.ops[0]));
    REQUIRE(hir.ops[0].is_dagger() == expected.ops[0].is_dagger());
    REQUIRE(hir.final_tableau == expected.final_tableau);
}

TEST_CASE("Wide phase collection stops at noncommuting rotations", "[optimizer]") {
    for (uint32_t width : {65, 128}) {
        for (bool constraint_boundary : {false, true}) {
            CAPTURE(width, constraint_boundary);
            std::string targets;
            for (uint32_t q = 0; q < width; ++q)
                targets += " " + std::to_string(q);
            const auto ancilla = std::to_string(width);
            std::string source = "H" + targets + "\nT" + targets + "\n";
            if (constraint_boundary) {
                // Use the fixed ancilla to equate the first two rotation axes.
                source += "CX " + ancilla + " 0\nT 0\nCX " + ancilla + " 0\n";
                for (uint32_t q = 1; q < width; ++q)
                    source += "T " + std::to_string(q) + "\n";
            } else {
                source += "T" + targets + "\n";
            }
            const auto suffix =
                constraint_boundary ? "H " + ancilla + "\nT " + ancilla + "\n" : "T 0\n";
            source += "H" + targets + "\n" + suffix;
            const auto expected =
                trace(parse("H" + targets + "\nS" + targets + "\nH" + targets + "\n" + suffix));
            auto hir = trace(parse(source));
            PhasePolynomialPass pass({128});
            pass.run(hir);
            REQUIRE(pass.wide_rewrites() == 1);
            REQUIRE(hir.num_t_gates() == 1);
            REQUIRE(hir.ops.size() == expected.ops.size());
            REQUIRE(hir.mask_view(hir.ops[0]) == expected.mask_view(expected.ops[0]));
            REQUIRE(hir.ops[0].is_dagger() == expected.ops[0].is_dagger());
            REQUIRE(hir.final_tableau == expected.final_tableau);
        }
    }
}

TEST_CASE("Wide phase fallback retains narrow observer pullbacks", "[optimizer]") {
    for (const auto* observer : {"M 2", "CX rec[-1] 2", "EXP_VAL Z2", "X_ERROR(0.25) 2"}) {
        CAPTURE(observer);
        std::string source = "H 2\nM 2\nH 0 1\nT 0\nT 0\nT 1\n";
        source += std::string(observer) + "\nT 1\n";
        for (uint32_t q = 3; q < 67; ++q) {
            const auto target = std::to_string(q) + "\n";
            source += "H " + target + "T " + target;
        }
        auto hir = trace(parse(source));
        auto expected = hir;
        PhasePolynomialPass narrow({64});
        narrow.run(expected);
        REQUIRE(expected.num_t_gates() == 64);
        PhasePolynomialPass wide({128});
        wide.run(hir);
        REQUIRE(wide.wide_attempts() > 0);
        REQUIRE(wide.wide_rewrites() == 0);
        REQUIRE(wide.pauli_pullbacks() == narrow.pauli_pullbacks());
        REQUIRE(hir.num_t_gates() == 64);
        REQUIRE(hir.num_measurements == expected.num_measurements);
        REQUIRE(hir.ops.size() == expected.ops.size());
        REQUIRE(hir.final_tableau == expected.final_tableau);
        for (size_t i = 0; i < hir.ops.size(); ++i) {
            REQUIRE(hir.ops[i].op_type() == expected.ops[i].op_type());
            if (hir.ops[i].op_type() != OpType::NOISE) {
                REQUIRE(hir.mask_view(hir.ops[i]) == expected.mask_view(expected.ops[i]));
            }
        }
    }
}

TEST_CASE("Wide phase budgets preserve the ordinary fallback and completed candidates",
          "[optimizer]") {
    const uint32_t width = 65;
    HirModule original(width, 2 * width);
    original.final_tableau.emplace(width);
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        for (uint32_t q = 0; q < width; ++q) {
            original.append_tgate(false,
                                  [&](MutablePauliMaskView mask) { mask.x().bit_set(q, true); });
            original.source_map.push_back({q + repeat * width});
        }
    }
    for (auto options :
         {PhasePolynomialOptions{128, 0, 200000}, PhasePolynomialOptions{128, 1, 200000},
          PhasePolynomialOptions{128, 50000000, 1}}) {
        auto limited = original;
        PhasePolynomialPass pass(options);
        pass.run(limited);
        REQUIRE(pass.wide_rewrites() == 0);
        REQUIRE(limited.num_t_gates() == original.num_t_gates());
        REQUIRE(limited.final_tableau == original.final_tableau);
        REQUIRE(limited.source_map == original.source_map);
        for (size_t i = 0; i < original.ops.size(); ++i) {
            REQUIRE(limited.mask_view(limited.ops[i]) == original.mask_view(original.ops[i]));
        }
    }
    auto without_output = original;
    without_output.final_tableau.reset();
    phase_detail::wide::Budget budget;
    const KnownStabilizers fixed(width);
    const optimizer_detail::CliffordAbsorption frame;
    const auto baseline =
        phase_detail::wide::try_region(without_output, 0, fixed, frame, 128, budget);
    REQUIRE(baseline);
    REQUIRE(baseline->t_count == 0);
    const auto completed_work = budget.updates;
    auto kept = original;
    PhasePolynomialPass limited({128, completed_work, 200000});
    limited.run(kept);
    REQUIRE(limited.wide_budget_exhausted());
    REQUIRE(limited.wide_rewrites() == 1);
    REQUIRE(kept.num_t_gates() == 0);
    auto unlimited = original;
    PhasePolynomialPass({128}).run(unlimited);
    REQUIRE(kept.final_tableau == unlimited.final_tableau);
}

TEST_CASE("Wide phase cost rejection retains ordinary rewrites", "[optimizer]") {
    std::string measured, padding;
    for (uint32_t q = 1; q < 65; ++q) {
        measured += " " + std::to_string(q);
        if (q >= 4)
            padding += " " + std::to_string(q);
    }
    // The measured Z axes add support without active work. Replacing all
    // seven nonempty Z parities and X0 with seven mixed parities saves a T
    // gate but makes all seven rotations act on the active qubit.
    const auto source = "MX" + measured + "\nM" + measured + "\nT" + padding +
                        "\nTPP Z1\nTPP Z2\nTPP Z1*Z2\nTPP Z3\nTPP Z1*Z3\n"
                        "TPP Z2*Z3\nTPP Z1*Z2*Z3\nTPP X0\nT" +
                        padding + "\nM 0 1 2 3\nR_Z(0.137) 0\nT 65\nT 65\n";
    auto hir = trace(parse(source));
    SymbolicStabilizers known(hir);
    size_t start = 0;
    while (hir.ops[start].op_type() != OpType::T_GATE) {
        known.advance(hir, hir.ops[start++]);
    }
    phase_detail::wide::Budget budget;
    const optimizer_detail::CliffordAbsorption frame;
    const auto candidate =
        phase_detail::wide::try_region(hir, start, known.fixed_constraints(), frame, 128, budget);
    REQUIRE(candidate);
    REQUIRE(candidate->t_count == 7);
    REQUIRE(std::ranges::count_if(candidate->phases, [](const auto& term) {
                return (term.second & 1) && term.first.x().bit_get(0);
            }) == 7);

    auto expected = hir;
    PhasePolynomialPass narrow({64});
    narrow.run(expected);
    REQUIRE(expected.num_t_gates() == 129);
    PhasePolynomialPass wide({128});
    wide.run(hir);
    REQUIRE(wide.wide_attempts() == 1);
    REQUIRE_FALSE(wide.wide_budget_exhausted());
    REQUIRE(wide.wide_rewrites() == 0);
    // The completed wide candidate is counted even though the cost guard
    // rejects it; the returned result and statistics belong to the rerun.
    REQUIRE(wide.blocks_expanded() == narrow.blocks_expanded() + 1);
    REQUIRE(wide.blocks_reduced() == narrow.blocks_reduced());
    REQUIRE(wide.output_t_count() == narrow.output_t_count());
    REQUIRE(hir.num_t_gates() == expected.num_t_gates());
    REQUIRE(hir.num_measurements == expected.num_measurements);
    REQUIRE(hir.source_map == expected.source_map);
    REQUIRE(hir.final_tableau == expected.final_tableau);
    REQUIRE(hir.ops.size() == expected.ops.size());
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        REQUIRE(hir.ops[i].op_type() == expected.ops[i].op_type());
        REQUIRE(hir.ops[i].flags() == expected.ops[i].flags());
        REQUIRE(hir.mask_view(hir.ops[i]) == expected.mask_view(expected.ops[i]));
        if (hir.ops[i].op_type() == OpType::MEASURE) {
            REQUIRE(hir.ops[i].meas_record_idx() == expected.ops[i].meas_record_idx());
        }
        if (hir.ops[i].op_type() == OpType::PHASE_ROTATION) {
            REQUIRE(hir.ops[i].alpha() == expected.ops[i].alpha());
        }
    }
}

TEST_CASE("Wide phase work allowance is shared across separated regions", "[optimizer]") {
    HirModule hir(65, 261);
    hir.final_tableau.emplace(65);
    for (uint32_t region = 0; region < 2; ++region) {
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            for (uint32_t q = 0; q < 65; ++q) {
                hir.append_tgate(false,
                                 [&](MutablePauliMaskView mask) { mask.x().bit_set(q, true); });
            }
        }
        if (region == 0) {
            test::append_measure(hir, 0, 1, false, MeasRecordIdx{0});
            hir.num_measurements = 1;
        }
    }
    PhasePolynomialPass pass({128, 1, 200000});
    pass.run(hir);
    REQUIRE(pass.wide_attempts() == 1);
    REQUIRE(pass.wide_budget_exhausted());
    REQUIRE(pass.wide_work() <= 1);
    REQUIRE(pass.wide_rewrites() == 0);
}

TEST_CASE("Phase pass pulls back expectation values and preserves stabilizers across them",
          "[optimizer]") {
    auto hir =
        trace(parse("H 0\n"
                    "T 0\n"
                    "T 0\n"
                    "EXP_VAL X0\n"
                    "CX 1 0\n"
                    "T 0\n"
                    "CX 1 0\n"
                    "EXP_VAL Y0*Z1\n"
                    "M 0 1"));
    const auto baseline = hir;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(pass.pauli_pullbacks() >= 1);
    REQUIRE(hir.num_t_gates() == 1);
    Xoshiro256PlusPlus rng(0x51A5E01);
    test::check_exact_equivalent(baseline, hir, rng);
}

TEST_CASE("Phase pass synthesizes quadratic Clifford remainder terms with exact equivalence",
          "[optimizer]") {
    // Establish both basis generators before adding 2*(a XOR b). Its quadratic
    // monomial has coefficient 4 modulo 8. The X probes rewind to Z axes that
    // do not commute with the synthesized Clifford correction.
    auto hir =
        trace(parse("H 0 1\n"
                    "T 0 1\n"
                    "CX 0 1\nT 1\nCX 0 1\n"
                    "CX 0 1\nT 1\nCX 0 1\n"
                    "EXP_VAL X0\n"
                    "EXP_VAL X1\n"
                    "MX 0 1"));
    const auto baseline = hir;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.num_t_gates() == 2);
    for (const auto& op : hir.ops) {
        if (op.op_type() == OpType::EXP_VAL || op.op_type() == OpType::MEASURE) {
            const auto mask = hir.mask_view(op);
            REQUIRE(mask.x().popcount() == 1);
            REQUIRE(mask.z().popcount() == 1);
            REQUIRE(mask.sign());
        }
    }
    Xoshiro256PlusPlus rng(0x51A5E02);
    test::check_exact_equivalent(baseline, hir, rng);
}
