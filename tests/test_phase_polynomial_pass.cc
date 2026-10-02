#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/phase_polynomial_pass.h"

#include "instrument_test_helpers.h"
#include "test_helpers.h"

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

TEST_CASE("Phase pass does not assume an entry stabilizer", "[optimizer]") {
    HirModule hir(2, 2);
    hir.final_tableau.emplace(2);
    test::append_tgate(hir, 1, 0, false);
    test::append_tgate(hir, 1, 2, false);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.num_t_gates() == 2);
}

TEST_CASE("Phase pass composes noncommuting Clifford factors across regions", "[optimizer]") {
    auto hir = trace(parse("H 0\nT 0\nT 0\nH 0\nT 0\nT 0\nH 0\nT_DAG 0\nT_DAG 0"));
    const auto reference = trace(parse("H 0\nS 0\nH 0\nS 0\nH 0\nS_DAG 0"));
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.ops.empty());
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
        std::string source = "T 0\nT 0\n";
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
        REQUIRE(measured.sign());
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
    REQUIRE_THROWS_AS(PhasePolynomialPass(PhasePolynomialOptions{65}), std::invalid_argument);
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
}
