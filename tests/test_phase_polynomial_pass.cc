#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_registry.h"
#include "clifft/optimizer/phase_polynomial_pass.h"

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clifft;

namespace {

HirModule parity_block(uint32_t n = 4, const std::vector<uint32_t>& positions = {0, 1, 2, 3}) {
    HirModule hir(n, 16);
    hir.final_tableau.emplace(n);
    for (uint32_t parity = 1; parity <= 16; ++parity) {
        const uint32_t mask = parity == 16 ? 15 : parity;
        hir.append_tgate(false, [&](MutablePauliMaskView slot) {
            for (uint32_t j = 0; j < 4; ++j) {
                if (mask & (1U << j)) {
                    slot.x().bit_set(positions[j], true);
                }
            }
        });
        hir.source_map.push_back({parity});
    }
    return hir;
}

HirModule related_axes(double probability) {
    HirModule hir(2, 2, 1);
    hir.final_tableau.emplace(2);
    auto mask = test::claim_noise_channel_mask(hir, 2, 0);
    hir.noise_sites.push_back({probability, {{mask, probability}}});
    hir.append_noise(NoiseSiteIdx{0});
    test::append_tgate(hir, 1, 0, false);
    test::append_tgate(hir, 1, 2, false);
    return hir;
}

}  // namespace

TEST_CASE("Phase polynomial reduces distinct parities to one rotation", "[optimizer]") {
    auto hir = parity_block();
    REQUIRE(analyze_active_width(hir).peak_width == 4);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(pass.blocks_reduced() == 1);
    REQUIRE(pass.input_t_count() == 16);
    REQUIRE(pass.output_t_count() == 1);
    REQUIRE(pass.result_peak() == 1);
    REQUIRE(hir.num_t_gates() == 1);
    REQUIRE(hir.destab_mask(hir.ops[0]).words[0] == 15);
    REQUIRE(hir.stab_mask(hir.ops[0]).is_zero());
    REQUIRE(hir.source_map[0].size() == 16);
    REQUIRE(hir.final_tableau->satisfies_invariants());
}

TEST_CASE("Phase polynomial handles axes spanning multiple words", "[optimizer]") {
    auto hir = parity_block(130, {0, 63, 64, 129});
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(pass.result_peak() == 1);
    const auto x = hir.destab_mask(hir.ops[0]);
    REQUIRE(x.bit_get(0));
    REQUIRE(x.bit_get(63));
    REQUIRE(x.bit_get(64));
    REQUIRE(x.bit_get(129));
    REQUIRE(x.popcount() == 4);
}

TEST_CASE("Phase polynomial respects signs of commuting Pauli products", "[optimizer]") {
    HirModule hir(2, 2);
    hir.final_tableau.emplace(2);
    test::append_tgate(hir, 3, 3, false);
    test::append_tgate(hir, 3, 0, false);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(pass.result_peak() == 0);
}

TEST_CASE("Phase polynomial removes unknown noisy entry relations", "[optimizer]") {
    auto hir = related_axes(0.2);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.num_t_gates() == 2);
    REQUIRE(hir.noise_sites[0].channels[0].prob == 0.2);
}

TEST_CASE("Phase polynomial keeps deterministic noise signs", "[optimizer]") {
    auto hir = related_axes(1);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.num_t_gates() == 0);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
}

TEST_CASE("Phase polynomial retains commuting noise distributions", "[optimizer]") {
    auto hir = parity_block();
    hir.noise_channel_masks = PauliMaskArena(4, 2);
    const auto x = test::claim_noise_channel_mask(hir, 1, 0);
    const auto y = test::claim_noise_channel_mask(hir, 2, 0);
    hir.noise_sites.push_back({0.3, {{x, 0.1}, {y, 0.2}}});
    hir.ops.insert(hir.ops.begin() + 8, HeisenbergOp::make_noise(NoiseSiteIdx{0}));
    hir.source_map.insert(hir.source_map.begin() + 8, {17});
    const auto sites = hir.noise_sites;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(hir.num_t_gates() == 1);
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(hir.noise_sites == sites);
    REQUIRE(hir.ops[0].op_type() == OpType::NOISE);
    REQUIRE(hir.source_map[0] == std::vector<uint32_t>{17});
}

TEST_CASE("Phase polynomial does not cross anticommuting noise", "[optimizer]") {
    auto hir = trace(parse("H 0\nT 0\nX_ERROR(0.3) 0\nT 0\nMX 0"));
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.blocks_examined() == 2);
    REQUIRE(hir.num_t_gates() == 2);
}

TEST_CASE("Phase polynomial preserves measurement barriers and records", "[optimizer]") {
    auto hir = trace(parse("H 0\nT 0\nMX 0\nDETECTOR rec[-1]\nT 0\nMX 0"));
    const auto detectors = hir.detector_targets;
    const auto records = hir.num_measurements;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.num_measurements == records);
    REQUIRE(hir.detector_targets == detectors);
}

TEST_CASE("Phase polynomial leaves schedules with logical noise crossings intact", "[optimizer]") {
    auto hir = parity_block();
    hir.logical_noise_prefix.assign(hir.ops.size(), 1);
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.num_t_gates() == 16);
    REQUIRE(hir.logical_noise_prefix.size() == hir.ops.size());
}

TEST_CASE("Phase polynomial variable cap conservatively skips blocks", "[optimizer]") {
    auto hir = parity_block();
    PhasePolynomialPass pass({3});
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.oversized_blocks() == 1);
    REQUIRE(hir.num_t_gates() == 16);
    REQUIRE_THROWS_AS(PhasePolynomialPass(PhasePolynomialOptions{65}), std::invalid_argument);
}

TEST_CASE("Phase polynomial clears statistics between runs", "[optimizer]") {
    PhasePolynomialPass pass;
    auto reduced = parity_block();
    pass.run(reduced);
    REQUIRE(pass.applied());
    auto simple = trace(parse("H 0\nT 0"));
    pass.run(simple);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.blocks_reduced() == 0);
    REQUIRE(pass.input_t_count() == 1);
    REQUIRE(pass.output_t_count() == 1);
}

TEST_CASE("Phase polynomial is opt in and preserves record order", "[optimizer]") {
    const auto entry = std::ranges::find_if(
        kRegisteredPasses, [](const PassInfo& info) { return info.name == "PhasePolynomialPass"; });
    REQUIRE(entry != std::end(kRegisteredPasses));
    REQUIRE_FALSE(entry->default_enabled);
    REQUIRE(entry->record_order.preserved);
    REQUIRE_FALSE(is_trajectory_compatible(*entry));
}

TEST_CASE("Phase polynomial pulls back Pauli measurements with their signs", "[optimizer]") {
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
    REQUIRE(hir.ops[0].op_type() == OpType::MEASURE);
    REQUIRE(hir.destab_mask(hir.ops[0]).words[0] == 1);
    REQUIRE(hir.stab_mask(hir.ops[0]).words[0] == 1);
    REQUIRE_FALSE(hir.sign(hir.ops[0]));
    REQUIRE(hir.ops[0].meas_record_idx() == MeasRecordIdx{0});
    REQUIRE(hir.ops[2].meas_record_idx() == MeasRecordIdx{1});
}

TEST_CASE("Phase polynomial stops when a measurement changes the entry code", "[optimizer]") {
    HirModule hir(2, 3);
    hir.final_tableau.emplace(2);
    test::append_tgate(hir, 1, 0, false);
    test::append_measure(hir, 2, 0, false, MeasRecordIdx{0});
    test::append_tgate(hir, 1, 2, false);
    hir.num_measurements = 1;
    PhasePolynomialPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.num_t_gates() == 2);
    REQUIRE(pass.blocks_examined() == 2);
    REQUIRE(hir.ops[1].meas_record_idx() == MeasRecordIdx{0});
}
