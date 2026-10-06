#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/rotation_simplification_pass.h"

#include "instrument_test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>

using namespace clifft;

TEST_CASE("Rotation simplification removes signed input constraints", "[optimizer]") {
    for (const auto* prep : {"", "X 1\n"}) {
        const bool negative = !std::string(prep).empty();
        auto hir = trace(parse(std::string(prep) + "H 0\nR_Z(0.137) 0\nR_PAULI(" +
                               (negative ? "0.137" : "-0.137") + ") Z0*Z1"));
        RotationSimplificationPass pass;
        pass.run(hir);
        REQUIRE(pass.applied());
        REQUIRE(pass.rotations_removed() == 2);
        REQUIRE(hir.ops.empty());
        REQUIRE(hir.final_tableau->satisfies_invariants());
    }
}

TEST_CASE("Rotation simplification preserves merged source lines", "[optimizer]") {
    auto hir = trace(parse("H 0\nR_Z(0.137) 0\nR_PAULI(0.213) Z0*Z1"));
    const auto original = hir;
    RotationSimplificationPass pass({256, 1});
    pass.run(hir);
    REQUIRE(pass.applied());
    REQUIRE(pass.regions_capped() == 1);
    REQUIRE(hir.ops.size() == 1);
    REQUIRE(hir.source_map.size() == 1);
    auto expected = original.source_map[0];
    expected.insert(expected.end(), original.source_map[1].begin(), original.source_map[1].end());
    REQUIRE(hir.source_map[0] == expected);
}

TEST_CASE("Rotation simplification accepts axes spanning mask words", "[optimizer]") {
    for (uint32_t q : {1, 64, 129}) {
        auto hir = trace(parse("H 0\nR_Z(0.137) 0\nR_PAULI(-0.137) Z0*Z" + std::to_string(q)));
        RotationSimplificationPass pass;
        pass.run(hir);
        REQUIRE(hir.ops.empty());
        REQUIRE(analyze_active_width(hir).peak_width == 0);
    }
}

TEST_CASE("Rotation simplification absorbs Clifford factors through instruments", "[optimizer]") {
    const auto options = test::source_dependent_jump_options(false);
    auto hir = trace(parse("H 0\nR_Z(0.137) 0\nR_Z(0.363) 0\nH 0\nLEVEL_TRANSITION[jump] 0\nM 0"),
                     &options);
    auto reference = trace(parse("H 0\nS 0\nH 0\nLEVEL_TRANSITION[jump] 0\nM 0"), &options);
    RotationSimplificationPass pass;
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

TEST_CASE("Rotation simplification stops using facts after an instrument", "[optimizer]") {
    const auto options = test::source_dependent_jump_options(false);
    auto hir = trace(parse("LEVEL_TRANSITION[jump] 0\nR_Z(0.137) 0"), &options);
    const auto original = hir;
    RotationSimplificationPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.regions_examined() == 0);
    REQUIRE(hir.ops.size() == original.ops.size());
    REQUIRE(hir.final_tableau == original.final_tableau);
}

TEST_CASE("Rotation simplification respects bounds and noise scheduling", "[optimizer]") {
    auto hir = trace(parse("H 0\nR_Z(0.137) 0\nR_PAULI(-0.137) Z0*Z1"));
    RotationSimplificationPass disabled({0, 8});
    disabled.run(hir);
    REQUIRE_FALSE(disabled.applied());
    REQUIRE(disabled.regions_examined() == 0);
    hir.logical_noise_prefix.assign(hir.ops.size(), 1);
    RotationSimplificationPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(hir.ops.size() == 2);
    REQUIRE(hir.logical_noise_prefix.size() == 2);
    REQUIRE_THROWS_AS(RotationSimplificationPass({4097, 8}), std::invalid_argument);
    REQUIRE_THROWS_AS(RotationSimplificationPass({256, 0}), std::invalid_argument);
    REQUIRE_THROWS_AS(RotationSimplificationPass({256, 65}), std::invalid_argument);
}

TEST_CASE("Rotation simplification skips circuits without rotations and resets statistics",
          "[optimizer]") {
    RotationSimplificationPass pass;
    auto reducible = trace(parse("R_Z(0.137) 0"));
    pass.run(reducible);
    REQUIRE(pass.applied());
    auto clifford = trace(parse("H 0\nCX 0 1\nM 0 1"));
    pass.run(clifford);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.regions_examined() == 0);
    REQUIRE(pass.regions_capped() == 0);
    REQUIRE(pass.rotations_removed() == 0);
}

TEST_CASE("Rotation simplification stops scanning rotations after facts are lost", "[optimizer]") {
    std::string source = "H 0\nR_Z(0.137) 0\n";
    for (size_t i = 0; i < 100; ++i) {
        source += "R_X(0.213) 0\nR_Z(0.137) 0\n";
    }
    auto hir = trace(parse(source));
    RotationSimplificationPass pass;
    pass.run(hir);
    REQUIRE_FALSE(pass.applied());
    REQUIRE(pass.regions_examined() == 1);
    REQUIRE(hir.ops.size() == 201);
}
