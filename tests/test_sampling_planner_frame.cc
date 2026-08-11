#include "clifft/sampling/planner_frame.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

using clifft::sampling::AffineBool;
using clifft::sampling::SymbolId;
using clifft::sampling::internal::active_measurement_frame;
using clifft::sampling::internal::CoordinateFrame;
using clifft::sampling::internal::dormant_measurement_frame;
using clifft::sampling::internal::dormant_promotion_frame;
using clifft::sampling::internal::PlannerPauli;
using clifft::sampling::internal::PlannerTableau;
using clifft::sampling::internal::SymbolicPauliFrame;

TEST_CASE("Planner coordinate frame round trips composed basis changes") {
    CoordinateFrame coordinates(3);
    PlannerTableau first(3);
    first.inplace_scatter_append(stim::GATE_DATA.at("H").tableau<clifft::kStimWidth>(), {0});
    PlannerTableau second(3);
    second.inplace_scatter_append(stim::GATE_DATA.at("CX").tableau<clifft::kStimWidth>(), {0, 1});

    PlannerPauli initial(3);
    initial.xs[0] = true;
    initial.zs[1] = true;
    initial.sign = true;

    coordinates.change_basis(first);
    const PlannerPauli after_first = coordinates.to_current(initial);
    REQUIRE(after_first.xs[0] == false);
    REQUIRE(after_first.zs[0] == true);
    REQUIRE(after_first.zs[1] == true);
    REQUIRE(coordinates.to_initial(after_first) == initial);

    coordinates.change_basis(second);
    const PlannerPauli after_second = coordinates.to_current(initial);
    REQUIRE(coordinates.to_initial(after_second) == initial);
}

TEST_CASE("Planner coordinate frame matches explicit inverse for signed Paulis") {
    constexpr uint32_t kNumQubits = 3;
    CoordinateFrame coordinates(kNumQubits);
    PlannerTableau cumulative(kNumQubits);
    const std::vector<size_t> indices{0, 1, 2};

    const auto require_matches_inverse = [&] {
        const PlannerTableau inverse = cumulative.inverse();
        for (uint32_t body = 0; body < 1U << (2 * kNumQubits); ++body) {
            for (bool sign : {false, true}) {
                PlannerPauli initial(kNumQubits);
                for (uint32_t q = 0; q < kNumQubits; ++q) {
                    initial.xs[q] = (body >> q) & 1U;
                    initial.zs[q] = (body >> (q + kNumQubits)) & 1U;
                }
                initial.sign = sign;

                const PlannerPauli expected = inverse.scatter_eval(initial.ref(), indices);
                const PlannerPauli actual = coordinates.to_current(initial);
                REQUIRE(actual == expected);
                REQUIRE(coordinates.to_initial(actual) == initial);
            }
        }
    };

    require_matches_inverse();
    for (const auto& [gate, targets] : std::vector<std::pair<const char*, std::vector<size_t>>>{
             {"H", {0}}, {"S", {1}}, {"CX", {0, 1}}, {"SQRT_X", {2}}, {"CZ", {1, 2}}}) {
        PlannerTableau change(kNumQubits);
        change.inplace_scatter_append(stim::GATE_DATA.at(gate).tableau<clifft::kStimWidth>(),
                                      targets);
        coordinates.change_basis(change);
        cumulative = change.then(cumulative);
        require_matches_inverse();
    }

    PlannerPauli promoted(kNumQubits);
    promoted.zs[0] = true;
    promoted.xs[2] = true;
    const PlannerTableau promotion = dormant_promotion_frame(promoted, 1, 2);
    coordinates.change_basis(promotion);
    cumulative = promotion.then(cumulative);
    require_matches_inverse();

    PlannerPauli active(kNumQubits);
    active.xs[0] = true;
    active.zs[1] = true;
    const PlannerTableau removal = active_measurement_frame(active, 2, 0);
    coordinates.change_basis(removal);
    cumulative = removal.then(cumulative);
    require_matches_inverse();

    PlannerPauli dormant(kNumQubits);
    dormant.zs[0] = true;
    dormant.xs[2] = true;
    const PlannerTableau replacement = dormant_measurement_frame(dormant, 2);
    coordinates.change_basis(replacement);
    cumulative = replacement.then(cumulative);
    require_matches_inverse();
}

TEST_CASE("Planner coordinate frame handles multiple packed words") {
    constexpr uint32_t kNumQubits = 70;
    CoordinateFrame coordinates(kNumQubits);
    PlannerTableau change(kNumQubits);
    change.inplace_scatter_append(stim::GATE_DATA.at("H").tableau<clifft::kStimWidth>(), {64});
    change.inplace_scatter_append(stim::GATE_DATA.at("CX").tableau<clifft::kStimWidth>(), {0, 64});
    change.inplace_scatter_append(stim::GATE_DATA.at("S").tableau<clifft::kStimWidth>(), {69});
    change.inplace_scatter_append(stim::GATE_DATA.at("CZ").tableau<clifft::kStimWidth>(), {64, 69});
    coordinates.change_basis(change);

    PlannerPauli initial(kNumQubits);
    initial.xs[0] = true;
    initial.zs[64] = true;
    initial.xs[69] = true;
    initial.sign = true;

    std::vector<size_t> indices(kNumQubits);
    for (uint32_t q = 0; q < kNumQubits; ++q) {
        indices[q] = q;
    }
    const PlannerTableau inverse = change.inverse();
    REQUIRE(coordinates.to_current(initial) == inverse.scatter_eval(initial.ref(), indices));
    REQUIRE(coordinates.to_initial(coordinates.to_current(initial)) == initial);
}

TEST_CASE("Planner coordinate frame caches repeated reverse lookups") {
    constexpr uint32_t kNumQubits = 3;
    CoordinateFrame coordinates(kNumQubits);
    PlannerPauli initial(kNumQubits);
    initial.xs[0] = true;
    initial.zs[2] = true;

    for (uint32_t lookup = 0; lookup < 2 * kNumQubits; ++lookup) {
        REQUIRE(coordinates.to_current(initial) == initial);
        REQUIRE_FALSE(coordinates.has_cached_inverse());
    }
    REQUIRE(coordinates.to_current(initial) == initial);
    REQUIRE(coordinates.has_cached_inverse());

    PlannerTableau change(kNumQubits);
    change.inplace_scatter_append(stim::GATE_DATA.at("H").tableau<clifft::kStimWidth>(), {0});
    coordinates.change_basis(change);
    REQUIRE_FALSE(coordinates.has_cached_inverse());
    REQUIRE(coordinates.to_current(initial) ==
            change.inverse().scatter_eval(initial.ref(), {0, 1, 2}));
}

TEST_CASE("Planner symbolic frame composes affine Pauli corrections") {
    SymbolicPauliFrame frame(2, 130);
    const AffineBool wide_condition(true, {SymbolId{0}, SymbolId{64}, SymbolId{129}});

    PlannerPauli x_correction(2);
    x_correction.xs[0] = true;
    frame.apply(x_correction, wide_condition);

    PlannerPauli z_observable(2);
    z_observable.zs[0] = true;
    REQUIRE(frame.sign_for(z_observable) == wide_condition);

    PlannerPauli z_correction(2);
    z_correction.zs[0] = true;
    frame.apply(z_correction, AffineBool::symbol(SymbolId{1}));

    PlannerPauli y_observable(2);
    y_observable.xs[0] = true;
    y_observable.zs[0] = true;
    REQUIRE(frame.sign_for(y_observable) == (wide_condition ^ AffineBool::symbol(SymbolId{1})));

    frame.apply(x_correction, wide_condition);
    REQUIRE(frame.sign_for(z_observable) == AffineBool(false));
    REQUIRE(SymbolicPauliFrame::estimated_workspace_bytes(2, 130) == 124);
}

TEST_CASE("Planner symbolic frame rejects unknown symbols") {
    SymbolicPauliFrame frame(1, 1);
    PlannerPauli correction(1);
    correction.xs[0] = true;

    REQUIRE_THROWS_AS(frame.apply(correction, AffineBool::symbol(SymbolId{1})), std::logic_error);
}

TEST_CASE("Planner promotion frame localizes the promoted observable") {
    PlannerPauli promoted(3);
    promoted.zs[0] = true;
    promoted.xs[2] = true;

    const PlannerTableau frame = dormant_promotion_frame(promoted, 1, 2);
    REQUIRE(frame.satisfies_invariants());
    REQUIRE(frame.xs[1] == promoted);

    const PlannerTableau old_to_new = frame.inverse();
    const PlannerPauli localized =
        old_to_new.scatter_eval(promoted.ref(), std::vector<size_t>{0, 1, 2});
    PlannerPauli expected(3);
    expected.xs[1] = true;
    REQUIRE(localized == expected);
}

TEST_CASE("Planner measurement frames install the selected observables") {
    PlannerPauli dormant(3);
    dormant.zs[0] = true;
    dormant.xs[2] = true;
    const PlannerTableau dormant_frame = dormant_measurement_frame(dormant, 2);
    REQUIRE(dormant_frame.satisfies_invariants());
    REQUIRE(dormant_frame.zs[2] == dormant);

    PlannerPauli active(3);
    active.xs[0] = true;
    active.zs[1] = true;
    const PlannerTableau active_frame = active_measurement_frame(active, 2, 0);
    REQUIRE(active_frame.satisfies_invariants());
    REQUIRE(active_frame.zs[1] == active);
}
