#include "clifft/sampling/planner_frame.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>
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
