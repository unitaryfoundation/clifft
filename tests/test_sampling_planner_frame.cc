#include "clifft/sampling/planner_frame.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <optional>
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

namespace {

PlannerPauli signed_pauli(uint32_t num_qubits, uint32_t body, bool sign) {
    PlannerPauli result(num_qubits);
    for (uint32_t q = 0; q < num_qubits; ++q) {
        result.set_pauli(q, (body >> q) & 1U, (body >> (q + num_qubits)) & 1U);
    }
    result.set_sign(sign);
    return result;
}

PlannerTableau nontrivial_basis(uint32_t num_qubits) {
    PlannerTableau result(num_qubits);
    result.append_named_gate(clifft::GateType::H, {0});
    if (num_qubits > 1) {
        result.append_named_gate(clifft::GateType::S, {1});
        result.append_named_gate(clifft::GateType::CX, {0, num_qubits - 1});
    }
    return result;
}

void require_same_coordinate_map(const CoordinateFrame& direct, const CoordinateFrame& generic,
                                 uint32_t num_qubits) {
    for (uint32_t q = 0; q < num_qubits; ++q) {
        CAPTURE(q);
        PlannerPauli x(num_qubits);
        x.set_pauli(q, true, false);
        REQUIRE(direct.to_initial(x) == generic.to_initial(x));

        PlannerPauli z(num_qubits);
        z.set_pauli(q, false, true);
        REQUIRE(direct.to_initial(z) == generic.to_initial(z));
    }
}

std::optional<uint32_t> active_measurement_pivot(const PlannerPauli& measured,
                                                 uint32_t active_width) {
    for (uint32_t q = active_width; q > 0; --q) {
        if (measured.x().bit_get(q - 1)) {
            return q - 1;
        }
    }
    for (uint32_t q = 0; q < active_width; ++q) {
        if (measured.z().bit_get(q)) {
            return q;
        }
    }
    return std::nullopt;
}

}  // namespace

TEST_CASE("Planner coordinate frame round trips composed basis changes") {
    CoordinateFrame coordinates(3);
    PlannerTableau first(3);
    first.append_named_gate(clifft::GateType::H, {0});
    PlannerTableau second(3);
    second.append_named_gate(clifft::GateType::CX, {0, 1});

    PlannerPauli initial(3);
    initial.set_pauli(0, true, false);
    initial.set_pauli(1, false, true);
    initial.set_sign(true);

    coordinates.change_basis(first);
    const PlannerPauli after_first = coordinates.to_current(initial);
    REQUIRE(after_first.x().bit_get(0) == false);
    REQUIRE(after_first.z().bit_get(0) == true);
    REQUIRE(after_first.z().bit_get(1) == true);
    REQUIRE(coordinates.to_initial(after_first) == initial);

    coordinates.change_basis(second);
    const PlannerPauli after_second = coordinates.to_current(initial);
    REQUIRE(coordinates.to_initial(after_second) == initial);
}

TEST_CASE("Planner coordinate frame matches explicit inverse for signed Paulis") {
    constexpr uint32_t kNumQubits = 3;
    CoordinateFrame coordinates(kNumQubits);
    PlannerTableau cumulative(kNumQubits);
    const auto require_matches_inverse = [&] {
        const PlannerTableau inverse = cumulative.inverse();
        for (uint32_t body = 0; body < 1U << (2 * kNumQubits); ++body) {
            for (bool sign : {false, true}) {
                PlannerPauli initial(kNumQubits);
                for (uint32_t q = 0; q < kNumQubits; ++q) {
                    initial.set_pauli(q, (body >> q) & 1U, (body >> (q + kNumQubits)) & 1U);
                }
                initial.set_sign(sign);

                const PlannerPauli expected = inverse.apply(initial.view());
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
        std::vector<uint32_t> native_targets(targets.begin(), targets.end());
        change.append_named_gate(clifft::parse_gate_name(gate), native_targets);
        coordinates.change_basis(change);
        cumulative = change.then(cumulative);
        require_matches_inverse();
    }

    PlannerPauli promoted(kNumQubits);
    promoted.set_pauli(0, false, true);
    promoted.set_pauli(2, true, false);
    const PlannerTableau promotion = dormant_promotion_frame(promoted, 1, 2);
    coordinates.promote_dormant(promoted, 1, 2);
    cumulative = promotion.then(cumulative);
    require_matches_inverse();

    PlannerPauli active(kNumQubits);
    active.set_pauli(0, true, false);
    active.set_pauli(1, false, true);
    const PlannerTableau removal = active_measurement_frame(active, 2, 0);
    coordinates.measure_active(active, 2, 0);
    cumulative = removal.then(cumulative);
    require_matches_inverse();

    PlannerPauli dormant(kNumQubits);
    dormant.set_pauli(0, false, true);
    dormant.set_pauli(2, true, false);
    const PlannerTableau replacement = dormant_measurement_frame(dormant, 2);
    coordinates.measure_dormant(dormant, 2);
    cumulative = replacement.then(cumulative);
    require_matches_inverse();
}

TEST_CASE("Planner structured coordinate updates match generic frames") {
    constexpr uint32_t kNumQubits = 3;
    const PlannerTableau basis = nontrivial_basis(kNumQubits);

    for (uint32_t active_width = 0; active_width < kNumQubits; ++active_width) {
        for (uint32_t dormant_pivot = active_width; dormant_pivot < kNumQubits; ++dormant_pivot) {
            for (uint32_t body = 0; body < 1U << (2 * kNumQubits); ++body) {
                for (bool sign : {false, true}) {
                    const PlannerPauli promoted = signed_pauli(kNumQubits, body, sign);
                    bool is_selected_pivot = promoted.x().bit_get(dormant_pivot);
                    for (uint32_t q = active_width; q < dormant_pivot; ++q) {
                        is_selected_pivot &= !promoted.x().bit_get(q);
                    }
                    if (!is_selected_pivot) {
                        continue;
                    }
                    CAPTURE(active_width, dormant_pivot, body, sign);

                    CoordinateFrame direct(kNumQubits);
                    CoordinateFrame generic(kNumQubits);
                    direct.change_basis(basis);
                    generic.change_basis(basis);
                    direct.promote_dormant(promoted, active_width, dormant_pivot);
                    generic.change_basis(
                        dormant_promotion_frame(promoted, active_width, dormant_pivot));
                    require_same_coordinate_map(direct, generic, kNumQubits);
                }
            }
        }
    }

    for (uint32_t dormant_pivot = 0; dormant_pivot < kNumQubits; ++dormant_pivot) {
        for (uint32_t body = 0; body < 1U << (2 * kNumQubits); ++body) {
            for (bool sign : {false, true}) {
                const PlannerPauli measured = signed_pauli(kNumQubits, body, sign);
                if (!measured.x().bit_get(dormant_pivot)) {
                    continue;
                }
                CAPTURE(dormant_pivot, body, sign);

                CoordinateFrame direct(kNumQubits);
                CoordinateFrame generic(kNumQubits);
                direct.change_basis(basis);
                generic.change_basis(basis);
                direct.measure_dormant(measured, dormant_pivot);
                generic.change_basis(dormant_measurement_frame(measured, dormant_pivot));
                require_same_coordinate_map(direct, generic, kNumQubits);
            }
        }
    }

    for (uint32_t active_width = 1; active_width <= kNumQubits; ++active_width) {
        const uint32_t body_limit = 1U << (2 * active_width);
        for (uint32_t active_body = 1; active_body < body_limit; ++active_body) {
            for (bool sign : {false, true}) {
                PlannerPauli measured(kNumQubits);
                for (uint32_t q = 0; q < active_width; ++q) {
                    measured.set_pauli(q, (active_body >> q) & 1U,
                                       (active_body >> (q + active_width)) & 1U);
                }
                measured.set_sign(sign);
                const std::optional<uint32_t> pivot =
                    active_measurement_pivot(measured, active_width);
                REQUIRE(pivot.has_value());
                CAPTURE(active_width, active_body, sign, *pivot);

                CoordinateFrame direct(kNumQubits);
                CoordinateFrame generic(kNumQubits);
                direct.change_basis(basis);
                generic.change_basis(basis);
                direct.measure_active(measured, active_width, *pivot);
                generic.change_basis(active_measurement_frame(measured, active_width, *pivot));
                require_same_coordinate_map(direct, generic, kNumQubits);
            }
        }
    }
}

TEST_CASE("Planner coordinate frame matches inverse across packed words") {
    constexpr uint32_t kNumQubits = 70;
    CoordinateFrame coordinates(kNumQubits);
    PlannerTableau cumulative(kNumQubits);
    uint64_t random_state = 0x9e3779b97f4a7c15ULL;
    const auto next_bits = [&] {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 7;
        random_state ^= random_state << 17;
        return random_state;
    };
    const auto require_matches_inverse = [&] {
        const PlannerTableau inverse = cumulative.inverse();
        for (uint32_t sample = 0; sample < 32; ++sample) {
            PlannerPauli initial(kNumQubits);
            uint64_t x_bits = 0;
            uint64_t z_bits = 0;
            for (uint32_t q = 0; q < kNumQubits; ++q) {
                if ((q & 63U) == 0) {
                    x_bits = next_bits();
                    z_bits = next_bits();
                }
                initial.set_pauli(q, (x_bits >> (q & 63U)) & 1U, (z_bits >> (q & 63U)) & 1U);
            }
            initial.set_pauli(63, (sample & 1U) != 0, (sample & 2U) != 0);
            initial.set_pauli(64, (sample & 4U) != 0, (sample & 8U) != 0);
            initial.set_sign((sample & 16U) != 0);

            const PlannerPauli expected = inverse.apply(initial.view());
            const PlannerPauli actual = coordinates.to_current(initial);
            REQUIRE(actual == expected);
            REQUIRE(coordinates.to_initial(actual) == initial);
        }
    };

    const auto apply_change = [&](const PlannerTableau& change) {
        coordinates.change_basis(change);
        cumulative = change.then(cumulative);
        require_matches_inverse();
    };

    PlannerTableau gates(kNumQubits);
    gates.append_named_gate(clifft::GateType::H, {64});
    gates.append_named_gate(clifft::GateType::CX, {0, 64});
    gates.append_named_gate(clifft::GateType::S, {69});
    gates.append_named_gate(clifft::GateType::CZ, {64, 69});
    apply_change(gates);

    PlannerPauli promoted(kNumQubits);
    promoted.set_pauli(0, false, true);
    promoted.set_pauli(69, true, false);
    promoted.set_sign(true);
    PlannerTableau promotion = dormant_promotion_frame(promoted, 1, 69);
    coordinates.promote_dormant(promoted, 1, 69);
    cumulative = promotion.then(cumulative);
    require_matches_inverse();

    PlannerPauli active(kNumQubits);
    active.set_pauli(0, true, false);
    active.set_pauli(64, false, true);
    active.set_sign(true);
    PlannerTableau active_removal = active_measurement_frame(active, 65, 0);
    coordinates.measure_active(active, 65, 0);
    cumulative = active_removal.then(cumulative);
    require_matches_inverse();

    PlannerPauli diagonal(kNumQubits);
    diagonal.set_pauli(3, false, true);
    diagonal.set_pauli(63, false, true);
    diagonal.set_sign(true);
    PlannerTableau diagonal_removal = active_measurement_frame(diagonal, 64, 3);
    coordinates.measure_active(diagonal, 64, 3);
    cumulative = diagonal_removal.then(cumulative);
    require_matches_inverse();

    PlannerPauli dormant(kNumQubits);
    dormant.set_pauli(0, false, true);
    dormant.set_pauli(69, true, false);
    dormant.set_sign(true);
    PlannerTableau dormant_replacement = dormant_measurement_frame(dormant, 69);
    coordinates.measure_dormant(dormant, 69);
    cumulative = dormant_replacement.then(cumulative);
    require_matches_inverse();
}

TEST_CASE("Planner coordinate frame maintains reverse map across interleaved wide updates") {
    constexpr uint32_t kNumQubits = 195;
    CoordinateFrame coordinates(kNumQubits);
    PlannerTableau cumulative(kNumQubits);

    auto check_probe_paulis = [&] {
        const PlannerTableau inverse = cumulative.inverse();
        for (uint32_t probe : {0U, 63U, 64U, 127U, 128U, 194U}) {
            for (uint32_t kind = 1; kind <= 3; ++kind) {
                for (bool sign : {false, true}) {
                    PlannerPauli initial(kNumQubits);
                    initial.set_pauli(probe, (kind & 1U) != 0, (kind & 2U) != 0);
                    initial.set_pauli((probe + 65) % kNumQubits, (kind & 2U) != 0,
                                      (kind & 1U) != 0);
                    initial.set_sign(sign);
                    const PlannerPauli expected = inverse.apply(initial.view());
                    const PlannerPauli actual = coordinates.to_current(initial);
                    REQUIRE(actual == expected);
                    REQUIRE(coordinates.to_initial(actual) == initial);
                }
            }
        }
    };

    check_probe_paulis();

    PlannerPauli dormant_first(kNumQubits);
    dormant_first.set_pauli(0, false, true);
    dormant_first.set_pauli(64, true, true);
    dormant_first.set_pauli(194, true, false);
    dormant_first.set_sign(true);
    coordinates.measure_dormant(dormant_first, 194);
    cumulative = dormant_measurement_frame(dormant_first, 194).then(cumulative);
    check_probe_paulis();

    PlannerPauli promoted(kNumQubits);
    promoted.set_pauli(63, false, true);
    promoted.set_pauli(128, true, false);
    promoted.set_sign(true);
    coordinates.promote_dormant(promoted, 0, 128);
    cumulative = dormant_promotion_frame(promoted, 0, 128).then(cumulative);
    check_probe_paulis();

    PlannerPauli promoted_second(kNumQubits);
    promoted_second.set_pauli(0, true, true);
    promoted_second.set_pauli(65, true, false);
    promoted_second.set_sign(false);
    coordinates.promote_dormant(promoted_second, 1, 65);
    cumulative = dormant_promotion_frame(promoted_second, 1, 65).then(cumulative);
    check_probe_paulis();

    PlannerPauli active_measured(kNumQubits);
    active_measured.set_pauli(0, true, false);
    active_measured.set_pauli(1, true, true);
    active_measured.set_sign(true);
    coordinates.measure_active(active_measured, 2, 1);
    cumulative = active_measurement_frame(active_measured, 2, 1).then(cumulative);
    check_probe_paulis();
}

TEST_CASE("Planner symbolic frame composes affine Pauli corrections") {
    SymbolicPauliFrame frame(2, 130);
    const AffineBool wide_condition(true, {SymbolId{0}, SymbolId{64}, SymbolId{129}});

    PlannerPauli x_correction(2);
    x_correction.set_pauli(0, true, false);
    frame.apply(x_correction, wide_condition);

    PlannerPauli z_observable(2);
    z_observable.set_pauli(0, false, true);
    REQUIRE(frame.sign_for(z_observable) == wide_condition);

    PlannerPauli z_correction(2);
    z_correction.set_pauli(0, false, true);
    frame.apply(z_correction, AffineBool::symbol(SymbolId{1}));

    PlannerPauli y_observable(2);
    y_observable.set_pauli(0, true, true);
    REQUIRE(frame.sign_for(y_observable) == (wide_condition ^ AffineBool::symbol(SymbolId{1})));

    frame.apply(x_correction, wide_condition);
    REQUIRE(frame.sign_for(z_observable) == AffineBool(false));
}

TEST_CASE("Planner symbolic frame merges out-of-order and multi-row cancellations") {
    constexpr uint32_t kNumQubits = 128;
    constexpr uint32_t kNumSymbols = 20480;
    const uint32_t last_q = kNumQubits - 1;
    const uint32_t far_symbol = kNumSymbols - 1160;

    SymbolicPauliFrame frame(kNumQubits, kNumSymbols);
    PlannerPauli q0_x(kNumQubits), q1_x(kNumQubits), qlast_x(kNumQubits);
    q0_x.set_pauli(0, true, false);
    q1_x.set_pauli(1, true, false);
    qlast_x.set_pauli(last_q, true, false);

    // Populate 6 256-symbol blocks on q0, then cancel earlier blocks out of order.
    AffineBool expected_q0;
    for (uint32_t w = 0; w < 24; ++w) {
        const AffineBool term = AffineBool::symbol(SymbolId{64 * w + 3});
        frame.apply(q0_x, term);
        expected_q0 ^= term;
    }
    for (uint32_t w = 0; w < 18; ++w) {
        const AffineBool term = AffineBool::symbol(SymbolId{64 * (17 - w) + 3});
        frame.apply(q0_x, term);
        expected_q0 ^= term;
    }

    // Exercise 3-row cancellation and re-introduction of the same symbol word in sign_for.
    const AffineBool shared(true, {SymbolId{5}, SymbolId{far_symbol}});
    const AffineBool extra = AffineBool::symbol(SymbolId{far_symbol + 1});
    frame.apply(q1_x, shared);
    frame.apply(qlast_x, shared ^ extra);

    PlannerPauli obs_q0(kNumQubits);
    obs_q0.set_pauli(0, false, true);
    REQUIRE(frame.sign_for(obs_q0) == expected_q0);

    PlannerPauli obs_last(kNumQubits);
    obs_last.set_pauli(last_q, false, true);
    REQUIRE(frame.sign_for(obs_last) == (shared ^ extra));

    // Combine q0, q1, and qlast so word 0 and far_symbol's word cancel then reappear.
    frame.apply(q0_x, shared);
    PlannerPauli obs_all(kNumQubits);
    obs_all.set_pauli(0, false, true);
    obs_all.set_pauli(1, false, true);
    obs_all.set_pauli(last_q, false, true);
    REQUIRE(frame.sign_for(obs_all) == (expected_q0 ^ shared ^ extra));
}

TEST_CASE("Planner symbolic frame applies sparse corrections across packed words") {
    constexpr uint32_t kNumQubits = 70;
    SymbolicPauliFrame frame(kNumQubits, 130);
    const AffineBool condition(true, {SymbolId{0}, SymbolId{64}, SymbolId{129}});

    PlannerPauli correction(kNumQubits);
    correction.set_pauli(0, true, false);
    correction.set_pauli(1, false, true);
    correction.set_pauli(63, true, true);
    correction.set_pauli(64, true, false);
    correction.set_pauli(65, false, true);
    correction.set_pauli(69, true, true);
    frame.apply(correction, condition);

    for (uint32_t q : {0U, 63U, 64U, 69U}) {
        PlannerPauli observable(kNumQubits);
        observable.set_pauli(q, false, true);
        REQUIRE(frame.sign_for(observable) == condition);
    }
    for (uint32_t q : {1U, 63U, 65U, 69U}) {
        PlannerPauli observable(kNumQubits);
        observable.set_pauli(q, true, false);
        REQUIRE(frame.sign_for(observable) == condition);
    }

    PlannerPauli unaffected(kNumQubits);
    unaffected.set_pauli(62, true, false);
    unaffected.set_pauli(68, false, true);
    REQUIRE(frame.sign_for(unaffected) == AffineBool(false));

    frame.apply(correction, condition);
    PlannerPauli cancelled(kNumQubits);
    cancelled.set_pauli(69, true, true);
    REQUIRE(frame.sign_for(cancelled) == AffineBool(false));
}

TEST_CASE("Planner symbolic frame rejects unknown symbols") {
    SymbolicPauliFrame frame(1, 1);
    PlannerPauli correction(1);
    correction.set_pauli(0, true, false);

    REQUIRE_THROWS_AS(frame.apply(correction, AffineBool::symbol(SymbolId{1})), std::logic_error);
}

TEST_CASE("Planner symbolic frame handles partial final words") {
    constexpr uint32_t kNumQubits = 65;
    SymbolicPauliFrame frame(kNumQubits, 1);
    PlannerPauli correction(kNumQubits);
    correction.set_pauli(64, true, false);

    REQUIRE_NOTHROW(frame.apply(correction, AffineBool::symbol(SymbolId{0})));

    PlannerPauli observable(kNumQubits);
    observable.set_pauli(64, false, true);
    REQUIRE(frame.sign_for(observable) == AffineBool::symbol(SymbolId{0}));
}

TEST_CASE("Planner promotion frame localizes the promoted observable") {
    PlannerPauli promoted(3);
    promoted.set_pauli(0, false, true);
    promoted.set_pauli(2, true, false);

    const PlannerTableau frame = dormant_promotion_frame(promoted, 1, 2);
    REQUIRE(frame.satisfies_invariants());
    REQUIRE(clifft::PauliString(frame.x_output(1)) == promoted);

    const PlannerTableau old_to_new = frame.inverse();
    const PlannerPauli localized = old_to_new.apply(promoted.view());
    PlannerPauli expected(3);
    expected.set_pauli(1, true, false);
    REQUIRE(localized == expected);
}

TEST_CASE("Planner measurement frames install the selected observables") {
    PlannerPauli dormant(3);
    dormant.set_pauli(0, false, true);
    dormant.set_pauli(2, true, false);
    const PlannerTableau dormant_frame = dormant_measurement_frame(dormant, 2);
    REQUIRE(dormant_frame.satisfies_invariants());
    REQUIRE(clifft::PauliString(dormant_frame.z_output(2)) == dormant);

    PlannerPauli active(3);
    active.set_pauli(0, true, false);
    active.set_pauli(1, false, true);
    const PlannerTableau active_frame = active_measurement_frame(active, 2, 0);
    REQUIRE(active_frame.satisfies_invariants());
    REQUIRE(clifft::PauliString(active_frame.z_output(1)) == active);
}
