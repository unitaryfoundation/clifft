#include "clifft/tableau/tableau.h"

#include "reference_clifford.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <span>

namespace {

using clifft::test::ReferencePauli;
using clifft::test::ReferenceTableau;

void check_pauli(const ReferencePauli& reference, clifft::PauliStringView native) {
    REQUIRE(reference.num_qubits() == native.num_qubits());
    CHECK(reference.phase() == native.phase());
    for (uint32_t q = 0; q < reference.num_qubits(); ++q) {
        CHECK(reference.x(q) == native.x().bit_get(q));
        CHECK(reference.z(q) == native.z().bit_get(q));
    }
}

void check_tableau(const ReferenceTableau& reference, const clifft::Tableau& native) {
    REQUIRE(reference.num_qubits() == native.num_qubits());
    for (uint32_t q = 0; q < reference.num_qubits(); ++q) {
        check_pauli(reference.x_output(q), native.x_output(q));
        check_pauli(reference.z_output(q), native.z_output(q));
    }
}

clifft::PauliString native_pauli(const ReferencePauli& reference) {
    clifft::PauliString result(reference.num_qubits());
    for (uint32_t q = 0; q < reference.num_qubits(); ++q) {
        result.set_pauli(q, reference.x(q), reference.z(q));
    }
    result.set_sign(reference.sign());
    return result;
}

}  // namespace

TEST_CASE("Scalar Clifford reference matches native gate sequences", "[tableau]") {
    constexpr std::array<uint32_t, 8> widths{0, 1, 63, 64, 65, 127, 128, 129};
    for (uint32_t width : widths) {
        CAPTURE(width);
        ReferenceTableau reference(width);
        clifft::Tableau native(width);
        std::mt19937_64 rng(0x7265666572656e63ULL + width);

        for (uint32_t step = 0; step < 80 && width != 0; ++step) {
            const uint32_t first = static_cast<uint32_t>(rng() % width);
            const uint32_t kind = static_cast<uint32_t>(rng() % (width > 1 ? 4U : 3U));
            if (kind == 0) {
                reference.append_h(first);
                native.append_named_gate(clifft::GateType::H, {first});
            } else if (kind == 1) {
                reference.append_s(first);
                native.append_named_gate(clifft::GateType::S, {first});
            } else if (kind == 2) {
                reference.append_s_dag(first);
                native.append_named_gate(clifft::GateType::S_DAG, {first});
            } else {
                uint32_t second = first;
                while (second == first) {
                    second = static_cast<uint32_t>(rng() % width);
                }
                reference.append_cx(first, second);
                native.append_named_gate(clifft::GateType::CX, {first, second});
            }
        }
        check_tableau(reference, native);
    }
}

TEST_CASE("Scalar Clifford reference matches native composition", "[tableau]") {
    constexpr uint32_t width = 65;
    ReferenceTableau first(width);
    ReferenceTableau second(width);
    clifft::Tableau native_first(width);
    clifft::Tableau native_second(width);

    first.append_h(0);
    first.append_cx(0, 64);
    second.append_s(64);
    second.append_cx(64, 0);
    native_first.append_named_gate(clifft::GateType::H, {0});
    native_first.append_named_gate(clifft::GateType::CX, {0, 64});
    native_second.append_named_gate(clifft::GateType::S, {64});
    native_second.append_named_gate(clifft::GateType::CX, {64, 0});

    const ReferenceTableau reference_composed = first.then(second);
    const clifft::Tableau native_composed = native_first.then(native_second);
    check_tableau(reference_composed, native_composed);

    std::mt19937_64 rng(0x6170706c795f7265ULL);
    for (uint32_t sample = 0; sample < 100; ++sample) {
        ReferencePauli input(width);
        for (uint32_t q = 0; q < width; ++q) {
            const uint32_t pauli = static_cast<uint32_t>(rng() & 3U);
            input.set_pauli(q, (pauli & 1U) != 0, (pauli & 2U) != 0);
        }
        input.set_sign((rng() & 1U) != 0);
        clifft::PauliString native_input = native_pauli(input);
        check_pauli(reference_composed.apply(input),
                    native_composed.apply(native_input.view()).view());
    }
}

TEST_CASE("Scalar Clifford reference matches native inverse-state rewinding", "[tableau]") {
    ReferenceTableau reference(2);
    reference.append_cx(0, 1);
    reference.append_h(0);

    clifft::Tableau native(2);
    native.append_named_gate(clifft::GateType::CX, {0, 1});
    native.append_named_gate(clifft::GateType::H, {0});
    check_tableau(reference, native);
}

TEST_CASE("Scalar Clifford reference matches native Pauli rotations", "[tableau]") {
    constexpr uint32_t width = 129;
    constexpr std::array<uint32_t, 5> support{0, 63, 64, 127, 128};
    for (bool sign : {false, true}) {
        for (bool dagger : {false, true}) {
            CAPTURE(sign, dagger);
            ReferencePauli axis(width);
            for (uint32_t k = 0; k < support.size(); ++k) {
                const uint32_t pauli = k % 3;
                axis.set_pauli(support[k], pauli != 2, pauli != 0);
            }
            axis.set_sign(sign);
            const ReferenceTableau reference = ReferenceTableau::from_pauli_rotation(axis, dagger);
            const clifft::PauliString native_axis = native_pauli(axis);
            const clifft::Tableau native =
                clifft::Tableau::from_pauli_rotation(native_axis.view(), dagger);
            check_tableau(reference, native);
        }
    }
}
