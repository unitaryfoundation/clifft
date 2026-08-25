#include "clifft/tableau/tableau.h"

#include "reference_clifford.h"
#include "stim.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace {

using clifft::test::ReferencePauli;
using clifft::test::ReferenceTableau;

void check_pauli(const ReferencePauli& reference, clifft::PauliStringView native,
                 const stim::PauliStringRef<64>& oracle) {
    REQUIRE(reference.num_qubits() == native.num_qubits());
    REQUIRE(reference.num_qubits() == oracle.num_qubits);
    CHECK(reference.phase() == native.phase());
    CHECK(reference.sign() == oracle.sign);
    for (uint32_t q = 0; q < reference.num_qubits(); ++q) {
        CHECK(reference.x(q) == native.x().bit_get(q));
        CHECK(reference.z(q) == native.z().bit_get(q));
        CHECK(reference.x(q) == oracle.xs[q]);
        CHECK(reference.z(q) == oracle.zs[q]);
    }
}

void check_tableau(const ReferenceTableau& reference, const clifft::Tableau& native,
                   const stim::Tableau<64>& oracle) {
    REQUIRE(reference.num_qubits() == native.num_qubits());
    REQUIRE(reference.num_qubits() == oracle.num_qubits);
    for (uint32_t q = 0; q < reference.num_qubits(); ++q) {
        check_pauli(reference.x_output(q), native.x_output(q), oracle.xs[q]);
        check_pauli(reference.z_output(q), native.z_output(q), oracle.zs[q]);
    }
}

stim::Tableau<64> stim_pauli_rotation(const ReferencePauli& axis, bool dagger) {
    std::mt19937_64 rng(0);
    stim::TableauSimulator<64> simulator(std::move(rng), axis.num_qubits());
    std::vector<stim::GateTarget> targets;
    bool first = true;
    for (uint32_t q = 0; q < axis.num_qubits(); ++q) {
        if (!axis.x(q) && !axis.z(q)) {
            continue;
        }
        if (!first) {
            targets.push_back(stim::GateTarget::combiner());
        }
        targets.push_back(
            stim::GateTarget::pauli_xz(q, axis.x(q), axis.z(q), first && axis.sign()));
        first = false;
    }
    if (!targets.empty()) {
        const stim::CircuitInstruction instruction(
            dagger ? stim::GateType::SPP_DAG : stim::GateType::SPP, {}, targets, {});
        simulator.do_gate(instruction);
    }
    return simulator.inv_state.inverse();
}

clifft::PauliString native_pauli(const ReferencePauli& reference) {
    clifft::PauliString result(reference.num_qubits());
    for (uint32_t q = 0; q < reference.num_qubits(); ++q) {
        result.set_pauli(q, reference.x(q), reference.z(q));
    }
    result.set_sign(reference.sign());
    return result;
}

stim::PauliString<64> stim_pauli(const ReferencePauli& reference) {
    stim::PauliString<64> result(reference.num_qubits());
    for (uint32_t q = 0; q < reference.num_qubits(); ++q) {
        result.xs[q] = reference.x(q);
        result.zs[q] = reference.z(q);
    }
    result.sign = reference.sign();
    return result;
}

}  // namespace

TEST_CASE("Scalar Clifford reference matches Stim and native gate sequences", "[tableau]") {
    constexpr std::array<uint32_t, 8> widths{0, 1, 63, 64, 65, 127, 128, 129};
    for (uint32_t width : widths) {
        CAPTURE(width);
        ReferenceTableau reference(width);
        clifft::Tableau native(width);
        stim::Tableau<64> oracle(width);
        std::mt19937_64 rng(0x7265666572656e63ULL + width);

        for (uint32_t step = 0; step < 80 && width != 0; ++step) {
            const uint32_t first = static_cast<uint32_t>(rng() % width);
            const uint32_t kind = static_cast<uint32_t>(rng() % (width > 1 ? 4U : 3U));
            if (kind == 0) {
                reference.append_h(first);
                native.append_named_gate(clifft::GateType::H, {first});
                oracle.inplace_scatter_append(stim::GATE_DATA.at("H").tableau<64>(), {first});
            } else if (kind == 1) {
                reference.append_s(first);
                native.append_named_gate(clifft::GateType::S, {first});
                oracle.inplace_scatter_append(stim::GATE_DATA.at("S").tableau<64>(), {first});
            } else if (kind == 2) {
                reference.append_s_dag(first);
                native.append_named_gate(clifft::GateType::S_DAG, {first});
                oracle.inplace_scatter_append(stim::GATE_DATA.at("S_DAG").tableau<64>(), {first});
            } else {
                uint32_t second = first;
                while (second == first) {
                    second = static_cast<uint32_t>(rng() % width);
                }
                reference.append_cx(first, second);
                native.append_named_gate(clifft::GateType::CX, {first, second});
                oracle.inplace_scatter_append(stim::GATE_DATA.at("CX").tableau<64>(),
                                              {first, second});
            }
        }
        check_tableau(reference, native, oracle);
    }
}

TEST_CASE("Scalar Clifford reference matches Stim and native composition", "[tableau]") {
    constexpr uint32_t width = 65;
    ReferenceTableau first(width);
    ReferenceTableau second(width);
    clifft::Tableau native_first(width);
    clifft::Tableau native_second(width);
    stim::Tableau<64> oracle_first(width);
    stim::Tableau<64> oracle_second(width);

    first.append_h(0);
    first.append_cx(0, 64);
    second.append_s(64);
    second.append_cx(64, 0);
    native_first.append_named_gate(clifft::GateType::H, {0});
    native_first.append_named_gate(clifft::GateType::CX, {0, 64});
    native_second.append_named_gate(clifft::GateType::S, {64});
    native_second.append_named_gate(clifft::GateType::CX, {64, 0});
    oracle_first.inplace_scatter_append(stim::GATE_DATA.at("H").tableau<64>(), {0});
    oracle_first.inplace_scatter_append(stim::GATE_DATA.at("CX").tableau<64>(), {0, 64});
    oracle_second.inplace_scatter_append(stim::GATE_DATA.at("S").tableau<64>(), {64});
    oracle_second.inplace_scatter_append(stim::GATE_DATA.at("CX").tableau<64>(), {64, 0});

    const ReferenceTableau reference_composed = first.then(second);
    const clifft::Tableau native_composed = native_first.then(native_second);
    const stim::Tableau<64> oracle_composed = oracle_first.then(oracle_second);
    check_tableau(reference_composed, native_composed, oracle_composed);

    std::mt19937_64 rng(0x6170706c795f7265ULL);
    for (uint32_t sample = 0; sample < 100; ++sample) {
        ReferencePauli input(width);
        for (uint32_t q = 0; q < width; ++q) {
            const uint32_t pauli = static_cast<uint32_t>(rng() & 3U);
            input.set_pauli(q, (pauli & 1U) != 0, (pauli & 2U) != 0);
        }
        input.set_sign((rng() & 1U) != 0);
        clifft::PauliString native_input = native_pauli(input);
        stim::PauliString<64> oracle_input = stim_pauli(input);
        check_pauli(reference_composed.apply(input),
                    native_composed.apply(native_input.view()).view(),
                    oracle_composed(oracle_input));
    }
}

TEST_CASE("Scalar Clifford reference matches Stim inverse-state rewinding", "[tableau]") {
    ReferenceTableau reference(2);
    reference.append_cx(0, 1);
    reference.append_h(0);

    clifft::Tableau native(2);
    native.append_named_gate(clifft::GateType::CX, {0, 1});
    native.append_named_gate(clifft::GateType::H, {0});

    std::mt19937_64 rng(0);
    stim::TableauSimulator<64> simulator(std::move(rng), 2);
    simulator.inv_state.prepend_H_XZ(0);
    simulator.inv_state.prepend_ZCX(0, 1);
    check_tableau(reference, native, simulator.inv_state);
}

TEST_CASE("Scalar Clifford reference matches Stim and native Pauli rotations", "[tableau]") {
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
            const stim::Tableau<64> oracle = stim_pauli_rotation(axis, dagger);
            check_tableau(reference, native, oracle);
        }
    }
}
