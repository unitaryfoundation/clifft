#include "clifft/tableau/tableau.h"

#include "stim.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct GateCase {
    clifft::GateType gate;
    std::string_view name;
    uint32_t arity;
};

constexpr std::array kNamedCliffords{
    GateCase{clifft::GateType::H, "H", 1},
    GateCase{clifft::GateType::S, "S", 1},
    GateCase{clifft::GateType::S_DAG, "S_DAG", 1},
    GateCase{clifft::GateType::X, "X", 1},
    GateCase{clifft::GateType::Y, "Y", 1},
    GateCase{clifft::GateType::Z, "Z", 1},
    GateCase{clifft::GateType::SQRT_X, "SQRT_X", 1},
    GateCase{clifft::GateType::SQRT_X_DAG, "SQRT_X_DAG", 1},
    GateCase{clifft::GateType::SQRT_Y, "SQRT_Y", 1},
    GateCase{clifft::GateType::SQRT_Y_DAG, "SQRT_Y_DAG", 1},
    GateCase{clifft::GateType::H_XY, "H_XY", 1},
    GateCase{clifft::GateType::H_YZ, "H_YZ", 1},
    GateCase{clifft::GateType::H_NXY, "H_NXY", 1},
    GateCase{clifft::GateType::H_NXZ, "H_NXZ", 1},
    GateCase{clifft::GateType::H_NYZ, "H_NYZ", 1},
    GateCase{clifft::GateType::C_XYZ, "C_XYZ", 1},
    GateCase{clifft::GateType::C_ZYX, "C_ZYX", 1},
    GateCase{clifft::GateType::C_NXYZ, "C_NXYZ", 1},
    GateCase{clifft::GateType::C_NZYX, "C_NZYX", 1},
    GateCase{clifft::GateType::C_XNYZ, "C_XNYZ", 1},
    GateCase{clifft::GateType::C_XYNZ, "C_XYNZ", 1},
    GateCase{clifft::GateType::C_ZNYX, "C_ZNYX", 1},
    GateCase{clifft::GateType::C_ZYNX, "C_ZYNX", 1},
    GateCase{clifft::GateType::CX, "CX", 2},
    GateCase{clifft::GateType::CY, "CY", 2},
    GateCase{clifft::GateType::CZ, "CZ", 2},
    GateCase{clifft::GateType::SWAP, "SWAP", 2},
    GateCase{clifft::GateType::ISWAP, "ISWAP", 2},
    GateCase{clifft::GateType::ISWAP_DAG, "ISWAP_DAG", 2},
    GateCase{clifft::GateType::SQRT_XX, "SQRT_XX", 2},
    GateCase{clifft::GateType::SQRT_XX_DAG, "SQRT_XX_DAG", 2},
    GateCase{clifft::GateType::SQRT_YY, "SQRT_YY", 2},
    GateCase{clifft::GateType::SQRT_YY_DAG, "SQRT_YY_DAG", 2},
    GateCase{clifft::GateType::SQRT_ZZ, "SQRT_ZZ", 2},
    GateCase{clifft::GateType::SQRT_ZZ_DAG, "SQRT_ZZ_DAG", 2},
    GateCase{clifft::GateType::CXSWAP, "CXSWAP", 2},
    GateCase{clifft::GateType::CZSWAP, "CZSWAP", 2},
    GateCase{clifft::GateType::SWAPCX, "SWAPCX", 2},
    GateCase{clifft::GateType::XCX, "XCX", 2},
    GateCase{clifft::GateType::XCY, "XCY", 2},
    GateCase{clifft::GateType::XCZ, "XCZ", 2},
    GateCase{clifft::GateType::YCX, "YCX", 2},
    GateCase{clifft::GateType::YCY, "YCY", 2},
    GateCase{clifft::GateType::YCZ, "YCZ", 2},
};

template <typename StimPauli>
void check_pauli(clifft::PauliStringView actual, const StimPauli& expected) {
    REQUIRE(actual.is_hermitian());
    CHECK(actual.sign() == expected.sign);
    for (uint32_t q = 0; q < actual.num_qubits(); ++q) {
        CHECK(actual.x().bit_get(q) == expected.xs[q]);
        CHECK(actual.z().bit_get(q) == expected.zs[q]);
    }
    if (actual.num_qubits() % 64 != 0 && actual.num_qubits() != 0) {
        const uint64_t padding = ~((uint64_t{1} << (actual.num_qubits() % 64)) - 1);
        CHECK((actual.x().words.back() & padding) == 0);
        CHECK((actual.z().words.back() & padding) == 0);
    }
}

void check_tableau(const clifft::Tableau& actual, const stim::Tableau<64>& expected) {
    REQUIRE(actual.num_qubits() == expected.num_qubits);
    for (uint32_t q = 0; q < actual.num_qubits(); ++q) {
        check_pauli(actual.x_output(q), expected.xs[q]);
        check_pauli(actual.z_output(q), expected.zs[q]);
    }
}

stim::Tableau<64> stim_pauli_rotation(clifft::PauliStringView axis, bool dagger) {
    std::mt19937_64 rng(0);
    stim::TableauSimulator<64> simulator(std::move(rng), axis.num_qubits());
    std::vector<stim::GateTarget> targets;
    bool first = true;
    for (uint32_t q = 0; q < axis.num_qubits(); ++q) {
        if (!axis.x().bit_get(q) && !axis.z().bit_get(q)) {
            continue;
        }
        if (!first) {
            targets.push_back(stim::GateTarget::combiner());
        }
        targets.push_back(stim::GateTarget::pauli_xz(q, axis.x().bit_get(q), axis.z().bit_get(q),
                                                     first && axis.sign()));
        first = false;
    }
    if (!targets.empty()) {
        const stim::CircuitInstruction instruction(
            dagger ? stim::GateType::SPP_DAG : stim::GateType::SPP, {}, targets, {});
        simulator.do_gate(instruction);
    }
    return simulator.inv_state.inverse();
}

}  // namespace

TEST_CASE("Native Pauli phase convention preserves Hermitian signs", "[tableau]") {
    const clifft::PauliString x = clifft::PauliString::from_text("+X");
    const clifft::PauliString y = clifft::PauliString::from_text("+Y");
    const clifft::PauliString negative_y = clifft::PauliString::from_text("-Y");
    CHECK(x.phase() == 0);
    CHECK(y.phase() == 1);
    CHECK_FALSE(y.sign());
    CHECK(negative_y.phase() == 3);
    CHECK(negative_y.sign());

    clifft::PauliString product = x;
    product.right_multiply(y.view());
    CHECK(product.is_hermitian() == false);
    product.right_multiply(x.view());
    CHECK(product == negative_y);
}

TEST_CASE("Native named Clifford cases cover fixed gate metadata", "[tableau]") {
    constexpr size_t gate_count = static_cast<size_t>(clifft::GateType::UNKNOWN);
    std::array<bool, gate_count> covered{};

    for (const GateCase& gate : kNamedCliffords) {
        CAPTURE(gate.name);
        const size_t index = static_cast<size_t>(gate.gate);
        REQUIRE(index < covered.size());
        CHECK(clifft::is_clifford(gate.gate));
        const clifft::GateArity arity = clifft::gate_arity(gate.gate);
        CHECK((arity == clifft::GateArity::SINGLE || arity == clifft::GateArity::PAIR));
        CHECK(gate.arity == (arity == clifft::GateArity::SINGLE ? 1U : 2U));
        CHECK_FALSE(covered[index]);
        covered[index] = true;
    }

    for (size_t index = 0; index < covered.size(); ++index) {
        const auto gate = static_cast<clifft::GateType>(index);
        const clifft::GateArity arity = clifft::gate_arity(gate);
        if (!clifft::is_clifford(gate) ||
            (arity != clifft::GateArity::SINGLE && arity != clifft::GateArity::PAIR)) {
            continue;
        }
        CAPTURE(clifft::gate_name(gate));
        CHECK(covered[index]);
    }
}

TEST_CASE("Native named Clifford rows match Stim", "[tableau]") {
    for (const GateCase& gate : kNamedCliffords) {
        CAPTURE(gate.name);
        const clifft::Tableau actual = clifft::Tableau::from_named_gate(gate.gate);
        const stim::Tableau<64> expected = stim::GATE_DATA.at(gate.name).tableau<64>();
        check_tableau(actual, expected);
        check_tableau(actual.inverse(), expected.inverse());
    }
}

TEST_CASE("Native local composition matches Stim across mask words", "[tableau]") {
    constexpr std::array<uint32_t, 8> widths{0, 1, 63, 64, 65, 127, 128, 129};
    for (uint32_t width : widths) {
        CAPTURE(width);
        clifft::Tableau appended(width);
        clifft::Tableau prepended(width);
        stim::Tableau<64> expected_appended(width);
        stim::Tableau<64> expected_prepended(width);
        std::mt19937_64 rng(0x6c6f63616cULL + width);

        for (uint32_t step = 0; step < 80 && width != 0; ++step) {
            const GateCase& gate = kNamedCliffords[rng() % kNamedCliffords.size()];
            if (gate.arity > width) {
                continue;
            }
            const uint32_t first = static_cast<uint32_t>(rng() % width);
            uint32_t second = first;
            while (gate.arity == 2 && second == first) {
                second = static_cast<uint32_t>(rng() % width);
            }
            const std::array<uint32_t, 2> native_targets{first, second};
            const auto native_span = std::span(native_targets).first(gate.arity);
            std::vector<size_t> stim_targets{first};
            if (gate.arity == 2) {
                stim_targets.push_back(second);
            }
            const stim::Tableau<64> stim_gate = stim::GATE_DATA.at(gate.name).tableau<64>();

            appended.append_named_gate(gate.gate, native_span);
            prepended.prepend_named_gate(gate.gate, native_span);
            expected_appended.inplace_scatter_append(stim_gate, stim_targets);
            expected_prepended.inplace_scatter_prepend(stim_gate, stim_targets);
        }

        check_tableau(appended, expected_appended);
        check_tableau(prepended, expected_prepended);
        check_tableau(appended.inverse(), expected_appended.inverse());
        check_tableau(prepended.inverse(), expected_prepended.inverse());
    }
}

TEST_CASE("Native tableau application and composition match Stim", "[tableau]") {
    constexpr uint32_t width = 65;
    clifft::Tableau first(width);
    clifft::Tableau second(width);
    stim::Tableau<64> expected_first(width);
    stim::Tableau<64> expected_second(width);

    const std::array<uint32_t, 2> native_pair{0, 64};
    const std::vector<size_t> stim_first{0};
    const std::vector<size_t> stim_second{64};
    const std::vector<size_t> stim_pair{0, 64};
    first.append_named_gate(clifft::GateType::H, std::span(native_pair).first<1>());
    first.append_named_gate(clifft::GateType::CX, native_pair);
    second.append_named_gate(clifft::GateType::S, std::span(native_pair).last<1>());
    second.append_named_gate(clifft::GateType::CY, native_pair);
    expected_first.inplace_scatter_append(stim::GATE_DATA.at("H").tableau<64>(), stim_first);
    expected_first.inplace_scatter_append(stim::GATE_DATA.at("CX").tableau<64>(), stim_pair);
    expected_second.inplace_scatter_append(stim::GATE_DATA.at("S").tableau<64>(), stim_second);
    expected_second.inplace_scatter_append(stim::GATE_DATA.at("CY").tableau<64>(), stim_pair);

    check_tableau(first.then(second), expected_first.then(expected_second));

    std::mt19937_64 rng(0x6170706c79ULL);
    for (uint32_t sample = 0; sample < 100; ++sample) {
        clifft::PauliString input(width);
        stim::PauliString<64> expected_input(width);
        for (uint32_t q = 0; q < width; ++q) {
            const uint32_t pauli = static_cast<uint32_t>(rng() & 3U);
            input.set_pauli(q, (pauli & 1U) != 0, (pauli & 2U) != 0);
            expected_input.xs[q] = (pauli & 1U) != 0;
            expected_input.zs[q] = (pauli & 2U) != 0;
        }
        input.set_sign((rng() & 1U) != 0);
        expected_input.sign = input.sign();
        check_pauli(first.apply(input.view()).view(), expected_first(expected_input));
    }
}

TEST_CASE("Native Pauli rotations match Stim across mask words", "[tableau]") {
    constexpr uint32_t width = 129;
    constexpr std::array<uint32_t, 5> qubits{0, 63, 64, 127, 128};
    for (bool sign : {false, true}) {
        for (bool dagger : {false, true}) {
            clifft::PauliString axis(width);
            for (uint32_t k = 0; k < qubits.size(); ++k) {
                const uint32_t pauli = k % 3;
                axis.set_pauli(qubits[k], pauli != 2, pauli != 0);
            }
            axis.set_sign(sign);
            CAPTURE(sign, dagger);
            const stim::Tableau<64> expected_rotation = stim_pauli_rotation(axis.view(), dagger);
            check_tableau(clifft::Tableau::from_pauli_rotation(axis.view(), dagger),
                          expected_rotation);

            clifft::Tableau prepended(width);
            stim::Tableau<64> expected_before(width);
            prepended.append_named_gate(clifft::GateType::H, std::span(qubits).first<1>());
            prepended.append_named_gate(clifft::GateType::CX, std::span(qubits).subspan<1, 2>());
            expected_before.inplace_scatter_append(stim::GATE_DATA.at("H").tableau<64>(), {0});
            expected_before.inplace_scatter_append(stim::GATE_DATA.at("CX").tableau<64>(),
                                                   {63, 64});
            prepended.prepend_pauli_rotation(axis.view(), dagger);
            check_tableau(prepended, expected_rotation.then(expected_before));

            clifft::Tableau prepended_pauli(width);
            prepended_pauli.append_named_gate(clifft::GateType::H, std::span(qubits).first<1>());
            prepended_pauli.append_named_gate(clifft::GateType::CX,
                                              std::span(qubits).subspan<1, 2>());
            prepended_pauli.prepend_pauli(axis.view());
            check_tableau(prepended_pauli, stim_pauli_rotation(axis.view(), false)
                                               .then(stim_pauli_rotation(axis.view(), false))
                                               .then(expected_before));
        }
    }
}
