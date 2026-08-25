#include "clifft/tableau/tableau.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <span>
#include <string_view>

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

clifft::GateType expected_inverse(clifft::GateType gate) {
    using clifft::GateType;
    switch (gate) {
        case GateType::S:
            return GateType::S_DAG;
        case GateType::S_DAG:
            return GateType::S;
        case GateType::SQRT_X:
            return GateType::SQRT_X_DAG;
        case GateType::SQRT_X_DAG:
            return GateType::SQRT_X;
        case GateType::SQRT_Y:
            return GateType::SQRT_Y_DAG;
        case GateType::SQRT_Y_DAG:
            return GateType::SQRT_Y;
        case GateType::C_XYZ:
            return GateType::C_ZYX;
        case GateType::C_ZYX:
            return GateType::C_XYZ;
        case GateType::C_NXYZ:
            return GateType::C_ZYNX;
        case GateType::C_NZYX:
            return GateType::C_XYNZ;
        case GateType::C_XNYZ:
            return GateType::C_ZNYX;
        case GateType::C_XYNZ:
            return GateType::C_NZYX;
        case GateType::C_ZNYX:
            return GateType::C_XNYZ;
        case GateType::C_ZYNX:
            return GateType::C_NXYZ;
        case GateType::ISWAP:
            return GateType::ISWAP_DAG;
        case GateType::ISWAP_DAG:
            return GateType::ISWAP;
        case GateType::SQRT_XX:
            return GateType::SQRT_XX_DAG;
        case GateType::SQRT_XX_DAG:
            return GateType::SQRT_XX;
        case GateType::SQRT_YY:
            return GateType::SQRT_YY_DAG;
        case GateType::SQRT_YY_DAG:
            return GateType::SQRT_YY;
        case GateType::SQRT_ZZ:
            return GateType::SQRT_ZZ_DAG;
        case GateType::SQRT_ZZ_DAG:
            return GateType::SQRT_ZZ;
        case GateType::CXSWAP:
            return GateType::SWAPCX;
        case GateType::SWAPCX:
            return GateType::CXSWAP;
        default:
            return gate;
    }
}

void check_tableau_well_formed(const clifft::Tableau& tableau) {
    for (uint32_t q = 0; q < tableau.num_qubits(); ++q) {
        for (clifft::PauliStringView row : {tableau.x_output(q), tableau.z_output(q)}) {
            CHECK(row.is_hermitian());
            if (row.num_qubits() % 64 != 0 && row.num_qubits() != 0) {
                const uint64_t padding = ~((uint64_t{1} << (row.num_qubits() % 64)) - 1);
                CHECK((row.x().words.back() & padding) == 0);
                CHECK((row.z().words.back() & padding) == 0);
            }
        }
    }
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

TEST_CASE("Native named Clifford inverses compose to identity", "[tableau]") {
    for (const GateCase& gate : kNamedCliffords) {
        CAPTURE(gate.name);
        const clifft::Tableau forward = clifft::Tableau::from_named_gate(gate.gate);
        const clifft::Tableau inverse =
            clifft::Tableau::from_named_gate(expected_inverse(gate.gate));
        const clifft::Tableau identity(gate.arity);
        CHECK(forward.then(inverse) == identity);
        CHECK(inverse.then(forward) == identity);
    }
}

TEST_CASE("Native local composition stays well formed across mask words", "[tableau]") {
    constexpr std::array<uint32_t, 8> widths{0, 1, 63, 64, 65, 127, 128, 129};
    for (uint32_t width : widths) {
        CAPTURE(width);
        clifft::Tableau appended(width);
        clifft::Tableau prepended(width);
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

            appended.append_named_gate(gate.gate, native_span);
            prepended.prepend_named_gate(gate.gate, native_span);
        }

        check_tableau_well_formed(appended);
        check_tableau_well_formed(prepended);
        const clifft::Tableau identity(width);
        CHECK(appended.then(appended.inverse()) == identity);
        CHECK(appended.inverse().then(appended) == identity);
        CHECK(prepended.then(prepended.inverse()) == identity);
        CHECK(prepended.inverse().then(prepended) == identity);
    }
}

TEST_CASE("Native tableau application agrees with composition", "[tableau]") {
    constexpr uint32_t width = 65;
    clifft::Tableau first(width);
    clifft::Tableau second(width);

    const std::array<uint32_t, 2> native_pair{0, 64};
    first.append_named_gate(clifft::GateType::H, std::span(native_pair).first<1>());
    first.append_named_gate(clifft::GateType::CX, native_pair);
    second.append_named_gate(clifft::GateType::S, std::span(native_pair).last<1>());
    second.append_named_gate(clifft::GateType::CY, native_pair);
    const clifft::Tableau composed = first.then(second);

    std::mt19937_64 rng(0x6170706c79ULL);
    for (uint32_t sample = 0; sample < 100; ++sample) {
        clifft::PauliString input(width);
        for (uint32_t q = 0; q < width; ++q) {
            const uint32_t pauli = static_cast<uint32_t>(rng() & 3U);
            input.set_pauli(q, (pauli & 1U) != 0, (pauli & 2U) != 0);
        }
        input.set_sign((rng() & 1U) != 0);
        const clifft::PauliString intermediate = first.apply(input.view());
        CHECK(composed.apply(input.view()) == second.apply(intermediate.view()));
    }
}

TEST_CASE("Native Pauli rotations compose across mask words", "[tableau]") {
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
            const clifft::Tableau rotation =
                clifft::Tableau::from_pauli_rotation(axis.view(), dagger);
            const clifft::Tableau inverse_rotation =
                clifft::Tableau::from_pauli_rotation(axis.view(), !dagger);
            CHECK(rotation.then(inverse_rotation) == clifft::Tableau(width));

            clifft::Tableau prepended(width);
            prepended.append_named_gate(clifft::GateType::H, std::span(qubits).first<1>());
            prepended.append_named_gate(clifft::GateType::CX, std::span(qubits).subspan<1, 2>());
            const clifft::Tableau before = prepended;
            prepended.prepend_pauli_rotation(axis.view(), dagger);
            CHECK(prepended == rotation.then(before));

            clifft::Tableau prepended_pauli = before;
            prepended_pauli.prepend_pauli(axis.view());
            const clifft::Tableau pauli_rotation =
                clifft::Tableau::from_pauli_rotation(axis.view(), false);
            CHECK(prepended_pauli == pauli_rotation.then(pauli_rotation).then(before));
        }
    }
}
