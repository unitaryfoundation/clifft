#include "clifft/tableau/tableau.h"

#include <array>
#include <cassert>
#include <stdexcept>
#include <string>

namespace clifft {

namespace {

uint32_t row_index(bool z, uint32_t qubit, uint32_t num_qubits) {
    return (z ? num_qubits : 0) + qubit;
}

}  // namespace

Tableau::Tableau(uint32_t num_qubits)
    : num_qubits_(num_qubits),
      num_words_((num_qubits + 63) / 64),
      x_rows_(static_cast<size_t>(2) * num_qubits * num_words_, 0),
      z_rows_(x_rows_.size(), 0),
      phases_(static_cast<size_t>(2) * num_qubits, 0) {
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        x_rows_[static_cast<size_t>(q) * num_words_ + q / 64] |= uint64_t{1} << (q % 64);
        z_rows_[static_cast<size_t>(num_qubits_ + q) * num_words_ + q / 64] |= uint64_t{1}
                                                                               << (q % 64);
    }
}

Tableau Tableau::from_rows(std::initializer_list<std::string_view> rows) {
    if (rows.size() % 2 != 0) {
        throw std::invalid_argument("A tableau requires one X and one Z row per qubit");
    }
    Tableau result(static_cast<uint32_t>(rows.size() / 2));
    uint32_t q = 0;
    for (auto it = rows.begin(); it != rows.end();) {
        const PauliString x = PauliString::from_text(*it++);
        const PauliString z = PauliString::from_text(*it++);
        if (x.num_qubits() != result.num_qubits() || z.num_qubits() != result.num_qubits()) {
            throw std::invalid_argument("Tableau row width mismatch");
        }
        result.set_row(row_index(false, q, result.num_qubits()), x.view());
        result.set_row(row_index(true, q, result.num_qubits()), z.view());
        ++q;
    }
    return result;
}

Tableau Tableau::from_named_gate(GateType gate) {
    switch (gate) {
        case GateType::H:
            return from_rows({"+Z", "+X"});
        case GateType::S:
            return from_rows({"+Y", "+Z"});
        case GateType::S_DAG:
            return from_rows({"-Y", "+Z"});
        case GateType::X:
            return from_rows({"+X", "-Z"});
        case GateType::Y:
            return from_rows({"-X", "-Z"});
        case GateType::Z:
            return from_rows({"-X", "+Z"});
        case GateType::SQRT_X:
            return from_rows({"+X", "-Y"});
        case GateType::SQRT_X_DAG:
            return from_rows({"+X", "+Y"});
        case GateType::SQRT_Y:
            return from_rows({"-Z", "+X"});
        case GateType::SQRT_Y_DAG:
            return from_rows({"+Z", "-X"});
        case GateType::H_XY:
            return from_rows({"+Y", "-Z"});
        case GateType::H_YZ:
            return from_rows({"-X", "+Y"});
        case GateType::H_NXY:
            return from_rows({"-Y", "-Z"});
        case GateType::H_NXZ:
            return from_rows({"-Z", "-X"});
        case GateType::H_NYZ:
            return from_rows({"-X", "-Y"});
        case GateType::C_XYZ:
            return from_rows({"+Y", "+X"});
        case GateType::C_ZYX:
            return from_rows({"+Z", "+Y"});
        case GateType::C_NXYZ:
            return from_rows({"-Y", "-X"});
        case GateType::C_NZYX:
            return from_rows({"-Z", "-Y"});
        case GateType::C_XNYZ:
            return from_rows({"-Y", "+X"});
        case GateType::C_XYNZ:
            return from_rows({"+Y", "-X"});
        case GateType::C_ZNYX:
            return from_rows({"+Z", "-Y"});
        case GateType::C_ZYNX:
            return from_rows({"-Z", "+Y"});
        case GateType::I:
            return Tableau(1);
        case GateType::CX:
            return from_rows({"+XX", "+Z_", "+_X", "+ZZ"});
        case GateType::CY:
            return from_rows({"+XY", "+Z_", "+ZX", "+ZZ"});
        case GateType::CZ:
            return from_rows({"+XZ", "+Z_", "+ZX", "+_Z"});
        case GateType::SWAP:
            return from_rows({"+_X", "+_Z", "+X_", "+Z_"});
        case GateType::ISWAP:
            return from_rows({"+ZY", "+_Z", "+YZ", "+Z_"});
        case GateType::ISWAP_DAG:
            return from_rows({"-ZY", "+_Z", "-YZ", "+Z_"});
        case GateType::SQRT_XX:
            return from_rows({"+X_", "-YX", "+_X", "-XY"});
        case GateType::SQRT_XX_DAG:
            return from_rows({"+X_", "+YX", "+_X", "+XY"});
        case GateType::SQRT_YY:
            return from_rows({"-ZY", "+XY", "-YZ", "+YX"});
        case GateType::SQRT_YY_DAG:
            return from_rows({"+ZY", "-XY", "+YZ", "-YX"});
        case GateType::SQRT_ZZ:
            return from_rows({"+YZ", "+Z_", "+ZY", "+_Z"});
        case GateType::SQRT_ZZ_DAG:
            return from_rows({"-YZ", "+Z_", "-ZY", "+_Z"});
        case GateType::CXSWAP:
            return from_rows({"+XX", "+_Z", "+X_", "+ZZ"});
        case GateType::CZSWAP:
            return from_rows({"+ZX", "+_Z", "+XZ", "+Z_"});
        case GateType::SWAPCX:
            return from_rows({"+_X", "+ZZ", "+XX", "+Z_"});
        case GateType::XCX:
            return from_rows({"+X_", "+ZX", "+_X", "+XZ"});
        case GateType::XCY:
            return from_rows({"+X_", "+ZY", "+XX", "+XZ"});
        case GateType::XCZ:
            return from_rows({"+X_", "+ZZ", "+XX", "+_Z"});
        case GateType::YCX:
            return from_rows({"+XX", "+ZX", "+_X", "+YZ"});
        case GateType::YCY:
            return from_rows({"+XY", "+ZY", "+YX", "+YZ"});
        case GateType::YCZ:
            return from_rows({"+XZ", "+ZZ", "+YX", "+_Z"});
        case GateType::II:
            return Tableau(2);
        default:
            throw std::invalid_argument("Gate does not have a fixed Clifford tableau: " +
                                        std::string(gate_name(gate)));
    }
}

Tableau Tableau::from_pauli_rotation(PauliStringView axis, bool dagger) {
    if (!axis.is_hermitian()) {
        throw std::invalid_argument("Pauli rotation axis must be Hermitian");
    }
    Tableau result(axis.num_qubits());
    Tableau identity(axis.num_qubits());
    for (uint32_t q = 0; q < axis.num_qubits(); ++q) {
        for (bool z_generator : {false, true}) {
            const uint32_t index = row_index(z_generator, q, axis.num_qubits());
            const PauliStringView generator = identity.row(index);
            if (axis.commutes(generator)) {
                continue;
            }
            PauliString mapped(axis.num_qubits());
            mapped.set_phase(axis.phase());
            mapped.mut_x().xor_with(axis.x());
            mapped.mut_z().xor_with(axis.z());
            mapped.right_multiply(generator);
            mapped.add_phase(dagger ? 1 : 3);
            result.set_row(index, mapped.view());
        }
    }
    return result;
}

PauliStringView Tableau::row(uint32_t index) const {
    assert(index < 2 * num_qubits_);
    const size_t offset = static_cast<size_t>(index) * num_words_;
    return PauliStringView{MaskView{std::span<const uint64_t>(x_rows_).subspan(offset, num_words_)},
                           MaskView{std::span<const uint64_t>(z_rows_).subspan(offset, num_words_)},
                           phases_[index], num_qubits_};
}

PauliStringView Tableau::x_output(uint32_t qubit) const {
    assert(qubit < num_qubits_);
    return row(row_index(false, qubit, num_qubits_));
}

PauliStringView Tableau::z_output(uint32_t qubit) const {
    assert(qubit < num_qubits_);
    return row(row_index(true, qubit, num_qubits_));
}

PauliString Tableau::y_output(uint32_t qubit) const {
    PauliString input(num_qubits_);
    input.set_pauli(qubit, true, true);
    input.set_sign(false);
    return apply(input.view());
}

void Tableau::set_row(uint32_t index, PauliStringView value) {
    assert(index < 2 * num_qubits_);
    assert(value.num_qubits() == num_qubits_);
    const size_t offset = static_cast<size_t>(index) * num_words_;
    for (uint32_t w = 0; w < num_words_; ++w) {
        x_rows_[offset + w] = value.x().words[w];
        z_rows_[offset + w] = value.z().words[w];
    }
    phases_[index] = value.phase();
}

PauliString Tableau::apply(PauliStringView input) const {
    assert(input.num_qubits() == num_qubits_);
    PauliString result(num_qubits_);
    result.set_phase(input.phase());
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (input.x().bit_get(q)) {
            result.right_multiply(x_output(q));
        }
    }
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (input.z().bit_get(q)) {
            result.right_multiply(z_output(q));
        }
    }
    return result;
}

Tableau Tableau::then(const Tableau& next) const {
    if (num_qubits_ != next.num_qubits_) {
        throw std::invalid_argument("Cannot compose tableaus with different widths");
    }
    Tableau result(num_qubits_);
    for (uint32_t row_i = 0; row_i < 2 * num_qubits_; ++row_i) {
        const PauliString mapped = next.apply(row(row_i));
        result.set_row(row_i, mapped.view());
    }
    return result;
}

Tableau Tableau::inverse() const {
    Tableau result(num_qubits_);
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        PauliString inverse_x(num_qubits_);
        PauliString inverse_z(num_qubits_);
        for (uint32_t k = 0; k < num_qubits_; ++k) {
            inverse_x.set_pauli(k, z_output(k).z().bit_get(q), x_output(k).z().bit_get(q));
            inverse_z.set_pauli(k, z_output(k).x().bit_get(q), x_output(k).x().bit_get(q));
        }
        inverse_x.set_sign(false);
        inverse_z.set_sign(false);
        if (apply(inverse_x.view()).sign()) {
            inverse_x.negate();
        }
        if (apply(inverse_z.view()).sign()) {
            inverse_z.negate();
        }
        result.set_row(row_index(false, q, num_qubits_), inverse_x.view());
        result.set_row(row_index(true, q, num_qubits_), inverse_z.view());
    }
    return result;
}

void Tableau::append_local(const Tableau& gate, std::span<const uint32_t> targets) {
    if (gate.num_qubits() != targets.size()) {
        throw std::invalid_argument("Gate tableau width does not match target count");
    }
    for (uint32_t target : targets) {
        if (target >= num_qubits_) {
            throw std::invalid_argument("Gate target is outside the tableau");
        }
    }

    for (uint32_t row_i = 0; row_i < 2 * num_qubits_; ++row_i) {
        const PauliStringView old = row(row_i);
        PauliString local(gate.num_qubits());
        for (uint32_t local_q = 0; local_q < targets.size(); ++local_q) {
            local.set_pauli(local_q, old.x().bit_get(targets[local_q]),
                            old.z().bit_get(targets[local_q]));
        }
        local.set_sign(false);
        const PauliString mapped = gate.apply(local.view());

        PauliString replacement(num_qubits_);
        replacement.set_phase(old.phase());
        replacement.mut_x().xor_with(old.x());
        replacement.mut_z().xor_with(old.z());
        replacement.add_phase((4U - local.phase() + mapped.phase()) & 3U);
        for (uint32_t local_q = 0; local_q < targets.size(); ++local_q) {
            replacement.set_pauli(targets[local_q], mapped.x().bit_get(local_q),
                                  mapped.z().bit_get(local_q));
        }
        set_row(row_i, replacement.view());
    }
}

void Tableau::prepend_local(const Tableau& gate, std::span<const uint32_t> targets) {
    if (gate.num_qubits() != targets.size()) {
        throw std::invalid_argument("Gate tableau width does not match target count");
    }
    for (uint32_t target : targets) {
        if (target >= num_qubits_) {
            throw std::invalid_argument("Gate target is outside the tableau");
        }
    }

    std::vector<PauliString> replacements;
    replacements.reserve(2 * targets.size());
    for (bool z_generator : {false, true}) {
        for (uint32_t local_q = 0; local_q < targets.size(); ++local_q) {
            const PauliStringView gate_row =
                z_generator ? gate.z_output(local_q) : gate.x_output(local_q);
            PauliString scattered(num_qubits_);
            scattered.set_phase(gate_row.phase());
            for (uint32_t k = 0; k < targets.size(); ++k) {
                scattered.set_pauli(targets[k], gate_row.x().bit_get(k), gate_row.z().bit_get(k));
            }
            replacements.push_back(apply(scattered.view()));
        }
    }
    size_t replacement = 0;
    for (bool z_generator : {false, true}) {
        for (uint32_t target : targets) {
            set_row(row_index(z_generator, target, num_qubits_),
                    replacements[replacement++].view());
        }
    }
}

void Tableau::append_named_gate(GateType gate, std::span<const uint32_t> targets) {
    append_local(from_named_gate(gate), targets);
}

void Tableau::prepend_named_gate(GateType gate, std::span<const uint32_t> targets) {
    prepend_local(from_named_gate(gate), targets);
}

void Tableau::prepend_pauli(PauliStringView axis) {
    if (axis.num_qubits() != num_qubits_ || !axis.is_hermitian()) {
        throw std::invalid_argument("Pauli axis does not match the tableau");
    }
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (axis.z().bit_get(q)) {
            phases_[row_index(false, q, num_qubits_)] ^= 2U;
        }
        if (axis.x().bit_get(q)) {
            phases_[row_index(true, q, num_qubits_)] ^= 2U;
        }
    }
}

void Tableau::prepend_pauli_rotation(PauliStringView axis, bool dagger) {
    if (axis.num_qubits() != num_qubits_ || !axis.is_hermitian()) {
        throw std::invalid_argument("Pauli rotation axis does not match the tableau");
    }
    const PauliString mapped_axis = apply(axis);
    Tableau identity(num_qubits_);
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        for (bool z_generator : {false, true}) {
            const uint32_t index = row_index(z_generator, q, num_qubits_);
            if (axis.commutes(identity.row(index))) {
                continue;
            }
            PauliString mapped = mapped_axis;
            mapped.right_multiply(row(index));
            mapped.add_phase(dagger ? 1 : 3);
            set_row(index, mapped.view());
        }
    }
}

bool Tableau::operator==(const Tableau& other) const {
    return num_qubits_ == other.num_qubits_ && x_rows_ == other.x_rows_ &&
           z_rows_ == other.z_rows_ && phases_ == other.phases_;
}

}  // namespace clifft
