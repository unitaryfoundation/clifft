#include "clifft/tableau/pauli_string.h"

#include <bit>
#include <cassert>
#include <stdexcept>

namespace clifft {

namespace {

uint8_t y_phase(MaskView x, MaskView z) {
    uint32_t count = 0;
    assert(x.num_words() == z.num_words());
    for (uint32_t w = 0; w < x.num_words(); ++w) {
        count += std::popcount(x.words[w] & z.words[w]);
    }
    return static_cast<uint8_t>(count & 3U);
}

bool symplectic_parity(PauliStringView first, PauliStringView second) {
    assert(first.num_qubits() == second.num_qubits());
    bool parity = false;
    for (uint32_t w = 0; w < first.x().num_words(); ++w) {
        parity ^= (std::popcount((first.x().words[w] & second.z().words[w]) ^
                                 (first.z().words[w] & second.x().words[w])) &
                   1U) != 0;
    }
    return parity;
}

}  // namespace

bool PauliStringView::is_hermitian() const {
    const uint8_t delta = (phase_ - y_phase(x_, z_)) & 3U;
    return delta == 0 || delta == 2;
}

bool PauliStringView::sign() const {
    const uint8_t delta = (phase_ - y_phase(x_, z_)) & 3U;
    assert((delta == 0 || delta == 2) && "PauliStringView::sign requires a Hermitian Pauli");
    return delta == 2;
}

bool PauliStringView::commutes(PauliStringView other) const {
    return !symplectic_parity(*this, other);
}

PauliString::PauliString(uint32_t num_qubits)
    : num_qubits_(num_qubits), x_((num_qubits + 63) / 64, 0), z_(x_.size(), 0) {}

PauliString::PauliString(PauliStringView source) : PauliString(source.num_qubits()) {
    mut_x().xor_with(source.x());
    mut_z().xor_with(source.z());
    set_phase(source.phase());
}

PauliString PauliString::from_text(std::string_view text) {
    if (text.empty() || (text.front() != '+' && text.front() != '-')) {
        throw std::invalid_argument("Pauli text must start with a sign");
    }

    PauliString result(static_cast<uint32_t>(text.size() - 1));
    for (uint32_t q = 0; q < result.num_qubits(); ++q) {
        switch (text[q + 1]) {
            case '_':
            case 'I':
                break;
            case 'X':
                result.set_pauli(q, true, false);
                break;
            case 'Y':
                result.set_pauli(q, true, true);
                break;
            case 'Z':
                result.set_pauli(q, false, true);
                break;
            default:
                throw std::invalid_argument("Invalid Pauli character");
        }
    }
    result.set_sign(text.front() == '-');
    return result;
}

void PauliString::set_pauli(uint32_t qubit, bool x_value, bool z_value) {
    assert(qubit < num_qubits_);
    mut_x().bit_set(qubit, x_value);
    mut_z().bit_set(qubit, z_value);
}

void PauliString::set_sign(bool sign_value) {
    phase_ = (y_phase(x(), z()) + (sign_value ? 2U : 0U)) & 3U;
}

void PauliString::right_multiply(PauliStringView other) {
    assert(num_qubits_ == other.num_qubits());
    bool crossing_parity = false;
    for (uint32_t w = 0; w < x().num_words(); ++w) {
        crossing_parity ^= (std::popcount(z_[w] & other.x().words[w]) & 1U) != 0;
    }
    phase_ = (phase_ + other.phase() + (crossing_parity ? 2U : 0U)) & 3U;
    mut_x().xor_with(other.x());
    mut_z().xor_with(other.z());
    clear_padding();
}

bool PauliString::operator==(const PauliString& other) const {
    return num_qubits_ == other.num_qubits_ && phase_ == other.phase_ && x_ == other.x_ &&
           z_ == other.z_;
}

void PauliString::clear_padding() {
    if (num_qubits_ == 0 || num_qubits_ % 64 == 0) {
        return;
    }
    const uint64_t valid = (uint64_t{1} << (num_qubits_ % 64)) - 1;
    x_.back() &= valid;
    z_.back() &= valid;
}

}  // namespace clifft
