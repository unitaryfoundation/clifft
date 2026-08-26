#include "clifft/tableau/stabilizer_ch_form.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace clifft {
namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;

[[nodiscard]] std::complex<double> i_pow(uint8_t exponent) {
    switch (exponent & 3U) {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {0.0, 1.0};
        case 2:
            return {-1.0, 0.0};
        default:
            return {0.0, -1.0};
    }
}

}  // namespace

StabilizerChForm::BinaryMatrix::BinaryMatrix(uint32_t size, bool identity)
    : size_(size),
      words_per_row_((static_cast<size_t>(size) + 63U) / 64U),
      words_(static_cast<size_t>(size) * words_per_row_, 0) {
    if (identity) {
        for (uint32_t q = 0; q < size_; ++q) {
            words_[static_cast<size_t>(q) * words_per_row_ + q / 64U] |=
                uint64_t{1} << (q % 64U);
        }
    }
}

bool StabilizerChForm::BinaryMatrix::get(uint32_t row, uint32_t col) const {
    assert(row < size_ && col < size_);
    return ((words_[static_cast<size_t>(row) * words_per_row_ + col / 64U] >> (col % 64U)) &
            1U) != 0;
}

void StabilizerChForm::BinaryMatrix::xor_bit(uint32_t row, uint32_t col, bool value) {
    assert(row < size_ && col < size_);
    if (value) {
        words_[static_cast<size_t>(row) * words_per_row_ + col / 64U] ^=
            uint64_t{1} << (col % 64U);
    }
}

void StabilizerChForm::BinaryMatrix::xor_rows(uint32_t dst, uint32_t src) {
    assert(dst < size_ && src < size_);
    auto destination = mut_row(dst);
    const auto source = row(src);
    for (size_t w = 0; w < words_per_row_; ++w) {
        destination[w] ^= source[w];
    }
}

std::span<const uint64_t> StabilizerChForm::BinaryMatrix::row(uint32_t index) const {
    assert(index < size_);
    return std::span<const uint64_t>(words_).subspan(static_cast<size_t>(index) * words_per_row_,
                                                     words_per_row_);
}

std::span<uint64_t> StabilizerChForm::BinaryMatrix::mut_row(uint32_t index) {
    assert(index < size_);
    return std::span<uint64_t>(words_).subspan(static_cast<size_t>(index) * words_per_row_,
                                               words_per_row_);
}

StabilizerChForm::StabilizerChForm(uint32_t num_qubits)
    : num_qubits_(num_qubits),
      words_((static_cast<size_t>(num_qubits) + 63U) / 64U),
      g_(num_qubits, true),
      f_(num_qubits, true),
      m_(num_qubits),
      gamma_(num_qubits, 0),
      h_(words_, 0),
      basis_(words_, 0) {}

bool StabilizerChForm::bit(const Bits& bits, uint32_t index) const {
    assert(index < num_qubits_);
    return ((bits[index / 64U] >> (index % 64U)) & 1U) != 0;
}

void StabilizerChForm::set_bit(Bits& bits, uint32_t index, bool value) const {
    assert(index < num_qubits_);
    const uint64_t mask = uint64_t{1} << (index % 64U);
    if (value) {
        bits[index / 64U] |= mask;
    } else {
        bits[index / 64U] &= ~mask;
    }
}

void StabilizerChForm::xor_bit(Bits& bits, uint32_t index) const {
    assert(index < num_qubits_);
    bits[index / 64U] ^= uint64_t{1} << (index % 64U);
}

bool StabilizerChForm::bits_equal(const Bits& lhs, const Bits& rhs) const {
    return lhs == rhs;
}

bool StabilizerChForm::parity_and(std::span<const uint64_t> a,
                                  std::span<const uint64_t> b) const {
    assert(a.size() == words_ && b.size() == words_);
    bool parity = false;
    for (size_t w = 0; w < words_; ++w) {
        parity ^= (std::popcount(a[w] & b[w]) & 1U) != 0;
    }
    return parity;
}

bool StabilizerChForm::parity_and(const Bits& a, const Bits& b) const {
    return parity_and(std::span<const uint64_t>(a), std::span<const uint64_t>(b));
}

bool StabilizerChForm::parity_and3(std::span<const uint64_t> a, const Bits& b,
                                   const Bits& c) const {
    assert(a.size() == words_ && b.size() == words_ && c.size() == words_);
    bool parity = false;
    for (size_t w = 0; w < words_; ++w) {
        parity ^= (std::popcount(a[w] & b[w] & c[w]) & 1U) != 0;
    }
    return parity;
}

uint32_t StabilizerChForm::popcount(const Bits& bits) const {
    uint32_t count = 0;
    for (uint64_t word : bits) {
        count += std::popcount(word);
    }
    return count;
}

void StabilizerChForm::s_right(uint32_t qubit) {
    for (uint32_t row = 0; row < num_qubits_; ++row) {
        const bool f = f_.get(row, qubit);
        m_.xor_bit(row, qubit, f);
        gamma_[row] = static_cast<uint8_t>((gamma_[row] + (f ? 3U : 0U)) & 3U);
    }
}

void StabilizerChForm::cz_right(uint32_t q1, uint32_t q2) {
    for (uint32_t row = 0; row < num_qubits_; ++row) {
        const bool f1 = f_.get(row, q1);
        const bool f2 = f_.get(row, q2);
        m_.xor_bit(row, q1, f2);
        m_.xor_bit(row, q2, f1);
        if (f1 && f2) {
            gamma_[row] = static_cast<uint8_t>((gamma_[row] + 2U) & 3U);
        }
    }
}

void StabilizerChForm::cnot_right(uint32_t control, uint32_t target) {
    for (uint32_t row = 0; row < num_qubits_; ++row) {
        g_.xor_bit(row, control, g_.get(row, target));
        f_.xor_bit(row, target, f_.get(row, control));
        m_.xor_bit(row, control, m_.get(row, target));
    }
}

StabilizerChForm::HDecomposition StabilizerChForm::decompose_h_sum(bool has_h, bool y, bool z,
                                                                   uint8_t delta) {
    assert(y != z);
    HDecomposition result;
    if (!has_h) {
        result.omega = i_pow(static_cast<uint8_t>(delta * static_cast<uint8_t>(y)));
        const uint8_t adjusted =
            static_cast<uint8_t>((y ? (4U - delta) : delta) & 3U);
        result.basis = (adjusted >> 1U) != 0;
        result.apply_s = (adjusted & 1U) != 0;
        result.apply_h = true;
        return result;
    }

    if ((delta & 1U) == 0) {
        result.basis = (delta >> 1U) != 0;
        result.omega = (result.basis && y) ? std::complex<double>{-1.0, 0.0}
                                           : std::complex<double>{1.0, 0.0};
        return result;
    }

    result.omega = kInvSqrt2 * (std::complex<double>{1.0, 0.0} + i_pow(delta));
    result.apply_s = true;
    result.apply_h = true;
    result.basis = !(((delta >> 1U) != 0) ^ y);
    return result;
}

void StabilizerChForm::update_sum(Bits t, Bits u, uint8_t delta, bool alpha) {
    if (bits_equal(t, u)) {
        basis_ = std::move(t);
        omega_ *= kInvSqrt2 * (alpha ? -1.0 : 1.0) *
                  (std::complex<double>{1.0, 0.0} + i_pow(delta));
        return;
    }

    std::vector<uint32_t> set_zero;
    std::vector<uint32_t> set_one;
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (bit(t, q) == bit(u, q)) {
            continue;
        }
        (bit(h_, q) ? set_one : set_zero).push_back(q);
    }

    uint32_t pivot = 0;
    if (!set_zero.empty()) {
        pivot = set_zero.front();
        for (uint32_t q : set_zero) {
            if (q != pivot) {
                cnot_right(pivot, q);
            }
        }
        for (uint32_t q : set_one) {
            cz_right(pivot, q);
        }
    } else {
        assert(!set_one.empty());
        pivot = set_one.front();
        for (uint32_t q : set_one) {
            if (q != pivot) {
                cnot_right(q, pivot);
            }
        }
    }

    Bits y;
    Bits z;
    if (bit(t, pivot)) {
        y = u;
        xor_bit(y, pivot);
        z = std::move(u);
    } else {
        y = std::move(t);
        z = y;
        xor_bit(z, pivot);
    }

    const HDecomposition decomposition =
        decompose_h_sum(bit(h_, pivot), bit(y, pivot), bit(z, pivot), delta);
    basis_ = std::move(y);
    set_bit(basis_, pivot, decomposition.basis);
    omega_ *= (alpha ? -1.0 : 1.0) * decomposition.omega;

    if (decomposition.apply_s) {
        s_right(pivot);
    }
    set_bit(h_, pivot, decomposition.apply_h);
}

void StabilizerChForm::apply_h(uint32_t qubit) {
    if (qubit >= num_qubits_) {
        throw std::out_of_range("CH-form Hadamard target is outside the state");
    }

    Bits t = basis_;
    Bits u = basis_;
    const auto g = g_.row(qubit);
    const auto f = f_.row(qubit);
    const auto m = m_.row(qubit);
    for (size_t w = 0; w < words_; ++w) {
        t[w] ^= g[w] & h_[w];
        u[w] ^= (f[w] & ~h_[w]) ^ (m[w] & h_[w]);
    }

    bool alpha = false;
    bool beta = false;
    for (size_t w = 0; w < words_; ++w) {
        alpha ^= (std::popcount(g[w] & ~h_[w] & basis_[w]) & 1U) != 0;
        beta ^= (std::popcount(m[w] & ~h_[w] & basis_[w]) & 1U) != 0;
        beta ^= (std::popcount(f[w] & h_[w] & m[w]) & 1U) != 0;
        beta ^= (std::popcount(f[w] & h_[w] & basis_[w]) & 1U) != 0;
    }
    const uint8_t delta =
        static_cast<uint8_t>((gamma_[qubit] + 2U * (static_cast<uint8_t>(alpha) +
                                                    static_cast<uint8_t>(beta))) &
                             3U);
    update_sum(std::move(t), std::move(u), delta, alpha);
}

void StabilizerChForm::apply_s(uint32_t qubit) {
    if (qubit >= num_qubits_) {
        throw std::out_of_range("CH-form S target is outside the state");
    }
    auto m = m_.mut_row(qubit);
    const auto g = g_.row(qubit);
    for (size_t w = 0; w < words_; ++w) {
        m[w] ^= g[w];
    }
    gamma_[qubit] = static_cast<uint8_t>((gamma_[qubit] + 3U) & 3U);
}

void StabilizerChForm::apply_s_dag(uint32_t qubit) {
    apply_s(qubit);
    apply_s(qubit);
    apply_s(qubit);
}

void StabilizerChForm::apply_x(uint32_t qubit) {
    apply_h(qubit);
    apply_z(qubit);
    apply_h(qubit);
}

void StabilizerChForm::apply_y(uint32_t qubit) {
    apply_z(qubit);
    apply_h(qubit);
    apply_z(qubit);
    apply_h(qubit);
    omega_ *= std::complex<double>{0.0, 1.0};
}

void StabilizerChForm::apply_z(uint32_t qubit) {
    apply_s(qubit);
    apply_s(qubit);
}

void StabilizerChForm::apply_cx(uint32_t control, uint32_t target) {
    if (control >= num_qubits_ || target >= num_qubits_ || control == target) {
        throw std::invalid_argument("CH-form CX requires two distinct in-range qubits");
    }
    const bool parity = parity_and(m_.row(control), f_.row(target));
    gamma_[control] = static_cast<uint8_t>(
        (gamma_[control] + gamma_[target] + (parity ? 2U : 0U)) & 3U);
    g_.xor_rows(target, control);
    f_.xor_rows(control, target);
    m_.xor_rows(control, target);
}

void StabilizerChForm::apply_cz(uint32_t q1, uint32_t q2) {
    if (q1 >= num_qubits_ || q2 >= num_qubits_ || q1 == q2) {
        throw std::invalid_argument("CH-form CZ requires two distinct in-range qubits");
    }
    auto m1 = m_.mut_row(q1);
    const auto g2 = g_.row(q2);
    for (size_t w = 0; w < words_; ++w) {
        m1[w] ^= g2[w];
    }
    auto m2 = m_.mut_row(q2);
    const auto g1 = g_.row(q1);
    for (size_t w = 0; w < words_; ++w) {
        m2[w] ^= g1[w];
    }
}

void StabilizerChForm::apply_swap(uint32_t q1, uint32_t q2) {
    if (q1 == q2) {
        return;
    }
    apply_cx(q1, q2);
    apply_cx(q2, q1);
    apply_cx(q1, q2);
}

void StabilizerChForm::apply_global_phase(std::complex<double> phase) {
    omega_ *= phase;
}

std::complex<double> StabilizerChForm::amplitude(std::span<const uint64_t> basis) const {
    if (basis.size() != words_) {
        throw std::invalid_argument("CH-form basis mask width does not match the state");
    }
    if (!basis.empty() && num_qubits_ % 64U != 0) {
        const uint64_t valid = (uint64_t{1} << (num_qubits_ % 64U)) - 1U;
        if ((basis.back() & ~valid) != 0) {
            throw std::invalid_argument("CH-form basis mask sets unused high bits");
        }
    }

    uint8_t phase = 0;
    Bits u(words_, 0);
    for (uint32_t p = 0; p < num_qubits_; ++p) {
        if (((basis[p / 64U] >> (p % 64U)) & 1U) == 0) {
            continue;
        }
        phase = static_cast<uint8_t>((phase + gamma_[p]) & 3U);
        const auto f = f_.row(p);
        for (size_t w = 0; w < words_; ++w) {
            u[w] ^= f[w];
        }
        if (parity_and(m_.row(p), u)) {
            phase = static_cast<uint8_t>((phase + 2U) & 3U);
        }
    }

    for (size_t w = 0; w < words_; ++w) {
        if (((~h_[w]) & (u[w] ^ basis_[w])) != 0) {
            return {0.0, 0.0};
        }
    }

    const bool sign = parity_and3(std::span<const uint64_t>(h_), u, basis_);
    const double magnitude = std::exp2(-0.5 * static_cast<double>(popcount(h_)));
    return omega_ * magnitude * i_pow(phase) * (sign ? -1.0 : 1.0);
}

}  // namespace clifft
