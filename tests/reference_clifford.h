#pragma once

// Deliberately scalar Clifford algebra used only as a test reference.

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace clifft::test {

class ReferencePauli {
  public:
    explicit ReferencePauli(uint32_t num_qubits) : xs_(num_qubits, false), zs_(num_qubits, false) {}

    [[nodiscard]] uint32_t num_qubits() const { return static_cast<uint32_t>(xs_.size()); }
    [[nodiscard]] bool x(uint32_t qubit) const { return xs_[qubit]; }
    [[nodiscard]] bool z(uint32_t qubit) const { return zs_[qubit]; }
    [[nodiscard]] uint8_t phase() const { return phase_; }

    void set_pauli(uint32_t qubit, bool x, bool z) {
        xs_[qubit] = x;
        zs_[qubit] = z;
    }

    void set_sign(bool sign) {
        uint32_t y_count = 0;
        for (uint32_t q = 0; q < num_qubits(); ++q) {
            y_count += x(q) && z(q);
        }
        phase_ = static_cast<uint8_t>((y_count + (sign ? 2U : 0U)) & 3U);
    }

    void add_phase(uint8_t delta) { phase_ = static_cast<uint8_t>((phase_ + delta) & 3U); }

    [[nodiscard]] bool is_hermitian() const {
        uint32_t y_count = 0;
        for (uint32_t q = 0; q < num_qubits(); ++q) {
            y_count += x(q) && z(q);
        }
        return ((phase_ - y_count) & 1U) == 0;
    }

    [[nodiscard]] bool sign() const {
        assert(is_hermitian());
        uint32_t y_count = 0;
        for (uint32_t q = 0; q < num_qubits(); ++q) {
            y_count += x(q) && z(q);
        }
        return ((phase_ - y_count) & 3U) == 2U;
    }

    [[nodiscard]] bool commutes(const ReferencePauli& other) const {
        assert(num_qubits() == other.num_qubits());
        bool parity = false;
        for (uint32_t q = 0; q < num_qubits(); ++q) {
            parity ^= (x(q) && other.z(q)) ^ (z(q) && other.x(q));
        }
        return !parity;
    }

    void right_multiply(const ReferencePauli& other) {
        assert(num_qubits() == other.num_qubits());
        bool crossing_parity = false;
        for (uint32_t q = 0; q < num_qubits(); ++q) {
            crossing_parity ^= z(q) && other.x(q);
        }
        phase_ = static_cast<uint8_t>((phase_ + other.phase_ + (crossing_parity ? 2U : 0U)) & 3U);
        for (uint32_t q = 0; q < num_qubits(); ++q) {
            xs_[q] = xs_[q] != other.xs_[q];
            zs_[q] = zs_[q] != other.zs_[q];
        }
    }

    [[nodiscard]] static ReferencePauli generator(uint32_t num_qubits, uint32_t qubit, bool z) {
        ReferencePauli result(num_qubits);
        result.set_pauli(qubit, !z, z);
        return result;
    }

  private:
    std::vector<bool> xs_;
    std::vector<bool> zs_;
    uint8_t phase_ = 0;
};

class ReferenceTableau {
  public:
    explicit ReferenceTableau(uint32_t num_qubits) : num_qubits_(num_qubits), x_rows_(), z_rows_() {
        x_rows_.reserve(num_qubits);
        z_rows_.reserve(num_qubits);
        for (uint32_t q = 0; q < num_qubits; ++q) {
            x_rows_.push_back(ReferencePauli::generator(num_qubits, q, false));
            z_rows_.push_back(ReferencePauli::generator(num_qubits, q, true));
        }
    }

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    [[nodiscard]] const ReferencePauli& x_output(uint32_t qubit) const { return x_rows_[qubit]; }
    [[nodiscard]] const ReferencePauli& z_output(uint32_t qubit) const { return z_rows_[qubit]; }

    [[nodiscard]] ReferencePauli apply(const ReferencePauli& input) const {
        assert(input.num_qubits() == num_qubits_);
        ReferencePauli result(num_qubits_);
        result.add_phase(input.phase());
        for (uint32_t q = 0; q < num_qubits_; ++q) {
            if (input.x(q)) {
                result.right_multiply(x_rows_[q]);
            }
            if (input.z(q)) {
                result.right_multiply(z_rows_[q]);
            }
        }
        return result;
    }

    [[nodiscard]] ReferenceTableau then(const ReferenceTableau& next) const {
        assert(num_qubits_ == next.num_qubits_);
        ReferenceTableau result(num_qubits_);
        for (uint32_t q = 0; q < num_qubits_; ++q) {
            result.x_rows_[q] = next.apply(x_rows_[q]);
            result.z_rows_[q] = next.apply(z_rows_[q]);
        }
        return result;
    }

    void append_h(uint32_t qubit) { append_local(h_gate(num_qubits_, qubit)); }
    void append_s(uint32_t qubit) { append_local(s_gate(num_qubits_, qubit, false)); }
    void append_s_dag(uint32_t qubit) { append_local(s_gate(num_qubits_, qubit, true)); }
    void append_cx(uint32_t control, uint32_t target) {
        append_local(cx_gate(num_qubits_, control, target));
    }

    void append_pauli_rotation(const ReferencePauli& axis, bool dagger) {
        *this = then(from_pauli_rotation(axis, dagger));
    }

    void prepend_pauli_rotation(const ReferencePauli& axis, bool dagger) {
        *this = from_pauli_rotation(axis, dagger).then(*this);
    }

    [[nodiscard]] static ReferenceTableau from_pauli_rotation(const ReferencePauli& axis,
                                                              bool dagger) {
        assert(axis.is_hermitian());
        ReferenceTableau result(axis.num_qubits());
        for (uint32_t q = 0; q < axis.num_qubits(); ++q) {
            for (bool z_generator : {false, true}) {
                ReferencePauli generator =
                    ReferencePauli::generator(axis.num_qubits(), q, z_generator);
                if (axis.commutes(generator)) {
                    continue;
                }
                ReferencePauli mapped = axis;
                mapped.right_multiply(generator);
                mapped.add_phase(dagger ? 1U : 3U);
                (z_generator ? result.z_rows_ : result.x_rows_)[q] = std::move(mapped);
            }
        }
        return result;
    }

  private:
    static ReferenceTableau h_gate(uint32_t num_qubits, uint32_t qubit) {
        ReferenceTableau gate(num_qubits);
        gate.x_rows_[qubit] = ReferencePauli::generator(num_qubits, qubit, true);
        gate.z_rows_[qubit] = ReferencePauli::generator(num_qubits, qubit, false);
        return gate;
    }

    static ReferenceTableau s_gate(uint32_t num_qubits, uint32_t qubit, bool dagger) {
        ReferenceTableau gate(num_qubits);
        ReferencePauli mapped_x(num_qubits);
        mapped_x.set_pauli(qubit, true, true);
        mapped_x.set_sign(dagger);
        gate.x_rows_[qubit] = std::move(mapped_x);
        return gate;
    }

    static ReferenceTableau cx_gate(uint32_t num_qubits, uint32_t control, uint32_t target) {
        assert(control != target);
        ReferenceTableau gate(num_qubits);
        gate.x_rows_[control].set_pauli(target, true, false);
        gate.z_rows_[target].set_pauli(control, false, true);
        return gate;
    }

    void append_local(const ReferenceTableau& gate) { *this = then(gate); }

    uint32_t num_qubits_;
    std::vector<ReferencePauli> x_rows_;
    std::vector<ReferencePauli> z_rows_;
};

}  // namespace clifft::test
