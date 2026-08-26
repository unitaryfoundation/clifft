#pragma once

// Phase-aware stabilizer states in CH form.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

// Represents |psi> = omega U_C U_H |s> using the CH form of
// Bravyi et al., arXiv:1808.00128. Unlike a stabilizer tableau, this
// representation retains the state's global phase. It is used only by
// phase-sensitive compilation; ordinary tracing and execution stay unchanged.
class StabilizerChForm {
  public:
    explicit StabilizerChForm(uint32_t num_qubits);

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }

    void apply_h(uint32_t qubit);
    void apply_s(uint32_t qubit);
    void apply_s_dag(uint32_t qubit);
    void apply_x(uint32_t qubit);
    void apply_y(uint32_t qubit);
    void apply_z(uint32_t qubit);
    void apply_cx(uint32_t control, uint32_t target);
    void apply_cz(uint32_t q1, uint32_t q2);
    void apply_swap(uint32_t q1, uint32_t q2);
    void apply_global_phase(std::complex<double> phase);

    // Returns <basis|psi>. The mask uses the same little-endian qubit-bit
    // convention as the rest of Clifft and must contain ceil(n / 64) words.
    [[nodiscard]] std::complex<double> amplitude(std::span<const uint64_t> basis) const;

  private:
    class BinaryMatrix {
      public:
        explicit BinaryMatrix(uint32_t size, bool identity = false);

        [[nodiscard]] bool get(uint32_t row, uint32_t col) const;
        void xor_bit(uint32_t row, uint32_t col, bool value);
        void xor_rows(uint32_t dst, uint32_t src);
        [[nodiscard]] std::span<const uint64_t> row(uint32_t index) const;
        [[nodiscard]] std::span<uint64_t> mut_row(uint32_t index);

      private:
        uint32_t size_ = 0;
        size_t words_per_row_ = 0;
        std::vector<uint64_t> words_;
    };

    using Bits = std::vector<uint64_t>;

    [[nodiscard]] bool bit(const Bits& bits, uint32_t index) const;
    void set_bit(Bits& bits, uint32_t index, bool value) const;
    void xor_bit(Bits& bits, uint32_t index) const;
    [[nodiscard]] bool bits_equal(const Bits& lhs, const Bits& rhs) const;
    [[nodiscard]] bool parity_and(std::span<const uint64_t> a,
                                  std::span<const uint64_t> b) const;
    [[nodiscard]] bool parity_and(const Bits& a, const Bits& b) const;
    [[nodiscard]] bool parity_and3(std::span<const uint64_t> a, const Bits& b,
                                   const Bits& c) const;
    [[nodiscard]] uint32_t popcount(const Bits& bits) const;

    void s_right(uint32_t qubit);
    void cz_right(uint32_t q1, uint32_t q2);
    void cnot_right(uint32_t control, uint32_t target);
    void update_sum(Bits t, Bits u, uint8_t delta, bool alpha);

    struct HDecomposition {
        std::complex<double> omega;
        bool apply_s = false;
        bool apply_h = false;
        bool basis = false;
    };
    [[nodiscard]] static HDecomposition decompose_h_sum(bool has_h, bool y, bool z,
                                                         uint8_t delta);

    uint32_t num_qubits_ = 0;
    size_t words_ = 0;
    BinaryMatrix g_;
    BinaryMatrix f_;
    BinaryMatrix m_;
    std::vector<uint8_t> gamma_;
    Bits h_;
    Bits basis_;
    std::complex<double> omega_{1.0, 0.0};
};

}  // namespace clifft
