#pragma once

// Runtime-width Pauli strings used by the native Clifford tableau.

#include "clifft/util/mask_view.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace clifft {

class PauliStringView {
  public:
    PauliStringView(MaskView x, MaskView z, uint8_t phase, uint32_t num_qubits)
        : x_(x), z_(z), phase_(phase & 3U), num_qubits_(num_qubits) {}

    [[nodiscard]] MaskView x() const { return x_; }
    [[nodiscard]] MaskView z() const { return z_; }
    [[nodiscard]] uint8_t phase() const { return phase_; }
    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    [[nodiscard]] bool sign() const;
    [[nodiscard]] bool is_hermitian() const;
    [[nodiscard]] bool commutes(PauliStringView other) const;

  private:
    MaskView x_;
    MaskView z_;
    uint8_t phase_;
    uint32_t num_qubits_;
};

class PauliString {
  public:
    explicit PauliString(uint32_t num_qubits = 0);
    explicit PauliString(PauliStringView source);

    [[nodiscard]] static PauliString from_text(std::string_view text);

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    [[nodiscard]] MaskView x() const { return MaskView{x_}; }
    [[nodiscard]] MaskView z() const { return MaskView{z_}; }
    [[nodiscard]] MutableMaskView mut_x() { return MutableMaskView{x_}; }
    [[nodiscard]] MutableMaskView mut_z() { return MutableMaskView{z_}; }
    [[nodiscard]] uint8_t phase() const { return phase_; }
    [[nodiscard]] bool sign() const { return view().sign(); }
    [[nodiscard]] bool is_hermitian() const { return view().is_hermitian(); }
    [[nodiscard]] PauliStringView view() const {
        return PauliStringView{x(), z(), phase_, num_qubits_};
    }

    void set_pauli(uint32_t qubit, bool x, bool z);
    void set_phase(uint8_t phase) { phase_ = phase & 3U; }
    void add_phase(uint8_t phase) { phase_ = (phase_ + phase) & 3U; }
    void set_sign(bool sign);
    void negate() { add_phase(2); }
    void right_multiply(PauliStringView other);

    [[nodiscard]] bool operator==(const PauliString& other) const;

  private:
    void clear_padding();

    uint32_t num_qubits_;
    std::vector<uint64_t> x_;
    std::vector<uint64_t> z_;
    uint8_t phase_ = 0;
};

}  // namespace clifft
