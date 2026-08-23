#pragma once

// Native runtime-width Clifford tableau.

#include "clifft/circuit/gate_data.h"
#include "clifft/tableau/pauli_string.h"

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

namespace clifft {

class Tableau {
  public:
    explicit Tableau(uint32_t num_qubits = 0);

    [[nodiscard]] static Tableau from_named_gate(GateType gate);
    [[nodiscard]] static Tableau from_pauli_rotation(PauliStringView axis, bool dagger);

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    [[nodiscard]] PauliStringView x_output(uint32_t qubit) const;
    [[nodiscard]] PauliStringView z_output(uint32_t qubit) const;
    [[nodiscard]] PauliString y_output(uint32_t qubit) const;
    [[nodiscard]] PauliString apply(PauliStringView input) const;
    [[nodiscard]] Tableau then(const Tableau& next) const;
    [[nodiscard]] Tableau inverse() const;

    void append_local(const Tableau& gate, std::span<const uint32_t> targets);
    void prepend_local(const Tableau& gate, std::span<const uint32_t> targets);
    void append_named_gate(GateType gate, std::span<const uint32_t> targets);
    void prepend_named_gate(GateType gate, std::span<const uint32_t> targets);
    void prepend_pauli(PauliStringView axis);
    void prepend_pauli_rotation(PauliStringView axis, bool dagger);

    [[nodiscard]] bool operator==(const Tableau& other) const;

  private:
    [[nodiscard]] static Tableau from_rows(std::initializer_list<std::string_view> rows);
    [[nodiscard]] PauliStringView row(uint32_t index) const;
    void set_row(uint32_t index, PauliStringView value);

    uint32_t num_qubits_;
    uint32_t num_words_;
    std::vector<uint64_t> x_rows_;
    std::vector<uint64_t> z_rows_;
    std::vector<uint8_t> phases_;
};

}  // namespace clifft
