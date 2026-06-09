#pragma once

#include "clifft/optimizer/pauli_axis.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

struct Gf2Matrix {
    uint32_t n = 0;
    uint32_t num_words = 0;
    std::vector<uint64_t> cols;

    [[nodiscard]] size_t num_cols() const { return num_words == 0 ? 0 : cols.size() / num_words; }

    [[nodiscard]] const uint64_t* col(size_t j) const { return &cols[j * num_words]; }

    uint64_t* col_mut(size_t j) { return &cols[j * num_words]; }

    void resize(uint32_t n_qubits, size_t m);
    void append_col(const uint64_t* words);
};

size_t properize(Gf2Matrix& mat, std::vector<int>& coeffs_mod8,
                 std::vector<PauliAxis>* axes = nullptr);

bool todd_optimize(Gf2Matrix& mat, std::vector<int>& coeffs_mod8, uint32_t max_n, size_t max_m,
                   size_t max_rounds, std::vector<PauliAxis>* axes = nullptr);

}  // namespace clifft
