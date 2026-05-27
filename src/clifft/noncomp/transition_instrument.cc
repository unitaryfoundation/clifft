#include "clifft/noncomp/transition_instrument.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

namespace {

// Tolerance for entry / column-sum bounds and for the
// source-independence equality check. Generous compared to typical
// floating-point error in user-supplied matrices; we want to admit
// hand-written tables that sum to 1.0 - epsilon.
constexpr double kProbTolerance = 1e-12;

bool columns_equal(const std::vector<std::vector<double>>& matrix, uint8_t j1, uint8_t j2) {
    for (size_t to = 0; to < matrix.size(); ++to) {
        if (std::abs(matrix[to][j1] - matrix[to][j2]) > kProbTolerance) {
            return false;
        }
    }
    return true;
}

}  // namespace

TransitionInstrument TransitionInstrument::from_matrix(std::vector<std::vector<double>> matrix,
                                                       const LevelSet& levels) {
    const size_t n = levels.size();

    if (matrix.size() != n) {
        throw std::invalid_argument("TransitionInstrument::from_matrix: matrix has " +
                                    std::to_string(matrix.size()) + " rows; expected " +
                                    std::to_string(n) + " (one per level)");
    }
    for (size_t to = 0; to < n; ++to) {
        if (matrix[to].size() != n) {
            throw std::invalid_argument(
                "TransitionInstrument::from_matrix: row " + std::to_string(to) + " has " +
                std::to_string(matrix[to].size()) + " columns; expected " + std::to_string(n));
        }
        for (size_t from = 0; from < n; ++from) {
            const double v = matrix[to][from];
            if (v < 0.0 - kProbTolerance || v > 1.0 + kProbTolerance) {
                throw std::invalid_argument("TransitionInstrument::from_matrix: entry (" +
                                            std::to_string(to) + ", " + std::to_string(from) +
                                            ") = " + std::to_string(v) + " out of [0, 1]");
            }
        }
    }

    std::vector<double> column_sums(n, 0.0);
    for (size_t from = 0; from < n; ++from) {
        double sum = 0.0;
        for (size_t to = 0; to < n; ++to) {
            sum += matrix[to][from];
        }
        if (sum < 0.0 - kProbTolerance || sum > 1.0 + kProbTolerance) {
            throw std::invalid_argument("TransitionInstrument::from_matrix: column " +
                                        std::to_string(from) + " sum = " + std::to_string(sum) +
                                        " out of [0, 1]");
        }
        column_sums[from] = sum;
    }

    // is_source_independent_on_computational: every column whose source
    // level has category Computational must equal the first such column
    // within tolerance. Vacuously true if there are fewer than two
    // Computational levels (LevelSet validation guarantees exactly two:
    // basis_bit == Zero and basis_bit == One).
    bool flag = true;
    const auto levels_span = levels.levels();
    uint8_t first_comp = 0xFF;
    for (size_t from = 0; from < n; ++from) {
        if (levels_span[from].category == LevelCategory::Computational) {
            if (first_comp == 0xFF) {
                first_comp = static_cast<uint8_t>(from);
            } else if (!columns_equal(matrix, first_comp, static_cast<uint8_t>(from))) {
                flag = false;
                break;
            }
        }
    }

    return TransitionInstrument(std::move(matrix), std::move(column_sums), flag);
}

TransitionInstrument::TransitionInstrument(std::vector<std::vector<double>> matrix,
                                           std::vector<double> column_sums,
                                           bool is_source_independent_on_computational)
    : matrix_(std::move(matrix)),
      column_sums_(std::move(column_sums)),
      is_source_independent_on_computational_(is_source_independent_on_computational) {}

double TransitionInstrument::prob(uint8_t to, uint8_t from) const {
    if (to >= matrix_.size() || from >= matrix_.size()) {
        throw std::invalid_argument("TransitionInstrument::prob: index (" + std::to_string(to) +
                                    ", " + std::to_string(from) + ") out of range (num_levels " +
                                    std::to_string(matrix_.size()) + ")");
    }
    return matrix_[to][from];
}

double TransitionInstrument::column_sum(uint8_t from) const {
    if (from >= column_sums_.size()) {
        throw std::invalid_argument("TransitionInstrument::column_sum: index " +
                                    std::to_string(from) + " out of range (num_levels " +
                                    std::to_string(column_sums_.size()) + ")");
    }
    return column_sums_[from];
}

double TransitionInstrument::no_jump_weight(uint8_t from) const {
    return 1.0 - column_sum(from);
}

}  // namespace clifft
