#include "clifft/noncomp/transition_instrument.h"

#include "clifft/noncomp/numeric.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

TransitionInstrument TransitionInstrument::from_matrix(std::vector<std::vector<double>> matrix,
                                                       const LevelSet& levels) {
    const size_t n = levels.size();

    if (matrix.size() != n) {
        throw std::invalid_argument("TransitionInstrument::from_matrix: matrix has " +
                                    std::to_string(matrix.size()) + " rows; expected " +
                                    std::to_string(n) + " (one per level)");
    }

    // Single pass: validate row width and entry bounds while copying
    // into the flat row-major buffer.
    std::vector<double> flat(n * n);
    for (size_t to = 0; to < n; ++to) {
        if (matrix[to].size() != n) {
            throw std::invalid_argument(
                "TransitionInstrument::from_matrix: row " + std::to_string(to) + " has " +
                std::to_string(matrix[to].size()) + " columns; expected " + std::to_string(n));
        }
        for (size_t from = 0; from < n; ++from) {
            const double v = matrix[to][from];
            // Raw user entries must be finite and lie strictly in [0, 1]:
            // tolerance applies only to derived column sums below.
            // is_finite_robust runs first because -ffast-math folds
            // std::isfinite() / NaN-aware comparisons away.
            if (!is_finite_robust(v) || v < 0.0 || v > 1.0) {
                throw std::invalid_argument("TransitionInstrument::from_matrix: entry (" +
                                            std::to_string(to) + ", " + std::to_string(from) +
                                            ") = " + std::to_string(v) +
                                            " is not finite or is out of [0, 1]");
            }
            flat[to * n + from] = v;
        }
    }

    std::vector<double> column_sums(n, 0.0);
    for (size_t from = 0; from < n; ++from) {
        double sum = 0.0;
        for (size_t to = 0; to < n; ++to) {
            sum += flat[to * n + from];
        }
        // Reject sums that exceed 1 by more than floating drift.
        // Within tolerance, clamp so prob() and no_jump_weight()
        // never report values outside [0, 1].
        if (sum > 1.0 + kProbTolerance) {
            throw std::invalid_argument("TransitionInstrument::from_matrix: column " +
                                        std::to_string(from) + " sum = " + std::to_string(sum) +
                                        " exceeds 1");
        }
        column_sums[from] = sum > 1.0 ? 1.0 : sum;
    }

    return TransitionInstrument(std::move(flat), std::move(column_sums), levels.fingerprint());
}

TransitionInstrument::TransitionInstrument(std::vector<double> matrix_flat,
                                           std::vector<double> column_sums,
                                           uint64_t level_fingerprint)
    : matrix_flat_(std::move(matrix_flat)),
      column_sums_(std::move(column_sums)),
      level_fingerprint_(level_fingerprint) {}

double TransitionInstrument::prob(uint8_t to, uint8_t from) const {
    const size_t n = column_sums_.size();
    if (to >= n || from >= n) {
        throw std::invalid_argument("TransitionInstrument::prob: index (" + std::to_string(to) +
                                    ", " + std::to_string(from) + ") out of range (num_levels " +
                                    std::to_string(n) + ")");
    }
    return matrix_flat_[static_cast<size_t>(to) * n + static_cast<size_t>(from)];
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
