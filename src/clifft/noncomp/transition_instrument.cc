#include "clifft/noncomp/transition_instrument.h"

#include "clifft/noncomp/numeric.h"

#include <stdexcept>
#include <string>

namespace clifft {

TransitionInstrument TransitionInstrument::from_matrix(
    const std::vector<std::vector<double>>& matrix) {
    if (matrix.size() != kNumLevels) {
        throw std::invalid_argument("TransitionInstrument::from_matrix: matrix has " +
                                    std::to_string(matrix.size()) + " rows; expected " +
                                    std::to_string(kNumLevels) + " (one per level)");
    }

    // Single pass: validate row width and entry bounds while copying
    // into the flat row-major buffer.
    std::array<double, kNumLevels * kNumLevels> flat{};
    for (size_t to = 0; to < kNumLevels; ++to) {
        if (matrix[to].size() != kNumLevels) {
            throw std::invalid_argument("TransitionInstrument::from_matrix: row " +
                                        std::to_string(to) + " has " +
                                        std::to_string(matrix[to].size()) + " columns; expected " +
                                        std::to_string(kNumLevels));
        }
        for (size_t from = 0; from < kNumLevels; ++from) {
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
            flat[to * kNumLevels + from] = v;
        }
    }

    std::array<double, kNumLevels> column_sums{};
    for (size_t from = 0; from < kNumLevels; ++from) {
        double sum = 0.0;
        for (size_t to = 0; to < kNumLevels; ++to) {
            sum += flat[to * kNumLevels + from];
        }
        // Reject sums that exceed 1 by more than floating drift.
        // Within tolerance, clamp so prob() derived weights never
        // report values outside [0, 1].
        if (sum > 1.0 + kProbTolerance) {
            throw std::invalid_argument("TransitionInstrument::from_matrix: column " +
                                        std::to_string(from) + " sum = " + std::to_string(sum) +
                                        " exceeds 1");
        }
        column_sums[from] = sum > 1.0 ? 1.0 : sum;
    }

    return TransitionInstrument(flat, column_sums);
}

}  // namespace clifft
