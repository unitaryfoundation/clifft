#include "clifft/noncomp/classifier.h"

#include "clifft/noncomp/numeric.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace clifft {

MeasurementClassifier MeasurementClassifier::from_matrix(
    const std::vector<std::vector<double>>& matrix) {
    const size_t s_n = matrix.size();
    if (s_n != 2 && s_n != 3) {
        throw std::invalid_argument(
            "MeasurementClassifier::from_matrix: a classifier has two record symbols and an "
            "optional third herald symbol; got " +
            std::to_string(s_n) + " rows");
    }

    // Single pass: validate row width and entry bounds while copying
    // into the flat row-major buffer.
    std::vector<double> flat(s_n * kNumLevels);
    for (size_t s = 0; s < s_n; ++s) {
        if (matrix[s].size() != kNumLevels) {
            throw std::invalid_argument("MeasurementClassifier::from_matrix: row " +
                                        std::to_string(s) + " has " +
                                        std::to_string(matrix[s].size()) + " columns; expected " +
                                        std::to_string(kNumLevels) + " (one per level)");
        }
        for (size_t l = 0; l < kNumLevels; ++l) {
            const double v = matrix[s][l];
            // is_finite_robust runs first because -ffast-math folds
            // std::isfinite() / NaN-aware comparisons away.
            if (!is_finite_robust(v) || v < 0.0 || v > 1.0) {
                throw std::invalid_argument("MeasurementClassifier::from_matrix: entry (" +
                                            std::to_string(s) + ", " + std::to_string(l) +
                                            ") = " + std::to_string(v) +
                                            " is not finite or is out of [0, 1]");
            }
            flat[s * kNumLevels + l] = v;
        }
    }

    for (const Level level : kAllLevels) {
        const size_t l = static_cast<size_t>(level);
        double sum = 0.0;
        for (size_t s = 0; s < s_n; ++s) {
            sum += flat[s * kNumLevels + l];
        }
        // A measurement always produces a symbol: classifier reject
        // columns are not supported, so the column must sum to 1.
        if (std::abs(sum - 1.0) > kProbTolerance) {
            throw std::invalid_argument(
                "MeasurementClassifier::from_matrix: classifier reject columns are not "
                "supported; the column for level '" +
                std::string(level_name(level)) + "' sums to " + std::to_string(sum) +
                " (must sum to 1)");
        }
        if (category(level) == LevelCategory::Computational && s_n == 3) {
            const double herald_mass = flat[kHeraldSymbol * kNumLevels + l];
            if (herald_mass > kProbTolerance) {
                throw std::invalid_argument(
                    "MeasurementClassifier::from_matrix: a computational column must place all "
                    "its probability on the record symbols 0 and 1; the column for level '" +
                    std::string(level_name(level)) + "' puts " + std::to_string(herald_mass) +
                    " on the herald symbol");
            }
        }
    }

    return MeasurementClassifier(s_n, std::move(flat));
}

double MeasurementClassifier::prob(uint8_t symbol_idx, Level level) const {
    if (symbol_idx >= num_symbols_) {
        throw std::invalid_argument("MeasurementClassifier::prob: symbol index " +
                                    std::to_string(symbol_idx) + " out of range (num_symbols " +
                                    std::to_string(num_symbols_) + ")");
    }
    return matrix_flat_[static_cast<size_t>(symbol_idx) * kNumLevels + static_cast<size_t>(level)];
}

}  // namespace clifft
