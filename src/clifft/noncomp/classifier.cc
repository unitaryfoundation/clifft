#include "clifft/noncomp/classifier.h"

#include "clifft/noncomp/numeric.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace clifft {

MeasurementClassifier MeasurementClassifier::from_matrix(std::vector<std::string> symbols,
                                                         std::vector<std::vector<double>> matrix,
                                                         const LevelSet& levels) {
    if (symbols.empty()) {
        throw std::invalid_argument("MeasurementClassifier::from_matrix: symbols list is empty");
    }
    if (symbols.size() > 256) {
        // The public symbol index APIs use uint8_t. Allowing more than
        // 256 symbols would make indices >= 256 unaddressable through
        // prob() / symbol_label() and silently wrap on conversion.
        throw std::invalid_argument("MeasurementClassifier::from_matrix: symbols list has " +
                                    std::to_string(symbols.size()) +
                                    " entries; max supported is 256");
    }
    {
        std::unordered_set<std::string> seen;
        seen.reserve(symbols.size());
        for (const auto& s : symbols) {
            if (!seen.insert(s).second) {
                throw std::invalid_argument(
                    "MeasurementClassifier::from_matrix: duplicate symbol '" + s + "'");
            }
        }
    }

    const size_t s_n = symbols.size();
    const size_t l_n = levels.size();

    if (matrix.size() != s_n) {
        throw std::invalid_argument("MeasurementClassifier::from_matrix: matrix has " +
                                    std::to_string(matrix.size()) + " rows; expected " +
                                    std::to_string(s_n) + " (one per symbol)");
    }

    // Single pass: validate row width and entry bounds while copying
    // into the flat row-major buffer.
    std::vector<double> flat(s_n * l_n);
    for (size_t s = 0; s < s_n; ++s) {
        if (matrix[s].size() != l_n) {
            throw std::invalid_argument("MeasurementClassifier::from_matrix: row " +
                                        std::to_string(s) + " has " +
                                        std::to_string(matrix[s].size()) + " columns; expected " +
                                        std::to_string(l_n) + " (one per level)");
        }
        for (size_t l = 0; l < l_n; ++l) {
            const double v = matrix[s][l];
            // is_finite_robust runs first because -ffast-math folds
            // std::isfinite() / NaN-aware comparisons away.
            if (!is_finite_robust(v) || v < 0.0 || v > 1.0) {
                throw std::invalid_argument("MeasurementClassifier::from_matrix: entry (" +
                                            std::to_string(s) + ", " + std::to_string(l) +
                                            ") = " + std::to_string(v) +
                                            " is not finite or is out of [0, 1]");
            }
            flat[s * l_n + l] = v;
        }
    }

    std::vector<double> reject_probs(l_n, 0.0);
    for (size_t l = 0; l < l_n; ++l) {
        double sum = 0.0;
        for (size_t s = 0; s < s_n; ++s) {
            sum += flat[s * l_n + l];
        }
        if (sum > 1.0 + kProbTolerance) {
            throw std::invalid_argument("MeasurementClassifier::from_matrix: column " +
                                        std::to_string(l) + " sum = " + std::to_string(sum) +
                                        " exceeds 1");
        }
        // Deficit is the implicit reject probability, clamped to [0, 1]
        // so accessors never report a negative number under floating
        // drift in the column sum.
        const double clamped = sum > 1.0 ? 1.0 : sum;
        reject_probs[l] = 1.0 - clamped;
    }

    return MeasurementClassifier(std::move(symbols), std::move(flat), std::move(reject_probs),
                                 levels.fingerprint());
}

MeasurementClassifier::MeasurementClassifier(std::vector<std::string> symbols,
                                             std::vector<double> matrix_flat,
                                             std::vector<double> reject_probs,
                                             uint64_t level_fingerprint)
    : symbols_(std::move(symbols)),
      matrix_flat_(std::move(matrix_flat)),
      reject_probs_(std::move(reject_probs)),
      level_fingerprint_(level_fingerprint) {}

const std::string& MeasurementClassifier::symbol_label(uint8_t symbol_idx) const {
    if (symbol_idx >= symbols_.size()) {
        throw std::invalid_argument("MeasurementClassifier::symbol_label: index " +
                                    std::to_string(symbol_idx) + " out of range (num_symbols " +
                                    std::to_string(symbols_.size()) + ")");
    }
    return symbols_[symbol_idx];
}

double MeasurementClassifier::prob(uint8_t symbol_idx, uint8_t level_id) const {
    const size_t s_n = symbols_.size();
    const size_t l_n = reject_probs_.size();
    if (symbol_idx >= s_n || level_id >= l_n) {
        throw std::invalid_argument("MeasurementClassifier::prob: index (" +
                                    std::to_string(symbol_idx) + ", " + std::to_string(level_id) +
                                    ") out of range (num_symbols " + std::to_string(s_n) +
                                    ", num_levels " + std::to_string(l_n) + ")");
    }
    return matrix_flat_[static_cast<size_t>(symbol_idx) * l_n + static_cast<size_t>(level_id)];
}

double MeasurementClassifier::reject_probability(uint8_t level_id) const {
    if (level_id >= reject_probs_.size()) {
        throw std::invalid_argument("MeasurementClassifier::reject_probability: index " +
                                    std::to_string(level_id) + " out of range (num_levels " +
                                    std::to_string(reject_probs_.size()) + ")");
    }
    return reject_probs_[level_id];
}

}  // namespace clifft
