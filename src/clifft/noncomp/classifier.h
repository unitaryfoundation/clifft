#pragma once

// MeasurementClassifier: column-substochastic map from levels to
// user-facing measurement symbols.
//
// matrix[symbol_index][level_id] = P(symbol | level). For each level,
// the column (taken across symbols) sums to at most 1; the deficit
// per column is the implicit P(reject | level). A reject outcome
// means the model has no opinion about what symbol a measurement on
// a qubit at this level should produce, and the sampler raises
// instead of silently picking a bit.
//
// Construction binds the classifier to a LevelSet so the per-level
// column count matches the level table.

#include "clifft/noncomp/level.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace clifft {

class MeasurementClassifier {
  public:
    // Validates that:
    //   - symbols is non-empty and has no duplicate labels;
    //   - matrix has shape (symbols.size(), levels.size());
    //   - every entry is finite and lies in [0, 1] (strict);
    //   - every column sum lies in [0, 1] (the per-level deficit is
    //     the cached P(reject | level), clamped to 0 on overshoot
    //     within floating tolerance).
    static MeasurementClassifier from_matrix(std::vector<std::string> symbols,
                                             std::vector<std::vector<double>> matrix,
                                             const LevelSet& levels);

    size_t num_symbols() const { return symbols_.size(); }
    size_t num_levels() const { return reject_probs_.size(); }
    const std::string& symbol_label(uint8_t symbol_idx) const;

    // P(symbol | level). Throws on out-of-range indices.
    double prob(uint8_t symbol_idx, uint8_t level_id) const;

    // 1 - sum_s prob(s, level_id). Cached at construction and
    // clamped to [0, 1] so it never reports a negative value under
    // floating drift in the column sum.
    double reject_probability(uint8_t level_id) const;

    // Fingerprint of the LevelSet this classifier was built against.
    // A model rejects a classifier whose fingerprint does not match
    // its own level table.
    uint64_t level_fingerprint() const { return level_fingerprint_; }

  private:
    MeasurementClassifier(std::vector<std::string> symbols, std::vector<double> matrix_flat,
                          std::vector<double> reject_probs, uint64_t level_fingerprint);

    std::vector<std::string> symbols_;
    // Row-major flat storage: matrix_flat_[symbol * num_levels() + level_id].
    std::vector<double> matrix_flat_;
    std::vector<double> reject_probs_;
    uint64_t level_fingerprint_;
};

}  // namespace clifft
