#pragma once

#include "clifft/noncomp/level.h"

#include <cstddef>
#include <vector>

namespace clifft {
namespace test {

// Raw input shape used by model construction tests. It remains dynamic so
// validation tests can deliberately form malformed matrices.
using RawProbabilityMatrix = std::vector<std::vector<double>>;

constexpr size_t level_index(Level level) {
    return static_cast<size_t>(level);
}

inline RawProbabilityMatrix zero_transition_matrix() {
    return RawProbabilityMatrix(kNumLevels, std::vector<double>(kNumLevels, 0.0));
}

// A source-independent certain transition from either computational level.
inline RawProbabilityMatrix certain_transition_from_computational(Level destination) {
    auto matrix = zero_transition_matrix();
    matrix[level_index(destination)][level_index(Level::G)] = 1.0;
    matrix[level_index(destination)][level_index(Level::E)] = 1.0;
    return matrix;
}

inline std::vector<double> pure_initial_state(Level level) {
    std::vector<double> probabilities(kNumLevels, 0.0);
    probabilities[level_index(level)] = 1.0;
    return probabilities;
}

// Classifier with one selected column replaced. Computational readout is
// faithful, and unselected noncomputational levels report symbol 0.
inline RawProbabilityMatrix classifier_matrix_with_column(Level level,
                                                          std::vector<double> probabilities) {
    RawProbabilityMatrix matrix(probabilities.size(), std::vector<double>(kNumLevels, 0.0));
    for (size_t index = 0; index < kNumLevels; ++index) {
        matrix[0][index] = 1.0;
    }
    matrix[0][level_index(Level::E)] = 0.0;
    matrix[1][level_index(Level::E)] = 1.0;
    for (size_t symbol = 0; symbol < probabilities.size(); ++symbol) {
        matrix[symbol][level_index(level)] = probabilities[symbol];
    }
    return matrix;
}

}  // namespace test
}  // namespace clifft
