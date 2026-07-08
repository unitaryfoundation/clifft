#pragma once

// MeasurementClassifier: stochastic map from levels to user-facing
// measurement symbols.
//
// matrix[symbol_index][level] = P(symbol | level). Symbol indices are
// positional: index 0 and 1 are the recorded measurement bit, and an
// optional third symbol (kHeraldSymbol) is the herald -- "this readout
// is ambiguous", reported in the per-shot herald sidecar rather than
// the record.
//
// Every column must sum to 1: a measurement always produces a symbol.
// A substochastic column would reserve probability for a reject
// (heralded-abort) outcome no sampling path represents, so it is
// rejected here at construction rather than at first use. The
// computational columns must also place no mass on the herald symbol:
// a computational readout is a real Born measurement whose record can
// at most be misreported (readout confusion), not heralded away.

#include "clifft/noncomp/level.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace clifft {

class MeasurementClassifier {
  public:
    // Validates that:
    //   - matrix.size() (the row count) is two or three;
    //   - every row has kNumLevels columns;
    //   - every entry is finite and lies in [0, 1] (strict);
    //   - every column sums to 1 within tolerance;
    //   - the computational columns place no mass on the herald symbol.
    static MeasurementClassifier from_matrix(const std::vector<std::vector<double>>& matrix);

    // The positional index of the herald symbol, when one is present.
    static constexpr uint8_t kHeraldSymbol = 2;

    size_t num_symbols() const { return num_symbols_; }

    // True when a third, herald symbol is present (see kHeraldSymbol).
    bool has_herald() const { return num_symbols_ == 3; }

    // P(symbol | level). Throws on an out-of-range symbol index.
    double prob(uint8_t symbol_idx, Level level) const;

  private:
    MeasurementClassifier(size_t num_symbols, std::vector<double> matrix_flat)
        : num_symbols_(num_symbols), matrix_flat_(std::move(matrix_flat)) {}

    size_t num_symbols_;
    // Row-major flat storage: matrix_flat_[symbol * kNumLevels + level].
    std::vector<double> matrix_flat_;
};

}  // namespace clifft
