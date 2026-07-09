#pragma once

// TransitionInstrument: a per-qubit transition matrix over the five
// levels, with cached column sums.
//
// The matrix uses T[to, from] convention: matrix[to][from] is the
// probability of the qubit transitioning from level `from` to level
// `to`. Each column corresponds to one source level; the column sum
// gives the total jump probability out of that source, and the
// deficit `1 - column_sum` is the implicit no-jump weight.
//
// Convention: every matrix entry, diagonal included, represents a
// discrete *transition event*. A non-zero matrix[a][a] is NOT the
// probability that source `a` stays at level `a`; it is the
// probability of a transition event that happens to land back at
// the source level (e.g., a depolarizing error that ends in the
// same level). The "nothing happened" branch ("no-jump") lives in
// the column deficit, not the diagonal. This mirrors the Kraus
// structure: each jump is its own Kraus operator, and no-jump is
// the complement.
//
// Source-dependent matrices (where the g and e columns differ) are
// fully supported: the source level is resolved at sample time.

#include "clifft/noncomp/level.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

class TransitionInstrument {
  public:
    // Validates that:
    //   - matrix is square with dimension kNumLevels;
    //   - every entry lies in [0, 1];
    //   - every column sum lies in [0, 1].
    static TransitionInstrument from_matrix(const std::vector<std::vector<double>>& matrix);

    // T[to, from].
    double prob(Level to, Level from) const {
        return matrix_[static_cast<size_t>(to) * kNumLevels + static_cast<size_t>(from)];
    }

    // Sum of column `from`, the total jump probability out of that
    // source level. Cached at construction.
    double column_sum(Level from) const { return column_sums_[static_cast<size_t>(from)]; }

  private:
    TransitionInstrument(std::array<double, kNumLevels * kNumLevels> matrix,
                         std::array<double, kNumLevels> column_sums)
        : matrix_(matrix), column_sums_(column_sums) {}

    // Row-major flat storage: matrix_[to * kNumLevels + from].
    std::array<double, kNumLevels * kNumLevels> matrix_;
    std::array<double, kNumLevels> column_sums_;
};

}  // namespace clifft
