#pragma once

// TransitionInstrument: a per-qubit transition matrix with cached
// derived properties.
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
// structure in which each jump is its own Kraus operator and
// no-jump is the complement; it also makes the no-jump back-action
// under unknown-coherent sources (the aI + bZ filter) a single
// special case rather than something carved out of the diagonal.
//
// Construction binds the instrument to a LevelSet so the matrix
// shape can be checked against the level table and the
// is_source_independent_on_computational flag can be computed.
// Source-dependent matrices (where Computational-category columns
// differ) are valid here; whether they are applicable to a given
// qubit is enforced at sample time against the QubitStatusKind of
// the target qubit.

#include "clifft/noncomp/level.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

class TransitionInstrument {
  public:
    // Validates that:
    //   - matrix is square with dimension equal to levels.size();
    //   - every entry lies in [0, 1];
    //   - every column sum lies in [0, 1].
    // Computes and caches is_source_independent_on_computational (true
    // iff every column whose source level has category Computational
    // is bit-identical within tolerance to the others).
    static TransitionInstrument from_matrix(std::vector<std::vector<double>> matrix,
                                            const LevelSet& levels);

    size_t num_levels() const { return column_sums_.size(); }

    // T[to, from]. Throws on out-of-range indices.
    double prob(uint8_t to, uint8_t from) const;

    // Sum of column `from`, the total jump probability out of that
    // source level. Cached at construction.
    double column_sum(uint8_t from) const;

    // 1 - column_sum(from): the implicit no-jump weight for that
    // source level.
    double no_jump_weight(uint8_t from) const;

    bool is_source_independent_on_computational() const {
        return is_source_independent_on_computational_;
    }

  private:
    TransitionInstrument(std::vector<double> matrix_flat, std::vector<double> column_sums,
                         bool is_source_independent_on_computational);

    // Row-major flat storage: matrix_flat_[to * num_levels() + from].
    // One allocation, contiguous memory; column-traversal accessors
    // walk a single buffer instead of dereferencing per-row vectors.
    std::vector<double> matrix_flat_;
    std::vector<double> column_sums_;
    bool is_source_independent_on_computational_;
};

}  // namespace clifft
