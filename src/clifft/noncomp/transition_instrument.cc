#include "clifft/noncomp/transition_instrument.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft {

// is_finite_robust below assumes IEEE 754 doubles (every Clifft
// target satisfies this). Make the assumption explicit.
static_assert(std::numeric_limits<double>::is_iec559,
              "TransitionInstrument requires IEEE 754 doubles");

namespace {

// Tolerance for entry / column-sum bounds and for the
// source-independence equality check. Generous compared to typical
// floating-point error in user-supplied matrices; we want to admit
// hand-written tables that sum to 1.0 - epsilon.
constexpr double kProbTolerance = 1e-12;

// Release builds use -ffast-math, which implies -ffinite-math-only.
// That lets the compiler assume operands are finite, folding away
// std::isfinite() and turning `v >= 0.0 && v <= 1.0` into something
// that passes NaN through. Inspect the IEEE 754 bit pattern
// instead: a non-finite double has all exponent bits set.
bool is_finite_robust(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    return (bits & kExpMask) != kExpMask;
}

}  // namespace

TransitionInstrument TransitionInstrument::from_matrix(std::vector<std::vector<double>> matrix,
                                                       const LevelSet& levels) {
    const size_t n = levels.size();

    if (matrix.size() != n) {
        throw std::invalid_argument(
            std::format("TransitionInstrument::from_matrix: matrix has {} rows; expected {} "
                        "(one per level)",
                        matrix.size(), n));
    }

    // Single pass: validate row width and entry bounds while copying
    // into the flat row-major buffer.
    std::vector<double> flat(n * n);
    for (size_t to = 0; to < n; ++to) {
        if (matrix[to].size() != n) {
            throw std::invalid_argument(
                std::format("TransitionInstrument::from_matrix: row {} has {} columns; "
                            "expected {}",
                            to, matrix[to].size(), n));
        }
        for (size_t from = 0; from < n; ++from) {
            const double v = matrix[to][from];
            // Raw user entries must be finite and lie strictly in [0, 1]:
            // tolerance applies only to derived column sums below.
            // is_finite_robust runs first because -ffast-math folds
            // std::isfinite() / NaN-aware comparisons away.
            if (!is_finite_robust(v) || v < 0.0 || v > 1.0) {
                throw std::invalid_argument(
                    std::format("TransitionInstrument::from_matrix: entry ({}, {}) = {} "
                                "is not finite or is out of [0, 1]",
                                to, from, v));
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
            throw std::invalid_argument(std::format(
                "TransitionInstrument::from_matrix: column {} sum = {} exceeds 1", from, sum));
        }
        column_sums[from] = sum > 1.0 ? 1.0 : sum;
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
        if (levels_span[from].category != LevelCategory::Computational) {
            continue;
        }
        if (first_comp == 0xFF) {
            first_comp = static_cast<uint8_t>(from);
            continue;
        }
        for (size_t to = 0; to < n; ++to) {
            if (std::abs(flat[to * n + first_comp] - flat[to * n + from]) > kProbTolerance) {
                flag = false;
                break;
            }
        }
        if (!flag) {
            break;
        }
    }

    return TransitionInstrument(std::move(flat), std::move(column_sums), flag);
}

TransitionInstrument::TransitionInstrument(std::vector<double> matrix_flat,
                                           std::vector<double> column_sums,
                                           bool is_source_independent_on_computational)
    : matrix_flat_(std::move(matrix_flat)),
      column_sums_(std::move(column_sums)),
      is_source_independent_on_computational_(is_source_independent_on_computational) {}

double TransitionInstrument::prob(uint8_t to, uint8_t from) const {
    const size_t n = column_sums_.size();
    if (to >= n || from >= n) {
        throw std::invalid_argument(
            std::format("TransitionInstrument::prob: index ({}, {}) out of range "
                        "(num_levels {})",
                        static_cast<unsigned>(to), static_cast<unsigned>(from), n));
    }
    return matrix_flat_[static_cast<size_t>(to) * n + static_cast<size_t>(from)];
}

double TransitionInstrument::column_sum(uint8_t from) const {
    if (from >= column_sums_.size()) {
        throw std::invalid_argument(
            std::format("TransitionInstrument::column_sum: index {} out of range "
                        "(num_levels {})",
                        static_cast<unsigned>(from), column_sums_.size()));
    }
    return column_sums_[from];
}

double TransitionInstrument::no_jump_weight(uint8_t from) const {
    return 1.0 - column_sum(from);
}

}  // namespace clifft
