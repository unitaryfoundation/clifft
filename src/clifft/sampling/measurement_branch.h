#pragma once

#include "clifft/sampling/kernels.h"
#include "clifft/util/numeric.h"

#include <cassert>

namespace clifft::sampling {

// Reachability is shared by random sampling and forced-record replay. Collapse
// still uses the original population, including any numerical dust.
enum class MeasurementBranchKind : uint8_t { Random, DeterministicZero, DeterministicOne };

struct MeasurementBranchClassification {
    MeasurementBranchKind kind = MeasurementBranchKind::Random;
    bool clamped_dust = false;
};

[[nodiscard]] inline MeasurementBranchClassification classify_measurement_branch(
    MeasurementProbabilities probabilities) noexcept {
    const double total = probabilities.total();
    assert(is_finite_robust(probabilities.zero) && probabilities.zero >= 0.0 &&
           is_finite_robust(probabilities.one) && probabilities.one >= 0.0 &&
           is_finite_robust(total) && total > 0.0 &&
           "measurement probabilities must be finite, nonnegative, and nonzero");
    const double epsilon = kMeasurementDustEpsilon * total;
    if (probabilities.one <= epsilon) {
        return {.kind = MeasurementBranchKind::DeterministicZero,
                .clamped_dust = probabilities.one > 0.0};
    }
    if (probabilities.zero <= epsilon) {
        return {.kind = MeasurementBranchKind::DeterministicOne,
                .clamped_dust = probabilities.zero > 0.0};
    }
    return {.kind = MeasurementBranchKind::Random};
}

}  // namespace clifft::sampling
