#pragma once

#include "clifft/util/xoshiro.h"

#include <cstdint>
#include <span>

namespace clifft::sampling {

// Prepared once, then shared by immutable classical programs. The inverse
// hazard is only used for rare events; dense masks avoid one log per event.
class PreparedBernoulli {
  public:
    explicit PreparedBernoulli(double probability = 0.0);
    [[nodiscard]] double probability() const noexcept { return probability_; }
    [[nodiscard]] double inverse_hazard() const noexcept { return inverse_hazard_; }

  private:
    double probability_;
    double inverse_hazard_;
};

// Overwrites output with independent events on the eligible lanes only.
// Spans cover the same retained word capacity, do not alias, and eligibility
// contains no bits beyond lanes. No allocation, seeding, or planning occurs.
void sample_bernoulli_bits(const PreparedBernoulli& distribution,
                           std::span<const uint64_t> eligible, std::span<uint64_t> output,
                           uint32_t lanes, Xoshiro256PlusPlus& rng) noexcept;

}  // namespace clifft::sampling
