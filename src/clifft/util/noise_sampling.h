#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

namespace clifft {

inline constexpr uint32_t kNoNoiseSite = std::numeric_limits<uint32_t>::max();

// Converts one independent Bernoulli site into additive hazard space. The
// clamp keeps a probability rounded to one finite and aligned with the finest
// probability step of the repository's deterministic [0, 1) RNG conversion.
[[nodiscard]] inline double bernoulli_hazard(double probability) noexcept {
    assert(std::isfinite(probability) && probability >= 0.0 &&
           "noise probability must be finite and nonnegative");
    return -std::log1p(-std::min(probability, 1.0 - 0x1.0p-53));
}

// Uses one exponential gap to skip independent silent Bernoulli sites. The
// cumulative table may contain repeated entries for zero-probability sites;
// upper_bound skips those without special cases in the hot caller.
[[nodiscard]] inline uint32_t sample_next_noise_site(std::span<const double> cumulative_hazards,
                                                     uint32_t first_candidate,
                                                     double uniform_draw) noexcept {
    if (first_candidate >= cumulative_hazards.size()) {
        return kNoNoiseSite;
    }
    assert(uniform_draw >= 0.0 && uniform_draw < 1.0 && "noise gap draw must be in [0, 1)");
    const double current_hazard =
        first_candidate == 0 ? 0.0 : cumulative_hazards[first_candidate - 1];
    const double target_hazard = current_hazard - std::log(1.0 - uniform_draw);
    const auto candidate = std::upper_bound(cumulative_hazards.begin() + first_candidate,
                                            cumulative_hazards.end(), target_hazard);
    if (candidate == cumulative_hazards.end()) {
        return kNoNoiseSite;
    }
    return static_cast<uint32_t>(candidate - cumulative_hazards.begin());
}

}  // namespace clifft
