#include "clifft/sampling/batch/random_bits.h"

#include "clifft/util/noise_sampling.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <stdexcept>

namespace clifft::sampling {

PreparedBernoulli::PreparedBernoulli(double probability)
    : probability_(probability), inverse_hazard_(0.0) {
    if (!is_probability(probability)) {
        throw std::invalid_argument("packed Bernoulli probability must lie in [0, 1]");
    }
    // A conservative crossover limits logarithms on dense channels. Dyadic
    // probabilities below are generated a word at a time instead.
    constexpr double sparse_cutoff = 1.0 / 16.0;
    if (probability > 0.0 && probability < sparse_cutoff) {
        const double inverse = 1.0 / bernoulli_hazard(probability);
        if (is_finite_robust(inverse)) {
            inverse_hazard_ = inverse;
        }
    }
}

void sample_bernoulli_bits(const PreparedBernoulli& distribution,
                           std::span<const uint64_t> eligible, std::span<uint64_t> output,
                           uint32_t lanes, Xoshiro256PlusPlus& rng) noexcept {
    assert(eligible.size() == output.size() && uint64_t{lanes} <= output.size() * 64);
    const double probability = distribution.probability();
    std::ranges::fill(output, uint64_t{0});
    if (probability == 0.0) {
        return;
    }
    if (probability == 1.0) {
        std::ranges::copy(eligible, output.begin());
        return;
    }
    if (distribution.inverse_hazard() > 0.0) {
        // Sampling candidate positions before applying eligibility preserves
        // independent Bernoulli events without walking the silent lanes.
        if (std::ranges::all_of(eligible, [](uint64_t word) { return word == 0; })) {
            return;
        }
        uint64_t lane = 0;
        while (lane < lanes) {
            const double gap =
                sample_geometric_gap(distribution.inverse_hazard(), rng.next_double());
            if (gap >= static_cast<double>(lanes - lane)) {
                break;
            }
            lane += static_cast<uint64_t>(gap);
            output[lane >> 6] |= (uint64_t{1} << (lane & 63)) & eligible[lane >> 6];
            ++lane;
        }
        return;
    }
    for (size_t word = 0; word < output.size(); ++word) {
        uint64_t mask = eligible[word];
        if (mask == 0) {
            continue;
        }
        if (probability == 0.5) {
            output[word] = rng() & mask;
        } else if (probability == 0.25) {
            const uint64_t first = rng();
            output[word] = first & rng() & mask;
        } else if (probability == 0.75) {
            const uint64_t first = rng();
            output[word] = (first | rng()) & mask;
        } else {
            // Strong partner channels can have very few eligible lanes. Draw
            // only for those lanes instead of generating a dense random column.
            while (mask != 0) {
                const auto bit = std::countr_zero(mask);
                if (rng.next_double() < probability) {
                    output[word] |= uint64_t{1} << bit;
                }
                mask &= mask - 1;
            }
        }
    }
}

}  // namespace clifft::sampling
