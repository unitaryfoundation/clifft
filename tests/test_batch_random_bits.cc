#include "clifft/sampling/batch/bits.h"
#include "clifft/sampling/batch/random_bits.h"

#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace clifft;
using namespace clifft::sampling;

TEST_CASE("Packed Bernoulli sampling preserves endpoints masks and reuse") {
    const double probability = GENERATE(0.0, 1.0, 0.5, 0.25, 0.75, 0.001, 0.0625, 0.3, 0.9);
    PreparedBernoulli distribution(probability);
    Xoshiro256PlusPlus rng(72);
    const std::array<uint64_t, 3> eligible{0x8000000000000001ULL, 1, 0};
    std::array<uint64_t, 3> result;
    for (unsigned repeat = 0; repeat < 4; ++repeat) {
        result.fill(~uint64_t{0});
        sample_bernoulli_bits(distribution, eligible, result, 65, rng);
        for (size_t word = 0; word < result.size(); ++word) {
            CHECK((result[word] & ~eligible[word]) == 0);
            if (probability == 0.0)
                CHECK(result[word] == 0);
            if (probability == 1.0)
                CHECK(result[word] == eligible[word]);
        }
    }
    const std::array<uint64_t, 3> empty{};
    sample_bernoulli_bits(distribution, empty, result, 0, rng);
    CHECK(result == empty);
    sample_bernoulli_bits(distribution, {}, {}, 0, rng);
}

TEST_CASE("Packed Bernoulli events have the independent joint law") {
    const double probability = GENERATE(0.001, 0.02, 0.0625, 0.25, 0.3, 0.5, 0.75, 0.9);
    const auto distribution = PreparedBernoulli(probability);
    Xoshiro256PlusPlus rng(91);
    const std::array<uint64_t, 2> eligible{~uint64_t{0}, 1};
    std::array<uint64_t, 2> result{};
    constexpr uint32_t batches = 4096;
    uint64_t total = 0, paired = 0;
    for (uint32_t batch = 0; batch < batches; ++batch) {
        sample_bernoulli_bits(distribution, eligible, result, 65, rng);
        total += std::popcount(result[0]) + std::popcount(result[1]);
        paired += std::popcount(result[0] & (result[0] >> 1) & 0x5555555555555555ULL);
        paired += (result[0] >> 63) & result[1];
    }
    const double trials = batches * 65.0;
    CHECK(std::abs(total / trials - probability) <
          7 * std::sqrt(probability * (1 - probability) / trials) + 1 / trials);
    const double pair_trials = batches * 33.0;
    const double both = probability * probability;
    CHECK(std::abs(paired / pair_trials - both) <
          8 * std::sqrt(both * (1 - both) / pair_trials) + 2 / pair_trials);
}

TEST_CASE("Packed Bernoulli rejects invalid prepared probabilities") {
    for (double probability : {-0.1, 1.1, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        CHECK_THROWS_AS(PreparedBernoulli(probability), std::invalid_argument);
    }
}
