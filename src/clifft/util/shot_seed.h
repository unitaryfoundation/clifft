#pragma once

// Derives deterministic per-shot RNG states from one call-level seed root.
// A shot's stream depends only on the root, its global shot index, and a
// domain label, so scheduling and worker count cannot change seeded results.

#include "clifft/util/xoshiro.h"

#include <array>
#include <cstdint>
#include <optional>

namespace clifft {

// Ordinary and fixed-fault sampling use one executor stream per shot.
inline constexpr uint64_t kSamplingExecutorDomain = 0x01;

// Driver-side trajectory draws (initial levels, trap destinations, classical
// consults, and herald flags) must not shift in-executor Born randomness.
inline constexpr uint64_t kTrajectoryDriverDomain = 0x11;
inline constexpr uint64_t kTrajectoryExecutorDomain = 0x12;

struct SeedRoot {
    uint64_t w[4];
};

// Deterministic expansion of a user seed (SplitMix64 chain).
inline SeedRoot seed_root_from_seed(uint64_t seed) {
    SeedRoot root;
    uint64_t z = seed;
    for (uint64_t& word : root.w) {
        word = splitmix64(z);
    }
    return root;
}

// Seeded calls expand the user seed. Unseeded calls read OS entropy once and
// derive every shot from that root. Zero-shot calls deliberately read no
// entropy so their observable behavior stays empty and side-effect free.
inline SeedRoot make_seed_root(uint32_t shots, std::optional<uint64_t> seed) {
    if (seed.has_value()) {
        return seed_root_from_seed(*seed);
    }
    SeedRoot root{};
    if (shots != 0) {
        const std::array<uint64_t, 4> words = entropy_seed_words();
        root.w[0] = words[0];
        root.w[1] = words[1];
        root.w[2] = words[2];
        root.w[3] = words[3];
    }
    return root;
}

// The odd shot multiplier preserves distinct shot indices modulo 2^64.
// Including the word number in the stream label prevents a match in one
// state word from automatically repeating in all four.
inline std::array<uint64_t, 4> derive_state(const SeedRoot& root, uint64_t shot, uint64_t domain) {
    std::array<uint64_t, 4> s;
    for (uint64_t k = 0; k < 4; ++k) {
        uint64_t z = root.w[k] ^ (shot * 0x9E3779B97F4A7C15ULL) ^
                     (((domain << 2) | k) * 0xBF58476D1CE4E5B9ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s[k] = z ^ (z >> 31);
    }
    return s;
}

}  // namespace clifft
