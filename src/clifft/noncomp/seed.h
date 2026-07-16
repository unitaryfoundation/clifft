#pragma once

// Derives separate per-shot RNG states for the driver and VM. Keeping these
// streams separate means that adding a driver-side random draw cannot consume
// from or shift the VM's measurement sequence. Unseeded runs start with 256
// bits of OS entropy; seeded runs expand the user's 64-bit seed
// deterministically. Within either stream, different shots always receive
// different 256-bit states. Each state word is mixed differently, so a match
// in one word does not automatically repeat in the other three. For
// independently generated roots, a given pair of states matches with
// probability about 2^-256.

#include "clifft/util/xoshiro.h"

#include <array>
#include <cstdint>

namespace clifft {

// Driver-side draws (initial levels, trap destinations, classical
// consults, herald flags) vs. the in-VM Born measurement randomness.
inline constexpr uint64_t kExactDriverDomain = 0x11;
inline constexpr uint64_t kExactSvmDomain = 0x12;

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

// Uses the SplitMix64 golden-gamma constant and mixing finalizer:
// https://prng.di.unimi.it/splitmix64.c
//
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
