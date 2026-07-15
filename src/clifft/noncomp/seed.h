#pragma once

// Per-shot RNG derivation: a 256-bit root (expanded from the user seed,
// or drawn from OS entropy when unseeded) fans out into one 256-bit
// generator state per (shot, domain). Distinct shots in a domain provably
// get distinct states (word 0 is a bijection of the shot index); any other
// coincidence needs four unrelated 64-bit collisions at once (~2^-256).
// The word index is folded into the domain tag so those four conditions
// cannot collapse into one.

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
