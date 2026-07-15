#pragma once

// Per-shot RNG derivation: a 256-bit root (OS entropy when unseeded; a
// deterministic expansion of the 64-bit user seed otherwise) fans out into
// one 256-bit generator state per (shot, domain). For a fixed root and
// domain, distinct shots produce distinct states (word 0 is a bijection of
// the shot index). Word-indexed domain tags keep a single cross-domain
// word alias from expanding into a full-state collision, and independent
// entropy roots collide on a fixed pair of states with probability ~2^-256.

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
