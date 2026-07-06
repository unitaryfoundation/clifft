#pragma once

// Per-shot seed derivation shared by the noncomputational sampling
// paths. One global seed fans out into independent sub-streams, one per
// (shot, domain) pair; every domain tag lives here so their pairwise
// distinctness is visible in one place.

#include <cstdint>

namespace clifft {

// AOT orchestrator streams: trajectory pre-sampling, classifier draws,
// and the in-VM measurement randomness of the compiled module.
inline constexpr uint64_t kHistoryDomain = 0x1;
inline constexpr uint64_t kClassifierDomain = 0x2;
inline constexpr uint64_t kSvmDomain = 0x3;

// Exact-driver streams: the driver's own draws between VM runs (initial
// levels, trap destinations, classical consults, herald flags) and the
// in-VM measurement randomness of the modules it executes.
inline constexpr uint64_t kExactDriverDomain = 0x11;
inline constexpr uint64_t kExactSvmDomain = 0x12;

// SplitMix64 finalizer over a mix of the global seed, shot, and domain.
// Full-avalanche mixing: adjacent shots and sibling domains land in
// statistically unrelated generator states, which is the xoshiro
// authors' recommended seeding procedure.
inline uint64_t derive_seed(uint64_t global, uint64_t shot, uint64_t domain) {
    uint64_t z = global ^ (shot * 0x9E3779B97F4A7C15ULL) ^ (domain * 0xBF58476D1CE4E5B9ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace clifft
