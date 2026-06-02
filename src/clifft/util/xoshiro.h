#pragma once

// =============================================================================
// Scientific PRNG: xoshiro256++ (Seeded by SplitMix64)
// =============================================================================
//
// Reference implementation of xoshiro256++ 1.0 by David Blackman and
// Sebastiano Vigna (vigna@acm.org), described in:
//
//   Blackman, D. & Vigna, S. (2021). "Scrambled Linear Pseudorandom
//   Number Generators." ACM Trans. Math. Softw. 47(4), Article 36.
//   https://doi.org/10.1145/3460772
//
// Source: https://prng.di.unimi.it/xoshiro256plusplus.c
// Seeding: https://prng.di.unimi.it/splitmix64.c
// License: Public domain (CC0)
//
// Period: 2^256-1. State: 32 bytes (vs MT19937's 2504 bytes), making per-shot
// reseeding ~100x cheaper. Uses pure bitwise math to guarantee identical
// sequences across GCC/Clang/MSVC.

#include <bit>
#include <cstdint>

namespace clifft {

class Xoshiro256PlusPlus {
  public:
    explicit Xoshiro256PlusPlus(uint64_t seed_val = 0) { seed(seed_val); }

    // Seed from a single 64-bit value via SplitMix64 expansion.
    inline void seed(uint64_t seed_val) {
        uint64_t z = seed_val;
        s_[0] = splitmix64(z);
        s_[1] = splitmix64(z);
        s_[2] = splitmix64(z);
        s_[3] = splitmix64(z);
    }

    // Seed all 256 bits directly (e.g. from hardware entropy).
    inline void seed_full(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
        s_[0] = s0;
        s_[1] = s1;
        s_[2] = s2;
        s_[3] = s3;
    }

    // Seed from OS hardware entropy (std::random_device).
    // Uses the glibc /dev/urandom workaround from Stim for
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94087
    void seed_from_entropy();

    inline uint64_t operator()() {
        const uint64_t result = std::rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;

        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];

        s_[2] ^= t;
        s_[3] = std::rotl(s_[3], 45);

        return result;
    }

  private:
    uint64_t s_[4];

    static inline uint64_t splitmix64(uint64_t& state) {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

}  // namespace clifft
