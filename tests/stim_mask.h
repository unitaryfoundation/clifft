#pragma once

// Stim <-> MaskView interop used only by independent test oracles.

#include "clifft/util/mask_view.h"

#include "stim.h"

#include <cassert>
#include <cstdint>

namespace clifft {

constexpr size_t kStimWidth = 64;
using StimBitsRange = stim::simd_bits_range_ref<kStimWidth>;

inline void stim_to_mask_view(const StimBitsRange& bits, uint32_t n, MutableMaskView dst) {
    const uint32_t words = (n + 63) / 64;
    assert(words <= dst.num_words() && "stim_to_mask_view: destination too narrow");
    for (uint32_t w = 0; w < words; ++w) {
        dst.words[w] = bits.u64[w];
    }
    for (uint32_t w = words; w < dst.num_words(); ++w) {
        dst.words[w] = 0;
    }
}

inline void mask_view_to_stim(MaskView src, uint32_t n, StimBitsRange dst) {
    const uint32_t words = (n + 63) / 64;
    assert(words <= src.num_words() && "mask_view_to_stim: source too narrow");
    assert(words <= dst.num_u64_padded() && "mask_view_to_stim: destination too narrow");
    dst.clear();
    for (uint32_t w = 0; w < words; ++w) {
        dst.u64[w] = src.words[w];
    }
    if (words != 0 && n % 64 != 0) {
        dst.u64[words - 1] &= (uint64_t{1} << (n % 64)) - 1;
    }
}

}  // namespace clifft
