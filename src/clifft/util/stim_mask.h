#pragma once

// Stim <-> MaskView interop helpers.
//
// These bridge stim's simd_bits_range_ref Pauli rows and clifft's
// runtime-width MaskView storage. Kept in a separate header so neither
// hir.h nor mask_view.h needs to take a Stim dependency on its own.

#include "clifft/util/mask_view.h"

#include "stim.h"

#include <cassert>
#include <cstdint>

namespace clifft {

// Stim SIMD lane width. Stim's TableauSimulator<W> handles arbitrary qubit
// counts via dynamic heap allocation at any W; this controls only the SIMD
// register width for internal bit operations.
constexpr size_t kStimWidth = 64;

/// Copy the first `n` bits of a Stim PauliString row into a destination
/// MutableMaskView. Trailing destination words are zeroed. The destination
/// must be at least `(n + 63) / 64` words wide.
inline void stim_to_mask_view(const stim::simd_bits_range_ref<kStimWidth>& bits, uint32_t n,
                              MutableMaskView dst) {
    const uint32_t words = (n + 63) / 64;
    assert(words <= dst.num_words() && "stim_to_mask_view: destination too narrow");
    for (uint32_t w = 0; w < words; ++w)
        dst.words[w] = bits.u64[w];
    for (uint32_t w = words; w < dst.num_words(); ++w)
        dst.words[w] = 0;
}

}  // namespace clifft
