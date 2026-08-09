#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace clifft::sampling {

// Restores a zero at pivot in a basis index from which that bit was removed.
// Lower bits retain their positions and higher bits shift left by one.
inline uint64_t insert_zero_bit(uint64_t packed, uint32_t pivot) noexcept {
    const uint64_t lower_mask = (uint64_t{1} << pivot) - 1;
    return (packed & lower_mask) | ((packed & ~lower_mask) << 1);
}

// Packs the parities selected by masks into an index, with masks[i]
// contributing bit i.
inline size_t selector_index(uint64_t representative, std::span<const uint64_t> masks) noexcept {
    size_t selector = 0;
    for (size_t bit = 0; bit < masks.size(); ++bit) {
        selector |= static_cast<size_t>(std::popcount(representative & masks[bit]) & 1U) << bit;
    }
    return selector;
}

}  // namespace clifft::sampling
