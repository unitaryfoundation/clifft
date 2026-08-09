#pragma once

#include <cstdint>

namespace clifft::sampling {

// Restores a zero at pivot in a basis index from which that bit was removed.
// Lower bits retain their positions and higher bits shift left by one.
inline uint64_t insert_zero_bit(uint64_t packed, uint32_t pivot) noexcept {
    const uint64_t lower_mask = (uint64_t{1} << pivot) - 1;
    return (packed & lower_mask) | ((packed & ~lower_mask) << 1);
}

}  // namespace clifft::sampling
