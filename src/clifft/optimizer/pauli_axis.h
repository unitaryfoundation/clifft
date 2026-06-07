#pragma once

#include "clifft/util/mask_view.h"

#include <cstdint>
#include <vector>

namespace clifft {

/// Signless Pauli axis stored as full-width (x, z) masks.
struct PauliAxis {
    std::vector<uint64_t> x;
    std::vector<uint64_t> z;

    void resize(uint32_t num_words);
    void set_from(MaskView xv, MaskView zv);
    void xor_with(MaskView xv, MaskView zv);
    bool is_zero() const;
};

}  // namespace clifft
