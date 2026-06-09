#pragma once

#include "clifft/util/mask_view.h"

#include <cstdint>
#include <vector>

namespace clifft {

struct PauliAxis {
    std::vector<uint64_t> x;
    std::vector<uint64_t> z;

    void resize(uint32_t num_words);
    void set_from(MaskView xv, MaskView zv);
    void xor_with(MaskView xv, MaskView zv);
    [[nodiscard]] bool is_zero() const;
};

}  // namespace clifft
