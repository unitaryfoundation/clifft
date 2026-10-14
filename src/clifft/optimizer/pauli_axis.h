#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/tableau/pauli_string.h"

#include <algorithm>

namespace clifft::optimizer_detail {

inline uint32_t pivot_of(const PauliString& axis) {
    const uint32_t domain = axis.x().num_words() * 64;
    const uint32_t x = axis.x().lowest_bit();
    return x < domain ? x : domain + axis.z().lowest_bit();
}

inline bool body_bit(const PauliString& axis, uint32_t pivot) {
    const uint32_t domain = axis.x().num_words() * 64;
    return pivot < domain ? axis.x().bit_get(pivot) : axis.z().bit_get(pivot - domain);
}

inline PauliString copy_axis(PauliMaskView mask, uint32_t width) {
    PauliString axis(width);
    axis.mut_x().xor_with(mask.x());
    axis.mut_z().xor_with(mask.z());
    axis.set_sign(mask.sign());
    return axis;
}

inline void write_axis(MutablePauliMaskView mask, const PauliString& axis) {
    std::ranges::copy(axis.x().words, mask.x().words.begin());
    std::ranges::copy(axis.z().words, mask.z().words.begin());
    mask.set_sign(axis.sign());
}

}  // namespace clifft::optimizer_detail
