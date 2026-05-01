#pragma once

// PauliMaskArena: contiguous storage for runtime-width Pauli (X, Z, sign)
// masks, indexed by a uint32_t handle.
//
// All masks in an arena share the same `num_words` width, set at
// construction from the circuit's physical qubit count. Masks are
// allocated by handle and accessed via PauliMaskRef / PauliMaskConstRef
// triples that hand out raw uint64_t pointers plus the shared width.
//
// Each handle owns its own (X, Z, sign) slot -- no deduplication. This
// matches the existing per-op mask ownership model in the HIR.
//
// Storage layout: parallel `x_` and `z_` vectors, each strided by
// `num_words_`, plus a `signs_` byte vector. Parallel arrays (rather than
// interleaved [x|z]) keep the hot APPLY_PAULI / NOISE kernel reading two
// contiguous strides, which auto-vectorizes more cleanly than interleaved
// layout when num_words is large.
//
// PR1 introduces the type alongside BitMask<N>. Callers (HirModule,
// ConstantPool) migrate in a later PR.

#include "clifft/util/mask_view.h"

#include <cassert>
#include <cstdint>
#include <vector>

namespace clifft {

struct PauliMaskRef {
    uint64_t* x;
    uint64_t* z;
    uint8_t* sign;
    uint32_t num_words;

    [[nodiscard]] MutableMaskView mut_x() { return {x, num_words}; }
    [[nodiscard]] MutableMaskView mut_z() { return {z, num_words}; }
    [[nodiscard]] MaskView view_x() const { return {x, num_words}; }
    [[nodiscard]] MaskView view_z() const { return {z, num_words}; }
};

struct PauliMaskConstRef {
    const uint64_t* x;
    const uint64_t* z;
    const uint8_t* sign;
    uint32_t num_words;

    [[nodiscard]] MaskView view_x() const { return {x, num_words}; }
    [[nodiscard]] MaskView view_z() const { return {z, num_words}; }
};

class PauliMaskArena {
  public:
    explicit PauliMaskArena(uint32_t num_qubits) : num_words_((num_qubits + 63) / 64) {
        // num_words_ == 0 is allowed (empty circuits): alloc still hands
        // out unique handles, but the per-mask slot has zero width.
    }

    [[nodiscard]] uint32_t num_words() const { return num_words_; }
    [[nodiscard]] size_t num_masks() const { return signs_.size(); }

    /// Allocate a new zero-initialized mask. Returns its handle.
    uint32_t alloc_zero() {
        uint32_t handle = static_cast<uint32_t>(signs_.size());
        x_.resize(x_.size() + num_words_, 0);
        z_.resize(z_.size() + num_words_, 0);
        signs_.push_back(0);
        return handle;
    }

    [[nodiscard]] PauliMaskRef get(uint32_t handle) {
        assert(handle < signs_.size());
        size_t off = static_cast<size_t>(handle) * num_words_;
        return {x_.data() + off, z_.data() + off, &signs_[handle], num_words_};
    }

    [[nodiscard]] PauliMaskConstRef get(uint32_t handle) const {
        assert(handle < signs_.size());
        size_t off = static_cast<size_t>(handle) * num_words_;
        return {x_.data() + off, z_.data() + off, &signs_[handle], num_words_};
    }

  private:
    uint32_t num_words_;
    std::vector<uint64_t> x_;
    std::vector<uint64_t> z_;
    std::vector<uint8_t> signs_;
};

}  // namespace clifft
