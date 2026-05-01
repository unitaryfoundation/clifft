#pragma once

// PauliMaskArena: contiguous storage for fixed-width Pauli (X, Z, sign)
// masks, indexed by an opaque PauliMaskHandle.
//
// All masks share the same width set at construction. Capacity is fixed at
// construction so view references (PauliMaskView, MutablePauliMaskView)
// returned by at()/mut_at() remain valid for the arena's lifetime; the
// arena does not resize.
//
// Each handle owns its own (X, Z, sign) slot -- no deduplication. Storage
// uses parallel x_/z_ word arrays plus a signs_ byte vector, which keeps
// the read-mostly hot paths over either x or z alone reading a single
// contiguous stride.

#include "clifft/util/mask_view.h"

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

/// Opaque handle into a PauliMaskArena.
enum class PauliMaskHandle : uint32_t {};

/// Read-only view of a single (X, Z, sign) entry in an arena.
class PauliMaskView {
  public:
    PauliMaskView(MaskView x, MaskView z, const uint8_t* sign) : x_(x), z_(z), sign_(sign) {}

    [[nodiscard]] MaskView x() const { return x_; }
    [[nodiscard]] MaskView z() const { return z_; }
    [[nodiscard]] bool sign() const { return *sign_ != 0; }

  private:
    MaskView x_;
    MaskView z_;
    const uint8_t* sign_;
};

/// Mutable view of a single (X, Z, sign) entry in an arena. Implicitly
/// converts to PauliMaskView for read-only call sites.
class MutablePauliMaskView {
  public:
    MutablePauliMaskView(MutableMaskView x, MutableMaskView z, uint8_t* sign)
        : x_(x), z_(z), sign_(sign) {}

    [[nodiscard]] MutableMaskView x() const { return x_; }
    [[nodiscard]] MutableMaskView z() const { return z_; }
    [[nodiscard]] bool sign() const { return *sign_ != 0; }
    void set_sign(bool s) { *sign_ = s ? 1 : 0; }

    operator PauliMaskView() const { return PauliMaskView(x_, z_, sign_); }

  private:
    MutableMaskView x_;
    MutableMaskView z_;
    uint8_t* sign_;
};

class PauliMaskArena {
  public:
    /// Construct with fixed capacity. All masks initialized to zero.
    PauliMaskArena(uint32_t num_qubits, size_t num_masks)
        : num_words_((num_qubits + 63) / 64),
          x_(num_masks * num_words_, 0),
          z_(num_masks * num_words_, 0),
          signs_(num_masks, 0) {}

    [[nodiscard]] uint32_t num_words() const { return num_words_; }
    [[nodiscard]] size_t size() const { return signs_.size(); }

    [[nodiscard]] PauliMaskView at(PauliMaskHandle h) const {
        size_t i = static_cast<size_t>(h);
        assert(i < signs_.size());
        return PauliMaskView(slice(x_, i), slice(z_, i), &signs_[i]);
    }

    [[nodiscard]] MutablePauliMaskView mut_at(PauliMaskHandle h) {
        size_t i = static_cast<size_t>(h);
        assert(i < signs_.size());
        return MutablePauliMaskView(slice(x_, i), slice(z_, i), &signs_[i]);
    }

  private:
    [[nodiscard]] std::span<uint64_t> slice(std::vector<uint64_t>& v, size_t i) {
        // std::span from an iterator + size is well-formed when num_words_ == 0,
        // even if v.data() is nullptr.
        return std::span<uint64_t>{v.begin() + i * num_words_, num_words_};
    }
    [[nodiscard]] std::span<const uint64_t> slice(const std::vector<uint64_t>& v, size_t i) const {
        return std::span<const uint64_t>{v.begin() + i * num_words_, num_words_};
    }

    uint32_t num_words_;
    // Capacity is fixed at construction. Resizing these vectors after
    // construction would invalidate any outstanding PauliMaskView /
    // MutablePauliMaskView, since their span/pointer fields reference
    // the underlying storage directly. Any future change that needs to
    // grow an arena post-construction must move to a stable-handle
    // representation (e.g. chunked storage) before doing so.
    std::vector<uint64_t> x_;
    std::vector<uint64_t> z_;
    std::vector<uint8_t> signs_;
};

}  // namespace clifft
