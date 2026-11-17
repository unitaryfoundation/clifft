#pragma once

// PauliMaskArena: packed storage for sparse Pauli masks, indexed by an
// opaque PauliMaskHandle.
//
// Each handle owns one (X, Z, sign) mask stored as its nonzero words, so a
// mask on a wide register costs memory in proportion to its support. Words
// live in chunks that are never reallocated: a PauliMaskView returned by
// at() stays valid until that same handle is reassigned, even while other
// masks are appended or reassigned.
//
// Reassigning a mask reuses its words in place when the new mask is no
// larger than the handle's capacity and otherwise moves it to fresh storage;
// the old words are reclaimed when the arena is copied or destroyed.
//
// A view holds the sign it read: set_sign() and assign() on a handle do not
// update views of that handle taken earlier.

#include "clifft/util/mask_view.h"
#include "clifft/util/pauli_mask.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft {

/// Opaque handle into a PauliMaskArena.
enum class PauliMaskHandle : uint32_t {};

class PauliMaskArena {
  public:
    PauliMaskArena() = default;
    explicit PauliMaskArena(uint32_t num_qubits) : num_words_((num_qubits + 63) / 64) {}

    PauliMaskArena(PauliMaskArena&&) = default;
    PauliMaskArena& operator=(PauliMaskArena&&) = default;

    // A copy shares the layout of the original unless reassignment has
    // abandoned storage there, in which case it repacks the live masks.
    PauliMaskArena(const PauliMaskArena& other) : num_words_(other.num_words_) {
        if (other.abandoned_words_ == 0) {
            slots_ = other.slots_;
            chunks_ = other.chunks_;
            return;
        }
        slots_.reserve(other.slots_.size());
        size_t total = 0;
        for (const Slot& slot : other.slots_) {
            total += slot.count;
        }
        if (total != 0) {
            chunks_.emplace_back().reserve(total);
        }
        for (size_t i = 0; i < other.slots_.size(); ++i) {
            append(other.at(static_cast<PauliMaskHandle>(i)));
        }
    }
    PauliMaskArena& operator=(const PauliMaskArena& other) {
        if (this != &other) {
            *this = PauliMaskArena(other);
        }
        return *this;
    }

    /// Register width in 64-qubit words. Every stored word index is below it.
    [[nodiscard]] uint32_t num_words() const { return num_words_; }
    [[nodiscard]] size_t size() const { return slots_.size(); }

    [[nodiscard]] PauliMaskView at(PauliMaskHandle h) const {
        const Slot& slot = slot_at(h);
        return PauliMaskView{{words_of(slot), slot.count}, slot.sign != 0};
    }

    PauliMaskHandle append(PauliMaskView mask) {
        const size_t index = slots_.size();
        assert(index < ~uint32_t{0} && "PauliMaskArena handle space exhausted");
        slots_.emplace_back();
        store(slots_.back(), mask);
        return static_cast<PauliMaskHandle>(index);
    }

    /// Replace a stored mask. Invalidates views of this handle only.
    void assign(PauliMaskHandle h, PauliMaskView mask) {
        Slot& slot = slots_[static_cast<size_t>(h)];
        if (mask.words().data() == words_of(slot)) {
            // Self-assignment, possibly with a different sign.
            assert(mask.words().size() == slot.count);
            slot.sign = mask.sign();
            return;
        }
        store(slot, mask);
    }

    /// Replace a stored mask with a dense Pauli, keeping only its nonzero
    /// words. Invalidates views of this handle only.
    void assign(PauliMaskHandle h, MaskView x, MaskView z, bool sign) {
        assert(x.num_words() == z.num_words() && x.num_words() <= num_words_);
        size_t count = 0;
        for (uint32_t w = 0; w < x.num_words(); ++w) {
            count += (x.words[w] | z.words[w]) != 0;
        }
        Slot& slot = slots_[static_cast<size_t>(h)];
        if (count > slot.capacity) {
            allocate(slot, count);
        }
        PauliWord* out = count == 0 ? nullptr : chunks_[slot.chunk].data() + slot.offset;
        for (uint32_t w = 0; w < x.num_words(); ++w) {
            if ((x.words[w] | z.words[w]) != 0) {
                *out++ = {w, x.words[w], z.words[w]};
            }
        }
        slot.count = static_cast<uint32_t>(count);
        slot.sign = sign;
    }

    void set_sign(PauliMaskHandle h, bool sign) { slots_[static_cast<size_t>(h)].sign = sign; }

    /// Bytes needed to store `masks` masks holding `words` nonzero words in total.
    [[nodiscard]] static constexpr size_t storage_bytes(size_t masks, size_t words) {
        return masks * sizeof(Slot) + words * sizeof(PauliWord);
    }

  private:
    // First chunk size in words; later chunks double up to kMaxChunkWords so
    // small modules stay small and large ones make few allocations.
    static constexpr size_t kMinChunkWords = 64;
    static constexpr size_t kMaxChunkWords = size_t{1} << 16;

    struct Slot {
        uint32_t chunk = 0;
        uint32_t offset = 0;
        uint32_t count = 0;
        uint32_t capacity : 31 = 0;
        uint32_t sign : 1 = 0;
    };

    [[nodiscard]] const Slot& slot_at(PauliMaskHandle h) const {
        assert(static_cast<size_t>(h) < slots_.size());
        return slots_[static_cast<size_t>(h)];
    }

    [[nodiscard]] const PauliWord* words_of(const Slot& slot) const {
        return slot.capacity == 0 ? nullptr : chunks_[slot.chunk].data() + slot.offset;
    }

    void store(Slot& slot, PauliMaskView mask) {
        const std::span<const PauliWord> words = mask.words();
        assert((words.empty() || words.back().index < num_words_) &&
               "Pauli mask is wider than the arena's register");
        assert(words.size() < (uint32_t{1} << 31));
        if (words.size() > slot.capacity) {
            allocate(slot, words.size());
        }
        if (!words.empty()) {
            std::ranges::copy(words, chunks_[slot.chunk].data() + slot.offset);
        }
        slot.count = static_cast<uint32_t>(words.size());
        slot.sign = mask.sign();
    }

    // Point the slot at `count` unused words. A chunk is filled only within
    // its reserved capacity, so its buffer never moves.
    void allocate(Slot& slot, size_t count) {
        if (chunks_.empty() || chunks_.back().capacity() - chunks_.back().size() < count) {
            const size_t grown = chunks_.empty() ? kMinChunkWords : 2 * chunks_.back().capacity();
            chunks_.emplace_back().reserve(std::max(count, std::min(grown, kMaxChunkWords)));
        }
        std::vector<PauliWord>& chunk = chunks_.back();
        abandoned_words_ += slot.capacity;
        slot.chunk = static_cast<uint32_t>(chunks_.size() - 1);
        slot.offset = static_cast<uint32_t>(chunk.size());
        slot.capacity = static_cast<uint32_t>(count);
        chunk.resize(chunk.size() + count);
    }

    uint32_t num_words_ = 0;
    std::vector<Slot> slots_;
    std::vector<std::vector<PauliWord>> chunks_;
    // Words left behind when a reassigned mask outgrew its storage.
    size_t abandoned_words_ = 0;
};

}  // namespace clifft
