#pragma once

// PauliMaskArena: packed storage for sparse Pauli masks, indexed by an
// opaque PauliMaskHandle.
//
// Each handle owns one (X, Z, sign) mask stored as its nonzero words, so a
// mask on a wide register costs memory in proportion to its support. Storage
// never moves: a PauliMaskView returned by at() stays valid until that same
// handle is reassigned, even while other masks are appended or reassigned.
//
// A view holds the size and sign it read: set_sign() and assign() on a
// handle do not update views of that handle taken earlier.
//
// Each mask has a slot that points at its words in a chunk. Reassigning a
// mask reuses its words in place when the new mask is no larger than the
// slot's capacity and otherwise moves it to fresh storage; the old words are
// reclaimed when the arena is copied or destroyed.

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

    // A copy packs the masks into one chunk, which also drops storage that
    // reassignment abandoned in the original.
    PauliMaskArena(const PauliMaskArena& other)
        : num_words_(other.num_words_), slots_(other.slots_) {
        size_t total = 0;
        for (const Slot& slot : slots_) {
            total += slot.count;
        }
        PauliWord* out = total == 0 ? nullptr : chunks_.emplace_back(total).data();
        for (Slot& slot : slots_) {
            // Each copied slot still points at the original's words. Most
            // masks on a narrow register hold exactly one word.
            if (slot.count == 1) {
                *out = *slot.words;
            } else {
                std::copy_n(slot.words, slot.count, out);
            }
            slot.words = slot.count == 0 ? nullptr : out;
            slot.capacity = slot.count;
            out += slot.count;
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
        assert(static_cast<size_t>(h) < slots_.size());
        const Slot& slot = slots_[static_cast<size_t>(h)];
        return PauliMaskView{{slot.words, slot.count}, slot.sign != 0};
    }

    PauliMaskHandle append(PauliMaskView mask) {
        const size_t index = slots_.size();
        assert(index < ~uint32_t{0} && "PauliMaskArena handle space exhausted");
        const std::span<const PauliWord> words = mask.words();
        assert((words.empty() || words.back().index < num_words_) &&
               "Pauli mask is wider than the arena's register");
        Slot& slot = slots_.emplace_back();
        if (!words.empty()) {
            std::vector<PauliWord>& chunk = chunk_with_room(words.size());
            slot.words = chunk.data() + chunk.size();
            chunk.insert(chunk.end(), words.begin(), words.end());
            slot.count = static_cast<uint32_t>(words.size());
            slot.capacity = slot.count;
        }
        slot.sign = mask.sign();
        return static_cast<PauliMaskHandle>(index);
    }

    /// Replace a stored mask. Invalidates views of this handle only.
    void assign(PauliMaskHandle h, PauliMaskView mask) {
        assert(static_cast<size_t>(h) < slots_.size());
        const std::span<const PauliWord> words = mask.words();
        assert((words.empty() || words.back().index < num_words_) &&
               "Pauli mask is wider than the arena's register");
        Slot& slot = slots_[static_cast<size_t>(h)];
        if (words.data() == slot.words) {
            // The mask's own words, possibly with a different sign.
            assert(words.size() == slot.count);
        } else {
            if (words.size() > slot.capacity) {
                allocate(slot, words.size());
            }
            std::ranges::copy(words, slot.words);
            slot.count = static_cast<uint32_t>(words.size());
        }
        slot.sign = mask.sign();
    }

    /// Replace a stored mask with a dense Pauli, keeping only its nonzero
    /// words. Invalidates views of this handle only.
    void assign(PauliMaskHandle h, MaskView x, MaskView z, bool sign) {
        assert(static_cast<size_t>(h) < slots_.size());
        assert(x.num_words() == z.num_words() && x.num_words() <= num_words_);
        size_t count = 0;
        for (uint32_t w = 0; w < x.num_words(); ++w) {
            count += (x.words[w] | z.words[w]) != 0;
        }
        Slot& slot = slots_[static_cast<size_t>(h)];
        if (count > slot.capacity) {
            allocate(slot, count);
        }
        PauliWord* out = slot.words;
        for (uint32_t w = 0; w < x.num_words(); ++w) {
            if ((x.words[w] | z.words[w]) != 0) {
                *out++ = {w, x.words[w], z.words[w]};
            }
        }
        slot.count = static_cast<uint32_t>(count);
        slot.sign = sign;
    }

    void set_sign(PauliMaskHandle h, bool sign) {
        assert(static_cast<size_t>(h) < slots_.size());
        slots_[static_cast<size_t>(h)].sign = sign;
    }

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
        // Null while the slot has no capacity. Points into a chunk otherwise.
        PauliWord* words = nullptr;
        uint32_t count = 0;
        uint32_t capacity : 31 = 0;
        uint32_t sign : 1 = 0;
    };

    // The chunk that can take `count` more words. A chunk is filled only
    // within its reserved capacity, so its buffer never moves.
    std::vector<PauliWord>& chunk_with_room(size_t count) {
        assert(count < (uint32_t{1} << 31));
        if (chunks_.empty() || chunks_.back().capacity() - chunks_.back().size() < count) {
            const size_t grown = chunks_.empty() ? kMinChunkWords : 2 * chunks_.back().capacity();
            chunks_.emplace_back().reserve(std::max(count, std::min(grown, kMaxChunkWords)));
        }
        return chunks_.back();
    }

    // Point the slot at `count` unused words.
    void allocate(Slot& slot, size_t count) {
        std::vector<PauliWord>& chunk = chunk_with_room(count);
        const size_t offset = chunk.size();
        chunk.resize(offset + count);
        slot.words = chunk.data() + offset;
        slot.capacity = static_cast<uint32_t>(count);
    }

    uint32_t num_words_ = 0;
    std::vector<Slot> slots_;
    std::vector<std::vector<PauliWord>> chunks_;
};

}  // namespace clifft
