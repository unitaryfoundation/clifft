#pragma once

#include "clifft/util/page_allocation.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace clifft::sampling {

// Non-owning handle to the words of a PackedBitColumns, for loops that set many
// bits and must not re-read the owner between them. It copies the storage
// pointer and shape and borrows the owner's storage, so it is valid until the
// owner is destroyed, assigned, or moved from; the owner allocates once at
// construction and never reallocates. Copying the view and its bit operations
// never allocate or throw.
class MutableBitColumnsView {
  public:
    MutableBitColumnsView() = default;

    [[nodiscard]] bool bit(size_t column, uint32_t lane) const noexcept {
        return ((words_[word_index(column, lane)] >> (lane & 63)) & uint64_t{1}) != 0;
    }

    void set_bit(size_t column, uint32_t lane) const noexcept {
        words_[word_index(column, lane)] |= uint64_t{1} << (lane & 63);
    }

  private:
    friend class PackedBitColumns;

    MutableBitColumnsView(uint64_t* words, size_t columns, uint32_t lane_capacity,
                          size_t word_capacity) noexcept
        : words_(words),
          columns_(columns),
          word_capacity_(word_capacity),
          lane_capacity_(lane_capacity) {}

    // Column c and lane s live at words_[c * word_capacity_ + s / 64]; lanes at
    // or beyond lane_capacity_ in the final word are padding.
    [[nodiscard]] size_t word_index(size_t column, uint32_t lane) const noexcept {
        assert(column < columns_ && "packed bit column index must be in range");
        assert(lane < lane_capacity_ && "packed bit lane must be in range");
        return column * word_capacity_ + (lane >> 6);
    }

    uint64_t* words_ = nullptr;
    size_t columns_ = 0;
    size_t word_capacity_ = 0;
    uint32_t lane_capacity_ = 0;
};

// Dense bit matrix with one stored Boolean value per column and one shot per
// lane:
//
//                         shot lane
//   value/column c  [0 1 ... 63] [64 ...]
//
// Column c and lane s live at words_[c * word_capacity_ + s / 64]. Construction
// owns all allocation, so hot operations only overwrite existing words.
class PackedBitColumns {
  public:
    PackedBitColumns() = default;
    PackedBitColumns(size_t columns, uint32_t lane_capacity);

    [[nodiscard]] size_t num_columns() const noexcept { return columns_; }

    // See MutableBitColumnsView for the lifetime rule.
    [[nodiscard]] MutableBitColumnsView mutable_view() noexcept {
        return {words_, columns_, lane_capacity_, word_capacity_};
    }

    [[nodiscard]] std::span<uint64_t> column(size_t column) noexcept;
    [[nodiscard]] std::span<const uint64_t> column(size_t column) const noexcept;

    [[nodiscard]] bool bit(size_t column, uint32_t lane) const noexcept;
    void set_bit(size_t column, uint32_t lane) noexcept;
    void clear() noexcept;

    // Replace a column with bits, restricting padding and rejected lanes to
    // zero. Source and live_mask must cover the fixed word capacity.
    void assign(size_t column, std::span<const uint64_t> source,
                std::span<const uint64_t> live_mask) noexcept;
    void assign_xor(size_t column, std::span<const uint64_t> left, std::span<const uint64_t> right,
                    std::span<const uint64_t> live_mask) noexcept;
    void copy(size_t destination, size_t source) noexcept;
    void xor_into(size_t column, std::span<const uint64_t> source) noexcept;

    // Stable-compacts every column using keep_mask. scratch is one fixed-size
    // word row prepared by the owning executor before dispatch.
    void compact(std::span<const uint64_t> keep_mask, uint32_t old_lanes, uint32_t new_lanes,
                 std::span<uint64_t> scratch) noexcept;

    // Columns outside the prefix retain their original contents and lane order.
    void compact_prefix(size_t columns, std::span<const uint64_t> keep_mask, uint32_t old_lanes,
                        uint32_t new_lanes, std::span<uint64_t> scratch) noexcept;

  private:
    // Shares addressing with mutable_view(); words_ is non-const, so a const
    // owner can still build the view for reads.
    [[nodiscard]] MutableBitColumnsView view() const noexcept {
        return {words_, columns_, lane_capacity_, word_capacity_};
    }

    size_t columns_ = 0;
    uint32_t lane_capacity_ = 0;
    size_t word_capacity_ = 0;
    PageAlignedAllocation storage_;
    uint64_t* words_ = nullptr;
};

[[nodiscard]] size_t packed_word_count(uint32_t lanes) noexcept;
[[nodiscard]] size_t packed_bit_columns_storage_bytes(size_t columns, uint32_t lane_capacity);
[[nodiscard]] uint64_t low_lane_mask(uint32_t bits) noexcept;
void fill_low_lane_mask(std::span<uint64_t> output, uint32_t lanes) noexcept;

}  // namespace clifft::sampling
