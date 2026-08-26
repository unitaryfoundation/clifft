#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::sampling {

// Fixed-capacity symbol-major bit columns used by packed cross-shot execution.
// Construction owns all allocation; hot operations only overwrite existing
// words. Column c and lane s live at words_[c * word_capacity_ + s / 64].
class PackedBitColumns {
  public:
    PackedBitColumns() = default;
    PackedBitColumns(size_t columns, uint32_t lane_capacity);

    [[nodiscard]] size_t num_columns() const noexcept { return columns_; }

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

  private:
    size_t columns_ = 0;
    uint32_t lane_capacity_ = 0;
    size_t word_capacity_ = 0;
    std::vector<uint64_t> words_;
};

[[nodiscard]] size_t packed_word_count(uint32_t lanes) noexcept;
[[nodiscard]] uint64_t low_lane_mask(uint32_t bits) noexcept;
void fill_low_lane_mask(std::span<uint64_t> output, uint32_t lanes) noexcept;

}  // namespace clifft::sampling
