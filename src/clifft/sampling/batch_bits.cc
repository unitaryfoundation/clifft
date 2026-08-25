#include "clifft/sampling/batch_bits.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <utility>

namespace clifft::sampling {

namespace {

uint64_t compress_bits_portable(uint64_t bits, uint64_t keep) noexcept {
    uint64_t output = 0;
    uint64_t destination = 1;
    while (keep != 0) {
        const uint64_t source = keep & (~keep + 1);
        if ((bits & source) != 0) {
            output |= destination;
        }
        keep &= keep - 1;
        destination <<= 1;
    }
    return output;
}

}  // namespace

size_t packed_word_count(uint32_t lanes) noexcept {
    return (static_cast<size_t>(lanes) + 63) / 64;
}

uint64_t low_lane_mask(uint32_t bits) noexcept {
    if (bits == 0) {
        return 0;
    }
    if (bits >= 64) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (uint64_t{1} << bits) - 1;
}

void fill_low_lane_mask(std::span<uint64_t> output, uint32_t lanes) noexcept {
    const size_t live_words = packed_word_count(lanes);
    for (size_t word = 0; word < live_words; ++word) {
        const uint32_t remaining = lanes - static_cast<uint32_t>(word * 64);
        output[word] = low_lane_mask(remaining);
    }
    std::fill(output.begin() + static_cast<std::ptrdiff_t>(live_words), output.end(), uint64_t{0});
}

uint32_t count_lane_bits(std::span<const uint64_t> bits, uint32_t lanes) noexcept {
    const size_t live_words = packed_word_count(lanes);
    uint32_t count = 0;
    for (size_t word = 0; word < live_words; ++word) {
        const uint32_t remaining = lanes - static_cast<uint32_t>(word * 64);
        count += static_cast<uint32_t>(std::popcount(bits[word] & low_lane_mask(remaining)));
    }
    return count;
}

PackedBitColumns::PackedBitColumns(size_t columns, uint32_t lane_capacity)
    : columns_(columns),
      lane_capacity_(lane_capacity),
      word_capacity_(packed_word_count(lane_capacity)) {
    if (columns_ != 0 && word_capacity_ > std::numeric_limits<size_t>::max() / columns_) {
        throw std::length_error("packed batch bit-column allocation exceeds size_t range");
    }
    words_.resize(columns_ * word_capacity_, 0);
}

std::span<uint64_t> PackedBitColumns::column(size_t column_index) noexcept {
    assert(column_index < columns_ && "packed bit column index must be in range");
    return std::span<uint64_t>(words_).subspan(column_index * word_capacity_, word_capacity_);
}

std::span<const uint64_t> PackedBitColumns::column(size_t column_index) const noexcept {
    assert(column_index < columns_ && "packed bit column index must be in range");
    return std::span<const uint64_t>(words_).subspan(column_index * word_capacity_, word_capacity_);
}

bool PackedBitColumns::bit(size_t column_index, uint32_t lane) const noexcept {
    assert(lane < lane_capacity_ && "packed bit lane must be in range");
    return ((column(column_index)[lane >> 6] >> (lane & 63)) & uint64_t{1}) != 0;
}

void PackedBitColumns::set_bit(size_t column_index, uint32_t lane) noexcept {
    assert(lane < lane_capacity_ && "packed bit lane must be in range");
    column(column_index)[lane >> 6] |= uint64_t{1} << (lane & 63);
}

void PackedBitColumns::clear() noexcept {
    std::ranges::fill(words_, uint64_t{0});
}

void PackedBitColumns::assign(size_t column_index, std::span<const uint64_t> source,
                              std::span<const uint64_t> live_mask) noexcept {
    assert(source.size() >= word_capacity_ && live_mask.size() >= word_capacity_ &&
           "packed assignment inputs must cover the fixed word capacity");
    std::span<uint64_t> destination = column(column_index);
    for (size_t word = 0; word < word_capacity_; ++word) {
        destination[word] = source[word] & live_mask[word];
    }
}

void PackedBitColumns::assign_xor(size_t column_index, std::span<const uint64_t> left,
                                  std::span<const uint64_t> right,
                                  std::span<const uint64_t> live_mask) noexcept {
    assert(left.size() >= word_capacity_ && right.size() >= word_capacity_ &&
           live_mask.size() >= word_capacity_ &&
           "packed XOR assignment inputs must cover the fixed word capacity");
    std::span<uint64_t> destination = column(column_index);
    for (size_t word = 0; word < word_capacity_; ++word) {
        destination[word] = (left[word] ^ right[word]) & live_mask[word];
    }
}

void PackedBitColumns::copy(size_t destination_index, size_t source_index) noexcept {
    const std::span<const uint64_t> source = std::as_const(*this).column(source_index);
    std::ranges::copy(source, column(destination_index).begin());
}

void PackedBitColumns::xor_into(size_t column_index, std::span<const uint64_t> source) noexcept {
    assert(source.size() >= word_capacity_ &&
           "packed XOR input must cover the fixed word capacity");
    std::span<uint64_t> destination = column(column_index);
    for (size_t word = 0; word < word_capacity_; ++word) {
        destination[word] ^= source[word];
    }
}

void PackedBitColumns::compact(std::span<const uint64_t> keep_mask, uint32_t old_lanes,
                               uint32_t new_lanes, std::span<uint64_t> scratch) noexcept {
    assert(old_lanes <= lane_capacity_ && new_lanes <= old_lanes &&
           keep_mask.size() >= word_capacity_ && scratch.size() >= word_capacity_ &&
           count_lane_bits(keep_mask, old_lanes) == new_lanes &&
           "packed compaction inputs must describe the retained lanes");
    const size_t old_words = packed_word_count(old_lanes);
    for (size_t column_index = 0; column_index < columns_; ++column_index) {
        std::ranges::fill(scratch, uint64_t{0});
        const std::span<const uint64_t> source = column(column_index);
        uint32_t destination_bit = 0;
        for (size_t source_word = 0; source_word < old_words; ++source_word) {
            const uint32_t remaining = old_lanes - static_cast<uint32_t>(source_word * 64);
            const uint64_t keep = keep_mask[source_word] & low_lane_mask(remaining);
            const uint32_t kept = static_cast<uint32_t>(std::popcount(keep));
            if (kept == 0) {
                continue;
            }
            const uint64_t compressed = compress_bits_portable(source[source_word], keep);
            const size_t destination_word = destination_bit >> 6;
            const uint32_t shift = destination_bit & 63;
            scratch[destination_word] |= compressed << shift;
            if (shift != 0 && kept > 64 - shift) {
                scratch[destination_word + 1] |= compressed >> (64 - shift);
            }
            destination_bit += kept;
        }
        assert(destination_bit == new_lanes && "packed compaction must retain every live lane");
        std::ranges::copy(scratch, column(column_index).begin());
    }
}

}  // namespace clifft::sampling
