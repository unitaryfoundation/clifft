#include "clifft/sampling/batch/bits.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <utility>

namespace clifft::sampling {

namespace {

size_t packed_storage_bytes(size_t columns, size_t word_capacity) {
    if (columns != 0 && word_capacity > std::numeric_limits<size_t>::max() / columns) {
        throw std::length_error("packed batch bit-column allocation exceeds size_t range");
    }
    const size_t words = columns * word_capacity;
    if (words > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
        throw std::length_error("packed batch bit-column allocation exceeds size_t range");
    }
    return words * sizeof(uint64_t);
}

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

// The input bytes are eight bit columns across eight lanes. The output bytes
// are the corresponding eight lane rows. This orientation also preserves the
// little-endian column order used by Stim's b8 matrices.
uint64_t transpose_8x8(uint64_t bits) noexcept {
    uint64_t swap = (bits ^ (bits >> 7)) & 0x00AA00AA00AA00AAULL;
    bits ^= swap ^ (swap << 7);
    swap = (bits ^ (bits >> 14)) & 0x0000CCCC0000CCCCULL;
    bits ^= swap ^ (swap << 14);
    swap = (bits ^ (bits >> 28)) & 0x00000000F0F0F0F0ULL;
    return bits ^ swap ^ (swap << 28);
}

#ifndef NDEBUG
uint32_t count_lane_bits(std::span<const uint64_t> bits, uint32_t lanes) noexcept {
    const size_t live_words = packed_word_count(lanes);
    uint32_t count = 0;
    for (size_t word = 0; word < live_words; ++word) {
        const uint32_t remaining = lanes - static_cast<uint32_t>(word * 64);
        count += static_cast<uint32_t>(std::popcount(bits[word] & low_lane_mask(remaining)));
    }
    return count;
}
#endif

}  // namespace

size_t packed_word_count(uint32_t lanes) noexcept {
    return (static_cast<size_t>(lanes) + 63) / 64;
}

size_t packed_bit_columns_storage_bytes(size_t columns, uint32_t lane_capacity) {
    return PageAlignedAllocation::allocation_size(
        packed_storage_bytes(columns, packed_word_count(lane_capacity)),
        PageAlignedAllocation::Alignment::BasePage);
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

PackedBitColumns::PackedBitColumns(size_t columns, uint32_t lane_capacity)
    : columns_(columns),
      lane_capacity_(lane_capacity),
      word_capacity_(packed_word_count(lane_capacity)),
      storage_(packed_bit_columns_storage_bytes(columns_, lane_capacity_),
               PageAlignedAllocation::Alignment::BasePage),
      words_(static_cast<uint64_t*>(storage_.data())) {
    if (!storage_.zero_initialized()) {
        std::ranges::fill(std::span<uint64_t>(words_, columns_ * word_capacity_), uint64_t{0});
    }
}

std::span<uint64_t> PackedBitColumns::column(size_t column_index) noexcept {
    assert(column_index < columns_ && "packed bit column index must be in range");
    return std::span<uint64_t>(words_, columns_ * word_capacity_)
        .subspan(column_index * word_capacity_, word_capacity_);
}

std::span<const uint64_t> PackedBitColumns::column(size_t column_index) const noexcept {
    assert(column_index < columns_ && "packed bit column index must be in range");
    return std::span<const uint64_t>(words_, columns_ * word_capacity_)
        .subspan(column_index * word_capacity_, word_capacity_);
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
    std::ranges::fill(std::span<uint64_t>(words_, columns_ * word_capacity_), uint64_t{0});
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
#if defined(CLIFFT_USE_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
    for (size_t word = 0; word < word_capacity_; ++word) {
        destination[word] ^= source[word];
    }
}

void PackedBitColumns::write_unpacked_rows(uint32_t lanes, size_t columns,
                                           std::span<uint8_t> destination, size_t row_stride,
                                           size_t column_offset) const noexcept {
    assert(lanes <= lane_capacity_ && columns <= columns_ && column_offset <= row_stride &&
           columns <= row_stride - column_offset &&
           (lanes == 0 || destination.size() >= static_cast<size_t>(lanes) * row_stride) &&
           "unpacked row destination must cover every requested lane and column");
    for (uint32_t lane_base = 0; lane_base < lanes; lane_base += 8) {
        const size_t word = lane_base >> 6;
        const uint32_t shift = lane_base & 63;
        const uint32_t lane_count = std::min(uint32_t{8}, lanes - lane_base);
        for (size_t column_base = 0; column_base < columns; column_base += 8) {
            const size_t column_count = std::min(size_t{8}, columns - column_base);
            uint64_t column_bytes = 0;
            for (size_t column_offset_in_block = 0; column_offset_in_block < column_count;
                 ++column_offset_in_block) {
                const uint8_t lane_bits = static_cast<uint8_t>(
                    column(column_base + column_offset_in_block)[word] >> shift);
                column_bytes |= static_cast<uint64_t>(lane_bits) << (8 * column_offset_in_block);
            }
            const uint64_t row_bytes = transpose_8x8(column_bytes);
            for (uint32_t lane = 0; lane < lane_count; ++lane) {
                const uint8_t bits = static_cast<uint8_t>(row_bytes >> (8 * lane));
                uint8_t* row = destination.data() +
                               static_cast<size_t>(lane_base + lane) * row_stride + column_offset;
                for (size_t column = 0; column < column_count; ++column) {
                    row[column_base + column] = (bits >> column) & uint8_t{1};
                }
            }
        }
    }
}

void PackedBitColumns::write_packed_rows(uint32_t lanes, size_t columns,
                                         std::span<uint8_t> destination, size_t row_stride,
                                         size_t bit_offset) const noexcept {
    const size_t end_bit = bit_offset + columns;
    const size_t required_stride = end_bit / 8 + static_cast<size_t>((end_bit & 7) != 0);
    assert(lanes <= lane_capacity_ && columns <= columns_ && bit_offset <= end_bit &&
           row_stride >= required_stride &&
           (lanes == 0 || destination.size() >= static_cast<size_t>(lanes) * row_stride) &&
           "packed row destination must cover every requested lane and column");
    (void)required_stride;
    for (uint32_t lane_base = 0; lane_base < lanes; lane_base += 8) {
        const size_t word = lane_base >> 6;
        const uint32_t shift = lane_base & 63;
        const uint32_t lane_count = std::min(uint32_t{8}, lanes - lane_base);
        for (size_t column_base = 0; column_base < columns; column_base += 8) {
            const size_t column_count = std::min(size_t{8}, columns - column_base);
            uint64_t column_bytes = 0;
            for (size_t column_in_block = 0; column_in_block < column_count; ++column_in_block) {
                const uint8_t lane_bits =
                    static_cast<uint8_t>(column(column_base + column_in_block)[word] >> shift);
                column_bytes |= static_cast<uint64_t>(lane_bits) << (8 * column_in_block);
            }
            const uint64_t row_bytes = transpose_8x8(column_bytes);
            const size_t destination_bit = bit_offset + column_base;
            const size_t destination_byte = destination_bit >> 3;
            const uint32_t destination_shift = destination_bit & 7;
            for (uint32_t lane = 0; lane < lane_count; ++lane) {
                uint8_t* row =
                    destination.data() + static_cast<size_t>(lane_base + lane) * row_stride;
                const uint16_t bits = static_cast<uint8_t>(row_bytes >> (8 * lane));
                row[destination_byte] |= static_cast<uint8_t>(bits << destination_shift);
                if (destination_shift != 0 && destination_byte + 1 < row_stride &&
                    column_count > 8 - destination_shift) {
                    row[destination_byte + 1] |=
                        static_cast<uint8_t>(bits >> (8 - destination_shift));
                }
            }
        }
    }
}

void PackedBitColumns::compact(std::span<const uint64_t> keep_mask, uint32_t old_lanes,
                               uint32_t new_lanes, std::span<uint64_t> scratch) noexcept {
    assert(old_lanes <= lane_capacity_ && new_lanes <= old_lanes &&
           keep_mask.size() >= word_capacity_ && scratch.size() >= word_capacity_ &&
           count_lane_bits(keep_mask, old_lanes) == new_lanes &&
           "packed compaction inputs must describe the retained lanes");
    (void)new_lanes;
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
