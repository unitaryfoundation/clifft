#include "clifft/sampling/batch/bits.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <span>
#include <vector>

using clifft::sampling::fill_low_lane_mask;
using clifft::sampling::MutableBitColumnsView;
using clifft::sampling::packed_bit_columns_storage_bytes;
using clifft::sampling::packed_word_count;
using clifft::sampling::PackedBitColumns;
using clifft::sampling::transpose_bit_columns_to_rows;

TEST_CASE("Packed batch columns preserve lane boundaries") {
    for (uint32_t lanes : std::array<uint32_t, 7>{1, 2, 63, 64, 65, 127, 128}) {
        PackedBitColumns columns(3, lanes);
        columns.set_bit(0, 0);
        columns.set_bit(1, lanes - 1);
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            if ((lane % 3) == 1) {
                columns.set_bit(2, lane);
            }
        }
        CAPTURE(lanes);
        REQUIRE(columns.bit(0, 0));
        REQUIRE(columns.bit(1, lanes - 1));
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            REQUIRE(columns.bit(2, lane) == ((lane % 3) == 1));
        }
    }
}

TEST_CASE("Mutable bit column views share owner addressing") {
    PackedBitColumns bits(3, 70);
    const MutableBitColumnsView view = bits.mutable_view();

    view.set_bit(2, 69);
    view.set_bit(0, 0);

    REQUIRE(bits.bit(2, 69));
    REQUIRE(bits.bit(0, 0));
    REQUIRE_FALSE(bits.bit(2, 68));
    REQUIRE_FALSE(bits.bit(1, 69));
    REQUIRE(view.bit(2, 69));
    REQUIRE_FALSE(view.bit(1, 69));

    // Lane 69 is bit 5 of the second word; every other word stays zero.
    REQUIRE(bits.column(0)[0] == 1);
    REQUIRE(bits.column(0)[1] == 0);
    REQUIRE(bits.column(1)[0] == 0);
    REQUIRE(bits.column(1)[1] == 0);
    REQUIRE(bits.column(2)[0] == 0);
    REQUIRE(bits.column(2)[1] == (uint64_t{1} << 5));

    // The view holds addressing only, so owner writes made after it was taken
    // are visible through it.
    bits.set_bit(1, 64);
    REQUIRE(view.bit(1, 64));
    REQUIRE(bits.column(1)[1] == 1);
}

TEST_CASE("Packed batch column footprint includes page alignment") {
    constexpr size_t columns = 3;
    constexpr uint32_t lanes = 65;
    const size_t raw_bytes = columns * packed_word_count(lanes) * sizeof(uint64_t);
    const size_t storage_bytes = packed_bit_columns_storage_bytes(columns, lanes);
    REQUIRE(storage_bytes >= raw_bytes);
    REQUIRE(storage_bytes % clifft::PageAlignedAllocation::kBaseAlignment == 0);
    REQUIRE(packed_bit_columns_storage_bytes(0, lanes) == 0);
}

TEST_CASE("Packed batch column compaction preserves unselected columns") {
    constexpr uint32_t lanes = 130;
    PackedBitColumns columns(4, lanes);
    const size_t compact_columns = GENERATE(0U, 2U, 4U);
    CAPTURE(compact_columns);
    std::vector<uint64_t> keep(packed_word_count(lanes), 0);
    std::vector<uint32_t> sources;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        if ((lane % 5) != 1 && lane != 64) {
            keep[lane >> 6] |= uint64_t{1} << (lane & 63);
            sources.push_back(lane);
        }
        for (size_t column = 0; column < columns.num_columns(); ++column) {
            if (((lane * 7 + static_cast<uint32_t>(column)) % 11) < 4) {
                columns.set_bit(column, lane);
            }
        }
    }

    std::vector<uint64_t> scratch(packed_word_count(lanes), 0);
    if (compact_columns == columns.num_columns()) {
        columns.compact(keep, lanes, static_cast<uint32_t>(sources.size()), scratch);
    } else {
        columns.compact_prefix(compact_columns, keep, lanes, static_cast<uint32_t>(sources.size()),
                               scratch);
    }
    for (size_t column = 0; column < columns.num_columns(); ++column) {
        for (uint32_t destination = 0; destination < lanes; ++destination) {
            CAPTURE(column, destination);
            if (column < compact_columns && destination >= sources.size()) {
                REQUIRE_FALSE(columns.bit(column, destination));
            } else {
                const uint32_t source =
                    column < compact_columns ? sources[destination] : destination;
                REQUIRE(columns.bit(column, destination) ==
                        (((source * 7 + static_cast<uint32_t>(column)) % 11) < 4));
            }
        }
    }
}

TEST_CASE("Packed batch low masks clear padding") {
    std::vector<uint64_t> words(3, ~uint64_t{0});
    fill_low_lane_mask(words, 65);
    REQUIRE(words[0] == ~uint64_t{0});
    REQUIRE(words[1] == 1);
    REQUIRE(words[2] == 0);
}

namespace {

uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

constexpr uint8_t kSentinel = 0xAA;

}  // namespace

TEST_CASE("Transposed bit columns match per-bit reads") {
    const size_t num_columns = GENERATE(0U, 1U, 7U, 63U, 64U, 65U, 130U, 385U);
    const uint32_t lanes = GENERATE(1U, 63U, 64U, 65U, 100U, 2048U);
    CAPTURE(num_columns, lanes);

    PackedBitColumns columns(num_columns, lanes);
    uint64_t state = (static_cast<uint64_t>(num_columns) << 32) | lanes;
    for (size_t column = 0; column < num_columns; ++column) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            if ((splitmix64(state) & 3) == 0) {
                columns.set_bit(column, lane);
            }
        }
    }

    // 37 is coprime with every lane count above, so this permutes the rows.
    std::vector<uint32_t> rows(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        rows[lane] = (lane * 37) % lanes;
    }
    const size_t stride = num_columns + 3;
    std::vector<uint8_t> output(static_cast<size_t>(lanes) * stride, kSentinel);

    transpose_bit_columns_to_rows(columns, num_columns, rows, output, stride);

    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const size_t row_start = static_cast<size_t>(rows[lane]) * stride;
        for (size_t column = 0; column < num_columns; ++column) {
            REQUIRE(output[row_start + column] == static_cast<uint8_t>(columns.bit(column, lane)));
        }
        for (size_t padding = num_columns; padding < stride; ++padding) {
            REQUIRE(output[row_start + padding] == kSentinel);
        }
    }
}

TEST_CASE("Transposed bit columns ignore lanes beyond the requested rows") {
    constexpr uint32_t lane_capacity = 128;
    constexpr uint32_t live_lanes = 70;
    constexpr uint32_t output_rows = 100;
    constexpr size_t num_columns = 70;
    constexpr size_t stride = num_columns + 3;

    PackedBitColumns columns(num_columns, lane_capacity);
    uint64_t state = 12345;
    for (size_t column = 0; column < num_columns; ++column) {
        for (uint32_t lane = 0; lane < lane_capacity; ++lane) {
            if ((splitmix64(state) & 1) == 0) {
                columns.set_bit(column, lane);
            }
        }
    }
    // Lanes 70..127 are set in every column so stale bits would be visible.
    for (size_t column = 0; column < num_columns; ++column) {
        for (uint32_t lane = live_lanes; lane < lane_capacity; ++lane) {
            columns.set_bit(column, lane);
        }
    }

    std::vector<uint32_t> rows(live_lanes);
    for (uint32_t lane = 0; lane < live_lanes; ++lane) {
        rows[lane] = (lane * 37) % live_lanes;
    }
    std::vector<uint8_t> output(static_cast<size_t>(output_rows) * stride, kSentinel);

    transpose_bit_columns_to_rows(columns, num_columns, rows, output, stride);

    for (uint32_t lane = 0; lane < live_lanes; ++lane) {
        const size_t row_start = static_cast<size_t>(rows[lane]) * stride;
        for (size_t column = 0; column < num_columns; ++column) {
            REQUIRE(output[row_start + column] == static_cast<uint8_t>(columns.bit(column, lane)));
        }
        for (size_t padding = num_columns; padding < stride; ++padding) {
            REQUIRE(output[row_start + padding] == kSentinel);
        }
    }
    for (size_t index = static_cast<size_t>(live_lanes) * stride; index < output.size(); ++index) {
        REQUIRE(output[index] == kSentinel);
    }
}
