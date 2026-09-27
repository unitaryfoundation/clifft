#include "clifft/sampling/batch/bits.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <vector>

using clifft::sampling::fill_low_lane_mask;
using clifft::sampling::packed_bit_columns_storage_bytes;
using clifft::sampling::packed_word_count;
using clifft::sampling::PackedBitColumns;

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
