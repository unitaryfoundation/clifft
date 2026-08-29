#include "clifft/sampling/batch/bits.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
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

TEST_CASE("Packed batch columns compact every sidecar stably") {
    constexpr uint32_t lanes = 130;
    PackedBitColumns columns(4, lanes);
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
    columns.compact(keep, lanes, static_cast<uint32_t>(sources.size()), scratch);
    for (size_t column = 0; column < columns.num_columns(); ++column) {
        for (uint32_t destination = 0; destination < sources.size(); ++destination) {
            CAPTURE(column, destination, sources[destination]);
            REQUIRE(columns.bit(column, destination) ==
                    (((sources[destination] * 7 + static_cast<uint32_t>(column)) % 11) < 4));
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

TEST_CASE("Packed batch columns transpose directly into row outputs") {
    constexpr size_t columns = 13;
    for (uint32_t lanes : std::array<uint32_t, 8>{1, 7, 8, 9, 63, 64, 65, 130}) {
        PackedBitColumns values(columns, lanes);
        for (size_t column = 0; column < columns; ++column) {
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                if (((lane * 17 + column * 5) % 11) < 4) {
                    values.set_bit(column, lane);
                }
            }
        }

        constexpr size_t unpacked_offset = 2;
        constexpr size_t unpacked_stride = unpacked_offset + columns + 1;
        std::vector<uint8_t> unpacked(static_cast<size_t>(lanes) * unpacked_stride, 0xA5);
        values.write_unpacked_rows(lanes, columns, unpacked, unpacked_stride, unpacked_offset);

        constexpr size_t packed_offset = 3;
        constexpr size_t packed_stride = 3;
        std::vector<uint8_t> packed(static_cast<size_t>(lanes) * packed_stride, 0);
        values.write_packed_rows(lanes, columns, packed, packed_stride, packed_offset);

        CAPTURE(lanes);
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            REQUIRE(unpacked[static_cast<size_t>(lane) * unpacked_stride] == 0xA5);
            REQUIRE(unpacked[static_cast<size_t>(lane) * unpacked_stride + 1] == 0xA5);
            for (size_t column = 0; column < columns; ++column) {
                const bool expected = values.bit(column, lane);
                CAPTURE(lane, column);
                REQUIRE(unpacked[static_cast<size_t>(lane) * unpacked_stride + unpacked_offset +
                                 column] == static_cast<uint8_t>(expected));
                REQUIRE(((packed[static_cast<size_t>(lane) * packed_stride +
                                 ((packed_offset + column) >> 3)] >>
                          ((packed_offset + column) & 7)) &
                         uint8_t{1}) == static_cast<uint8_t>(expected));
            }
        }
    }
}
