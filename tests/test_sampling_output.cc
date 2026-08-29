#include "clifft/sampling/output_writer.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <sstream>
#include <string>

using clifft::sampling::SamplingFileFormat;
using clifft::sampling::SamplingRowWriter;

TEST_CASE("Sampling row writer streams common Stim formats") {
    constexpr std::array<uint8_t, 6> rows{0x55, 0x01, 0xAA, 0x02, 0xFF, 0x03};

    std::ostringstream text;
    SamplingRowWriter text_writer(text, SamplingFileFormat::Format01, 10, 3);
    text_writer.write_packed_rows(rows, 2, 2);
    text_writer.write_packed_rows(std::span<const uint8_t>(rows).subspan(4), 1, 2);
    REQUIRE(text.str() == "1010101010\n0101010101\n1111111111\n");

    std::ostringstream binary;
    SamplingRowWriter binary_writer(binary, SamplingFileFormat::B8, 10, 3);
    constexpr std::array<uint8_t, 9> padded_rows{0x55, 0x01, 0xCC, 0xAA, 0x02,
                                                 0xCC, 0xFF, 0x03, 0xCC};
    binary_writer.write_packed_rows(padded_rows, 3, 3);
    REQUIRE(binary.str() == std::string("\x55\x01\xAA\x02\xFF\x03", 6));
}

TEST_CASE("Sampling row writer preserves empty row semantics") {
    std::ostringstream text;
    SamplingRowWriter text_writer(text, SamplingFileFormat::Format01, 0, 3);
    text_writer.write_packed_rows({}, 3, 0);
    REQUIRE(text.str() == "\n\n\n");

    std::ostringstream binary;
    SamplingRowWriter binary_writer(binary, SamplingFileFormat::B8, 0, 3);
    binary_writer.write_packed_rows({}, 3, 0);
    REQUIRE(binary.str().empty());
}

TEST_CASE("Sampling row writer validates retained batch bounds") {
    std::ostringstream output;
    SamplingRowWriter writer(output, SamplingFileFormat::B8, 9, 2);
    constexpr std::array<uint8_t, 4> rows{};
    REQUIRE_THROWS_WITH(writer.write_packed_rows(rows, 3, 2),
                        "sampling writer batch exceeds retained capacity");
    REQUIRE_THROWS_WITH(writer.write_packed_rows(rows, 2, 1),
                        "sampling writer row stride is too small");
}
