#include "clifft/sampling/inspection_format.h"

#include <array>
#include <bit>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace clifft::sampling {

namespace {

bool bit_identical(double left, double right) {
    uint64_t left_bits = 0;
    uint64_t right_bits = 0;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

std::string format_with_precision(double value, int precision) {
    std::array<char, 64> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.*g", precision, value);
    return std::string(buffer.data());
}

}  // namespace

std::string format_double_roundtrip(double value) {
    for (const int precision : {15, 16, 17}) {
        std::string candidate = format_with_precision(value, precision);
        if (bit_identical(std::strtod(candidate.c_str(), nullptr), value)) {
            return candidate;
        }
    }
    return format_with_precision(value, 17);
}

std::string format_pauli_product(uint64_t x, uint64_t z) {
    return format_pauli_product(std::span<const uint64_t>(&x, 1), std::span<const uint64_t>(&z, 1));
}

std::string format_pauli_product(std::span<const uint64_t> x, std::span<const uint64_t> z) {
    assert(x.size() == z.size());
    std::string out;
    for (size_t word = 0; word < x.size(); ++word) {
        uint64_t remaining = x[word] | z[word];
        while (remaining != 0) {
            const int bit_index = std::countr_zero(remaining);
            const uint64_t bit = uint64_t{1} << bit_index;
            remaining &= remaining - 1;
            if (!out.empty()) {
                out += '*';
            }
            if ((x[word] & bit) != 0 && (z[word] & bit) != 0) {
                out += 'Y';
            } else if ((x[word] & bit) != 0) {
                out += 'X';
            } else {
                out += 'Z';
            }
            out += std::to_string(64 * word + static_cast<size_t>(bit_index));
        }
    }
    return out.empty() ? "I" : out;
}

std::string format_width_prefix(uint32_t before, uint32_t after) {
    if (before == after) {
        return "w" + std::to_string(before);
    }
    return "w" + std::to_string(before) + "->" + std::to_string(after);
}

}  // namespace clifft::sampling
