#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>

namespace clifft::sampling::folded {

// Fixed physical support for the supported codes, including the 85-data-qubit
// folded patch. Keeping two words avoids compiler-specific 128-bit integers.
struct Mask {
    uint64_t low = 0, high = 0;
    constexpr Mask(uint64_t low = 0, uint64_t high = 0) : low(low), high(high) {}
    constexpr explicit operator uint64_t() const noexcept { return low; }
    constexpr explicit operator bool() const noexcept { return low || high; }
    friend constexpr bool operator==(Mask, Mask) = default;
    friend constexpr auto operator<=>(Mask a, Mask b) noexcept {
        return a.high == b.high ? a.low <=> b.low : a.high <=> b.high;
    }
    friend constexpr Mask operator^(Mask a, Mask b) noexcept {
        return {a.low ^ b.low, a.high ^ b.high};
    }
    friend constexpr Mask operator&(Mask a, Mask b) noexcept {
        return {a.low & b.low, a.high & b.high};
    }
    friend constexpr Mask operator|(Mask a, Mask b) noexcept {
        return {a.low | b.low, a.high | b.high};
    }
    friend constexpr Mask operator~(Mask a) noexcept { return {~a.low, ~a.high}; }
    friend constexpr Mask operator<<(Mask a, size_t n) noexcept {
        if (n >= 128)
            return {};
        if (n >= 64)
            return {0, a.low << (n - 64)};
        if (n == 0)
            return a;
        return {a.low << n, (a.high << n) | (a.low >> (64 - n))};
    }
    friend constexpr Mask operator>>(Mask a, size_t n) noexcept {
        if (n >= 128)
            return {};
        if (n >= 64)
            return {a.high >> (n - 64), 0};
        if (n == 0)
            return a;
        return {(a.low >> n) | (a.high << (64 - n)), a.high >> n};
    }
    friend constexpr Mask operator+(Mask a, Mask b) noexcept {
        const auto low = a.low + b.low;
        return {low, a.high + b.high + uint64_t(low < a.low)};
    }
    friend constexpr Mask operator-(Mask a, Mask b) noexcept {
        return {a.low - b.low, a.high - b.high - uint64_t(a.low < b.low)};
    }
    constexpr Mask& operator^=(Mask other) noexcept { return *this = *this ^ other; }
};

}  // namespace clifft::sampling::folded
