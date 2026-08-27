#include "clifft/util/numeric.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <limits>

using namespace clifft;

TEST_CASE("Unsigned arithmetic saturates at the representable limit") {
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();

    CHECK(saturating_add_u64(2, 3) == 5);
    CHECK(saturating_add_u64(maximum, 1) == maximum);
    CHECK(saturating_add_u64(maximum - 1, 1) == maximum);
    CHECK(saturating_multiply_u64(3, 4) == 12);
    CHECK(saturating_multiply_u64(0, maximum) == 0);
    CHECK(saturating_multiply_u64(maximum, 2) == maximum);
}

TEST_CASE("Robust finite checks survive finite-math optimization") {
    CHECK(is_finite_robust(0.0));
    CHECK(is_finite_robust(std::numeric_limits<double>::lowest()));
    CHECK(is_finite_robust(std::numeric_limits<double>::max()));
    CHECK_FALSE(is_finite_robust(clifft::test::opaque_nan()));
    CHECK_FALSE(is_finite_robust(clifft::test::opaque_infinity()));
    CHECK_FALSE(is_finite_robust(clifft::test::opaque_nonfinite(0xFFF0000000000000ULL)));
}

TEST_CASE("Probability checks survive finite-math optimization") {
    CHECK(is_probability(0.0));
    CHECK(is_probability(-0.0));
    CHECK(is_probability(0.5));
    CHECK(is_probability(1.0));
    CHECK_FALSE(is_probability(-0.01));
    CHECK_FALSE(is_probability(1.01));
    CHECK_FALSE(is_probability(clifft::test::opaque_nan()));
    CHECK_FALSE(is_probability(clifft::test::opaque_infinity()));
    CHECK_FALSE(is_probability(clifft::test::opaque_nonfinite(0xFFF0000000000000ULL)));
}
