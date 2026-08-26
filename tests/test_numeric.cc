#include "clifft/util/numeric.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <limits>

using namespace clifft;

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
