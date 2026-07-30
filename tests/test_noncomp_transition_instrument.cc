#include "clifft/noncomp/level.h"
#include "clifft/noncomp/transition_instrument.h"

#include "noncomp_test_helpers.h"
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::Level;
using clifft::TransitionInstrument;
using clifft::test::opaque_infinity;
using clifft::test::opaque_nan;
using clifft::test::RawProbabilityMatrix;
using clifft::test::zero_transition_matrix;

// Shape validation

TEST_CASE("TransitionInstrument: accepts a zero matrix") {
    REQUIRE_NOTHROW(TransitionInstrument::from_matrix(zero_transition_matrix()));
}

TEST_CASE("TransitionInstrument: rejects wrong row count") {
    RawProbabilityMatrix bad(4, std::vector<double>(5, 0.0));
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(bad),
                        ContainsSubstring("4 rows") && ContainsSubstring("expected 5"));
}

TEST_CASE("TransitionInstrument: rejects jagged row width") {
    RawProbabilityMatrix bad = zero_transition_matrix();
    bad[2].pop_back();  // row 2 now has 4 columns
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(bad),
                        ContainsSubstring("row 2") && ContainsSubstring("4 columns"));
}

// Entry / column-sum validation

TEST_CASE("TransitionInstrument: rejects negative entry") {
    RawProbabilityMatrix m = zero_transition_matrix();
    m[1][0] = -0.1;
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(m),
                        ContainsSubstring("entry") && ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("TransitionInstrument: rejects entry above 1") {
    RawProbabilityMatrix m = zero_transition_matrix();
    m[1][0] = 1.5;
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(m), ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("TransitionInstrument: rejects column sum above 1") {
    RawProbabilityMatrix m = zero_transition_matrix();
    m[2][0] = 0.6;
    m[3][0] = 0.6;  // column 0 sums to 1.2
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(m),
                        ContainsSubstring("column 0") && ContainsSubstring("exceeds 1"));
}

TEST_CASE("TransitionInstrument: rejects NaN entry") {
    RawProbabilityMatrix m = zero_transition_matrix();
    m[1][0] = opaque_nan();
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(m),
                        ContainsSubstring("not finite") || ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("TransitionInstrument: rejects positive infinity entry") {
    RawProbabilityMatrix m = zero_transition_matrix();
    m[1][0] = opaque_infinity();
    REQUIRE_THROWS_AS(TransitionInstrument::from_matrix(m), std::invalid_argument);
}

TEST_CASE("TransitionInstrument: entry validation has no tolerance slack") {
    RawProbabilityMatrix m = zero_transition_matrix();
    // Just barely above 1: with no tolerance on entries, this must reject.
    m[1][0] = std::nextafter(1.0, 2.0);
    REQUIRE_THROWS_AS(TransitionInstrument::from_matrix(m), std::invalid_argument);
}

TEST_CASE("TransitionInstrument: column sum within tolerance above 1 is clamped") {
    RawProbabilityMatrix m = zero_transition_matrix();
    // Two entries whose sum is the next representable double above 1.0
    // (1.0 + 2^-52, ~2.22e-16). Each entry passes the strict [0, 1]
    // entry check, the sum is strictly > 1.0 yet well within
    // kProbTolerance, and the addition is exact (no rounding) so the
    // test behaves identically in Debug and Release.
    const double a = std::nextafter(std::nextafter(0.5, 1.0), 1.0);  // 0.5 + 2^-52
    const double b = 0.5;
    REQUIRE(a + b > 1.0);  // guards against the test going stale if the inputs change
    REQUIRE(a + b < 1.0 + 1e-12);
    m[0][0] = a;
    m[1][0] = b;

    auto instr = TransitionInstrument::from_matrix(m);
    // column_sum is clamped to [0, 1] so derived no-jump weights stay
    // non-negative.
    REQUIRE(instr.column_sum(Level::G) == 1.0);
}

// Accessors

TEST_CASE("TransitionInstrument: column_sum matches the matrix") {
    RawProbabilityMatrix m = zero_transition_matrix();
    // From g: 30% jump to leak_g, 10% jump to lost -> column sum 0.4.
    m[2][0] = 0.3;
    m[4][0] = 0.1;
    // From e: 70% jump to leak_e -> column sum 0.7.
    m[3][1] = 0.7;

    auto instr = TransitionInstrument::from_matrix(m);

    REQUIRE_THAT(instr.column_sum(Level::G), WithinAbs(0.4, 1e-12));
    REQUIRE_THAT(instr.column_sum(Level::E), WithinAbs(0.7, 1e-12));
    REQUIRE_THAT(instr.column_sum(Level::LeakG), WithinAbs(0.0, 1e-12));
}

TEST_CASE("TransitionInstrument: prob follows the destination-source convention") {
    RawProbabilityMatrix m = zero_transition_matrix();
    m[3][1] = 0.25;  // P(leak_e | from e) = 0.25
    auto instr = TransitionInstrument::from_matrix(m);

    REQUIRE_THAT(instr.prob(Level::LeakE, Level::E), WithinAbs(0.25, 1e-12));
    REQUIRE_THAT(instr.prob(Level::E, Level::LeakE), WithinAbs(0.0, 1e-12));
}
