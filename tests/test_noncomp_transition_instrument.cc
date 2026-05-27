#include "clifft/noncomp/level.h"
#include "clifft/noncomp/transition_instrument.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::LevelSet;
using clifft::TransitionInstrument;

namespace {

// 5x5 zero matrix sized for the default level set.
std::vector<std::vector<double>> zero5() {
    return std::vector<std::vector<double>>(5, std::vector<double>(5, 0.0));
}

}  // namespace

// =========================================================================
// Shape validation
// =========================================================================

TEST_CASE("TransitionInstrument: accepts a zero matrix") {
    LevelSet levels = LevelSet::default_set();
    REQUIRE_NOTHROW(TransitionInstrument::from_matrix(zero5(), levels));
}

TEST_CASE("TransitionInstrument: rejects wrong row count") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> bad(4, std::vector<double>(5, 0.0));
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(std::move(bad), levels),
                        ContainsSubstring("4 rows") && ContainsSubstring("expected 5"));
}

TEST_CASE("TransitionInstrument: rejects jagged row width") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> bad = zero5();
    bad[2].pop_back();  // row 2 now has 4 columns
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(std::move(bad), levels),
                        ContainsSubstring("row 2") && ContainsSubstring("4 columns"));
}

// =========================================================================
// Entry / column-sum validation
// =========================================================================

TEST_CASE("TransitionInstrument: rejects negative entry") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    m[1][0] = -0.1;
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(std::move(m), levels),
                        ContainsSubstring("entry") && ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("TransitionInstrument: rejects entry above 1") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    m[1][0] = 1.5;
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(std::move(m), levels),
                        ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("TransitionInstrument: rejects column sum above 1") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    m[2][0] = 0.6;
    m[3][0] = 0.6;  // column 0 sums to 1.2
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(std::move(m), levels),
                        ContainsSubstring("column 0") && ContainsSubstring("exceeds 1"));
}

TEST_CASE("TransitionInstrument: rejects NaN entry") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    m[1][0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_WITH(TransitionInstrument::from_matrix(std::move(m), levels),
                        ContainsSubstring("not finite") || ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("TransitionInstrument: rejects positive infinity entry") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    m[1][0] = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(TransitionInstrument::from_matrix(std::move(m), levels),
                      std::invalid_argument);
}

TEST_CASE("TransitionInstrument: entry validation has no tolerance slack") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    // Just barely above 1: with no tolerance on entries, this must reject.
    m[1][0] = std::nextafter(1.0, 2.0);
    REQUIRE_THROWS_AS(TransitionInstrument::from_matrix(std::move(m), levels),
                      std::invalid_argument);
}

TEST_CASE("TransitionInstrument: column sum within tolerance above 1 is clamped") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
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

    auto instr = TransitionInstrument::from_matrix(std::move(m), levels);
    // column_sum is clamped to [0, 1] so no_jump_weight is non-negative.
    REQUIRE(instr.column_sum(0) == 1.0);
    REQUIRE(instr.no_jump_weight(0) == 0.0);
}

// =========================================================================
// Accessors
// =========================================================================

TEST_CASE("TransitionInstrument: column_sum and no_jump_weight match the matrix") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    // Source 0: 30% jump to level 2, 10% jump to level 4 -> column sum 0.4.
    m[2][0] = 0.3;
    m[4][0] = 0.1;
    // Source 1: 70% jump to level 3 -> column sum 0.7.
    m[3][1] = 0.7;

    auto instr = TransitionInstrument::from_matrix(std::move(m), levels);

    REQUIRE(instr.num_levels() == 5);
    REQUIRE_THAT(instr.column_sum(0), WithinAbs(0.4, 1e-12));
    REQUIRE_THAT(instr.column_sum(1), WithinAbs(0.7, 1e-12));
    REQUIRE_THAT(instr.column_sum(2), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(instr.no_jump_weight(0), WithinAbs(0.6, 1e-12));
    REQUIRE_THAT(instr.no_jump_weight(1), WithinAbs(0.3, 1e-12));
    REQUIRE_THAT(instr.no_jump_weight(2), WithinAbs(1.0, 1e-12));
}

TEST_CASE("TransitionInstrument: prob returns the entry under T[to, from] convention") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    m[3][1] = 0.25;  // P(level 3 | from level 1) = 0.25
    auto instr = TransitionInstrument::from_matrix(std::move(m), levels);

    REQUIRE_THAT(instr.prob(3, 1), WithinAbs(0.25, 1e-12));
    REQUIRE_THAT(instr.prob(1, 3), WithinAbs(0.0, 1e-12));
}

TEST_CASE("TransitionInstrument: prob rejects out-of-range index") {
    LevelSet levels = LevelSet::default_set();
    auto instr = TransitionInstrument::from_matrix(zero5(), levels);
    REQUIRE_THROWS_AS(instr.prob(7, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(instr.prob(0, 7), std::invalid_argument);
}

TEST_CASE("TransitionInstrument: column_sum rejects out-of-range index") {
    LevelSet levels = LevelSet::default_set();
    auto instr = TransitionInstrument::from_matrix(zero5(), levels);
    REQUIRE_THROWS_AS(instr.column_sum(99), std::invalid_argument);
}

// =========================================================================
// is_source_independent_on_computational
// =========================================================================

TEST_CASE("TransitionInstrument: source-independent flag true for all-zero matrix") {
    LevelSet levels = LevelSet::default_set();
    auto instr = TransitionInstrument::from_matrix(zero5(), levels);
    REQUIRE(instr.is_source_independent_on_computational());
}

TEST_CASE("TransitionInstrument: source-independent flag true when g/e columns match") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    // g and e columns identical: both jump 0.2 to leak_g, 0.1 to lost.
    m[2][0] = 0.2;
    m[4][0] = 0.1;
    m[2][1] = 0.2;
    m[4][1] = 0.1;
    auto instr = TransitionInstrument::from_matrix(std::move(m), levels);
    REQUIRE(instr.is_source_independent_on_computational());
}

TEST_CASE("TransitionInstrument: source-independent flag false when g/e columns differ") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    // g column jumps to leak_g; e column jumps to leak_e. Different.
    m[2][0] = 0.1;
    m[3][1] = 0.1;
    auto instr = TransitionInstrument::from_matrix(std::move(m), levels);
    REQUIRE_FALSE(instr.is_source_independent_on_computational());
}

TEST_CASE("TransitionInstrument: only Computational-source columns affect the flag") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    // g and e columns identical (both empty); leak_g column has non-trivial
    // content. The flag should still be true since leak_g is not Computational.
    m[4][2] = 0.5;  // leak_g -> lost with prob 0.5
    auto instr = TransitionInstrument::from_matrix(std::move(m), levels);
    REQUIRE(instr.is_source_independent_on_computational());
}

TEST_CASE("TransitionInstrument: tolerance - sub-1e-12 column difference still independent") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    m[2][0] = 0.3;
    m[2][1] = 0.3 + 1e-13;  // within tolerance
    auto instr = TransitionInstrument::from_matrix(std::move(m), levels);
    REQUIRE(instr.is_source_independent_on_computational());
}

TEST_CASE("TransitionInstrument: tolerance - 1e-9 column difference is not independent") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = zero5();
    m[2][0] = 0.3;
    m[2][1] = 0.3 + 1e-9;  // exceeds tolerance
    auto instr = TransitionInstrument::from_matrix(std::move(m), levels);
    REQUIRE_FALSE(instr.is_source_independent_on_computational());
}
