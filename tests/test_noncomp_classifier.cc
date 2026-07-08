#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::Level;
using clifft::MeasurementClassifier;
using clifft::test::opaque_infinity;
using clifft::test::opaque_nan;

namespace {

// Identity readout on g/e; leak_g/lost read "0", leak_e reads "1".
// Every column sums to 1, as construction requires.
std::vector<std::vector<double>> default_matrix() {
    return {
        {1, 0, 1, 0, 1},  // P("0" | level) over [g, e, leak_g, leak_e, lost]
        {0, 1, 0, 1, 0},  // P("1" | level)
    };
}

}  // namespace

// =========================================================================
// Construction validation
// =========================================================================

TEST_CASE("MeasurementClassifier: accepts a stochastic two-symbol matrix") {
    auto classifier = MeasurementClassifier::from_matrix(default_matrix());
    REQUIRE(classifier.num_symbols() == 2);
    REQUIRE_FALSE(classifier.has_herald());
}

TEST_CASE("MeasurementClassifier: accepts a three-symbol matrix with a noncomp herald") {
    // The herald symbol may carry mass only on noncomputational columns.
    std::vector<std::vector<double>> m = {
        {1, 0, 0.5, 0, 0},
        {0, 1, 0.2, 1, 0},
        {0, 0, 0.3, 0, 1},  // herald: leak_g sometimes, lost always
    };
    auto classifier = MeasurementClassifier::from_matrix(m);
    REQUIRE(classifier.num_symbols() == 3);
    REQUIRE(classifier.has_herald());
    REQUIRE_THAT(classifier.prob(MeasurementClassifier::kHeraldSymbol, Level::Lost),
                 WithinAbs(1.0, 1e-12));
}

TEST_CASE("MeasurementClassifier: rejects symbol counts other than two or three") {
    std::vector<std::vector<double>> one = {{1, 1, 1, 1, 1}};
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(one),
                        ContainsSubstring("two record symbols") && ContainsSubstring("got 1"));
    std::vector<std::vector<double>> four = {
        {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(four), ContainsSubstring("got 4"));
}

TEST_CASE("MeasurementClassifier: rejects wrong row count") {
    std::vector<std::vector<double>> m = {
        {1, 1, 1, 1, 1},
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m), ContainsSubstring("got 1"));
}

TEST_CASE("MeasurementClassifier: rejects wrong column count") {
    std::vector<std::vector<double>> m = {
        {1, 0, 1, 0},  // only 4 columns
        {0, 1, 0, 1},
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                        ContainsSubstring("4 columns") && ContainsSubstring("expected 5"));
}

TEST_CASE("MeasurementClassifier: rejects negative entry") {
    std::vector<std::vector<double>> m = default_matrix();
    m[0][0] = -0.1;
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                        ContainsSubstring("entry") && ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("MeasurementClassifier: rejects entry above 1") {
    std::vector<std::vector<double>> m = default_matrix();
    m[0][0] = 1.5;
    REQUIRE_THROWS_AS(MeasurementClassifier::from_matrix(m), std::invalid_argument);
}

TEST_CASE("MeasurementClassifier: rejects NaN entry") {
    std::vector<std::vector<double>> m = default_matrix();
    m[0][0] = opaque_nan();
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                        ContainsSubstring("not finite") || ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("MeasurementClassifier: rejects infinity entry") {
    std::vector<std::vector<double>> m = default_matrix();
    m[0][0] = opaque_infinity();
    REQUIRE_THROWS_AS(MeasurementClassifier::from_matrix(m), std::invalid_argument);
}

TEST_CASE("MeasurementClassifier: rejects a substochastic (reject) column") {
    std::vector<std::vector<double>> m = default_matrix();
    m[0][2] = 0.5;  // leak_g column now sums to 0.5
    m[1][2] = 0.0;
    REQUIRE_THROWS_WITH(
        MeasurementClassifier::from_matrix(m),
        ContainsSubstring("reject columns are not supported") && ContainsSubstring("leak_g"));
}

TEST_CASE("MeasurementClassifier: rejects a column sum above 1") {
    std::vector<std::vector<double>> m = default_matrix();
    m[0][0] = 0.6;
    m[1][0] = 0.6;  // g column sums to 1.2
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m), ContainsSubstring("must sum to 1"));
}

TEST_CASE("MeasurementClassifier: accepts a column sum within tolerance of 1") {
    std::vector<std::vector<double>> m = default_matrix();
    // Two entries whose sum is the next representable double above 1.0
    // (1.0 + 2^-52), well within kProbTolerance. Each entry passes the
    // strict [0, 1] check.
    const double a = std::nextafter(std::nextafter(0.5, 1.0), 1.0);  // 0.5 + 2^-52
    const double b = 0.5;
    REQUIRE(a + b > 1.0);
    REQUIRE(a + b < 1.0 + 1e-12);
    m[0][0] = a;
    m[1][0] = b;
    REQUIRE_NOTHROW(MeasurementClassifier::from_matrix(m));
}

TEST_CASE("MeasurementClassifier: rejects herald mass on a computational column") {
    std::vector<std::vector<double>> m = {
        {1, 0, 1, 0, 1}, {0, 0.8, 0, 1, 0}, {0, 0.2, 0, 0, 0},  // e column puts 0.2 on the herald
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                        ContainsSubstring("record symbols 0 and 1") && ContainsSubstring("'e'"));
}

// =========================================================================
// Accessors
// =========================================================================

TEST_CASE("MeasurementClassifier: prob returns the entry under (symbol, level) convention") {
    std::vector<std::vector<double>> m = default_matrix();
    m[0][2] = 0.25;  // P("0" | leak_g) = 0.25
    m[1][2] = 0.75;  // P("1" | leak_g) = 0.75
    auto classifier = MeasurementClassifier::from_matrix(m);

    REQUIRE_THAT(classifier.prob(0, Level::G), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(classifier.prob(1, Level::E), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(classifier.prob(0, Level::LeakG), WithinAbs(0.25, 1e-12));
    REQUIRE_THAT(classifier.prob(1, Level::LeakG), WithinAbs(0.75, 1e-12));
    REQUIRE_THAT(classifier.prob(1, Level::Lost), WithinAbs(0.0, 1e-12));
}

TEST_CASE("MeasurementClassifier: out-of-range symbol probability throws") {
    auto classifier = MeasurementClassifier::from_matrix(default_matrix());
    REQUIRE_THROWS_AS(classifier.prob(99, Level::G), std::invalid_argument);
}
