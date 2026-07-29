#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"

#include "noncomp_test_helpers.h"
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
using clifft::test::classifier_matrix_with_column;
using clifft::test::opaque_infinity;
using clifft::test::opaque_nan;
using clifft::test::RawProbabilityMatrix;

// Construction validation

TEST_CASE("MeasurementClassifier: accepts a stochastic two-symbol matrix") {
    auto classifier =
        MeasurementClassifier::from_matrix(classifier_matrix_with_column(Level::LeakE, {0.0, 1.0}));
    REQUIRE(classifier.num_symbols() == 2);
    REQUIRE_FALSE(classifier.has_herald());
}

TEST_CASE("MeasurementClassifier: accepts a three-symbol matrix with a noncomp herald") {
    // The herald symbol may carry mass only on noncomputational columns.
    RawProbabilityMatrix m = {
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
    RawProbabilityMatrix one = {{1, 1, 1, 1, 1}};
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(one),
                        ContainsSubstring("two record symbols") && ContainsSubstring("got 1"));
    RawProbabilityMatrix four = {
        {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(four), ContainsSubstring("got 4"));
}

TEST_CASE("MeasurementClassifier: rejects wrong row count") {
    RawProbabilityMatrix m = {
        {1, 1, 1, 1, 1},
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m), ContainsSubstring("got 1"));
}

TEST_CASE("MeasurementClassifier: rejects wrong column count") {
    RawProbabilityMatrix m = {
        {1, 0, 1, 0},  // only 4 columns
        {0, 1, 0, 1},
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                        ContainsSubstring("4 columns") && ContainsSubstring("expected 5"));
}

TEST_CASE("MeasurementClassifier: rejects negative entry") {
    RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
    m[0][0] = -0.1;
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                        ContainsSubstring("entry") && ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("MeasurementClassifier: rejects entry above 1") {
    RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
    m[0][0] = 1.5;
    REQUIRE_THROWS_AS(MeasurementClassifier::from_matrix(m), std::invalid_argument);
}

TEST_CASE("MeasurementClassifier: rejects NaN entry") {
    RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
    m[0][0] = opaque_nan();
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                        ContainsSubstring("not finite") || ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("MeasurementClassifier: rejects infinity entry") {
    RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
    m[0][0] = opaque_infinity();
    REQUIRE_THROWS_AS(MeasurementClassifier::from_matrix(m), std::invalid_argument);
}

TEST_CASE("MeasurementClassifier: rejects a substochastic reject column") {
    RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
    m[0][2] = 0.5;  // leak_g column now sums to 0.5
    m[1][2] = 0.0;
    REQUIRE_THROWS_WITH(
        MeasurementClassifier::from_matrix(m),
        ContainsSubstring("reject columns are not supported") && ContainsSubstring("leak_g"));
}

TEST_CASE("MeasurementClassifier: rejects a column sum above 1") {
    RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
    m[0][0] = 0.6;
    m[1][0] = 0.6;  // g column sums to 1.2
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m), ContainsSubstring("must sum to 1"));
}

TEST_CASE("MeasurementClassifier: column sum tolerance accepts inside and rejects outside") {
    // kProbTolerance = 1e-12. A sum of 1 + 1e-13 is clearly inside the band
    // and must be accepted; a sum of 1 + 1e-11 is clearly outside and must
    // be rejected. Both individual entries are in [0, 1] so the per-entry
    // check cannot fire.
    SECTION("clearly accepted: sum = 1 + 1e-13") {
        RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
        m[0][0] = 0.5 + 1e-13;  // entry in [0,1]; the sum exceeds 1 by 1e-13, inside the tolerance
        m[1][0] = 0.5;
        REQUIRE_NOTHROW(MeasurementClassifier::from_matrix(m));
    }
    SECTION("clearly rejected: sum = 1 + 1e-11") {
        RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
        m[0][0] = 0.5 + 1e-11;  // entry in [0,1]; the sum exceeds 1 by 1e-11, outside the tolerance
        m[1][0] = 0.5;
        REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                            ContainsSubstring("must sum to 1"));
    }
}

TEST_CASE("MeasurementClassifier: rejects herald mass on a computational column") {
    RawProbabilityMatrix m = {
        {1, 0, 1, 0, 1}, {0, 0.8, 0, 1, 0}, {0, 0.2, 0, 0, 0},  // e column puts 0.2 on the herald
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix(m),
                        ContainsSubstring("record symbols 0 and 1") && ContainsSubstring("'e'"));
}

// Accessors

TEST_CASE("MeasurementClassifier: prob follows the symbol-level convention") {
    RawProbabilityMatrix m = classifier_matrix_with_column(Level::LeakE, {0.0, 1.0});
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
    auto classifier =
        MeasurementClassifier::from_matrix(classifier_matrix_with_column(Level::LeakE, {0.0, 1.0}));
    REQUIRE_THROWS_AS(classifier.prob(99, Level::G), std::invalid_argument);
}
