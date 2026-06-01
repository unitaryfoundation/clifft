#include "clifft/noncomp/classifier.h"
#include "clifft/noncomp/level.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::LevelSet;
using clifft::MeasurementClassifier;

namespace {

// "Identity + reject" classifier for the default 5-level set:
// g -> "0", e -> "1", leak_g/leak_e/lost -> reject.
std::vector<std::vector<double>> default_identity_matrix() {
    return {
        {1, 0, 0, 0, 0},  // P("0" | level) over [g, e, leak_g, leak_e, lost]
        {0, 1, 0, 0, 0},  // P("1" | level)
    };
}

}  // namespace

// =========================================================================
// Construction validation
// =========================================================================

TEST_CASE("MeasurementClassifier: accepts identity + reject for the default level set") {
    LevelSet levels = LevelSet::default_set();
    auto classifier =
        MeasurementClassifier::from_matrix({"0", "1"}, default_identity_matrix(), levels);
    REQUIRE(classifier.num_symbols() == 2);
    REQUIRE(classifier.num_levels() == 5);
}

TEST_CASE("MeasurementClassifier: rejects empty symbols list") {
    LevelSet levels = LevelSet::default_set();
    REQUIRE_THROWS_WITH(
        MeasurementClassifier::from_matrix({}, std::vector<std::vector<double>>{}, levels),
        ContainsSubstring("symbols list is empty"));
}

TEST_CASE("MeasurementClassifier: accepts the 256-symbol boundary (uint8_t addressable)") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::string> symbols;
    symbols.reserve(256);
    for (size_t i = 0; i < 256; ++i) {
        symbols.push_back("s" + std::to_string(i));
    }
    // All-zero matrix passes shape and column-sum checks trivially.
    std::vector<std::vector<double>> m(256, std::vector<double>(5, 0.0));
    auto classifier = MeasurementClassifier::from_matrix(std::move(symbols), std::move(m), levels);
    REQUIRE(classifier.num_symbols() == 256);
    // Highest index is addressable through the uint8_t API.
    REQUIRE(classifier.symbol_label(255) == "s255");
}

TEST_CASE("MeasurementClassifier: rejects 257 symbols (above uint8_t addressability)") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::string> symbols;
    symbols.reserve(257);
    for (size_t i = 0; i < 257; ++i) {
        symbols.push_back("s" + std::to_string(i));
    }
    std::vector<std::vector<double>> m(257, std::vector<double>(5, 0.0));
    REQUIRE_THROWS_WITH(
        MeasurementClassifier::from_matrix(std::move(symbols), std::move(m), levels),
        ContainsSubstring("257") && ContainsSubstring("256"));
}

TEST_CASE("MeasurementClassifier: rejects duplicate symbol labels") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = {
        {1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix({"0", "0"}, std::move(m), levels),
                        ContainsSubstring("duplicate symbol") && ContainsSubstring("'0'"));
}

TEST_CASE("MeasurementClassifier: rejects wrong row count") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = {
        {1, 0, 0, 0, 0},
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels),
                        ContainsSubstring("1 rows") && ContainsSubstring("expected 2"));
}

TEST_CASE("MeasurementClassifier: rejects wrong column count") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = {
        {1, 0, 0, 0},  // only 4 columns
        {0, 1, 0, 0},
    };
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels),
                        ContainsSubstring("4 columns") && ContainsSubstring("expected 5"));
}

TEST_CASE("MeasurementClassifier: rejects negative entry") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    m[0][0] = -0.1;
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels),
                        ContainsSubstring("entry") && ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("MeasurementClassifier: rejects entry above 1") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    m[0][0] = 1.5;
    REQUIRE_THROWS_AS(MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels),
                      std::invalid_argument);
}

TEST_CASE("MeasurementClassifier: rejects NaN entry") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    m[0][0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels),
                        ContainsSubstring("not finite") || ContainsSubstring("out of [0, 1]"));
}

TEST_CASE("MeasurementClassifier: rejects infinity entry") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    m[0][0] = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels),
                      std::invalid_argument);
}

TEST_CASE("MeasurementClassifier: entry validation has no tolerance slack") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    m[0][0] = std::nextafter(1.0, 2.0);
    REQUIRE_THROWS_AS(MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels),
                      std::invalid_argument);
}

TEST_CASE("MeasurementClassifier: rejects column sum above 1") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    m[0][0] = 0.6;
    m[1][0] = 0.6;  // column 0 sums to 1.2
    REQUIRE_THROWS_WITH(MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels),
                        ContainsSubstring("column 0") && ContainsSubstring("exceeds 1"));
}

TEST_CASE("MeasurementClassifier: column sum within tolerance above 1 clamps reject to 0") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    // Two entries whose sum is the next representable double above 1.0
    // (1.0 + 2^-52), well within kProbTolerance. Each entry passes the
    // strict [0, 1] check.
    const double a = std::nextafter(std::nextafter(0.5, 1.0), 1.0);  // 0.5 + 2^-52
    const double b = 0.5;
    REQUIRE(a + b > 1.0);
    REQUIRE(a + b < 1.0 + 1e-12);
    m[0][0] = a;
    m[1][0] = b;
    auto classifier = MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels);
    REQUIRE(classifier.reject_probability(0) == 0.0);
}

// =========================================================================
// Accessors
// =========================================================================

TEST_CASE("MeasurementClassifier: prob returns the entry under (symbol, level) convention") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    m[0][2] = 0.25;  // P("0" | leak_g) = 0.25
    m[1][2] = 0.25;  // P("1" | leak_g) = 0.25; deficit 0.5 = reject prob
    auto classifier = MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels);

    REQUIRE_THAT(classifier.prob(0, 0), WithinAbs(1.0, 1e-12));   // P("0" | g)
    REQUIRE_THAT(classifier.prob(1, 1), WithinAbs(1.0, 1e-12));   // P("1" | e)
    REQUIRE_THAT(classifier.prob(0, 2), WithinAbs(0.25, 1e-12));  // P("0" | leak_g)
    REQUIRE_THAT(classifier.prob(1, 2), WithinAbs(0.25, 1e-12));  // P("1" | leak_g)
    REQUIRE_THAT(classifier.prob(0, 4), WithinAbs(0.0, 1e-12));   // P("0" | lost) = 0
}

TEST_CASE("MeasurementClassifier: symbol_label returns the constructed strings") {
    LevelSet levels = LevelSet::default_set();
    auto classifier =
        MeasurementClassifier::from_matrix({"zero", "one"}, default_identity_matrix(), levels);
    REQUIRE(classifier.symbol_label(0) == "zero");
    REQUIRE(classifier.symbol_label(1) == "one");
}

TEST_CASE("MeasurementClassifier: out-of-range accessors throw") {
    LevelSet levels = LevelSet::default_set();
    auto classifier =
        MeasurementClassifier::from_matrix({"0", "1"}, default_identity_matrix(), levels);
    REQUIRE_THROWS_AS(classifier.symbol_label(99), std::invalid_argument);
    REQUIRE_THROWS_AS(classifier.prob(99, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(classifier.prob(0, 99), std::invalid_argument);
    REQUIRE_THROWS_AS(classifier.reject_probability(99), std::invalid_argument);
}

// =========================================================================
// reject_probability
// =========================================================================

TEST_CASE("MeasurementClassifier: identity matrix gives 0 reject on g/e and 1 on leak/lost") {
    LevelSet levels = LevelSet::default_set();
    auto classifier =
        MeasurementClassifier::from_matrix({"0", "1"}, default_identity_matrix(), levels);
    REQUIRE_THAT(classifier.reject_probability(0), WithinAbs(0.0, 1e-12));  // g
    REQUIRE_THAT(classifier.reject_probability(1), WithinAbs(0.0, 1e-12));  // e
    REQUIRE_THAT(classifier.reject_probability(2), WithinAbs(1.0, 1e-12));  // leak_g
    REQUIRE_THAT(classifier.reject_probability(3), WithinAbs(1.0, 1e-12));  // leak_e
    REQUIRE_THAT(classifier.reject_probability(4), WithinAbs(1.0, 1e-12));  // lost
}

TEST_CASE("MeasurementClassifier: all-zero matrix gives reject = 1 for every level") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m(2, std::vector<double>(5, 0.0));
    auto classifier = MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels);
    for (uint8_t l = 0; l < 5; ++l) {
        REQUIRE_THAT(classifier.reject_probability(l), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("MeasurementClassifier: random-bit-on-lost policy via 0.5/0.5 column") {
    LevelSet levels = LevelSet::default_set();
    std::vector<std::vector<double>> m = default_identity_matrix();
    m[0][4] = 0.5;  // P("0" | lost) = 0.5
    m[1][4] = 0.5;  // P("1" | lost) = 0.5
    auto classifier = MeasurementClassifier::from_matrix({"0", "1"}, std::move(m), levels);
    REQUIRE_THAT(classifier.reject_probability(4), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(classifier.prob(0, 4), WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(classifier.prob(1, 4), WithinAbs(0.5, 1e-12));
}
