#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/qubit_status.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <stdexcept>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using clifft::DampingPolicy;
using clifft::kInvalidLevel;
using clifft::Level;
using clifft::LevelCategory;
using clifft::LevelSet;
using clifft::NonComputationalPolicy;
using clifft::QubitStatus;
using clifft::QubitStatusKind;

// =========================================================================
// LevelSet construction / validation
// =========================================================================

TEST_CASE("LevelSet: default_set validates and exposes the expected levels") {
    LevelSet set = LevelSet::default_set();
    REQUIRE(set.size() == 5);
    REQUIRE(set.at(0).category == LevelCategory::Computational);
    REQUIRE(set.computational_zero_id() == 0);
    REQUIRE(set.at(1).category == LevelCategory::Computational);
    REQUIRE(set.computational_one_id() == 1);
    REQUIRE(set.at(2).category == LevelCategory::Leaked);
    REQUIRE(set.at(4).category == LevelCategory::Lost);
}

TEST_CASE("LevelSet: rejects empty level set") {
    REQUIRE_THROWS_AS(LevelSet(std::vector<Level>{}), std::invalid_argument);
}

TEST_CASE("LevelSet: accepts a custom set with exactly two Computational levels") {
    std::vector<Level> levels = {
        Level{"g", LevelCategory::Computational},
        Level{"e", LevelCategory::Computational},
        Level{"leak", LevelCategory::Leaked},
        Level{"lost", LevelCategory::Lost},
    };
    LevelSet set(std::move(levels));
    REQUIRE(set.computational_zero_id() == 0);
    REQUIRE(set.computational_one_id() == 1);
}

TEST_CASE("LevelSet: rejects level set above the 128-entry cap") {
    std::vector<Level> levels;
    levels.reserve(129);
    for (size_t i = 0; i < 129; ++i) {
        levels.push_back(Level{"L" + std::to_string(i), LevelCategory::Computational});
    }
    REQUIRE_THROWS_WITH(LevelSet(std::move(levels)), ContainsSubstring("128"));
}

TEST_CASE("LevelSet: rejects unrecognized LevelCategory enum value") {
    std::vector<Level> levels = {
        Level{"weird", static_cast<LevelCategory>(99)},
    };
    REQUIRE_THROWS_WITH(LevelSet(std::move(levels)),
                        ContainsSubstring("unrecognized LevelCategory"));
}

TEST_CASE("LevelSet: rejects more than two Computational levels") {
    std::vector<Level> levels = {
        Level{"g", LevelCategory::Computational},
        Level{"e", LevelCategory::Computational},
        Level{"x", LevelCategory::Computational},
    };
    REQUIRE_THROWS_WITH(LevelSet(std::move(levels)),
                        ContainsSubstring("two Computational") && ContainsSubstring("got 3"));
}

TEST_CASE("LevelSet: rejects fewer than two Computational levels") {
    std::vector<Level> levels = {
        Level{"g", LevelCategory::Computational},
        Level{"lost", LevelCategory::Lost},
    };
    REQUIRE_THROWS_WITH(LevelSet(std::move(levels)),
                        ContainsSubstring("two Computational") && ContainsSubstring("got 1"));
}

// =========================================================================
// LevelSet status factories
// =========================================================================

TEST_CASE("LevelSet::leaked accepts a Leaked level id, rejects others") {
    LevelSet set = LevelSet::default_set();
    REQUIRE_NOTHROW(set.leaked(2));   // leak_g
    REQUIRE_NOTHROW(set.leaked(3));   // leak_e
    REQUIRE_THROWS_AS(set.leaked(0),  // g
                      std::invalid_argument);
    REQUIRE_THROWS_AS(set.leaked(4),  // lost
                      std::invalid_argument);
}

TEST_CASE("LevelSet::lost accepts a Lost level id, rejects others") {
    LevelSet set = LevelSet::default_set();
    REQUIRE_NOTHROW(set.lost(4));   // lost
    REQUIRE_THROWS_AS(set.lost(0),  // g
                      std::invalid_argument);
    REQUIRE_THROWS_AS(set.lost(2),  // leak_g
                      std::invalid_argument);
}

// =========================================================================
// QubitStatus
// =========================================================================

TEST_CASE("QubitStatus::computational carries kInvalidLevel") {
    QubitStatus s = QubitStatus::computational();
    REQUIRE(s.kind() == QubitStatusKind::Computational);
    REQUIRE(s.level_id() == kInvalidLevel);
}

TEST_CASE("QubitStatus::is_computational is true only for the computational kind") {
    LevelSet set = LevelSet::default_set();
    REQUIRE(QubitStatus::computational().is_computational());
    REQUIRE_FALSE(set.leaked(2).is_computational());
    REQUIRE_FALSE(set.lost(4).is_computational());
}

TEST_CASE("QubitStatus _unchecked factories build without table validation") {
    // These are the only paths that don't require a LevelSet. Useful
    // for interior code and tests; the name flags the responsibility.
    QubitStatus t = QubitStatus::leaked_unchecked(9);
    REQUIRE(t.kind() == QubitStatusKind::Leaked);
    REQUIRE(t.level_id() == 9);

    QubitStatus u = QubitStatus::lost_unchecked(11);
    REQUIRE(u.kind() == QubitStatusKind::Lost);
    REQUIRE(u.level_id() == 11);
}

// =========================================================================
// NonComputationalPolicy
// =========================================================================

TEST_CASE("NonComputationalPolicy: defaults are exact and conservative") {
    NonComputationalPolicy policy;
    REQUIRE(policy.reset_restores_lost == false);
    REQUIRE(policy.damping == DampingPolicy::Exact);
}

TEST_CASE("NonComputationalPolicy: explicit overrides round-trip") {
    NonComputationalPolicy policy;
    policy.reset_restores_lost = true;
    REQUIRE(policy.reset_restores_lost == true);
}
