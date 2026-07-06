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

TEST_CASE("LevelSet::computational_known accepts a Computational level id") {
    LevelSet set = LevelSet::default_set();
    QubitStatus s = set.computational_known(0);
    REQUIRE(s.kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(s.level_id() == 0);

    QubitStatus t = set.computational_known(1);
    REQUIRE(t.kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(t.level_id() == 1);
}

TEST_CASE("LevelSet::computational_known rejects non-Computational ids") {
    LevelSet set = LevelSet::default_set();
    REQUIRE_THROWS_AS(set.computational_known(2),  // leak_g
                      std::invalid_argument);
    REQUIRE_THROWS_AS(set.computational_known(4),  // lost
                      std::invalid_argument);
}

TEST_CASE("LevelSet::computational_known rejects out-of-range id") {
    LevelSet set = LevelSet::default_set();
    REQUIRE_THROWS_WITH(set.computational_known(99), ContainsSubstring("out of range"));
}

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

TEST_CASE("QubitStatus::computational_unknown carries kInvalidLevel") {
    QubitStatus s = QubitStatus::computational_unknown();
    REQUIRE(s.kind() == QubitStatusKind::ComputationalUnknown);
    REQUIRE(s.level_id() == kInvalidLevel);
}

TEST_CASE("QubitStatus::is_unknown_computational is true only for Unknown") {
    LevelSet set = LevelSet::default_set();
    REQUIRE(QubitStatus::computational_unknown().is_unknown_computational());
    REQUIRE_FALSE(set.computational_known(0).is_unknown_computational());
    REQUIRE_FALSE(set.leaked(2).is_unknown_computational());
    REQUIRE_FALSE(set.lost(4).is_unknown_computational());
}

TEST_CASE("QubitStatus::known_source_level returns nullopt on Unknown, id otherwise") {
    LevelSet set = LevelSet::default_set();

    REQUIRE_FALSE(QubitStatus::computational_unknown().known_source_level().has_value());

    auto k = set.computational_known(1).known_source_level();
    REQUIRE(k.has_value());
    REQUIRE(*k == 1);

    auto leaked = set.leaked(2).known_source_level();
    REQUIRE(leaked.has_value());
    REQUIRE(*leaked == 2);

    auto lost = set.lost(4).known_source_level();
    REQUIRE(lost.has_value());
    REQUIRE(*lost == 4);
}

TEST_CASE("QubitStatus::require_classical_source_level throws on Unknown") {
    LevelSet set = LevelSet::default_set();

    REQUIRE_THROWS_AS(QubitStatus::computational_unknown().require_classical_source_level(),
                      std::invalid_argument);

    REQUIRE(set.computational_known(0).require_classical_source_level() == 0);
    REQUIRE(set.leaked(3).require_classical_source_level() == 3);
    REQUIRE(set.lost(4).require_classical_source_level() == 4);
}

TEST_CASE("QubitStatus _unchecked factories build without table validation") {
    // These are the only paths that don't require a LevelSet. Useful
    // for interior code and tests; the name flags the responsibility.
    QubitStatus s = QubitStatus::computational_known_unchecked(7);
    REQUIRE(s.kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(s.level_id() == 7);

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
