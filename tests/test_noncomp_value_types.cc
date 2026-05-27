#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/qubit_status.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <stdexcept>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using clifft::is_unknown_computational;
using clifft::kInvalidLevel;
using clifft::known_source_level;
using clifft::Level;
using clifft::LevelCategory;
using clifft::make_computational_known;
using clifft::make_computational_unknown;
using clifft::make_leaked;
using clifft::make_lost;
using clifft::NonComputationalPolicy;
using clifft::QubitStatusKind;
using clifft::require_classical_source_level;
using clifft::UnknownSourcePolicy;

// =========================================================================
// Level
// =========================================================================

TEST_CASE("Level: default set matches the sqale-aligned schema") {
    auto levels = clifft::default_levels();

    REQUIRE(levels.size() == 5);

    REQUIRE(levels[0].label == "g");
    REQUIRE(levels[0].category == LevelCategory::Computational);
    REQUIRE(levels[0].basis_bit.has_value());
    REQUIRE(*levels[0].basis_bit == 0);

    REQUIRE(levels[1].label == "e");
    REQUIRE(levels[1].category == LevelCategory::Computational);
    REQUIRE(*levels[1].basis_bit == 1);

    REQUIRE(levels[2].label == "leak_g");
    REQUIRE(levels[2].category == LevelCategory::Leaked);
    REQUIRE_FALSE(levels[2].basis_bit.has_value());

    REQUIRE(levels[3].label == "leak_e");
    REQUIRE(levels[3].category == LevelCategory::Leaked);
    REQUIRE_FALSE(levels[3].basis_bit.has_value());

    REQUIRE(levels[4].label == "lost");
    REQUIRE(levels[4].category == LevelCategory::Lost);
    REQUIRE_FALSE(levels[4].basis_bit.has_value());
}

TEST_CASE("validate_levels: accepts the default set") {
    auto levels = clifft::default_levels();
    REQUIRE_NOTHROW(clifft::validate_levels(levels));
}

TEST_CASE("validate_levels: rejects empty level set") {
    std::vector<Level> empty;
    REQUIRE_THROWS_AS(clifft::validate_levels(empty), std::invalid_argument);
}

TEST_CASE("validate_levels: rejects Computational level missing basis_bit") {
    std::vector<Level> levels = {
        Level{"g", LevelCategory::Computational, std::nullopt},
    };
    REQUIRE_THROWS_WITH(clifft::validate_levels(levels),
                        ContainsSubstring("Computational") && ContainsSubstring("basis_bit"));
}

TEST_CASE("validate_levels: rejects Computational level with basis_bit > 1") {
    std::vector<Level> levels = {
        Level{"weird", LevelCategory::Computational, uint8_t{2}},
    };
    REQUIRE_THROWS_WITH(clifft::validate_levels(levels), ContainsSubstring("basis_bit"));
}

TEST_CASE("validate_levels: rejects Leaked level carrying a basis_bit") {
    std::vector<Level> levels = {
        Level{"leak_g", LevelCategory::Leaked, uint8_t{0}},
    };
    REQUIRE_THROWS_WITH(clifft::validate_levels(levels),
                        ContainsSubstring("non-Computational") && ContainsSubstring("basis_bit"));
}

TEST_CASE("validate_levels: rejects Lost level carrying a basis_bit") {
    std::vector<Level> levels = {
        Level{"lost", LevelCategory::Lost, uint8_t{0}},
    };
    REQUIRE_THROWS_AS(clifft::validate_levels(levels), std::invalid_argument);
}

TEST_CASE("validate_levels: rejects level sets above the 128-entry cap") {
    std::vector<Level> levels;
    levels.reserve(129);
    for (size_t i = 0; i < 129; ++i) {
        levels.push_back(Level{"L" + std::to_string(i), LevelCategory::Computational, uint8_t{0}});
    }
    REQUIRE_THROWS_WITH(clifft::validate_levels(levels), ContainsSubstring("128"));
}

// =========================================================================
// NonComputationalPolicy
// =========================================================================

TEST_CASE("NonComputationalPolicy: defaults are conservative") {
    NonComputationalPolicy policy;
    REQUIRE(policy.reset_restores_lost == false);
    REQUIRE(policy.unknown_source_policy == UnknownSourcePolicy::Reject);
}

TEST_CASE("NonComputationalPolicy: explicit overrides round-trip") {
    NonComputationalPolicy policy;
    policy.reset_restores_lost = true;
    REQUIRE(policy.reset_restores_lost == true);
}

// =========================================================================
// QubitStatus factories: invariants are structural
// =========================================================================

TEST_CASE("make_computational_unknown: carries kInvalidLevel") {
    auto s = make_computational_unknown();
    REQUIRE(s.kind == QubitStatusKind::ComputationalUnknown);
    REQUIRE(s.level_id == kInvalidLevel);
}

TEST_CASE("make_computational_known: accepts a Computational level") {
    auto levels = clifft::default_levels();
    auto s = make_computational_known(0, levels);  // g
    REQUIRE(s.kind == QubitStatusKind::ComputationalKnown);
    REQUIRE(s.level_id == 0);

    auto t = make_computational_known(1, levels);  // e
    REQUIRE(t.kind == QubitStatusKind::ComputationalKnown);
    REQUIRE(t.level_id == 1);
}

TEST_CASE("make_computational_known: rejects a Leaked level") {
    auto levels = clifft::default_levels();
    REQUIRE_THROWS_AS(make_computational_known(2, levels),  // leak_g
                      std::invalid_argument);
}

TEST_CASE("make_computational_known: rejects a Lost level") {
    auto levels = clifft::default_levels();
    REQUIRE_THROWS_AS(make_computational_known(4, levels),  // lost
                      std::invalid_argument);
}

TEST_CASE("make_computational_known: rejects out-of-range level_id") {
    auto levels = clifft::default_levels();
    REQUIRE_THROWS_WITH(make_computational_known(99, levels), ContainsSubstring("out of range"));
}

TEST_CASE("make_leaked: accepts a Leaked level, rejects others") {
    auto levels = clifft::default_levels();
    REQUIRE_NOTHROW(make_leaked(2, levels));   // leak_g
    REQUIRE_NOTHROW(make_leaked(3, levels));   // leak_e
    REQUIRE_THROWS_AS(make_leaked(0, levels),  // g
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make_leaked(4, levels),  // lost
                      std::invalid_argument);
}

TEST_CASE("make_lost: accepts a Lost level, rejects others") {
    auto levels = clifft::default_levels();
    REQUIRE_NOTHROW(make_lost(4, levels));   // lost
    REQUIRE_THROWS_AS(make_lost(0, levels),  // g
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make_lost(2, levels),  // leak_g
                      std::invalid_argument);
}

// =========================================================================
// QubitStatus accessors
// =========================================================================

TEST_CASE("is_unknown_computational: true only for ComputationalUnknown") {
    auto levels = clifft::default_levels();
    REQUIRE(is_unknown_computational(make_computational_unknown()));
    REQUIRE_FALSE(is_unknown_computational(make_computational_known(0, levels)));
    REQUIRE_FALSE(is_unknown_computational(make_leaked(2, levels)));
    REQUIRE_FALSE(is_unknown_computational(make_lost(4, levels)));
}

TEST_CASE("known_source_level: nullopt on Unknown, the level id otherwise") {
    auto levels = clifft::default_levels();

    REQUIRE_FALSE(known_source_level(make_computational_unknown()).has_value());

    auto k = known_source_level(make_computational_known(1, levels));
    REQUIRE(k.has_value());
    REQUIRE(*k == 1);

    auto leaked = known_source_level(make_leaked(2, levels));
    REQUIRE(leaked.has_value());
    REQUIRE(*leaked == 2);

    auto lost = known_source_level(make_lost(4, levels));
    REQUIRE(lost.has_value());
    REQUIRE(*lost == 4);
}

TEST_CASE("require_classical_source_level: throws on Unknown, returns id otherwise") {
    auto levels = clifft::default_levels();

    REQUIRE_THROWS_AS(require_classical_source_level(make_computational_unknown()),
                      std::invalid_argument);

    REQUIRE(require_classical_source_level(make_computational_known(0, levels)) == 0);
    REQUIRE(require_classical_source_level(make_leaked(3, levels)) == 3);
    REQUIRE(require_classical_source_level(make_lost(4, levels)) == 4);
}
