#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>

using clifft::DampingPolicy;
using clifft::is_computational;
using clifft::is_leaked;
using clifft::is_lost;
using clifft::kNumLevels;
using clifft::Level;
using clifft::level_name;
using clifft::noncomp_level;
using clifft::NonComputationalPolicy;
using clifft::QubitStatus;
using clifft::status_for;

// Level structure

TEST_CASE("Level: predicates partition the five levels") {
    REQUIRE(kNumLevels == 5);
    REQUIRE(is_computational(Level::G));
    REQUIRE(is_computational(Level::E));
    REQUIRE(is_leaked(Level::LeakG));
    REQUIRE(is_leaked(Level::LeakE));
    REQUIRE(is_lost(Level::Lost));
    REQUIRE_FALSE(is_leaked(Level::G));
    REQUIRE_FALSE(is_lost(Level::LeakG));
    REQUIRE_FALSE(is_computational(Level::Lost));
    REQUIRE(std::string(level_name(Level::G)) == "g");
    REQUIRE(std::string(level_name(Level::LeakE)) == "leak_e");
    REQUIRE(std::string(level_name(Level::Lost)) == "lost");
}

// QubitStatus

TEST_CASE("QubitStatus: status_for collapses the computational levels") {
    REQUIRE(status_for(Level::G) == QubitStatus::Computational);
    REQUIRE(status_for(Level::E) == QubitStatus::Computational);
    REQUIRE(status_for(Level::LeakG) == QubitStatus::LeakG);
    REQUIRE(status_for(Level::LeakE) == QubitStatus::LeakE);
    REQUIRE(status_for(Level::Lost) == QubitStatus::Lost);
}

TEST_CASE("QubitStatus: category predicates partition the statuses") {
    REQUIRE(is_computational(QubitStatus::Computational));
    REQUIRE_FALSE(is_leaked(QubitStatus::Computational));
    REQUIRE_FALSE(is_lost(QubitStatus::Computational));
    REQUIRE(is_leaked(QubitStatus::LeakG));
    REQUIRE(is_leaked(QubitStatus::LeakE));
    REQUIRE(is_lost(QubitStatus::Lost));
    REQUIRE_FALSE(is_leaked(QubitStatus::Lost));
}

TEST_CASE("QubitStatus: noncomp_level names definite levels and throws on computational") {
    REQUIRE(noncomp_level(QubitStatus::LeakG) == Level::LeakG);
    REQUIRE(noncomp_level(QubitStatus::LeakE) == Level::LeakE);
    REQUIRE(noncomp_level(QubitStatus::Lost) == Level::Lost);
    REQUIRE_THROWS_AS(noncomp_level(QubitStatus::Computational), std::logic_error);
}

// NonComputationalPolicy

TEST_CASE("NonComputationalPolicy: defaults are exact and conservative") {
    NonComputationalPolicy policy;
    REQUIRE(policy.reset_restores_lost == false);
    REQUIRE(policy.damping == DampingPolicy::Exact);
}
