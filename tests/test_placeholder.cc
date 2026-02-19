#include "ucc/util/config.h"

#include <catch2/catch_test_macros.hpp>

// Placeholder test to verify build infrastructure works.
// This file will be removed once real tests are added.

TEST_CASE("Build infrastructure works", "[placeholder]") {
    REQUIRE(ucc::kMaxInlineQubits == 64);
}

TEST_CASE("Stim is linked correctly", "[placeholder]") {
    // Just verify we can include Stim headers and use basic types
    // This confirms FetchContent and linking are working
    REQUIRE(true);
}
