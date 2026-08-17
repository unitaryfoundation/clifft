#include "clifft/util/runtime_isa.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string_view>

using Catch::Matchers::ContainsSubstring;
using clifft::internal::runtime_isa;
using clifft::internal::runtime_isa_name;
using clifft::internal::RuntimeIsa;
using clifft::internal::validate_runtime_isa;

TEST_CASE("Runtime ISA names are stable") {
    REQUIRE(std::string_view(runtime_isa_name(RuntimeIsa::Scalar)) == "scalar");
    REQUIRE(std::string_view(runtime_isa_name(RuntimeIsa::Avx2)) == "avx2");
    REQUIRE(std::string_view(runtime_isa_name(RuntimeIsa::Avx512)) == "avx512");
    REQUIRE(std::string_view(runtime_isa_name(RuntimeIsa::TrapAvx2)) == "trap:avx2");
    REQUIRE(std::string_view(runtime_isa_name(RuntimeIsa::TrapAvx512)) == "trap:avx512");
    REQUIRE(std::string_view(runtime_isa_name(RuntimeIsa::TrapUnknown)) == "trap:unknown");
}

TEST_CASE("Runtime ISA validates executable selections") {
    REQUIRE_NOTHROW(validate_runtime_isa(RuntimeIsa::Scalar));
    REQUIRE_NOTHROW(validate_runtime_isa(RuntimeIsa::Avx2));
    REQUIRE_NOTHROW(validate_runtime_isa(RuntimeIsa::Avx512));
}

TEST_CASE("Runtime ISA reports forced selection errors") {
    REQUIRE_THROWS_WITH(validate_runtime_isa(RuntimeIsa::TrapAvx2),
                        ContainsSubstring("CLIFFT_FORCE_ISA=avx2 requested"));
    REQUIRE_THROWS_WITH(validate_runtime_isa(RuntimeIsa::TrapAvx512),
                        ContainsSubstring("CLIFFT_FORCE_ISA=avx512 requested"));
    REQUIRE_THROWS_WITH(validate_runtime_isa(RuntimeIsa::TrapUnknown),
                        ContainsSubstring("unrecognized value"));
}
