#include "clifft/api/program.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"

#include <catch2/catch_test_macros.hpp>

namespace clifft::test {

TEST_CASE("Program preserves the original plan across bounded reconstruction failures",
          "[reconstruction]") {
    const auto source = parse("H 0 1\nT 0 1\nH 0 1\nM 0 1\nDETECTOR rec[-1]\n");
    const auto hir = trace(source);
    const auto original = Program::compile(source, hir);
    REQUIRE(original.original().peak_active_width() == 2);
    for (bool keep_records : {false, true}) {
        REQUIRE_FALSE(original.uses_reconstruction(keep_records));
        REQUIRE(&original.sampling_plan(keep_records) == &original.original());
    }
    const auto selected = Program::compile(source, hir, true);
    for (bool keep_records : {false, true}) {
        REQUIRE(selected.uses_reconstruction(keep_records));
        REQUIRE(selected.sampling_plan(keep_records).peak_active_width() == 1);
    }
    REQUIRE(selected.original().inspect() == original.original().inspect());
    auto limits = ReconstructionLimits{};
    limits.max_dilation_qubits = 1;
    const auto bounded = Program::compile(source, hir, true, true, limits);
    const auto transformed = Program::compile(source, hir, true, false);
    for (const auto* fallback : {&bounded, &transformed}) {
        for (bool keep_records : {false, true}) {
            REQUIRE_FALSE(fallback->uses_reconstruction(keep_records));
            REQUIRE_FALSE(fallback->reconstruction_fallback_reason(keep_records).empty());
            REQUIRE(&fallback->sampling_plan(keep_records) == &fallback->original());
            REQUIRE(fallback->original().inspect() == original.original().inspect());
        }
    }
}

}  // namespace clifft::test
