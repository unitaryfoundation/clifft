#pragma once

#include "clifft/sampling/results.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace clifft::test {

struct GpuReplayCase {
    std::string_view name;
    std::string_view circuit;
    uint32_t visible;
    uint32_t hidden;
    uint32_t min_active_width;
};

inline constexpr std::array kGpuReplayCases{
    GpuReplayCase{"multiple Pauli measurements", R"(
        H 0
        H 1
        T 0
        T 1
        CX 0 1
        MPP Y0*Z1
        R_PAULI(0.17) X0*Y1
        M 0
        CX rec[-1] 2
        EXP_VAL Z2
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    )",
                  2, 0, 2},
    // Unequal branch probabilities and post-measurement expectations expose
    // incorrect branch selection and collapse in both measurement paths.
    GpuReplayCase{"biased active and dormant measurements", R"(
        H 0 1 2
        T 0 1 2
        CX 0 1
        H 0
        M 0
        EXP_VAL X1*X2
        H 3
        M 3
        EXP_VAL Z3
    )",
                  2, 0, 3},
    GpuReplayCase{"uniform reset", R"(
        H 0
        R 0
        H 0
        T 0
        M 0
        EXP_VAL Z0
        OBSERVABLE_INCLUDE(0) rec[-1]
    )",
                  1, 1, 0},
    GpuReplayCase{"deterministic reset", R"(
        X 0
        R 0
        H 0
        M 0
    )",
                  1, 1, 0},
    GpuReplayCase{"feedback reset", R"(
        H 0
        M 0
        CX rec[-1] 1
        R 1
        M 1
        DETECTOR rec[-1]
    )",
                  2, 1, 0},
    // Z1 after the reset depends on the hidden branch, while its final value
    // depends on the visible result. Checking both exposes misplaced records.
    GpuReplayCase{"active reset", R"(
        H 0
        T 0
        H 0
        CX 0 1
        R 0
        EXP_VAL Z0
        EXP_VAL Z1
        H 1
        T 1
        H 1
        M 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        EXP_VAL Z1
    )",
                  1, 1, 1},
};

// Two visible columns and one hidden slot distinguish the output row stride
// from the internal record stride. The other outputs identify each visible row.
inline constexpr std::string_view kResetBatchCircuit = R"(
    H 0
    T 0
    H 0
    CX 0 1
    R 0
    H 1
    T 1
    H 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-1]
    EXP_VAL Z0
    EXP_VAL Z1
)";

inline void require_reset_batch_rows(const sampling::SamplingResult& rows, uint32_t shots) {
    REQUIRE(rows.measurements.size() == 2 * shots);
    REQUIRE(rows.detectors.size() == shots);
    REQUIRE(rows.observables.size() == shots);
    REQUIRE(rows.exp_vals.size() == 2 * shots);
    uint32_t ones = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        CAPTURE(shot);
        const uint8_t measured = rows.measurements[2 * shot + 1];
        REQUIRE(measured <= 1);
        REQUIRE(rows.measurements[2 * shot] == 0);
        REQUIRE(rows.detectors[shot] == measured);
        REQUIRE(rows.observables[shot] == measured);
        REQUIRE_THAT(rows.exp_vals[2 * shot], Catch::Matchers::WithinAbs(1.0, 2e-5));
        REQUIRE_THAT(rows.exp_vals[2 * shot + 1],
                     Catch::Matchers::WithinAbs(1.0 - 2.0 * measured, 2e-5));
        ones += measured;
    }
    // Summing the hidden branches gives P(m1 = 1) = 1/4.
    const double tolerance = 6.0 * std::sqrt(0.25 * 0.75 / shots) + 1e-3;
    REQUIRE_THAT(static_cast<double>(ones) / shots, Catch::Matchers::WithinAbs(0.25, tolerance));
}

}  // namespace clifft::test
