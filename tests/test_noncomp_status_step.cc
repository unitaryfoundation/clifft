#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/status_step.h"

#include <catch2/catch_test_macros.hpp>

using clifft::GateType;
using clifft::NonComputationalPolicy;
using clifft::normal_post_op_status;
using clifft::OperandRole;
using clifft::QubitStatus;

namespace {

constexpr OperandRole kPhysical = OperandRole::Physical;
constexpr OperandRole kFeedbackX = OperandRole::FeedbackX;
constexpr OperandRole kFeedbackZ = OperandRole::FeedbackZ;

}  // namespace

TEST_CASE("normal_post_op_status: normal operations keep a computational qubit computational") {
    // Gates, measurements, resets, and probes act within or onto H_C;
    // none of them moves a qubit between categories.
    NonComputationalPolicy policy;
    for (GateType gate : {GateType::H, GateType::X, GateType::Z, GateType::CZ, GateType::M,
                          GateType::R, GateType::RX, GateType::MR, GateType::EXP_VAL}) {
        QubitStatus out =
            normal_post_op_status(QubitStatus::Computational, gate, kPhysical, policy);
        REQUIRE(out == QubitStatus::Computational);
    }
}

TEST_CASE("normal_post_op_status: a gate leaves a noncomputational qubit untouched") {
    NonComputationalPolicy policy;
    QubitStatus out = normal_post_op_status(QubitStatus::LeakG, GateType::H, kPhysical, policy);
    REQUIRE(out == QubitStatus::LeakG);
}

TEST_CASE("normal_post_op_status: leaked reset restores; lost reset is policy-gated") {
    // Every reset flavor restores a leaked qubit to computational.
    NonComputationalPolicy policy;
    for (GateType reset : {GateType::R, GateType::MR, GateType::RX}) {
        QubitStatus leaked_r = normal_post_op_status(QubitStatus::LeakG, reset, kPhysical, policy);
        REQUIRE(leaked_r == QubitStatus::Computational);
    }

    QubitStatus lost_default =
        normal_post_op_status(QubitStatus::Lost, GateType::R, kPhysical, policy);
    REQUIRE(lost_default == QubitStatus::Lost);

    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    QubitStatus lost_restored =
        normal_post_op_status(QubitStatus::Lost, GateType::R, kPhysical, restore);
    REQUIRE(lost_restored == QubitStatus::Computational);
}

TEST_CASE("normal_post_op_status: feedback corrections change no status") {
    NonComputationalPolicy policy;
    // A virtual correction acts within H_C at most; in particular a
    // conditional X is not a reset, so it never restores a vacated site.
    QubitStatus comp =
        normal_post_op_status(QubitStatus::Computational, GateType::CX, kFeedbackX, policy);
    REQUIRE(comp == QubitStatus::Computational);
    QubitStatus leaked =
        normal_post_op_status(QubitStatus::LeakG, GateType::CX, kFeedbackX, policy);
    REQUIRE(leaked == QubitStatus::LeakG);
    QubitStatus z = normal_post_op_status(QubitStatus::LeakG, GateType::CZ, kFeedbackZ, policy);
    REQUIRE(z == QubitStatus::LeakG);
}
