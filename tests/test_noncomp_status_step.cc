#include "clifft/circuit/gate_data.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/qubit_status.h"
#include "clifft/noncomp/status_step.h"

#include <catch2/catch_test_macros.hpp>

using clifft::GateType;
using clifft::LevelSet;
using clifft::NonComputationalPolicy;
using clifft::normal_post_op_status;
using clifft::OperandRole;
using clifft::QubitStatus;
using clifft::QubitStatusKind;
using clifft::step_status;
using clifft::TransitionOutcome;

namespace {

// Default-set ids: g=0, e=1, leak_g=2, leak_e=3, lost=4.
constexpr uint8_t kLeakG = 2;
constexpr uint8_t kLost = 4;

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
            normal_post_op_status(QubitStatus::computational(), gate, kPhysical, policy);
        REQUIRE(out.kind() == QubitStatusKind::Computational);
    }
}

TEST_CASE("normal_post_op_status: a gate leaves a noncomputational qubit untouched") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    QubitStatus out = normal_post_op_status(levels.leaked(kLeakG), GateType::H, kPhysical, policy);
    REQUIRE(out.kind() == QubitStatusKind::Leaked);
    REQUIRE(out.level_id() == kLeakG);
}

TEST_CASE("normal_post_op_status: leaked reset restores; lost reset is policy-gated") {
    LevelSet levels = LevelSet::default_set();

    // Every reset flavor restores a leaked qubit to computational.
    NonComputationalPolicy policy;
    for (GateType reset : {GateType::R, GateType::MR, GateType::RX}) {
        QubitStatus leaked_r =
            normal_post_op_status(levels.leaked(kLeakG), reset, kPhysical, policy);
        REQUIRE(leaked_r.kind() == QubitStatusKind::Computational);
    }

    QubitStatus lost_default =
        normal_post_op_status(levels.lost(kLost), GateType::R, kPhysical, policy);
    REQUIRE(lost_default.kind() == QubitStatusKind::Lost);

    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    QubitStatus lost_restored =
        normal_post_op_status(levels.lost(kLost), GateType::R, kPhysical, restore);
    REQUIRE(lost_restored.kind() == QubitStatusKind::Computational);
}

TEST_CASE("normal_post_op_status: feedback corrections change no status") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    // A virtual correction acts within H_C at most; in particular a
    // conditional X is not a reset, so it never restores a vacated site.
    QubitStatus comp =
        normal_post_op_status(QubitStatus::computational(), GateType::CX, kFeedbackX, policy);
    REQUIRE(comp.kind() == QubitStatusKind::Computational);
    QubitStatus leaked =
        normal_post_op_status(levels.leaked(kLeakG), GateType::CX, kFeedbackX, policy);
    REQUIRE(leaked.kind() == QubitStatusKind::Leaked);
    REQUIRE(leaked.level_id() == kLeakG);
    QubitStatus z = normal_post_op_status(levels.leaked(kLeakG), GateType::CZ, kFeedbackZ, policy);
    REQUIRE(z.kind() == QubitStatusKind::Leaked);
}

TEST_CASE("step_status: a jump destination wins over the normal post-op status") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    TransitionOutcome jump{true, kLeakG};
    QubitStatus out =
        step_status(QubitStatus::computational(), GateType::H, kPhysical, jump, policy, levels);
    REQUIRE(out.kind() == QubitStatusKind::Leaked);
    REQUIRE(out.level_id() == kLeakG);
}

TEST_CASE("step_status: no jump falls back to the normal post-op status") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    TransitionOutcome no_jump{false, clifft::kInvalidLevel};
    QubitStatus out =
        step_status(levels.leaked(kLeakG), GateType::R, kPhysical, no_jump, policy, levels);
    REQUIRE(out.kind() == QubitStatusKind::Computational);
}
