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
constexpr uint8_t kG = 0;
constexpr uint8_t kE = 1;
constexpr uint8_t kLeakG = 2;
constexpr uint8_t kLost = 4;

constexpr OperandRole kPhysical = OperandRole::Physical;
constexpr OperandRole kFeedback = OperandRole::Feedback;

}  // namespace

TEST_CASE("normal_post_op_status: a quantum gate demotes a known computational qubit") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    QubitStatus out = normal_post_op_status(levels.computational_known(kG), GateType::H, kPhysical,
                                            policy, levels);
    REQUIRE(out.kind() == QubitStatusKind::ComputationalUnknown);
}

TEST_CASE("normal_post_op_status: a Z-basis M preserves the pre-SVM-known status") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    QubitStatus known = normal_post_op_status(levels.computational_known(kE), GateType::M,
                                              kPhysical, policy, levels);
    REQUIRE(known.kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(known.level_id() == kE);
    QubitStatus unknown = normal_post_op_status(QubitStatus::computational_unknown(), GateType::M,
                                                kPhysical, policy, levels);
    REQUIRE(unknown.kind() == QubitStatusKind::ComputationalUnknown);
}

TEST_CASE("normal_post_op_status: Z-basis reset yields Known(g), X/Y reset yields Unknown") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    QubitStatus r = normal_post_op_status(QubitStatus::computational_unknown(), GateType::R,
                                          kPhysical, policy, levels);
    REQUIRE(r.kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(r.level_id() == kG);
    QubitStatus rx = normal_post_op_status(levels.computational_known(kG), GateType::RX, kPhysical,
                                           policy, levels);
    REQUIRE(rx.kind() == QubitStatusKind::ComputationalUnknown);
}

TEST_CASE("normal_post_op_status: a non-destructive probe preserves status") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    QubitStatus out = normal_post_op_status(levels.computational_known(kG), GateType::EXP_VAL,
                                            kPhysical, policy, levels);
    REQUIRE(out.kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(out.level_id() == kG);
}

TEST_CASE("normal_post_op_status: a gate leaves a noncomputational qubit untouched") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    QubitStatus out =
        normal_post_op_status(levels.leaked(kLeakG), GateType::H, kPhysical, policy, levels);
    REQUIRE(out.kind() == QubitStatusKind::Leaked);
    REQUIRE(out.level_id() == kLeakG);
}

TEST_CASE("normal_post_op_status: leaked reset restores; lost reset is policy-gated") {
    LevelSet levels = LevelSet::default_set();

    NonComputationalPolicy policy;
    QubitStatus leaked_r =
        normal_post_op_status(levels.leaked(kLeakG), GateType::R, kPhysical, policy, levels);
    REQUIRE(leaked_r.kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(leaked_r.level_id() == kG);

    QubitStatus lost_default =
        normal_post_op_status(levels.lost(kLost), GateType::R, kPhysical, policy, levels);
    REQUIRE(lost_default.kind() == QubitStatusKind::Lost);

    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    QubitStatus lost_restored =
        normal_post_op_status(levels.lost(kLost), GateType::R, kPhysical, restore, levels);
    REQUIRE(lost_restored.kind() == QubitStatusKind::ComputationalKnown);
    REQUIRE(lost_restored.level_id() == kG);
}

TEST_CASE("normal_post_op_status: feedback demotes a known qubit and never restores") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    // A conditional Pauli on a known qubit may flip g<->e unknowably, so
    // it demotes -- even for a gate (CX) that would otherwise act.
    QubitStatus known = normal_post_op_status(levels.computational_known(kG), GateType::CX,
                                              kFeedback, policy, levels);
    REQUIRE(known.kind() == QubitStatusKind::ComputationalUnknown);
    // Noncomputational qubits are untouched by a virtual correction (no
    // reset-restore on feedback).
    QubitStatus leaked =
        normal_post_op_status(levels.leaked(kLeakG), GateType::CX, kFeedback, policy, levels);
    REQUIRE(leaked.kind() == QubitStatusKind::Leaked);
    REQUIRE(leaked.level_id() == kLeakG);
}

TEST_CASE("step_status: a jump destination wins over the normal post-op status") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    TransitionOutcome jump{true, kLeakG};
    QubitStatus out =
        step_status(levels.computational_known(kG), GateType::H, kPhysical, jump, policy, levels);
    REQUIRE(out.kind() == QubitStatusKind::Leaked);
    REQUIRE(out.level_id() == kLeakG);
}

TEST_CASE("step_status: no jump falls back to the normal post-op status") {
    LevelSet levels = LevelSet::default_set();
    NonComputationalPolicy policy;
    TransitionOutcome no_jump{false, clifft::kInvalidLevel};
    QubitStatus out = step_status(levels.computational_known(kG), GateType::H, kPhysical, no_jump,
                                  policy, levels);
    REQUIRE(out.kind() == QubitStatusKind::ComputationalUnknown);
}
