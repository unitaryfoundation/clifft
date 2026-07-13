#include "clifft/circuit/circuit.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/status_walk.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <stdexcept>
#include <vector>

using Catch::Matchers::StartsWith;
using clifft::advance_ordinary_node;
using clifft::AstNode;
using clifft::Circuit;
using clifft::GateType;
using clifft::Level;
using clifft::NonComputationalPolicy;
using clifft::normal_post_op_status;
using clifft::OperandRole;
using clifft::OrdinaryStep;
using clifft::parse;
using clifft::qubit_operands;
using clifft::QubitOperand;
using clifft::QubitStatus;
using clifft::Target;

namespace {

AstNode node(GateType gate, std::vector<Target> targets, std::vector<double> args = {}) {
    return AstNode{gate, std::move(targets), std::move(args), 0};
}

}  // namespace

TEST_CASE("qubit_operands: physical gate yields Physical operands in target order") {
    std::vector<QubitOperand> ops =
        qubit_operands(node(GateType::CX, {Target::qubit(0), Target::qubit(1)}));
    REQUIRE(ops.size() == 2);
    REQUIRE(ops[0].qubit == 0);
    REQUIRE(ops[0].role == OperandRole::Physical);
    REQUIRE(ops[1].qubit == 1);
    REQUIRE(ops[1].role == OperandRole::Physical);
}

TEST_CASE("qubit_operands: MPP yields its Pauli-tagged qubits in target order as Physical") {
    Circuit c = parse("MPP X0*Z1*Y2");
    std::vector<QubitOperand> ops = qubit_operands(c.nodes.back());
    REQUIRE(ops.size() == 3);
    REQUIRE(ops[0].qubit == 0);
    REQUIRE(ops[1].qubit == 1);
    REQUIRE(ops[2].qubit == 2);
    REQUIRE(ops[0].role == OperandRole::Physical);
    REQUIRE(ops[1].role == OperandRole::Physical);
    REQUIRE(ops[2].role == OperandRole::Physical);
}

TEST_CASE("qubit_operands: CX feedback yields a single Feedback operand") {
    std::vector<QubitOperand> ops =
        qubit_operands(node(GateType::CX, {Target::rec(0), Target::qubit(1)}));
    REQUIRE(ops.size() == 1);
    REQUIRE(ops[0].qubit == 1);
    REQUIRE(ops[0].role == OperandRole::Feedback);
}

TEST_CASE("qubit_operands: CZ feedback yields a single Feedback operand") {
    std::vector<QubitOperand> ops =
        qubit_operands(node(GateType::CZ, {Target::rec(0), Target::qubit(1)}));
    REQUIRE(ops.size() == 1);
    REQUIRE(ops[0].qubit == 1);
    REQUIRE(ops[0].role == OperandRole::Feedback);
}

TEST_CASE("qubit_operands: parsed feedback matches the hand-built classification") {
    Circuit c = parse("M 0\nCX rec[-1] 1\n");
    std::vector<QubitOperand> ops = qubit_operands(c.nodes.back());
    REQUIRE(ops.size() == 1);
    REQUIRE(ops[0].qubit == 1);
    REQUIRE(ops[0].role == OperandRole::Feedback);
}

TEST_CASE("qubit_operands: non-qubit operations produce no operands") {
    REQUIRE(qubit_operands(node(GateType::MPAD, {Target::qubit(1)})).empty());
    REQUIRE(qubit_operands(node(GateType::DETECTOR, {Target::rec(0)})).empty());
    REQUIRE(qubit_operands(node(GateType::OBSERVABLE_INCLUDE, {Target::rec(0)})).empty());
    REQUIRE(qubit_operands(node(GateType::READOUT_NOISE, {Target::rec(0)}, {0.1})).empty());
}

TEST_CASE("qubit_operands: a record control with a qubit on a non-CX/CZ gate is rejected") {
    REQUIRE_THROWS_AS(qubit_operands(node(GateType::H, {Target::rec(0), Target::qubit(1)})),
                      std::invalid_argument);
}

TEST_CASE("advance_ordinary_node: two-qubit gate with one lost operand drops whole") {
    // CX 0 1 where qubit 1 is lost: the op drops, both statuses hold.
    Circuit c = parse("CX 0 1\n");
    const AstNode& node = c.nodes[0];
    std::vector<QubitStatus> status = {QubitStatus::Computational, QubitStatus::Lost};
    NonComputationalPolicy policy;
    OrdinaryStep step = advance_ordinary_node(node, 0, status, policy, "test");
    REQUIRE(step.dropped);
    REQUIRE(status[0] == QubitStatus::Computational);
    REQUIRE(status[1] == QubitStatus::Lost);
}

TEST_CASE("advance_ordinary_node: M on a leaked qubit reports the level and keeps status") {
    Circuit c = parse("M 0\n");
    const AstNode& node = c.nodes[0];
    std::vector<QubitStatus> status = {QubitStatus::LeakG};
    NonComputationalPolicy policy;
    OrdinaryStep step = advance_ordinary_node(node, 0, status, policy, "test");
    REQUIRE_FALSE(step.dropped);
    REQUIRE(step.measured_noncomp_level.has_value());
    REQUIRE(*step.measured_noncomp_level == Level::LeakG);
    REQUIRE(status[0] == QubitStatus::LeakG);  // measurement does not reset
}

TEST_CASE("advance_ordinary_node: MPP on a lost operand throws with caller prefix") {
    Circuit c = parse("MPP X0*Z1\n");
    const AstNode& node = c.nodes[0];
    std::vector<QubitStatus> status = {QubitStatus::Computational, QubitStatus::Lost};
    NonComputationalPolicy policy;
    REQUIRE_THROWS_WITH(advance_ordinary_node(node, 0, status, policy, "myfunc"),
                        StartsWith("myfunc:"));
}

TEST_CASE("normal_post_op_status: normal operations keep a computational qubit computational") {
    NonComputationalPolicy policy;
    for (GateType gate : {GateType::H, GateType::X, GateType::Z, GateType::CZ, GateType::M,
                          GateType::R, GateType::RX, GateType::MR, GateType::EXP_VAL}) {
        QubitStatus out = normal_post_op_status(QubitStatus::Computational, gate,
                                                OperandRole::Physical, policy);
        REQUIRE(out == QubitStatus::Computational);
    }
}

TEST_CASE("normal_post_op_status: a gate leaves a noncomputational qubit untouched") {
    NonComputationalPolicy policy;
    QubitStatus out = normal_post_op_status(QubitStatus::LeakG, GateType::H,
                                            OperandRole::Physical, policy);
    REQUIRE(out == QubitStatus::LeakG);
}

TEST_CASE("normal_post_op_status: leaked reset restores; lost reset is policy-gated") {
    NonComputationalPolicy policy;
    for (GateType reset : {GateType::R, GateType::MR, GateType::RX}) {
        QubitStatus leaked_r = normal_post_op_status(QubitStatus::LeakG, reset,
                                                     OperandRole::Physical, policy);
        REQUIRE(leaked_r == QubitStatus::Computational);
    }

    QubitStatus lost_default = normal_post_op_status(QubitStatus::Lost, GateType::R,
                                                     OperandRole::Physical, policy);
    REQUIRE(lost_default == QubitStatus::Lost);

    NonComputationalPolicy restore;
    restore.reset_restores_lost = true;
    QubitStatus lost_restored = normal_post_op_status(QubitStatus::Lost, GateType::R,
                                                      OperandRole::Physical, restore);
    REQUIRE(lost_restored == QubitStatus::Computational);
}

TEST_CASE("normal_post_op_status: feedback corrections change no status") {
    NonComputationalPolicy policy;
    QubitStatus comp = normal_post_op_status(QubitStatus::Computational, GateType::CX,
                                             OperandRole::Feedback, policy);
    REQUIRE(comp == QubitStatus::Computational);
    QubitStatus leaked = normal_post_op_status(QubitStatus::LeakG, GateType::CX,
                                               OperandRole::Feedback, policy);
    REQUIRE(leaked == QubitStatus::LeakG);
    QubitStatus z = normal_post_op_status(QubitStatus::LeakG, GateType::CZ,
                                          OperandRole::Feedback, policy);
    REQUIRE(z == QubitStatus::LeakG);
}
