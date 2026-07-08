#include "clifft/circuit/circuit.h"
#include "clifft/circuit/gate_data.h"
#include "clifft/circuit/parser.h"
#include "clifft/circuit/target.h"
#include "clifft/noncomp/op_role.h"
#include "clifft/noncomp/status_step.h"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

using clifft::AstNode;
using clifft::Circuit;
using clifft::GateType;
using clifft::OperandRole;
using clifft::parse;
using clifft::qubit_operands;
using clifft::QubitOperand;
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
