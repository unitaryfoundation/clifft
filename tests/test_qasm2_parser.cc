#include "clifft/circuit/parser.h"
#include "clifft/circuit/qasm2_parser.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <string>
#include <vector>

using namespace clifft;

TEST_CASE("OpenQASM 2 imports the ABSTRACTS gate vocabulary") {
    const Qasm2Import imported = parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        s q[0];
        t q[1];
        cx q[0], q[2];
        rx(0.5*pi) q[3];
    )");

    REQUIRE(imported.circuit.num_qubits == 4);
    REQUIRE(imported.circuit.nodes.size() == 4);
    CHECK(imported.circuit.nodes[0].gate == GateType::S);
    CHECK(imported.circuit.nodes[1].gate == GateType::T);
    CHECK(imported.circuit.nodes[2].gate == GateType::CX);
    CHECK(imported.circuit.nodes[2].targets[0].value() == 0);
    CHECK(imported.circuit.nodes[2].targets[1].value() == 2);
    CHECK(imported.circuit.nodes[3].gate == GateType::R_X);
    REQUIRE(imported.circuit.nodes[3].args.size() == 1);
    CHECK(imported.circuit.nodes[3].args[0] == Catch::Approx(0.5));
    CHECK(imported.global_phase_half_turns == 0.0);
}

TEST_CASE("OpenQASM 2 preserves declared widths and broadcasts registers") {
    const Qasm2Import imported = parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg control[2];
        qreg target[2];
        h control;
        cx control, target;
        cz control[0], target;
    )");

    REQUIRE(imported.circuit.num_qubits == 4);
    REQUIRE(imported.circuit.nodes.size() == 6);
    CHECK(imported.circuit.nodes[0].targets[0].value() == 0);
    CHECK(imported.circuit.nodes[1].targets[0].value() == 1);
    CHECK(imported.circuit.nodes[2].targets[0].value() == 0);
    CHECK(imported.circuit.nodes[2].targets[1].value() == 2);
    CHECK(imported.circuit.nodes[3].targets[0].value() == 1);
    CHECK(imported.circuit.nodes[3].targets[1].value() == 3);
    CHECK(imported.circuit.nodes[4].targets[0].value() == 0);
    CHECK(imported.circuit.nodes[4].targets[1].value() == 2);
    CHECK(imported.circuit.nodes[5].targets[0].value() == 0);
    CHECK(imported.circuit.nodes[5].targets[1].value() == 3);
}

TEST_CASE("OpenQASM 2 evaluates constant angle expressions") {
    const Qasm2Import imported = parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        rx(sqrt(4)*pi/4) q[0];
        ry(2^3*pi/8) q[1];
        rz(cos(0)*pi) q[2];
        rz(-2^2) q[3];
        rz(2^-1) q[4];
        rz(2^3^2) q[5];
    )");

    REQUIRE(imported.circuit.nodes.size() == 6);
    CHECK(imported.circuit.nodes[0].args[0] == Catch::Approx(0.5));
    CHECK(imported.circuit.nodes[1].args[0] == Catch::Approx(1.0));
    CHECK(imported.circuit.nodes[2].args[0] == Catch::Approx(1.0));
    CHECK(imported.circuit.nodes[3].args[0] == Catch::Approx(-4.0 / std::numbers::pi));
    CHECK(imported.circuit.nodes[4].args[0] == Catch::Approx(0.5 / std::numbers::pi));
    CHECK(imported.circuit.nodes[5].args[0] == Catch::Approx(512.0 / std::numbers::pi));
}

TEST_CASE("OpenQASM 2 lowers Euler gates with their source phase") {
    const Qasm2Import u1 = parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u1(pi/2) q[0];
    )");
    REQUIRE(u1.circuit.nodes.size() == 1);
    CHECK(u1.circuit.nodes[0].gate == GateType::U3);
    REQUIRE(u1.circuit.nodes[0].args.size() == 3);
    CHECK(u1.circuit.nodes[0].args[0] == 0.0);
    CHECK(u1.circuit.nodes[0].args[1] == 0.0);
    CHECK(u1.circuit.nodes[0].args[2] == Catch::Approx(0.5));
    CHECK(u1.global_phase_half_turns == Catch::Approx(0.25));

    const Qasm2Import u2 = parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u2(pi/3, pi/7) q[0];
    )");
    REQUIRE(u2.circuit.nodes.size() == 1);
    REQUIRE(u2.circuit.nodes[0].args.size() == 3);
    CHECK(u2.circuit.nodes[0].args[0] == Catch::Approx(0.5));
    CHECK(u2.circuit.nodes[0].args[1] == Catch::Approx(1.0 / 3.0));
    CHECK(u2.circuit.nodes[0].args[2] == Catch::Approx(1.0 / 7.0));
    CHECK(u2.global_phase_half_turns == Catch::Approx(5.0 / 21.0));

    const Qasm2Import builtin = parse_qasm2(R"(
        OPENQASM 2.0;
        qreg q[1];
        U(pi/3, pi/5, -pi/7) q[0];
    )");
    REQUIRE(builtin.circuit.nodes.size() == 1);
    REQUIRE(builtin.circuit.nodes[0].args.size() == 3);
    CHECK(builtin.circuit.nodes[0].args[0] == Catch::Approx(1.0 / 3.0));
    CHECK(builtin.circuit.nodes[0].args[1] == Catch::Approx(1.0 / 5.0));
    CHECK(builtin.circuit.nodes[0].args[2] == Catch::Approx(-1.0 / 7.0));
    CHECK(builtin.global_phase_half_turns == Catch::Approx(1.0 / 35.0));
}

TEST_CASE("OpenQASM 2 phase corrections include register broadcast") {
    const Qasm2Import imported = parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        u1(pi/2) q;
    )");
    CHECK(imported.circuit.nodes.size() == 3);
    CHECK(imported.global_phase_half_turns == Catch::Approx(0.75));
}

TEST_CASE("OpenQASM 2 accepts comments barriers and builtins") {
    const std::string source =
        "/* author \xC3\xA9 */ OPENQASM 2.0;\n"
        "qreg q[2]; // \xCE\xB8 rotation\n"
        "U(0, 0, 0) q[0];\n"
        "CX q[0], q[1];\n"
        "barrier q[0], q[1];\n";
    const Qasm2Import imported = parse_qasm2(source);
    REQUIRE(imported.circuit.nodes.size() == 2);
    CHECK(imported.circuit.nodes[0].source_line == 3);
    CHECK(imported.circuit.nodes[1].source_line == 4);
}

TEST_CASE("OpenQASM 2 reports non-ASCII syntax at its source line") {
    const std::string source =
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "\xCE\xB8 q[0];\n";
    try {
        (void)parse_qasm2(source);
        FAIL("Expected ParseError");
    } catch (const ParseError& error) {
        CHECK(error.line() == 3);
    }
}

TEST_CASE("OpenQASM 2 identity preserves metadata without an operation") {
    const Qasm2Import imported = parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[9];
        id q;
    )");
    CHECK(imported.circuit.num_qubits == 9);
    CHECK(imported.circuit.nodes.empty());
}

TEST_CASE("OpenQASM 2 rejects unsupported language features") {
    const std::vector<std::string> sources = {
        "OPENQASM 3.0;",
        "OPENQASM 2.0; include \"other.inc\";",
        "OPENQASM 2.0; creg c[1];",
        "OPENQASM 2.0; qreg q[1]; reset q[0];",
        "OPENQASM 2.0; qreg q[1]; measure q[0];",
        "OPENQASM 2.0; gate custom a { U(0,0,0) a; }",
        "OPENQASM 2.0; opaque custom a;",
    };
    for (const std::string& source : sources) {
        CAPTURE(source);
        CHECK_THROWS_AS(parse_qasm2(source), ParseError);
    }
}

TEST_CASE("OpenQASM 2 validates registers gates and expressions") {
    const std::vector<std::string> sources = {
        "OPENQASM 2.0; qreg q[0];",
        "OPENQASM 2.0; qreg q[1]; qreg q[1];",
        "OPENQASM 2.0; qreg q[1]; x q[0];",
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; x q[1];",
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; rx(1/0) q[0];",
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; rx(foo(1)) q[0];",
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; ch q[0],q[0];",
        "OPENQASM 2.0; qreg q[1]; CX q[0],q[0];",
    };
    for (const std::string& source : sources) {
        CAPTURE(source);
        CHECK_THROWS_AS(parse_qasm2(source), ParseError);
    }
}

TEST_CASE("OpenQASM 2 validates register broadcast and operation limits") {
    CHECK_THROWS_AS(parse_qasm2(R"(
        OPENQASM 2.0;
        qreg a[2];
        qreg b[3];
        CX a,b;
    )"),
                    ParseError);

    CHECK_THROWS_AS(parse_qasm2(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q;
    )",
                                2),
                    ParseError);
}
