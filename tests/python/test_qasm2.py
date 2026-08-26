"""Native unitary OpenQASM 2 import and compilation contracts."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import assert_statevectors_equiv
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector

import clifft

ABSTRACTS_STYLE_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
s q[0];
t q[1];
cx q[0], q[2];
rx(0.5*pi) q[3];
"""


def test_parse_qasm2_exposes_lowered_circuit_and_phase() -> None:
    """The public import retains both the reusable AST and source phase."""
    imported = clifft.parse_qasm2(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u1(pi/2) q[0];
        """
    )

    assert isinstance(imported, clifft.Qasm2Import)
    assert isinstance(imported.circuit, clifft.Circuit)
    assert imported.num_qubits == 1
    assert len(imported) == 1
    assert imported.global_phase_half_turns == pytest.approx(0.25)
    assert imported.circuit.nodes[0].gate == clifft.GateType.U3


def test_compile_accepts_qasm2_as_an_explicit_input_format() -> None:
    """QASM input flows through the ordinary compiler after native lowering."""
    qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
    """
    program = clifft.compile(qasm, input_format="qasm2")
    probabilities = clifft.basis_probabilities(program, ["00", "01", "10", "11"])
    np.testing.assert_allclose(probabilities, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


def test_abstracts_style_qasm_matches_qiskit_statevector() -> None:
    """The benchmark's complete gate vocabulary agrees with an independent parser."""
    program = clifft.compile(ABSTRACTS_STYLE_QASM, hir_passes=None, input_format="qasm2")
    actual = clifft.get_statevector(program)
    expected = np.asarray(Statevector.from_instruction(qasm2.loads(ABSTRACTS_STYLE_QASM)).data)
    assert_statevectors_equiv(actual, expected)


def test_qasm2_euler_and_register_broadcast_match_qiskit() -> None:
    """Parameterized source semantics survive native register expansion."""
    source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg a[2];
        qreg b[2];
        h a;
        u3(pi/3, -pi/5, pi/7) b;
        cx a,b;
        cz a[0],b[1];
    """
    program = clifft.compile(source, hir_passes=None, input_format="qasm2")
    actual = clifft.get_statevector(program)
    expected = np.asarray(Statevector.from_instruction(qasm2.loads(source)).data)
    assert_statevectors_equiv(actual, expected)


@pytest.mark.parametrize("expression", ["-2^2", "2^-1", "2^3^2"])
def test_qasm2_power_precedence_matches_qiskit(expression: str) -> None:
    """Unary signs and right-associative powers follow Qiskit's grammar."""
    source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; ' f"rz({expression}) q[0];"
    imported = clifft.parse_qasm2(source)
    actual = imported.circuit.nodes[0].args[0] * np.pi
    expected = float(qasm2.loads(source).data[0].operation.params[0])
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "source",
    [
        "OPENQASM 2.0; qreg q[1]; U(0.31, -0.47, 0.59) q[0];",
        ('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; ' "u1(0.37) q[0];"),
        ('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; ' "u2(-0.23, 0.41) q[0];"),
        ('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; ' "u3(0.31, -0.47, 0.59) q[0];"),
        ('OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; ' "u3(0.31, -0.47, 0.59) q;"),
    ],
)
def test_qasm2_phase_sidecar_reconstructs_exact_qiskit_statevector(source: str) -> None:
    """The sidecar restores source amplitudes without phase alignment."""
    imported = clifft.parse_qasm2(source)
    internal = QuantumCircuit(imported.num_qubits)
    for node in imported.circuit.nodes:
        assert node.gate == clifft.GateType.U3
        theta, phi, lam = (angle * np.pi for angle in node.args)
        qubit = node.targets[0].value
        internal.rz(lam, qubit)
        internal.ry(theta, qubit)
        internal.rz(phi, qubit)

    internal_state = np.asarray(Statevector.from_instruction(internal).data)
    actual = np.exp(1j * np.pi * imported.global_phase_half_turns) * internal_state
    expected = np.asarray(Statevector.from_instruction(qasm2.loads(source)).data)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_parse_qasm2_file() -> None:
    """The file entry point returns the same import metadata."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as handle:
        handle.write(ABSTRACTS_STYLE_QASM)
        path = Path(handle.name)

    try:
        imported = clifft.parse_qasm2_file(str(path))
        assert imported.num_qubits == 4
        assert len(imported) == 4
    finally:
        path.unlink()


def test_compile_rejects_unknown_input_format() -> None:
    """Input selection is explicit rather than content-sniffed."""
    with pytest.raises(ValueError, match="input_format"):
        clifft.compile("H 0", input_format="qasm3")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "statement",
    [
        "creg c[1];",
        "reset q[0];",
        "measure q[0] -> c[0];",
        "if(c==1) x q[0];",
    ],
)
def test_qasm2_unitary_scope_rejects_dynamic_statements(statement: str) -> None:
    """The initial importer never assigns accidental semantics to dynamic circuits."""
    source = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        {statement}
    """
    with pytest.raises(clifft.ParseError, match="Non-unitary and classical"):
        clifft.parse_qasm2(source)


def test_qasm3_header_has_a_specific_error() -> None:
    """QASM 3 is reserved for a separately specified importer extension."""
    with pytest.raises(clifft.ParseError, match="OpenQASM 3 is not supported"):
        clifft.parse_qasm2("OPENQASM 3.0;")
