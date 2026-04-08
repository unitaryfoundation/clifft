"""Smoke tests for the Clifft Python API."""

import tempfile
from pathlib import Path

import pytest

import clifft


def test_version() -> None:
    """Test that version() returns a valid version string."""
    v = clifft.version()
    assert isinstance(v, str)
    assert len(v) > 0
    # Should be a semver-like string (e.g., "0.1.0")
    parts = v.split(".")
    assert len(parts) >= 2, f"Expected semver-like version, got: {v}"


def test_max_sim_qubits() -> None:
    """Test that max_sim_qubits() returns the expected value."""
    max_qubits = clifft.max_sim_qubits()
    assert max_qubits == 64


def test_module_version_attribute() -> None:
    """Test that __version__ matches version()."""
    assert clifft.__version__ == clifft.version()


# --------------------------------------------------------------------------
# Parser tests
# --------------------------------------------------------------------------


def test_parse_simple_circuit() -> None:
    """Test parsing a simple circuit."""
    circuit = clifft.parse("H 0\nCX 0 1\nM 0 1")
    assert len(circuit) == 4  # H, CX, M, M
    assert circuit.num_qubits == 2
    assert circuit.num_measurements == 2


def test_parse_circuit_nodes() -> None:
    """Test that circuit nodes have correct structure."""
    circuit = clifft.parse("H 0\nCX 0 1")
    assert len(circuit.nodes) == 2

    # Check H gate
    h_node = circuit.nodes[0]
    assert h_node.gate == clifft.GateType.H
    assert len(h_node.targets) == 1
    assert h_node.targets[0].value == 0
    assert not h_node.targets[0].is_rec

    # Check CX gate
    cx_node = circuit.nodes[1]
    assert cx_node.gate == clifft.GateType.CX
    assert len(cx_node.targets) == 2
    assert cx_node.targets[0].value == 0
    assert cx_node.targets[1].value == 1


def test_parse_mpp() -> None:
    """Test parsing MPP with Pauli-tagged targets."""
    circuit = clifft.parse("MPP X0*Z1")
    assert len(circuit) == 1
    assert circuit.nodes[0].gate == clifft.GateType.MPP
    assert len(circuit.nodes[0].targets) == 2

    # Check Pauli tags
    t0 = circuit.nodes[0].targets[0]
    assert t0.has_pauli
    assert t0.pauli_char == "X"
    assert t0.value == 0

    t1 = circuit.nodes[0].targets[1]
    assert t1.pauli_char == "Z"
    assert t1.value == 1


def test_parse_feedback() -> None:
    """Test parsing classical feedback with rec targets."""
    circuit = clifft.parse("M 0\nCX rec[-1] 1")
    assert len(circuit) == 2

    cx_node = circuit.nodes[1]
    assert cx_node.gate == clifft.GateType.CX
    assert cx_node.targets[0].is_rec
    assert cx_node.targets[0].value == 0  # rec[-1] resolved to 0
    assert not cx_node.targets[1].is_rec
    assert cx_node.targets[1].value == 1


def test_parse_error_unknown_gate() -> None:
    """Test that ParseError is raised for unknown gates."""
    with pytest.raises(clifft.ParseError) as exc_info:
        clifft.parse("BADGATE 0")
    assert "Unknown gate" in str(exc_info.value)
    assert "Line 1" in str(exc_info.value)


def test_parse_error_rec_out_of_bounds() -> None:
    """Test that ParseError is raised for out-of-bounds rec references."""
    with pytest.raises(clifft.ParseError) as exc_info:
        clifft.parse("CX rec[-1] 0")  # No measurements yet
    assert "out of bounds" in str(exc_info.value)


def test_repeat_unrolling() -> None:
    """Test that REPEAT blocks are unrolled correctly."""
    c = clifft.parse("REPEAT 3 {\nH 0\n}")
    assert len(c.nodes) == 3
    for node in c.nodes:
        assert node.gate == clifft.GateType.H


def test_repeat_safety_limit() -> None:
    """Test that exceeding max_ops raises ParseError."""
    with pytest.raises(clifft.ParseError) as exc_info:
        clifft.parse("REPEAT 10 {\nH 0\n}", max_ops=5)
    assert "exceeds maximum" in str(exc_info.value)


def test_parse_file() -> None:
    """Test parsing a circuit from a file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".stim", delete=False) as f:
        f.write("H 0\nCX 0 1\nM 0 1")
        f.flush()
        path = Path(f.name)

    try:
        circuit = clifft.parse_file(str(path))
        assert len(circuit) == 4
        assert circuit.num_qubits == 2
    finally:
        path.unlink()


def test_parse_file_not_found() -> None:
    """Test that RuntimeError is raised for missing files."""
    with pytest.raises(RuntimeError) as exc_info:
        clifft.parse_file("/nonexistent/path/circuit.stim")
    assert "Cannot open" in str(exc_info.value)


def test_circuit_repr() -> None:
    """Test Circuit __repr__."""
    circuit = clifft.parse("H 0\nM 0")
    repr_str = repr(circuit)
    assert "2 ops" in repr_str
    assert "1 qubits" in repr_str
    assert "1 measurements" in repr_str


def test_astnode_repr() -> None:
    """Test AstNode __repr__."""
    circuit = clifft.parse("CX 0 1")
    repr_str = repr(circuit.nodes[0])
    assert "CX" in repr_str
    assert "0" in repr_str
    assert "1" in repr_str


def test_target_repr() -> None:
    """Test Target __repr__."""
    circuit = clifft.parse("M 0\nCX rec[-1] 1")
    cx_node = circuit.nodes[1]

    # rec target
    rec_repr = repr(cx_node.targets[0])
    assert "rec[" in rec_repr

    # qubit target
    qubit_repr = repr(cx_node.targets[1])
    assert "1" in qubit_repr
