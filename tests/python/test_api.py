"""Smoke tests for the UCC Python API."""

import ucc


def test_version() -> None:
    """Test that version() returns a valid version string."""
    v = ucc.version()
    assert isinstance(v, str)
    assert len(v) > 0
    # Should be a semver-like string (e.g., "0.1.0")
    parts = v.split(".")
    assert len(parts) >= 2, f"Expected semver-like version, got: {v}"


def test_max_sim_qubits() -> None:
    """Test that max_sim_qubits() returns the expected value."""
    max_qubits = ucc.max_sim_qubits()
    assert max_qubits == 64


def test_module_version_attribute() -> None:
    """Test that __version__ matches version()."""
    assert ucc.__version__ == ucc.version()
