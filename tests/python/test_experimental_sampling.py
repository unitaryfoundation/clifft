"""Conformance and boundary tests for the explicitly selected sampling backend."""

import numpy as np
import pytest

import clifft
import clifft.experimental as experimental


def test_program_is_separate_from_legacy_bytecode() -> None:
    legacy = clifft.compile("H 0\nT 0\nM 0")
    program = experimental.compile("H 0\nT 0\nM 0")

    assert isinstance(legacy, clifft.Program)
    assert isinstance(program, experimental.Program)
    assert not isinstance(program, clifft.Program)
    assert program.num_qubits == 1
    assert program.num_measurements == 1
    assert program.num_hidden_measurements == 0
    assert program.num_actions > 0


@pytest.mark.parametrize(
    "circuit",
    [
        "H 0\nM 0\nM 0",
        "H 0\nCX 0 1\nM 0 1",
        "H 0\nT 0\nS 0\nMX 0",
        "H 0\nT 0\nR_Z(0.3) 0\nM 0",
        "H 0\nH 1\nT 0\nT 1\nCX 0 1\nMPP Y0*Z1",
    ],
)
def test_seeded_samples_match_legacy_for_curated_circuits(circuit: str) -> None:
    legacy = clifft.sample(clifft.compile(circuit), shots=128, seed=1234).measurements
    actual = experimental.sample(experimental.compile(circuit), shots=128, seed=1234).measurements
    np.testing.assert_array_equal(actual, legacy)


@pytest.mark.parametrize(
    "circuit,operation",
    [
        ("X_ERROR(0.1) 0\nM 0", "NOISE"),
        ("H 0\nM 0\nCX rec[-1] 1", "CONDITIONAL_PAULI"),
        ("R 0", "CONDITIONAL_PAULI"),
        ("M 0\nDETECTOR rec[-1]", "DETECTOR"),
    ],
)
def test_unsupported_capabilities_fail_during_compile(circuit: str, operation: str) -> None:
    with pytest.raises(ValueError, match=operation):
        experimental.compile(circuit)
