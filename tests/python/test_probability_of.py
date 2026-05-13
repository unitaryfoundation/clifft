"""Tests for clifft.probability_of(): exact measurement-record probabilities.

probability_of() returns the probability sample() would assign to each
measurement record under a compiled program with measurements. This module
covers the Python wrapper's contract: input polymorphism, return shapes,
return_log behavior, cross-checks against probabilities() and qiskit,
sampling consistency, and rejection paths.
"""

from typing import Any

import numpy as np
import pytest
from conftest import random_dense_clifford_t_circuit
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

import clifft

# =============================================================================
# Basic correctness: closed-form probabilities.
# =============================================================================


def test_bell_state_probabilities() -> None:
    prog = clifft.compile("H 0\nCX 0 1\nM 0 1")
    probs = clifft.probability_of(prog, ["00", "01", "10", "11"])
    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)
    assert probs.dtype == np.float64
    assert probs.shape == (4,)


def test_single_qubit_plus_state() -> None:
    prog = clifft.compile("H 0\nM 0")
    probs = clifft.probability_of(prog, ["0", "1"])
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-12)


def test_unreachable_records_are_zero() -> None:
    prog = clifft.compile("M 0")
    probs = clifft.probability_of(prog, ["0", "1"])
    np.testing.assert_allclose(probs, [1.0, 0.0], atol=1e-12)


def test_feedback_circuit_returns_joint_trajectory_probability() -> None:
    prog = clifft.compile("H 0\n" "M 0\n" "CX rec[-1] 1\n" "M 1\n")
    probs = clifft.probability_of(prog, ["00", "01", "10", "11"])
    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


# =============================================================================
# Input polymorphism.
# =============================================================================


def test_single_string_returns_length_one_array() -> None:
    prog = clifft.compile("H 0\nM 0")
    probs = clifft.probability_of(prog, "0")
    assert probs.shape == (1,)
    np.testing.assert_allclose(probs, [0.5], atol=1e-12)


def test_sequence_of_strings() -> None:
    prog = clifft.compile("H 0\nM 0")
    probs = clifft.probability_of(prog, ("0", "1"))
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8])
def test_array_input_matches_string_input(dtype: np.dtype) -> None:
    prog = clifft.compile("H 0\nCX 0 1\nM 0 1")
    records = np.array([[0, 0], [1, 1]], dtype=dtype)
    np.testing.assert_allclose(
        clifft.probability_of(prog, records),
        clifft.probability_of(prog, ["00", "11"]),
    )


def test_empty_record_batch() -> None:
    prog = clifft.compile("H 0\nM 0")
    probs = clifft.probability_of(prog, [])
    assert probs.shape == (0,)
    assert probs.dtype == np.float64


# =============================================================================
# return_log option.
# =============================================================================


def test_return_log_returns_natural_log() -> None:
    prog = clifft.compile("H 0\nCX 0 1\nM 0 1")
    log_probs = clifft.probability_of(prog, ["00", "01", "10", "11"], return_log=True)
    assert np.isclose(log_probs[0], np.log(0.5))
    assert log_probs[1] == -np.inf
    assert log_probs[2] == -np.inf
    assert np.isclose(log_probs[3], np.log(0.5))


def test_return_log_default_false() -> None:
    prog = clifft.compile("H 0\nM 0")
    probs = clifft.probability_of(prog, ["0"])
    # 0.5 (linear), not log(0.5).
    np.testing.assert_allclose(probs, [0.5], atol=1e-12)


# =============================================================================
# Cross-check against probabilities() on terminal-M-all circuits.
# =============================================================================


def test_matches_probabilities_on_clifford_circuit() -> None:
    bitstrings = ["00", "01", "10", "11"]
    unitary = clifft.compile("H 0\nCX 0 1")
    measured = clifft.compile("H 0\nCX 0 1\nM 0 1")

    expected = clifft.probabilities(unitary, bitstrings)
    actual = clifft.probability_of(measured, bitstrings)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_matches_probabilities_on_clifford_t_circuit() -> None:
    bitstrings = ["0", "1"]
    unitary = clifft.compile("H 0\nT 0\nH 0")
    measured = clifft.compile("H 0\nT 0\nH 0\nM 0")

    expected = clifft.probabilities(unitary, bitstrings)
    actual = clifft.probability_of(measured, bitstrings)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("num_qubits,seed", [(2, 101), (3, 202), (4, 303)])
def test_matches_qiskit_for_random_small_circuits(num_qubits: int, seed: int) -> None:
    circuit = random_dense_clifford_t_circuit(num_qubits, depth=18, seed=seed)
    measured = clifft.compile(circuit + "\nM " + " ".join(str(q) for q in range(num_qubits)))
    qiskit_sv = qiskit_statevector(stim_to_qiskit_noiseless(circuit))

    bitstrings = [
        "".join(str((i >> q) & 1) for q in range(num_qubits)) for i in range(1 << num_qubits)
    ]
    actual = clifft.probability_of(measured, bitstrings)
    np.testing.assert_allclose(actual, np.abs(qiskit_sv) ** 2, atol=1e-12)


# =============================================================================
# Empirical sampling consistency.
# =============================================================================


def test_sample_frequencies_match_probability_of() -> None:
    # Run sample() many shots, count frequencies, compare to probability_of().
    # Chi-squared style sanity check; not a deep statistical test.
    prog = clifft.compile("H 0\nT 0\nH 0\nM 0")

    shots = 200_000
    measurements = clifft.sample(prog, shots=shots, seed=42).measurements
    freq_1 = float(measurements.sum()) / shots
    freq_0 = 1.0 - freq_1

    probs = clifft.probability_of(prog, ["0", "1"])
    # 5 sigma binomial half-width on shots=2e5, p~0.85 is ~0.004.
    assert abs(freq_0 - probs[0]) < 0.01
    assert abs(freq_1 - probs[1]) < 0.01


# =============================================================================
# Rejection paths.
# =============================================================================


def test_rejects_zero_measurement_program() -> None:
    prog = clifft.compile("H 0\nT 0")
    with pytest.raises(ValueError, match="at least one measurement"):
        clifft.probability_of(prog, [])


def test_rejects_hidden_measurement_slots() -> None:
    prog = clifft.compile("M 0\nR 1\nM 1")
    assert prog.num_measurements == 2
    with pytest.raises(ValueError, match="hidden measurement slots"):
        clifft.probability_of(prog, ["00", "11"])


def test_rejects_noise_opcodes() -> None:
    prog = clifft.compile("X_ERROR(0.1) 0\nM 0")
    with pytest.raises(ValueError, match="pure-state evolution"):
        clifft.probability_of(prog, ["0"])


def test_rejects_detector_opcodes() -> None:
    prog = clifft.compile("M 0\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match="pure-state evolution"):
        clifft.probability_of(prog, ["0"])


def test_rejects_observable_opcodes() -> None:
    prog = clifft.compile("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    with pytest.raises(ValueError, match="pure-state evolution"):
        clifft.probability_of(prog, ["0"])


def test_rejects_record_string_with_wrong_length() -> None:
    prog = clifft.compile("M 0 1")
    with pytest.raises(ValueError, match="length 1, expected 2"):
        clifft.probability_of(prog, "0")


def test_rejects_record_string_with_invalid_chars() -> None:
    prog = clifft.compile("M 0")
    with pytest.raises(ValueError, match="expected only '0' and '1'"):
        clifft.probability_of(prog, ["x"])


def test_rejects_array_with_wrong_columns() -> None:
    prog = clifft.compile("M 0 1")
    arr = np.array([[0, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="3 columns, expected 2"):
        clifft.probability_of(prog, arr)


def test_rejects_array_with_non_bit_values() -> None:
    prog = clifft.compile("M 0")
    arr = np.array([[2]], dtype=np.uint8)
    with pytest.raises(ValueError, match="contain only 0 and 1"):
        clifft.probability_of(prog, arr)


def test_rejects_invalid_array_dtype() -> None:
    prog = clifft.compile("M 0")
    invalid: Any = np.array([[0]], dtype=np.int64)
    with pytest.raises(TypeError, match="dtype must be bool or uint8"):
        clifft.probability_of(prog, invalid)


def test_rejects_invalid_array_dim() -> None:
    prog = clifft.compile("M 0")
    with pytest.raises(ValueError, match="must be 2D"):
        clifft.probability_of(prog, np.array([0, 1], dtype=np.uint8))


def test_rejects_invalid_input_type() -> None:
    prog = clifft.compile("M 0")
    bad: Any = 42
    with pytest.raises(TypeError, match="strings or a 2D"):
        clifft.probability_of(prog, bad)
