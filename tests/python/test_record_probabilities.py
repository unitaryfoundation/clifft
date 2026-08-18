"""Tests for clifft.record_probabilities(): exact measurement-record probabilities.

record_probabilities() returns the probability sample() would assign to each
measurement record under a compiled program with measurements. This module
covers the Python wrapper's contract: input polymorphism, return shapes,
return_log behavior, cross-checks against basis_probabilities() and qiskit,
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


def test_bell_state_basis_probabilities(sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nCX 0 1\nM 0 1")
    probs = sampling_api.record_probabilities(prog, ["00", "01", "10", "11"])
    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)
    assert probs.dtype == np.float64
    assert probs.shape == (4,)


def test_single_qubit_plus_state(sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nM 0")
    probs = sampling_api.record_probabilities(prog, ["0", "1"])
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-12)


def test_unreachable_records_are_zero(sampling_api: Any) -> None:
    prog = sampling_api.compile("M 0")
    probs = sampling_api.record_probabilities(prog, ["0", "1"])
    np.testing.assert_allclose(probs, [1.0, 0.0], atol=1e-12)


def test_feedback_circuit_returns_joint_trajectory_probability() -> None:
    prog = clifft.compile("H 0\n" "M 0\n" "CX rec[-1] 1\n" "M 1\n")
    probs = clifft.record_probabilities(prog, ["00", "01", "10", "11"])
    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


# =============================================================================
# Input polymorphism.
# =============================================================================


def test_single_string_returns_length_one_array(sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nM 0")
    probs = sampling_api.record_probabilities(prog, "0")
    assert probs.shape == (1,)
    np.testing.assert_allclose(probs, [0.5], atol=1e-12)


def test_sequence_of_strings(sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nM 0")
    probs = sampling_api.record_probabilities(prog, ("0", "1"))
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8])
def test_array_input_matches_string_input(dtype: np.dtype, sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nCX 0 1\nM 0 1")
    records = np.array([[0, 0], [1, 1]], dtype=dtype)
    np.testing.assert_allclose(
        sampling_api.record_probabilities(prog, records),
        sampling_api.record_probabilities(prog, ["00", "11"]),
    )


def test_empty_record_batch(sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nM 0")
    probs = sampling_api.record_probabilities(prog, [])
    assert probs.shape == (0,)
    assert probs.dtype == np.float64


# =============================================================================
# return_log option.
# =============================================================================


def test_return_log_returns_natural_log(sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nCX 0 1\nM 0 1")
    log_probs = sampling_api.record_probabilities(prog, ["00", "01", "10", "11"], return_log=True)
    assert np.isclose(log_probs[0], np.log(0.5))
    assert log_probs[1] == -np.inf
    assert log_probs[2] == -np.inf
    assert np.isclose(log_probs[3], np.log(0.5))


def test_return_log_default_false(sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nM 0")
    probs = sampling_api.record_probabilities(prog, ["0"])
    # 0.5 (linear), not log(0.5).
    np.testing.assert_allclose(probs, [0.5], atol=1e-12)


# =============================================================================
# Cross-check against basis_probabilities() on terminal-M-all circuits.
# =============================================================================


def test_matches_probabilities_on_clifford_circuit(sampling_api: Any) -> None:
    bitstrings = ["00", "01", "10", "11"]
    unitary = clifft.compile("H 0\nCX 0 1")
    measured = sampling_api.compile("H 0\nCX 0 1\nM 0 1")

    expected = clifft.basis_probabilities(unitary, bitstrings)
    actual = sampling_api.record_probabilities(measured, bitstrings)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_matches_probabilities_on_clifford_t_circuit(sampling_api: Any) -> None:
    bitstrings = ["0", "1"]
    unitary = clifft.compile("H 0\nT 0\nH 0")
    measured = sampling_api.compile("H 0\nT 0\nH 0\nM 0")

    expected = clifft.basis_probabilities(unitary, bitstrings)
    actual = sampling_api.record_probabilities(measured, bitstrings)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("num_qubits,seed", [(2, 101), (3, 202), (4, 303)])
def test_matches_qiskit_for_random_small_circuits(
    num_qubits: int, seed: int, sampling_api: Any
) -> None:
    circuit = random_dense_clifford_t_circuit(num_qubits, depth=18, seed=seed)
    measured = sampling_api.compile(circuit + "\nM " + " ".join(str(q) for q in range(num_qubits)))
    qiskit_sv = qiskit_statevector(stim_to_qiskit_noiseless(circuit))

    bitstrings = [
        "".join(str((i >> q) & 1) for q in range(num_qubits)) for i in range(1 << num_qubits)
    ]
    actual = sampling_api.record_probabilities(measured, bitstrings)
    np.testing.assert_allclose(actual, np.abs(qiskit_sv) ** 2, atol=1e-12)


@pytest.mark.parametrize(
    "circuit,num_qubits",
    [
        ("H 0\nR_Z(0.25) 0\nH 0", 1),
        ("R_X(0.5) 0", 1),
        ("R_Y(0.3) 0", 1),
        ("U3(0.5, 0.25, 0.125) 0", 1),
        ("H 0\nH 1\nR_ZZ(0.3) 0 1\nH 0\nH 1", 2),
        ("R_XX(0.4) 0 1", 2),
        ("R_YY(0.2) 0 1", 2),
        ("R_PAULI(0.1) X0*Y1*Z2", 3),
    ],
    ids=["rz", "rx", "ry", "u3", "rzz", "rxx", "ryy", "r-pauli"],
)
def test_matches_qiskit_for_arbitrary_rotation_circuits(
    circuit: str, num_qubits: int, sampling_api: Any
) -> None:
    measured = sampling_api.compile(circuit + "\nM " + " ".join(map(str, range(num_qubits))))
    qiskit_sv = qiskit_statevector(stim_to_qiskit_noiseless(circuit))
    bitstrings = [
        "".join(str((basis >> q) & 1) for q in range(num_qubits))
        for basis in range(1 << num_qubits)
    ]

    actual = sampling_api.record_probabilities(measured, bitstrings)
    np.testing.assert_allclose(actual, np.abs(qiskit_sv) ** 2, atol=1e-12)


# =============================================================================
# Empirical sampling consistency.
# =============================================================================


def test_sample_frequencies_match_record_probabilities(sampling_api: Any) -> None:
    # Run sample() many shots, count frequencies, compare to record_probabilities().
    # Chi-squared style sanity check; not a deep statistical test.
    prog = sampling_api.compile("H 0\nT 0\nH 0\nM 0")

    shots = 200_000
    measurements = sampling_api.sample(prog, shots=shots, seed=42).measurements
    freq_1 = float(measurements.sum()) / shots
    freq_0 = 1.0 - freq_1

    probs = sampling_api.record_probabilities(prog, ["0", "1"])
    # 5 sigma binomial half-width on shots=2e5, p~0.85 is ~0.004.
    assert abs(freq_0 - probs[0]) < 0.01
    assert abs(freq_1 - probs[1]) < 0.01


# =============================================================================
# Rejection paths.
# =============================================================================


def test_rejects_zero_measurement_program(sampling_api: Any) -> None:
    prog = sampling_api.compile("H 0\nT 0")
    with pytest.raises(ValueError, match="at least one measurement"):
        sampling_api.record_probabilities(prog, [])


def test_rejects_zero_measurement_program_with_real_records(sampling_api: Any) -> None:
    # The wrapper formats records against program.num_measurements before
    # calling into C++. Without an explicit zero-measurement guard, this
    # surfaces as a confusing record-length mismatch ("expected 0") instead
    # of pointing the user at basis_probabilities().
    prog = sampling_api.compile("H 0")
    with pytest.raises(ValueError, match="use clifft.basis_probabilities"):
        sampling_api.record_probabilities(prog, ["0"])


def test_rejects_hidden_measurement_slots() -> None:
    prog = clifft.compile("M 0\nR 1\nM 1")
    assert prog.num_measurements == 2
    with pytest.raises(ValueError, match="hidden measurement slots"):
        clifft.record_probabilities(prog, ["00", "11"])


def test_rejects_noise_operations() -> None:
    prog = clifft.compile("X_ERROR(0.1) 0\nM 0")
    with pytest.raises(ValueError, match="pure-state evolution"):
        clifft.record_probabilities(prog, ["0"])


def test_rejects_detector_operations() -> None:
    prog = clifft.compile("M 0\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match="pure-state evolution"):
        clifft.record_probabilities(prog, ["0"])


def test_rejects_observable_operations() -> None:
    prog = clifft.compile("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    with pytest.raises(ValueError, match="pure-state evolution"):
        clifft.record_probabilities(prog, ["0"])


def test_rejects_record_string_with_wrong_length(sampling_api: Any) -> None:
    prog = sampling_api.compile("M 0 1")
    with pytest.raises(ValueError, match="length 1, expected 2"):
        sampling_api.record_probabilities(prog, "0")


def test_rejects_record_string_with_invalid_chars(sampling_api: Any) -> None:
    prog = sampling_api.compile("M 0")
    with pytest.raises(ValueError, match="expected only '0' and '1'"):
        sampling_api.record_probabilities(prog, ["x"])


def test_rejects_array_with_wrong_columns(sampling_api: Any) -> None:
    prog = sampling_api.compile("M 0 1")
    arr = np.array([[0, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="3 columns, expected 2"):
        sampling_api.record_probabilities(prog, arr)


def test_rejects_array_with_non_bit_values(sampling_api: Any) -> None:
    prog = sampling_api.compile("M 0")
    arr = np.array([[2]], dtype=np.uint8)
    with pytest.raises(ValueError, match="contain only 0 and 1"):
        sampling_api.record_probabilities(prog, arr)


def test_rejects_invalid_array_dtype(sampling_api: Any) -> None:
    prog = sampling_api.compile("M 0")
    invalid: Any = np.array([[0]], dtype=np.int64)
    with pytest.raises(TypeError, match="dtype must be bool or uint8"):
        sampling_api.record_probabilities(prog, invalid)


def test_rejects_invalid_array_dim(sampling_api: Any) -> None:
    prog = sampling_api.compile("M 0")
    with pytest.raises(ValueError, match="must be 2D"):
        sampling_api.record_probabilities(prog, np.array([0, 1], dtype=np.uint8))


def test_rejects_invalid_input_type(sampling_api: Any) -> None:
    prog = sampling_api.compile("M 0")
    bad: Any = 42
    with pytest.raises(TypeError, match="strings or a 2D"):
        sampling_api.record_probabilities(prog, bad)
