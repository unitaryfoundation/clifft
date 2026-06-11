"""Tests for the Pauli-twirl helpers and Pauli-noise passthrough.

The helper math is checked against closed forms. The twirl is an
approximation -- it preserves the Pauli-transfer-matrix diagonal, not circuit
statistics -- so the integration tests pin both sides of that contract
against clifft's exact near-Clifford simulation: one selected configuration
where the twirled channel and the exact unitary coincide, and two
counterexamples (a single twirled Hadamard, and coherent accumulation of two
S gates) where they measurably do not.

The passthrough tests confirm ordinary Pauli noise instructions behave
identically through the noncomputational path: statistically unchanged on
computational qubits, dropped on leaked/lost operands.
"""

from __future__ import annotations

import numpy as np
import pytest

import clifft
from clifft import noncomp, twirl

BAND = 0.04  # ~7 sigma at 8000 shots for p near 0.5
SHOTS = 8000

S_MATRIX = np.array([[1, 0], [0, 1j]], dtype=complex)


def _p1(text: str, slot: int = 0, seed: int = 1) -> float:
    meas = np.asarray(clifft.sample(clifft.compile(text), SHOTS, seed).measurements)
    return float(meas[:, slot].mean())


# --- Helper math -------------------------------------------------------------


def test_rotation_closed_form():
    theta = 0.37
    p = np.sin(theta / 2) ** 2
    assert twirl.rotation("Z", theta) == pytest.approx((0.0, 0.0, p))
    assert twirl.rotation("X", theta) == pytest.approx((p, 0.0, 0.0))
    assert twirl.rotation("y", theta) == pytest.approx((0.0, p, 0.0))
    with pytest.raises(ValueError, match="axis"):
        twirl.rotation("Q", theta)


def test_pauli_probabilities_matches_rotation():
    theta = 0.81
    rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
    assert twirl.pauli_probabilities(rz) == pytest.approx(twirl.rotation("Z", theta))


def test_pauli_probabilities_known_cases():
    # X twirls to a certain X flip; H splits evenly between X and Z; the
    # result ignores global phase (S equals RZ(pi/2) up to one).
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    assert twirl.pauli_probabilities(x) == pytest.approx((1.0, 0.0, 0.0))
    assert twirl.pauli_probabilities(h) == pytest.approx((0.5, 0.0, 0.5))
    assert twirl.pauli_probabilities(S_MATRIX) == pytest.approx((0.0, 0.0, 0.5))


def test_pauli_probabilities_validates_input():
    with pytest.raises(ValueError, match="2x2"):
        twirl.pauli_probabilities(np.eye(3))
    with pytest.raises(ValueError, match="unitary"):
        twirl.pauli_probabilities([[1, 0], [0, 2]])


# --- The twirl is an approximation: one matching case, two counterexamples ----


def test_selected_case_where_twirl_matches_exact():
    # One S between Hadamards happens to coincide: the coherence the twirl
    # discards does not reach this measurement, so the exact simulation and
    # the twirled Z_ERROR(0.5) give the same marginal. A selected case, not
    # a general single-use property -- see the counterexamples below.
    (_, _, p_z) = twirl.pauli_probabilities(S_MATRIX)
    exact = _p1("H 0\nS 0\nH 0\nM 0\n")
    twirled = _p1(f"H 0\nZ_ERROR({p_z}) 0\nH 0\nM 0\n")
    assert abs(exact - twirled) < BAND


def test_single_application_counterexample():
    # Twirling even one application can be wrong: exact H H is the identity
    # (P(M=1) = 0), while twirling the second H into its Pauli channel
    # (0.5, 0, 0.5) scrambles the |+> state into a coin flip.
    (px, py, pz) = twirl.pauli_probabilities(
        np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    )
    exact = _p1("H 0\nH 0\nM 0\n")
    twirled = _p1(f"H 0\nPAULI_CHANNEL_1({px}, {py}, {pz}) 0\nM 0\n")
    assert exact == pytest.approx(0.0)
    assert abs(twirled - 0.5) < BAND


def test_coherent_accumulation_counterexample():
    # Two S gates compose to an exact Z (P(M=1) = 1 between Hadamards); two
    # independent twirled channels only flip with probability 1/2.
    (_, _, p_z) = twirl.pauli_probabilities(S_MATRIX)
    exact = _p1("H 0\nS 0\nS 0\nH 0\nM 0\n")
    twirled = _p1(f"H 0\nZ_ERROR({p_z}) 0\nZ_ERROR({p_z}) 0\nH 0\nM 0\n")
    assert exact == pytest.approx(1.0)
    assert abs(twirled - 0.5) < BAND


# --- Pauli noise passes through the noncomputational path ---------------------


def test_pauli_noise_unchanged_through_noncomp_path():
    lossless = noncomp.Model(initial_state=[1.0, 0.0, 0.0, 0.0, 0.0])
    r = noncomp.sample("H 0\nZ_ERROR(0.3) 0\nH 0\nM 0\n", lossless, shots=SHOTS, seed=2)
    assert abs(np.asarray(r.measurements)[:, 0].mean() - 0.3) < BAND

    (px, py, pz) = twirl.pauli_probabilities(S_MATRIX)
    r = noncomp.sample(
        f"H 0\nPAULI_CHANNEL_1({px}, {py}, {pz}) 0\nH 0\nM 0\n", lossless, shots=SHOTS, seed=3
    )
    assert abs(np.asarray(r.measurements)[:, 0].mean() - 0.5) < BAND


def test_pauli_noise_dropped_on_a_lost_qubit():
    # The X_ERROR lands after the loss, so it is dropped and the record is
    # whatever the classifier pins for the lost level.
    lost_col = [[0.0] * 5 for _ in range(2)]
    for lvl in range(5):
        lost_col[0][lvl] = 1.0
    to_lost = [[0.0] * 5 for _ in range(5)]
    to_lost[noncomp.Level.LOST][noncomp.Level.G] = 1.0
    to_lost[noncomp.Level.LOST][noncomp.Level.E] = 1.0
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": to_lost},
        classifier=noncomp.Classifier(["0", "1"], lost_col),
    )
    r = noncomp.sample("H 0\nS 0\nX_ERROR(1) 0\nM 0\n", model, shots=64, seed=4)
    assert not np.asarray(r.measurements).any()
