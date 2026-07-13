"""Worked examples for the categories our model supports exactly, plus the
ones it intentionally rejects. Documents the supported boundary.

These complement the unit coverage in test_noncomp.py and the oracle
cross-checks in test_noncomp_oracle.py; here the emphasis is on small,
readable end-to-end scenarios -- especially initial noncomputational
populations, which the other files do not exercise.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import noncomp_classifier_matrix_with_column, noncomp_transition_matrix

from clifft import noncomp

Level = noncomp.Level
COMPUTATIONAL = noncomp.QubitStatus.COMPUTATIONAL
LOST = noncomp.QubitStatus.LOST
SHOTS = 8000
BAND = 0.04


def _classifier(level: int, col: list[float]) -> noncomp.Classifier:
    return noncomp.Classifier(noncomp_classifier_matrix_with_column(level, col))


# --- Supported categories ----------------------------------------------------


def test_example_initial_leaked_population():
    # 30% of shots start in leak_g; the classifier (leak_g -> 1) sets their
    # record bit, and g (-> 0) sets the rest. So P(record=1) == P(leak_g) == 0.3.
    model = noncomp.Model(
        initial_state=[0.7, 0.0, 0.3, 0.0, 0.0],
        classifier=_classifier(Level.LEAK_G, [0.0, 1.0]),
    )
    r = noncomp.sample("M 0\n", model, shots=SHOTS, seed=1)
    assert abs(np.asarray(r.measurements)[:, 0].mean() - 0.3) < BAND
    assert abs((np.asarray(r.final_status) == noncomp.QubitStatus.LEAK_G).mean() - 0.3) < BAND


def test_example_initial_lost_population():
    # Half the shots start lost; final status reflects it.
    model = noncomp.Model(
        initial_state=[0.5, 0.0, 0.0, 0.0, 0.5],
        classifier=_classifier(Level.LOST, [1.0, 0.0]),
    )
    r = noncomp.sample("M 0\n", model, shots=SHOTS, seed=2)
    assert abs((np.asarray(r.final_status) == LOST).mean() - 0.5) < BAND


def test_example_after_gate_leakage_with_classifier():
    # A gate leaks the qubit; its measurement is classified.
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={
            "S": noncomp_transition_matrix(
                {(Level.LEAK_G, Level.G): 1.0, (Level.LEAK_G, Level.E): 1.0}
            )
        },
        classifier=_classifier(Level.LEAK_G, [0.0, 1.0]),
    )
    r = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=SHOTS, seed=3)
    assert (np.asarray(r.measurements)[:, 0] == 1).all()
    assert (np.asarray(r.final_status) == noncomp.QubitStatus.LEAK_G).all()


def test_example_relaxation_to_ground_on_known_qubit():
    # A known |1> qubit relaxes to g at the S gate; the M must read the
    # relaxed 0, not the stale |1>, and the qubit stays computational.
    model = noncomp.Model(
        initial_state=[0.0, 1.0, 0.0, 0.0, 0.0],
        transitions={"S": noncomp_transition_matrix({(Level.G, Level.E): 1.0})},
    )
    r = noncomp.sample("S 0\nM 0\n", model, shots=64, seed=6)
    assert (np.asarray(r.measurements)[:, 0] == 0).all()
    assert (np.asarray(r.final_status) == COMPUTATIONAL).all()


# --- Intentionally unsupported -----------------------------------------------


def test_reject_parity_measurement_under_capable_model():
    # A parity measurement (MPP) is not supported when the model can leak
    # or lose qubits, rejected before sampling begins.
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={
            "S": noncomp_transition_matrix(
                {(Level.LEAK_G, Level.G): 1.0, (Level.LEAK_G, Level.E): 1.0}
            )
        },
        classifier=noncomp.Classifier([[1.0] * 5, [0.0] * 5]),
    )
    with pytest.raises(ValueError, match="not supported"):
        noncomp.sample("H 0\nS 0\nMPP Z0*Z1\n", model, shots=8, seed=5)
