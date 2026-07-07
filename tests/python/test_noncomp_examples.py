"""Worked examples for the categories our model supports exactly, plus the
ones it intentionally rejects. Mirrors sqale-sim style scenarios and documents
the supported boundary.

These complement the unit coverage in test_noncomp.py and the oracle
cross-checks in test_noncomp_oracle.py; here the emphasis is on small,
readable end-to-end scenarios -- especially initial noncomputational
populations, which the other files do not exercise.
"""

from __future__ import annotations

import numpy as np
import pytest

from clifft import noncomp

Level = noncomp.Level
COMPUTATIONAL = noncomp.QubitStatusKind.COMPUTATIONAL
LEAKED = noncomp.QubitStatusKind.LEAKED
LOST = noncomp.QubitStatusKind.LOST
SHOTS = 8000
BAND = 0.04


def _classifier(level: int, col: list[float]) -> noncomp.Classifier:
    m = [[0.0] * 5 for _ in range(2)]  # P[symbol][level]
    for lvl in range(5):
        m[0][lvl] = 1.0
    m[0][level], m[1][level] = col[0], col[1]
    return noncomp.Classifier(["0", "1"], m)


def _transition(entries: dict[tuple[int, int], float]) -> list[list[float]]:
    m = [[0.0] * 5 for _ in range(5)]  # T[to][from]
    for (to, frm), p in entries.items():
        m[to][frm] = p
    return m


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
    assert abs((np.asarray(r.final_status) == LEAKED).mean() - 0.3) < BAND


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
            "S": _transition({(Level.LEAK_G, Level.G): 1.0, (Level.LEAK_G, Level.E): 1.0})
        },
        classifier=_classifier(Level.LEAK_G, [0.0, 1.0]),
    )
    r = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=SHOTS, seed=3)
    assert (np.asarray(r.measurements)[:, 0] == 1).all()
    assert (np.asarray(r.final_status) == LEAKED).all()


def test_example_relaxation_to_ground_on_known_qubit():
    # A known |1> qubit relaxes to g at the S gate; the M must read the
    # relaxed 0, not the stale |1>, and the qubit stays computational.
    model = noncomp.Model(
        initial_state=[0.0, 1.0, 0.0, 0.0, 0.0],
        transitions={"S": _transition({(Level.G, Level.E): 1.0})},
    )
    r = noncomp.sample("S 0\nM 0\n", model, shots=64, seed=6)
    assert (np.asarray(r.measurements)[:, 0] == 0).all()
    assert (np.asarray(r.final_status) == COMPUTATIONAL).all()


# --- Intentionally unsupported -----------------------------------------------


def test_reject_xy_measurement_on_noncomputational_qubit():
    # Once qubit 0 is leaked, its downstream gates drop, but an X/Y-basis (or
    # parity) measurement of it has no faithful single-bit form and is refused
    # -- a representability limit, not a policy the caller can turn off.
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={
            "S": _transition({(Level.LEAK_G, Level.G): 1.0, (Level.LEAK_G, Level.E): 1.0})
        },
    )
    with pytest.raises(ValueError, match="not representable"):
        noncomp.sample("H 0\nS 0\nMX 0\n", model, shots=8, seed=5)
