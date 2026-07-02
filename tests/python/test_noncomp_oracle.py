"""Cross-check clifft.noncomp sampling against an independent reference.

The reference (``utils_noncomp_oracle``) is a tiny numpy density-matrix
simulator built from first principles, plus explicit closed-form probabilities
for the classical events (initial level, transitions, classifier). It is
independent of clifft's sampler/rewriter/SVM. We first self-check the reference
against clifft's own simulator on lossless circuits, then use it to validate the
supported noncomputational subset within shot noise.

Scope note: comparing output distributions cannot isolate the hidden trace-out
(a lost qubit that never re-enters is observationally identical with or without
it); that mechanism is covered by the C++ structural test. Here the oracle
validates the supported output distributions: initial population, transition
probability, classifier replacement, and the survivor marginal as a partial
trace.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import utils_noncomp_oracle as oracle

import clifft
from clifft import noncomp

Level = noncomp.Level
BAND = 0.04  # ~7 sigma at 8000 shots for p near 0.5
SHOTS = 8000


def _transition(entries: dict[tuple[int, int], float]) -> list[list[float]]:
    """5x5 T[to][from]; a column's deficit below 1 is the no-jump (stay) weight."""
    m = [[0.0] * 5 for _ in range(5)]
    for (to, frm), p in entries.items():
        m[to][frm] = p
    return m


def _classifier(level: int, col: list[float]) -> noncomp.Classifier:
    m = [[0.0] * 5 for _ in range(2)]  # P[symbol][level], two symbols
    for lvl in range(5):
        m[0][lvl] = 1.0
    # Computational levels read out faithfully (no readout confusion).
    m[0][noncomp.Level.E], m[1][noncomp.Level.E] = 0.0, 1.0
    m[0][level], m[1][level] = col[0], col[1]
    return noncomp.Classifier(["0", "1"], m)


def _p1(result: noncomp.NonComputationalSample, slot: int) -> float:
    return float(np.asarray(result.measurements)[:, slot].mean())


# --- Self-check: the reference agrees with clifft on lossless circuits --------


def _matches_clifft(text: str, state: npt.NDArray[np.complex128], n: int) -> None:
    empirical = np.asarray(clifft.sample(clifft.compile(text), SHOTS, 7).measurements)
    for q in range(n):
        assert abs(oracle.prob_one(state, q, n) - empirical[:, q].mean()) < BAND


def test_oracle_quantum_core_matches_clifft():
    h = oracle.apply_1q(oracle.zero_state(1), "H", 0, 1)
    _matches_clifft("H 0\nM 0\n", h, 1)
    _matches_clifft("X 0\nM 0\n", oracle.apply_1q(oracle.zero_state(1), "X", 0, 1), 1)
    _matches_clifft("H 0\nS 0\nM 0\n", oracle.apply_1q(h, "S", 0, 1), 1)
    bell = oracle.apply_cx(oracle.apply_1q(oracle.zero_state(2), "H", 0, 2), 0, 1, 2)
    _matches_clifft("H 0\nCX 0 1\nM 0\nM 1\n", bell, 2)


def test_oracle_partial_trace_of_bell_is_maximally_mixed():
    state = oracle.apply_cx(oracle.apply_1q(oracle.zero_state(2), "H", 0, 2), 0, 1, 2)
    assert abs(oracle.marginal_one_after_trace_out(state, lost=0, survivor=1, n=2) - 0.5) < 1e-12


# --- Supported noncomputational subset vs the reference -----------------------


def test_lossless_matches_clifft_distribution():
    text = "H 0\nCX 0 1\nM 0\nM 1\n"
    model = noncomp.Model(initial_state=[1.0, 0.0, 0.0, 0.0, 0.0])
    nc = noncomp.sample(text, model, shots=SHOTS, seed=1)
    plain = np.asarray(clifft.sample(clifft.compile(text), SHOTS, 1).measurements)
    nc_m = np.asarray(nc.measurements)
    for q in range(2):
        assert abs(nc_m[:, q].mean() - plain[:, q].mean()) < BAND
    # Bell correlation preserved in both.
    assert (nc_m[:, 0] == nc_m[:, 1]).mean() > 0.99


def test_initial_population_sampling():
    # 70% g (-> 0), 30% e (-> 1 via X-prep); expected P(record=1) = 0.3.
    model = noncomp.Model(initial_state=[0.7, 0.3, 0.0, 0.0, 0.0])
    r = noncomp.sample("M 0\n", model, shots=SHOTS, seed=2)
    assert abs(_p1(r, 0) - 0.3) < BAND


def test_transition_probability_on_known_source():
    # On known g, S jumps to leak_g with prob 0.4 (deficit 0.6 stays g).
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": _transition({(Level.LEAK_G, Level.G): 0.4})},
    )
    r = noncomp.sample("S 0\n", model, shots=SHOTS, seed=3)
    leaked = (np.asarray(r.final_status) == noncomp.QubitStatusKind.LEAKED).mean()
    assert abs(leaked - 0.4) < BAND


@pytest.mark.parametrize("col,expected", [([0.0, 1.0], 1.0), ([1.0, 0.0], 0.0), ([0.5, 0.5], 0.5)])
def test_classifier_replacement_distribution(col, expected):
    # Always leak to leak_g, then the classifier's column sets the record bit.
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={
            "S": _transition({(Level.LEAK_G, Level.G): 1.0, (Level.LEAK_G, Level.E): 1.0})
        },
        classifier=_classifier(Level.LEAK_G, col),
    )
    r = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=SHOTS, seed=4)
    assert abs(_p1(r, 0) - expected) < BAND


def test_partial_relaxation_matches_analytic_mixture():
    # With probability p the S transition collapses the H-prepared |+> to g;
    # otherwise the carrier stays coherent. P(M=1) = (1 - p) * P(1 | H|0>).
    p = 0.3
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": _transition({(Level.G, Level.G): p, (Level.G, Level.E): p})},
    )
    r = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=SHOTS, seed=6)
    h = oracle.apply_1q(oracle.zero_state(1), "H", 0, 1)
    expected = (1 - p) * oracle.prob_one(h, 0, 1)
    assert abs(_p1(r, 0) - expected) < BAND


def test_survivor_marginal_equals_partial_trace():
    # Bell pair, lose qubit 0. The survivor's record marginal must equal the
    # reference partial trace (0.5), and the lost record follows the classifier.
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": _transition({(Level.LOST, Level.G): 1.0, (Level.LOST, Level.E): 1.0})},
        classifier=_classifier(Level.LOST, [0.5, 0.5]),
    )
    r = noncomp.sample("H 0\nCX 0 1\nS 0\nM 0\nM 1\n", model, shots=SHOTS, seed=5)

    bell = oracle.apply_cx(oracle.apply_1q(oracle.zero_state(2), "H", 0, 2), 0, 1, 2)
    expected_survivor = oracle.marginal_one_after_trace_out(bell, lost=0, survivor=1, n=2)
    assert abs(_p1(r, 1) - expected_survivor) < BAND  # survivor (M 1) == partial trace
    assert abs(_p1(r, 0) - 0.5) < BAND  # lost record (M 0) follows the [0.5, 0.5] classifier
