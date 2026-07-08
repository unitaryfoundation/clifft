"""Closed-form micro-probes for the exact sampling mode.

Each probe pins, in closed form built on the first-principles oracle, a
distinct property the exact mode must satisfy: zero fires from a gate-determined
source, destination-collapse correlations with an entangled partner, and the
sqrt(1 - p) no-fire coherence that separates ``damping="exact"`` from
``damping="neglect"``.
"""

from __future__ import annotations

import numpy as np
import utils_noncomp_oracle as oracle
from conftest import binomial_tolerance

from clifft import noncomp

Level = noncomp.Level
SHOTS = 20_000


def _transition(entries: dict[tuple[int, int], float]) -> list[list[float]]:
    m = [[0.0] * 5 for _ in range(5)]
    for (to, frm), p in entries.items():
        m[to][frm] = p
    return m


def _faithful_classifier() -> noncomp.Classifier:
    """g/leak_g read 0; e/leak_e read 1; lost reads a fair coin."""
    m = [[0.0] * 5 for _ in range(2)]
    m[0][Level.G] = m[1][Level.E] = 1.0
    m[0][Level.LEAK_G] = m[1][Level.LEAK_E] = 1.0
    m[0][Level.LOST] = m[1][Level.LOST] = 0.5
    return noncomp.Classifier(m)


def _model(transitions: dict, damping: str = "exact") -> noncomp.Model:
    return noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions=transitions,
        classifier=_faithful_classifier(),
        damping=damping,
    )


def test_gate_determined_source_fires_exactly_zero():
    """H then H returns the qubit to |g> by algebra; a leak that fires only
    from e must then never fire -- the fire draw conditions on the live
    state, where <P_e> is exactly 0. The certain rate (p = 1) makes any
    ahead-of-time source guess loud: a uniform draw would leak half the
    shots."""
    model = _model({"leak": _transition({(Level.LEAK_E, Level.E): 1.0})})
    r = noncomp.sample("H 0\nH 0\nLEVEL_TRANSITION[leak] 0\nM 0\n", model, shots=SHOTS, seed=11)
    status = np.asarray(r.final_status)
    assert (status == noncomp.QubitStatus.COMPUTATIONAL).all()
    assert (np.asarray(r.measurements) == 0).all()  # H H |0> measures 0


def test_bell_joint_correlation_has_tvd_zero():
    """Source-dependent certain destinations on a Bell half: the collapse
    that picks the source is the same collapse the partner's measurement
    sees, so the classified record and the partner agree on every shot.
    The exact joint is {00: 1/2, 11: 1/2}; any independent source draw
    puts mass on 01/10."""
    model = _model(
        {"leak": _transition({(Level.LEAK_G, Level.G): 1.0, (Level.LEAK_E, Level.E): 1.0})}
    )
    r = noncomp.sample(
        "H 0\nCX 0 1\nLEVEL_TRANSITION[leak] 0\nM 0\nM 1\n", model, shots=SHOTS, seed=12
    )
    m = np.asarray(r.measurements)
    status = np.asarray(r.final_status)
    # The certain fire really happened: an accidentally skipped transition
    # would also show perfect agreement (a plain Bell pair does).
    assert np.isin(status[:, 0], (noncomp.QubitStatus.LEAK_G, noncomp.QubitStatus.LEAK_E)).all()
    assert (m[:, 0] == m[:, 1]).all()  # off-diagonal mass is exactly 0
    assert abs(m[:, 0].mean() - 0.5) < binomial_tolerance(0.5, SHOTS)


def test_damping_boundary_probe_separates_exact_from_neglect():
    """|+> through one leak-from-e site, then an X-basis readout (H, M).

    The no-fire branch's back-action is the whole difference between the
    damping modes: exact applies the filter diag(1, sqrt(1 - p)), leaving
    coherence sqrt(1 - p) and a nonzero X-basis flip rate; neglect keeps
    |+> intact, so the no-fire branch reads 1 with probability exactly 0.
    Both expectations are computed from the oracle's channel primitives,
    and the closed forms are far enough apart that each sample can only
    match its own mode."""
    p = 0.84
    transitions = {"leak": _transition({(Level.LEAK_E, Level.E): p})}
    text = "H 0\nLEVEL_TRANSITION[leak] 0\nH 0\nM 0\n"

    plus = oracle.apply_1q(oracle.zero_state(1), "H", 0, 1)
    p_fire = p * oracle.prob_one(plus, 0, 1)  # fire weight: p * <P_e>

    # Exact: damp, rotate, read. The fired branch is leaked and reads 1.
    w0, damped = oracle.damp_no_fire(plus, 0, 0.0, p, 1)
    p1_no_fire = oracle.prob_one(oracle.apply_1q(damped, "H", 0, 1), 0, 1)
    expected_exact = p_fire * 1.0 + w0 * p1_no_fire

    # Neglect: the no-fire branch keeps |+>, and H|+> = |0> reads 1 never.
    expected_neglect = p_fire * 1.0

    tol_exact = binomial_tolerance(expected_exact, SHOTS)
    tol_neglect = binomial_tolerance(expected_neglect, SHOTS)
    assert abs(expected_exact - expected_neglect) > 2 * (
        tol_exact + tol_neglect
    ), "probe lost its discriminating power; adjust p or SHOTS"

    r_exact = noncomp.sample(text, _model(transitions, damping="exact"), shots=SHOTS, seed=13)
    r_neglect = noncomp.sample(text, _model(transitions, damping="neglect"), shots=SHOTS, seed=14)
    p1_exact = float(np.asarray(r_exact.measurements)[:, 0].mean())
    p1_neglect = float(np.asarray(r_neglect.measurements)[:, 0].mean())

    assert abs(p1_exact - expected_exact) < tol_exact
    assert abs(p1_neglect - expected_neglect) < tol_neglect
