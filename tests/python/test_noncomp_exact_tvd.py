"""Distributional cross-check of the exact mode against the dense reference.

The enumerator (``utils_noncomp_enumerator``) computes the exact joint
distribution of the visible record from first principles; the tests here
first self-check it against hand-derived closed forms, then compare
clifft's exact-mode sampling to it on a distance-3 repetition-code round
at cold-atom-magnitude rates, by total variation distance with a
shot-noise band calibrated from the reference distribution itself.
"""

from __future__ import annotations

import numpy as np
import utils_noncomp_enumerator as en
from conftest import binomial_tolerance

from clifft import noncomp

Level = noncomp.Level


def _transition(entries: dict[tuple[int, int], float]) -> list[list[float]]:
    m = [[0.0] * 5 for _ in range(5)]
    for (to, frm), p in entries.items():
        m[to][frm] = p
    return m


def _classifier_matrix() -> list[list[float]]:
    """Faithful computational readout; leak_g reads 0, leak_e 1, lost a coin."""
    m = [[0.0] * 5 for _ in range(2)]
    m[0][Level.G] = m[1][Level.E] = 1.0
    m[0][Level.LEAK_G] = m[1][Level.LEAK_E] = 1.0
    m[0][Level.LOST] = m[1][Level.LOST] = 0.5
    return m


def _model(initial: list, transitions: dict, damping: str = "exact") -> noncomp.Model:
    return noncomp.Model(
        initial_state=initial,
        transitions=transitions,
        classifier=noncomp.Classifier(["0", "1"], _classifier_matrix()),
        damping=damping,
    )


# --- Enumerator self-checks against hand-derived closed forms -----------------


def test_enumerator_lossless_bell_joint():
    dist = en.enumerate_exact(
        "H 0\nCX 0 1\nM 0\nM 1\n",
        initial=[1, 0, 0, 0, 0],
        transitions={},
        classifier=_classifier_matrix(),
    )
    assert abs(dist.record_probs[(0, 0)] - 0.5) < 1e-12
    assert abs(dist.record_probs[(1, 1)] - 0.5) < 1e-12
    assert set(dist.record_probs) == {(0, 0), (1, 1)}
    assert dist.dropped_mass < 1e-12


def test_enumerator_plus_state_marginal_closed_form():
    """|+> under leak-from-g (p = 0.4), leak reads 1: hand-derived forms.

    Exact: fire p/2 reads 1; no-fire has weight (2 - p)/2 and the damped
    state (sqrt(1-p)|g> + |e>)/norm reads 1 with probability 1/(2 - p),
    so P(M=1) = p/2 + 1/2 = 0.7. Neglect: the no-fire branch keeps |+>
    (reads 1 half the time), so P(M=1) = p/2 + (1 - p/2)/2 = 0.6.
    """
    p = 0.4
    circuit = "H 0\nLEVEL_TRANSITION[leak] 0\nM 0\n"
    transitions = {"leak": _transition({(Level.LEAK_E, Level.G): p})}

    exact = en.enumerate_exact(
        circuit,
        initial=[1, 0, 0, 0, 0],
        transitions=transitions,
        classifier=_classifier_matrix(),
        damping="exact",
    )
    p1 = sum(w for rec, w in exact.record_probs.items() if rec[0] == 1)
    assert abs(p1 - 0.7) < 1e-12

    neglect = en.enumerate_exact(
        circuit,
        initial=[1, 0, 0, 0, 0],
        transitions=transitions,
        classifier=_classifier_matrix(),
        damping="neglect",
    )
    p1 = sum(w for rec, w in neglect.record_probs.items() if rec[0] == 1)
    assert abs(p1 - 0.6) < 1e-12


def test_enumerator_initial_leak_and_recapture():
    """A leaked initial consults classically: seepage back to e is a
    recapture, and the record then reads the re-prepared |1>."""
    seep = 0.3
    dist = en.enumerate_exact(
        "LEVEL_TRANSITION[seep] 0\nM 0\n",
        initial=[0, 0, 1, 0, 0],  # starts at leak_g
        transitions={"seep": _transition({(Level.E, Level.LEAK_G): seep})},
        classifier=_classifier_matrix(),
    )
    # Recaptured (p = 0.3): reads 1. Still leaked at leak_g: reads 0.
    assert abs(dist.record_probs[(1,)] - seep) < 1e-12
    assert abs(dist.record_probs[(0,)] - (1 - seep)) < 1e-12


# --- Repetition-code round: TVD to the dense reference ------------------------

# Cold-atom-magnitude rates, scaled x3 so a correlation-class error would
# stand above the shot-noise band while staying in the published regime.
SCALE = 3.0
LEAK_2Q_E = 1.0e-3 * SCALE
LEAK_2Q_G = 1.0e-4 * SCALE
LEAK_1Q_E = 1.3e-3 * SCALE
LOSS_P = 3.9e-3 * SCALE
INITIAL = [1 - 0.015, 0.0, 0.015, 0.0, 0.0]

REP_CODE_ROUND = f"""
H 0 1 2
CX 0 3
LEVEL_TRANSITION[leak2q] 0 3
CX 1 3
LEVEL_TRANSITION[leak2q] 1 3
CX 1 4
LEVEL_TRANSITION[leak2q] 1 4
CX 2 4
LEVEL_TRANSITION[leak2q] 2 4
LEVEL_TRANSITION[leak1q] 0 1 2
LOSS({LOSS_P}) 0 1 2
MR 3 4
M 0 1 2
"""

TRANSITIONS = {
    "leak2q": _transition({(Level.LEAK_E, Level.E): LEAK_2Q_E, (Level.LEAK_G, Level.G): LEAK_2Q_G}),
    "leak1q": _transition({(Level.LEAK_E, Level.E): LEAK_1Q_E}),
}

SHOTS = 60_000


def _shot_noise_band(probs: dict, shots: int, draws: int = 300) -> float:
    """TVD band from multinomial resampling of the reference itself."""
    keys = list(probs)
    p = np.array([probs[k] for k in keys])
    p = p / p.sum()
    rng = np.random.default_rng(2026)
    tvds = [0.5 * np.abs(rng.multinomial(shots, p) / shots - p).sum() for _ in range(draws)]
    return 1.3 * float(np.max(tvds))


def _status_kind_fractions(noncomp_probs: dict[int, float]) -> tuple[float, float]:
    leaked = noncomp_probs.get(Level.LEAK_G, 0.0) + noncomp_probs.get(Level.LEAK_E, 0.0)
    lost = noncomp_probs.get(Level.LOST, 0.0)
    return leaked, lost


def _run_and_compare(damping: str, seed: int) -> None:
    reference = en.enumerate_exact(
        REP_CODE_ROUND,
        initial=INITIAL,
        transitions=TRANSITIONS,
        classifier=_classifier_matrix(),
        damping=damping,
    )
    assert reference.dropped_mass < 1e-6

    r = noncomp.sample(
        REP_CODE_ROUND, _model(INITIAL, TRANSITIONS, damping=damping), shots=SHOTS, seed=seed
    )
    empirical = en.empirical_record_probs(np.asarray(r.measurements))

    band = _shot_noise_band(reference.record_probs, SHOTS) + reference.dropped_mass
    assert band < 0.05, "band too loose to be meaningful; raise SHOTS"
    distance = en.tvd(reference.record_probs, empirical)
    assert distance < band, f"TVD {distance:.4f} exceeds shot-noise band {band:.4f}"

    # Final-status marginals: leaked/lost fractions per qubit.
    status = np.asarray(r.final_status)
    for q in range(5):
        want_leaked, want_lost = _status_kind_fractions(reference.noncomp_level_probs[q])
        got_leaked = float((status[:, q] == noncomp.QubitStatusKind.LEAKED).mean())
        got_lost = float((status[:, q] == noncomp.QubitStatusKind.LOST).mean())
        assert abs(got_leaked - want_leaked) < binomial_tolerance(max(want_leaked, 1e-4), SHOTS)
        assert abs(got_lost - want_lost) < binomial_tolerance(max(want_lost, 1e-4), SHOTS)


def test_rep_code_round_tvd_reaches_shot_noise():
    _run_and_compare("exact", seed=21)


def test_rep_code_round_tvd_under_neglect_matches_its_own_reference():
    """The neglect fallback must match the reference that omits the no-fire
    filter -- pinning the driver's neglect semantics distributionally, not
    just at micro-probes."""
    _run_and_compare("neglect", seed=22)
