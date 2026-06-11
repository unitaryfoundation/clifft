"""Quantify the two documented approximations of ``equalize_rates``.

The equalized-rates policy samples a source-dependent transition on an unknown
qubit by drawing the source uniformly, independent of the simulator's internal
collapse. These probes pin its accuracy envelope against closed-form references
built on the first-principles oracle (``utils_noncomp_oracle``):

1. Destination-collapse correlations: the independent source draw reproduces
   every per-qubit marginal of a genuinely unbiased source but discards the
   correlation between the leaked qubit's readout symbol and its entangled
   partner's record. A collapse-conditioned variant (Born-measure the qubit,
   pick the destination from the measured bit's column) keeps it. The probe
   computes both joint distributions in closed form, asserts the marginals
   agree, quantifies the joint gap, and checks clifft's samples against the
   independent-draw prediction -- and against neither matching the conditioned
   one.

2. Deterministic-but-untracked states: status tracking is instruction-known,
   so a qubit returned to a definite level by gate algebra still takes the
   approximate path. On a genuinely unbiased state the approximation's Z-basis
   marginals are exact (equalization only adds dephasing); on a
   gate-determined state they diverge by a closed-form amount.

These expectations encode the approximation as designed. If the sampler ever
starts conditioning destinations on runtime outcomes or promoting
gate-determined states, the corresponding probe must change with it.
"""

from __future__ import annotations

import numpy as np
import utils_noncomp_oracle as oracle

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


def _leak_classifier() -> noncomp.Classifier:
    """leak_g reads 0 and leak_e reads 1, mirroring the g/e split they record."""
    m = [[0.0] * 5 for _ in range(2)]
    for lvl in range(5):
        m[0][lvl] = 1.0
    m[0][Level.LEAK_E], m[1][Level.LEAK_E] = 0.0, 1.0
    return noncomp.Classifier(["0", "1"], m)


def _bell() -> np.ndarray:
    return oracle.apply_cx(oracle.apply_1q(oracle.zero_state(2), "H", 0, 2), 0, 1, 2)


def test_joint_correlation_probe_bell_pair():
    """Divergence 1: the leaked symbol decorrelates from the partner's record.

    Bell pair; the source-dependent leak sends g to leak_g (reads 0) and e to
    leak_e (reads 1) with certainty, so the leaked qubit's record reveals the
    source. Conditioning the destination on a Born measurement would tie that
    record to the partner's collapsed bit; the independent draw cannot.
    """
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={
            "S": _transition({(Level.LEAK_G, Level.G): 1.0, (Level.LEAK_E, Level.E): 1.0})
        },
        classifier=_leak_classifier(),
        unknown_source_policy="equalize_rates",
    )
    r = noncomp.sample("H 0\nCX 0 1\nS 0\nM 0\nM 1\n", model, shots=SHOTS, seed=11)
    meas = np.asarray(r.measurements)
    empirical = np.zeros((2, 2))
    for a in (0, 1):
        for b in (0, 1):
            empirical[a, b] = ((meas[:, 0] == a) & (meas[:, 1] == b)).mean()

    bell = _bell()
    # Independent-draw prediction (what clifft implements): the leaked symbol
    # is the uniform source draw; the partner's marginal is the partial trace.
    p_symbol_one = 0.5
    p_partner_one = oracle.marginal_one_after_trace_out(bell, lost=0, survivor=1, n=2)
    independent = np.outer([1 - p_symbol_one, p_symbol_one], [1 - p_partner_one, p_partner_one])
    # Collapse-conditioned prediction: the symbol equals the Born-measured bit,
    # so the joint is the state's own computational-basis distribution
    # (qubit 0 is the most significant index in the oracle's convention).
    conditioned = np.abs(bell.reshape(2, 2)) ** 2

    # Same per-qubit marginals, quantifiably different joints.
    assert np.allclose(independent.sum(axis=1), conditioned.sum(axis=1), atol=1e-12)
    assert np.allclose(independent.sum(axis=0), conditioned.sum(axis=0), atol=1e-12)
    tvd = 0.5 * np.abs(independent - conditioned).sum()
    assert abs(tvd - 0.5) < 1e-12

    # clifft matches the independent-draw joint, not the conditioned one.
    assert np.all(np.abs(empirical - independent) < BAND)
    equal_rate = (meas[:, 0] == meas[:, 1]).mean()
    assert abs(equal_rate - 0.5) < BAND  # conditioned semantics would give 1.0


def test_unbiased_state_marginals_are_exact():
    """Equalization only adds dephasing: on a genuinely unbiased source both
    the leak marginal and the Z-basis record marginal match the exact channel.
    """
    p = 0.6
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": _transition({(Level.LEAK_E, Level.E): p})},
        classifier=_leak_classifier(),
        unknown_source_policy="equalize_rates",
    )
    r = noncomp.sample("H 0\nS 0\nM 0\n", model, shots=SHOTS, seed=12)

    # Exact channel on |+>: fires with probability p * P(e); conditioned on no
    # jump the amplitudes pass through the damping filter diag(1, sqrt(1-p)).
    plus = oracle.apply_1q(oracle.zero_state(1), "H", 0, 1)
    p_fire_exact = p * oracle.prob_one(plus, 0, 1)
    no_jump = np.diag([1.0, np.sqrt(1.0 - p)]) @ plus
    p_one_exact = p_fire_exact + float(np.real(no_jump.conj() @ (np.diag([0.0, 1.0]) @ no_jump)))

    leaked = (np.asarray(r.final_status)[:, 0] == noncomp.QubitStatusKind.LEAKED).mean()
    assert abs(leaked - p_fire_exact) < BAND
    assert abs(np.asarray(r.measurements)[:, 0].mean() - p_one_exact) < BAND


def test_gate_determined_state_diverges_by_the_closed_form_amount():
    """Divergence 2: H H returns the qubit to |g> deterministically, but the
    instruction-known status tracker still says unknown. The exact channel
    (leaking only from e) could never fire there; the equalized draw fires at
    p and leaks on the half of draws that pick the e column.
    """
    p = 0.6
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": _transition({(Level.LEAK_E, Level.E): p})},
        classifier=_leak_classifier(),
        unknown_source_policy="equalize_rates",
    )
    r = noncomp.sample("H 0\nH 0\nS 0\nM 0\n", model, shots=SHOTS, seed=13)

    state = oracle.apply_1q(oracle.apply_1q(oracle.zero_state(1), "H", 0, 1), "H", 0, 1)
    p_fire_exact = p * oracle.prob_one(state, 0, 1)
    assert abs(p_fire_exact) < 1e-12  # the exact channel never fires on |g>

    leaked = (np.asarray(r.final_status)[:, 0] == noncomp.QubitStatusKind.LEAKED).mean()
    assert abs(leaked - p / 2) < BAND  # the documented approximation error
    # The pseudo-jump branch collapses to g and the no-fire branch is already
    # |g>, so the record stays 0 except on the leaked (reads 1) branch.
    assert abs(np.asarray(r.measurements)[:, 0].mean() - p / 2) < BAND
