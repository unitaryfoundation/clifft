"""Cross-check both noncomputational executors against an independent reference.

The reference (``utils_noncomp_oracle``) is a tiny numpy density-matrix
simulator built from first principles, plus explicit closed-form probabilities
for the classical events (initial level, transitions, classifier). It is
independent of Clifft's rewriter and executors. We first self-check the reference
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

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import utils_noncomp_oracle as oracle
from conftest import noncomp_classifier_matrix_with_column, noncomp_transition_matrix

import clifft
from clifft import noncomp

Level = noncomp.Level
BAND = 0.04  # ~7 sigma at 8000 shots for p near 0.5
SHOTS = 8000


@pytest.fixture(params=[clifft.noncomp.sample], ids=["symbolic"])
def noncomp_sampling_api(request: pytest.FixtureRequest) -> Any:
    """Run the independent-oracle checks against the production trajectory sampler."""
    return request.param


def _classifier(level: int, col: list[float]) -> noncomp.Classifier:
    return noncomp.Classifier(noncomp_classifier_matrix_with_column(level, col))


def _p1(result: noncomp.NonComputationalSample, slot: int) -> float:
    return float(np.asarray(result.measurements)[:, slot].mean())


# The reference agrees with Clifft on lossless circuits.


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


# Supported noncomputational behavior against the reference.


def test_lossless_matches_clifft_distribution(noncomp_sampling_api):
    text = "H 0\nCX 0 1\nM 0\nM 1\n"
    model = noncomp.Model(initial_state=[1.0, 0.0, 0.0, 0.0, 0.0])
    nc = noncomp_sampling_api(text, model, shots=SHOTS, seed=1)
    plain = np.asarray(clifft.sample(clifft.compile(text), SHOTS, 1).measurements)
    nc_m = np.asarray(nc.measurements)
    for q in range(2):
        assert abs(nc_m[:, q].mean() - plain[:, q].mean()) < BAND
    # Bell correlation preserved in both.
    assert (nc_m[:, 0] == nc_m[:, 1]).mean() > 0.99


def test_initial_population_sampling(noncomp_sampling_api):
    # 70% g (-> 0), 30% e (-> 1 via X-prep); expected P(record=1) = 0.3.
    model = noncomp.Model(initial_state=[0.7, 0.3, 0.0, 0.0, 0.0])
    r = noncomp_sampling_api("M 0\n", model, shots=SHOTS, seed=2)
    assert abs(_p1(r, 0) - 0.3) < BAND


def test_transition_probability_on_known_source(noncomp_sampling_api):
    # On known g, S jumps to leak_g with prob 0.4 (deficit 0.6 stays g).
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": noncomp_transition_matrix({(Level.LEAK_G, Level.G): 0.4})},
    )
    r = noncomp_sampling_api("S 0\n", model, shots=SHOTS, seed=3)
    leaked = np.isin(
        np.asarray(r.final_status), (noncomp.QubitStatus.LEAK_G, noncomp.QubitStatus.LEAK_E)
    ).mean()
    assert abs(leaked - 0.4) < BAND


def test_inline_leakage_matches_exact_enumerator(noncomp_sampling_api):
    """Source-preserving leakage agrees on entangled records and statuses."""
    import utils_noncomp_enumerator as en

    p = 0.35
    circuit = f"H 0\nCX 0 1\nLEAKAGE({p}) 0 1\nM 0\nM 1\n"
    classifier_matrix = [[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]]
    reference = en.enumerate_exact(
        circuit,
        initial=[1, 0, 0, 0, 0],
        transitions={},
        classifier=classifier_matrix,
    )
    assert reference.dropped_mass < 1e-12

    result = noncomp_sampling_api(
        circuit,
        noncomp.Model(classifier=noncomp.Classifier(classifier_matrix)),
        shots=SHOTS,
        seed=208,
    )
    empirical_records = en.empirical_record_probs(np.asarray(result.measurements))
    assert en.tvd(reference.record_probs, empirical_records) < BAND

    status_pairs = [
        (en.LEAK_G, noncomp.QubitStatus.LEAK_G),
        (en.LEAK_E, noncomp.QubitStatus.LEAK_E),
    ]
    for q in range(2):
        for level, status in status_pairs:
            expected = reference.noncomp_level_probs[q].get(level, 0.0)
            observed = float((result.final_status[:, q] == status).mean())
            assert abs(observed - expected) < BAND


def test_inline_leakage_measure_reset_reprepares_parked_factor(noncomp_sampling_api):
    """MR restores a leaked carrier at zero in both reference and sampler."""
    import utils_noncomp_enumerator as en

    circuit = "X 0\nLEAKAGE(1) 0\nMR 0\nM 0\n"
    classifier_matrix = [[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]]
    reference = en.enumerate_exact(
        circuit,
        initial=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={},
        classifier=classifier_matrix,
    )
    assert reference.record_probs == {(1, 0): 1.0}

    result = noncomp_sampling_api(
        circuit,
        noncomp.Model(classifier=noncomp.Classifier(classifier_matrix)),
        shots=32,
        seed=209,
    )
    assert np.array_equal(np.asarray(result.measurements), np.tile([1, 0], (32, 1)))


@pytest.mark.parametrize("col,expected", [([0.0, 1.0], 1.0), ([1.0, 0.0], 0.0), ([0.5, 0.5], 0.5)])
def test_classifier_replacement_distribution(col, expected, noncomp_sampling_api):
    # Always leak to leak_g, then the classifier's column sets the record bit.
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={
            "S": noncomp_transition_matrix(
                {(Level.LEAK_G, Level.G): 1.0, (Level.LEAK_G, Level.E): 1.0}
            )
        },
        classifier=_classifier(Level.LEAK_G, col),
    )
    r = noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=SHOTS, seed=4)
    assert abs(_p1(r, 0) - expected) < BAND


def test_partial_relaxation_matches_analytic_mixture(noncomp_sampling_api):
    # With probability p the S transition collapses the H-prepared |+> to g;
    # otherwise the carrier stays coherent. P(M=1) = (1 - p) * P(1 | H|0>).
    p = 0.3
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={
            "S": noncomp_transition_matrix({(Level.G, Level.G): p, (Level.G, Level.E): p})
        },
    )
    r = noncomp_sampling_api("H 0\nS 0\nM 0\n", model, shots=SHOTS, seed=6)
    h = oracle.apply_1q(oracle.zero_state(1), "H", 0, 1)
    expected = (1 - p) * oracle.prob_one(h, 0, 1)
    assert abs(_p1(r, 0) - expected) < BAND


def test_survivor_marginal_equals_partial_trace(noncomp_sampling_api):
    # Bell pair, lose qubit 0. The survivor's record marginal must equal the
    # reference partial trace (0.5), and the lost record follows the classifier.
    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={
            "S": noncomp_transition_matrix({(Level.LOST, Level.G): 1.0, (Level.LOST, Level.E): 1.0})
        },
        classifier=_classifier(Level.LOST, [0.5, 0.5]),
    )
    r = noncomp_sampling_api("H 0\nCX 0 1\nS 0\nM 0\nM 1\n", model, shots=SHOTS, seed=5)

    bell = oracle.apply_cx(oracle.apply_1q(oracle.zero_state(2), "H", 0, 2), 0, 1, 2)
    expected_survivor = oracle.marginal_one_after_trace_out(bell, lost=0, survivor=1, n=2)
    assert abs(_p1(r, 1) - expected_survivor) < BAND  # survivor (M 1) == partial trace
    assert abs(_p1(r, 0) - 0.5) < BAND  # lost record (M 0) follows the [0.5, 0.5] classifier


# Exact-channel primitive checks.


def test_channel_kraus_set_is_complete():
    # Fire weights plus the no-fire weight must sum to 1 on any state:
    # sum_s ptot_s * <P_s> + <K0' K0> = 1. Random states, random columns.
    rng = np.random.default_rng(9)
    for _ in range(20):
        state = rng.normal(size=4) + 1j * rng.normal(size=4)
        state = state / np.linalg.norm(state)
        ptot_g, ptot_e = rng.uniform(0.0, 1.0, size=2)
        for q in (0, 1):
            pop_g, _ = oracle.collapse(state, q, 0, 2)
            pop_e, _ = oracle.collapse(state, q, 1, 2)
            w0, _ = oracle.damp_no_fire(state, q, ptot_g, ptot_e, 2)
            total = ptot_g * pop_g + ptot_e * pop_e + w0
            assert abs(total - 1.0) < 1e-12


def test_damp_with_zero_rates_is_identity():
    h = oracle.apply_1q(oracle.zero_state(1), "H", 0, 1)
    w, post = oracle.damp_no_fire(h, 0, 0.0, 0.0, 1)
    assert abs(w - 1.0) < 1e-12
    assert np.allclose(post, h)


def test_collapse_reproduces_born_weights():
    h = oracle.apply_1q(oracle.zero_state(1), "H", 0, 1)
    w0, post0 = oracle.collapse(h, 0, 0, 1)
    w1, post1 = oracle.collapse(h, 0, 1, 1)
    assert abs(w0 - 0.5) < 1e-12 and abs(w1 - 0.5) < 1e-12
    assert abs(abs(post0[0]) - 1.0) < 1e-12  # renormalized |0>
    assert abs(abs(post1[1]) - 1.0) < 1e-12  # renormalized |1>


def test_set_collapsed_qubit_reprepares_destination():
    _, at_e = oracle.collapse(oracle.apply_1q(oracle.zero_state(1), "X", 0, 1), 0, 1, 1)
    moved = oracle.set_collapsed_qubit(at_e, 0, 1, 0, 1)
    assert abs(abs(moved[0]) - 1.0) < 1e-12


# Two-site exact-damping composition against the enumerator.


def test_two_site_exact_damping_composition_matches_enumerator(noncomp_sampling_api):
    """Two LEVEL_TRANSITION[leak] sites in a row with p=0.5 from e, exact damping.

    Circuit: H 0 / LEVEL_TRANSITION[leak] 0 / LEVEL_TRANSITION[leak] 0 / H 0 / M 0.
    The channel contains two consecutive source-dependent (e-only) sites; the
    composition of two non-scalar damp filters is the first multi-site exact-
    damping check at a rate where any TVD between exact and neglect would be
    observable.  Compare sampled record/status frequencies against the
    enumerator's exact distribution within the file's existing BAND tolerance.

    This exercises composed continuations under exact damping: each site adds
    one branch at run time, and the no-fire filter at each site accumulates
    multiplicatively.
    """
    import utils_noncomp_enumerator as en

    p = 0.5
    transitions = {"leak": noncomp_transition_matrix({(Level.LEAK_E, Level.E): p})}
    circuit_text = "H 0\nLEVEL_TRANSITION[leak] 0\nLEVEL_TRANSITION[leak] 0\nH 0\nM 0\n"
    classifier_matrix = [[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]]

    reference = en.enumerate_exact(
        circuit_text,
        initial=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions=transitions,
        classifier=classifier_matrix,
        damping="exact",
    )
    assert reference.dropped_mass < 1e-12

    model = noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions=transitions,
        classifier=_classifier(Level.LEAK_E, [0.0, 1.0]),
        damping="exact",
    )
    r = noncomp_sampling_api(circuit_text, model, shots=SHOTS, seed=31)

    empirical = en.empirical_record_probs(np.asarray(r.measurements))
    tvd = en.tvd(reference.record_probs, empirical)
    assert tvd < BAND, f"TVD {tvd:.4f} exceeds tolerance {BAND}"
