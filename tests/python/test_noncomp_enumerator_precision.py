"""Calibrate rare-event reference weights without statistical sampling."""

import math

import pytest
import utils_noncomp_enumerator as en

CLASSIFIER: list[list[float]] = [[1, 0, 1, 1, 1], [0, 1, 0, 0, 0]]


def test_initial_rare_population_requires_an_explicit_pruning_budget():
    probability = 1e-16
    initial = [1 - probability, probability, 0.0, 0.0, 0.0]
    pruned = en.enumerate_exact("M 0", initial=initial, transitions={}, classifier=CLASSIFIER)
    assert (1,) not in pruned.record_probs
    assert pruned.dropped_mass == pytest.approx(probability, rel=1e-14, abs=0)
    with pytest.raises(AssertionError, match="inconclusive reference: omitted_bound="):
        pruned.require_precision(smallest_probability=probability, relative_tolerance=1e-9)

    retained = en.enumerate_exact(
        "M 0", initial=initial, transitions={}, classifier=CLASSIFIER, prune_probability=0
    )
    retained.require_precision(smallest_probability=probability, relative_tolerance=1e-9)
    assert retained.record_probs[(1,)] == pytest.approx(probability, rel=1e-9, abs=0)
    assert retained.dropped_mass == 0


def test_joint_rare_losses_survive_enumeration_and_record_aggregation():
    probability = 1e-14
    result = en.enumerate_exact(
        f"LOSS({probability}) 0 1\nX 0 1\nM 0 1",
        initial=[1, 0, 0, 0, 0],
        transitions={},
        classifier=CLASSIFIER,
        prune_probability=0,
    )
    result.require_precision(smallest_probability=probability**2, relative_tolerance=1e-9)
    assert result.record_probs == pytest.approx(
        {
            (0, 0): probability**2,
            (0, 1): probability * (1 - probability),
            (1, 0): probability * (1 - probability),
            (1, 1): (1 - probability) ** 2,
        },
        rel=1e-9,
        abs=0,
    )
    for marginal in result.noncomp_level_probs:
        assert marginal[en.LOST] == pytest.approx(probability, rel=1e-9, abs=0)


def test_pruning_counts_joint_transition_mass_once():
    probability = 1e-14
    column = [[0.0] * 5 for _ in range(5)]
    column[en.LEAK_E][en.E] = probability
    circuit = "H 0\nLEVEL_TRANSITION[jump] 0\nM 0\nM 0"
    pruned = en.enumerate_exact(
        circuit, initial=[1, 0, 0, 0, 0], transitions={"jump": column}, classifier=CLASSIFIER
    )
    assert pruned.dropped_mass == pytest.approx(probability / 2, rel=1e-9, abs=0)
    with pytest.raises(AssertionError, match="inconclusive reference"):
        pruned.require_precision(smallest_probability=probability / 2, relative_tolerance=1e-9)
    retained = en.enumerate_exact(
        circuit,
        initial=[1, 0, 0, 0, 0],
        transitions={"jump": column},
        classifier=CLASSIFIER,
        prune_probability=0,
    )
    retained.require_precision(smallest_probability=probability / 2, relative_tolerance=1e-9)
    assert retained.noncomp_level_probs[0][en.LEAK_E] == pytest.approx(
        probability / 2, rel=1e-9, abs=0
    )
    assert retained.record_probs[(1, 1)] == pytest.approx((1 - probability) / 2, abs=1e-15)


@pytest.mark.parametrize("omitted_mass", [math.nan, math.inf, -1e-20, 1e-16])
def test_invalid_or_excessive_omissions_cannot_certify_a_signal(omitted_mass):
    result = en.ExactDistribution()
    result.dropped_mass = omitted_mass
    with pytest.raises(AssertionError, match="inconclusive reference"):
        result.require_precision(smallest_probability=1e-20, relative_tolerance=1e-9)
