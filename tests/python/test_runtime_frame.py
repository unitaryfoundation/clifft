"""Tests for sampling programs planned with a runtime Clifford frame.

The planner option is internal. These tests reach it through the private
``_lower_runtime_frame`` binding and compare its programs with ordinary ones
through the public sampling functions.
"""

import numpy as np
import pytest

import clifft
from clifft._clifft_core import _lower_runtime_frame

# A Bell pair whose T and T_DAG cancel unless the fault flips one sign.
SIGNED_PAIR = (
    "H 0\n"
    "CX 0 1\n"
    "X_ERROR(0.3) 0\n"
    "T 0\n"
    "T_DAG 1\n"
    "MPP X0*X1\n"
    "MPP Y0*X1\n"
    "M 0 1\n"
    "DETECTOR rec[-4]\n"
    "OBSERVABLE_INCLUDE(0) rec[-3]\n"
)

# The same structure where the sign comes from a random stabilizer outcome.
RANDOM_OUTCOME = "H 0 1\nMPP Z0*Z1\nT 0\nT_DAG 1\nH 0 1\nM 0 1\n"


def _programs(text: str, postselection_mask: list[int] | None = None) -> tuple:
    hir = clifft.trace(clifft.parse(text), postselection_mask=postselection_mask or [])
    return clifft.lower(hir), _lower_runtime_frame(hir)


def _pattern_counts(rows: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(rows.shape[1])
    return np.bincount(rows @ weights, minlength=1 << rows.shape[1])


def _assert_same_distribution(a: np.ndarray, b: np.ndarray) -> None:
    """Two-sample check at six standard deviations for every outcome pattern."""
    counts_a = _pattern_counts(a)
    counts_b = _pattern_counts(b)
    shots = a.shape[0]
    assert b.shape[0] == shots
    pooled = (counts_a + counts_b) / (2 * shots)
    sigma = np.sqrt(2 * pooled * (1 - pooled) / shots)
    assert np.all(np.abs(counts_a - counts_b) / shots <= 6 * sigma + 1e-12)


def test_frame_plan_needs_no_active_coordinate() -> None:
    fixed, frame = _programs(SIGNED_PAIR)
    assert fixed.peak_active_width == 1
    assert frame.peak_active_width == 0
    assert "FRAME_CLIFFORD" in frame.inspect()
    assert "FRAME_CLIFFORD" not in fixed.inspect()


def test_sample_follows_the_ordinary_plan() -> None:
    fixed, frame = _programs(SIGNED_PAIR)
    shots = 40_000
    expected = clifft.sample(fixed, shots, seed=11)
    actual = clifft.sample(frame, shots, seed=12)
    _assert_same_distribution(expected.measurements, actual.measurements)
    _assert_same_distribution(
        np.hstack([expected.detectors, expected.observables]),
        np.hstack([actual.detectors, actual.observables]),
    )


def test_sample_is_repeatable_and_ignores_batch_size() -> None:
    _, frame = _programs(SIGNED_PAIR)
    first = clifft.sample(frame, 512, seed=5)
    again = clifft.sample(frame, 512, seed=5)
    np.testing.assert_array_equal(first.measurements, again.measurements)
    # Frame plans always run on the scalar executor.
    batched = clifft.sample(frame, 512, seed=5, batch_size=64)
    np.testing.assert_array_equal(first.measurements, batched.measurements)


@pytest.mark.parametrize("keep_records", [False, True])
def test_sample_survivors_follows_the_ordinary_plan(keep_records: bool) -> None:
    fixed, frame = _programs(SIGNED_PAIR, postselection_mask=[1])
    shots = 40_000
    expected = clifft.sample_survivors(fixed, shots, seed=21, keep_records=keep_records)
    actual = clifft.sample_survivors(frame, shots, seed=22, keep_records=keep_records)
    for attribute in ("passed_shots", "logical_errors"):
        a = getattr(expected, attribute)
        b = getattr(actual, attribute)
        pooled = (a + b) / (2 * shots)
        assert abs(a - b) / shots <= 6 * np.sqrt(2 * pooled * (1 - pooled) / shots)
    if keep_records:
        assert actual.measurements.shape == (actual.passed_shots, 4)
        assert not actual.detectors.any()


def test_record_probabilities_match_the_ordinary_plan() -> None:
    fixed, frame = _programs(RANDOM_OUTCOME)
    assert frame.peak_active_width < fixed.peak_active_width
    records = [f"{value:03b}" for value in range(8)]
    expected = clifft.record_probabilities(fixed, records)
    actual = clifft.record_probabilities(frame, records)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual.sum() == pytest.approx(1.0)


def test_fixed_fault_sampling_rejects_frame_plans() -> None:
    _, frame = _programs(SIGNED_PAIR)
    with pytest.raises(ValueError, match="runtime Clifford frame"):
        clifft.sample_k(frame, 16, 1, seed=3)
    with pytest.raises(ValueError, match="runtime Clifford frame"):
        clifft.sample_k_survivors(frame, 16, 1, seed=3)
