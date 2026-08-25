"""Reusable CPU-oracle checks for experimental HIP kernel development."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

import clifft
from clifft.experimental import hip


def require_hip_device() -> None:
    """Skip the current test when no AMD device is visible."""
    if not hip.is_available():
        pytest.skip("requires an AMD GPU visible to HIP")


def assert_same_rows(left: clifft.SampleResult, right: clifft.SampleResult) -> None:
    """Compare all fixed-row outputs exactly."""
    np.testing.assert_array_equal(left.measurements, right.measurements)
    np.testing.assert_array_equal(left.detectors, right.detectors)
    np.testing.assert_array_equal(left.observables, right.observables)
    np.testing.assert_array_equal(left.exp_vals, right.exp_vals)


def assert_repeatable(sampler: hip.Sampler, shots: int, seed: int) -> None:
    """Check that retained execution and batching preserve seeded rows."""
    assert_same_rows(sampler.sample(shots, seed=seed), sampler.sample(shots, seed=seed))


def assert_distribution_matches(
    cpu_rows: npt.NDArray[np.uint8],
    hip_rows: npt.NDArray[np.uint8],
    *,
    sigma: float = 6.0,
    absolute_floor: float = 1e-3,
) -> None:
    """Compare complete empirical row distributions with two-sample tolerances."""
    if cpu_rows.ndim != 2 or hip_rows.ndim != 2 or cpu_rows.shape[1] != hip_rows.shape[1]:
        raise ValueError("row arrays must be 2D with the same number of columns")
    if cpu_rows.shape[0] == 0 or hip_rows.shape[0] == 0:
        raise ValueError("distribution comparisons require non-empty samples")

    width = cpu_rows.shape[1]
    if width > 63:
        raise ValueError("distribution helper supports at most 63 output bits")
    powers = np.left_shift(np.uint64(1), np.arange(width, dtype=np.uint64))
    cpu_keys = cpu_rows.astype(np.uint64) @ powers
    hip_keys = hip_rows.astype(np.uint64) @ powers
    for key in np.union1d(cpu_keys, hip_keys):
        cpu_probability = float(np.mean(cpu_keys == key))
        hip_probability = float(np.mean(hip_keys == key))
        variance = (
            cpu_probability
            * (1.0 - cpu_probability)
            * (1.0 / cpu_rows.shape[0] + 1.0 / hip_rows.shape[0])
        )
        tolerance = sigma * np.sqrt(variance) + absolute_floor
        assert hip_probability == pytest.approx(cpu_probability, abs=tolerance)


def assert_forced_record_probabilities(
    cpu_program: clifft.Program,
    hip_sampler: hip.Sampler,
    *,
    absolute_tolerance: float,
) -> None:
    """Enumerate every small record branch and compare reachability and probability."""
    num_records = hip_sampler.program.num_records
    assert num_records == cpu_program.num_measurements + cpu_program.num_hidden_measurements
    records = np.asarray(list(itertools.product((0, 1), repeat=num_records)), dtype=np.uint8)
    cpu_log_probabilities = clifft.record_probabilities(cpu_program, records, return_log=True)
    for record, log_probability in zip(records, cpu_log_probabilities, strict=True):
        replay = hip_sampler.replay_shot(record.tolist())
        assert replay.reachable == np.isfinite(log_probability)
        if replay.reachable:
            assert replay.log_probability == pytest.approx(log_probability, abs=absolute_tolerance)
            np.testing.assert_array_equal(replay.outputs.measurements, record[np.newaxis, :])
