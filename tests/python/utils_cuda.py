"""Reusable CPU-oracle checks for experimental CUDA kernel development."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from utils_gpu_replay import (
    assert_forced_record_probabilities as assert_forced_record_probabilities,
)

import clifft
from clifft.experimental import cuda


def require_cuda_device() -> None:
    """Skip the current test when no NVIDIA device is visible."""
    if not cuda.is_available():
        pytest.skip("requires an NVIDIA GPU visible to CUDA")


def assert_same_rows(left: clifft.SampleResult, right: clifft.SampleResult) -> None:
    """Compare all fixed-row outputs exactly."""
    np.testing.assert_array_equal(left.measurements, right.measurements)
    np.testing.assert_array_equal(left.detectors, right.detectors)
    np.testing.assert_array_equal(left.observables, right.observables)
    np.testing.assert_array_equal(left.exp_vals, right.exp_vals)


def assert_repeatable(sampler: cuda.Sampler, shots: int, seed: int) -> None:
    """Check that retained execution and batching preserve seeded rows."""
    assert_same_rows(sampler.sample(shots, seed=seed), sampler.sample(shots, seed=seed))


def assert_rate_matches(
    cpu_count: int,
    cpu_total: int,
    gpu_count: int,
    gpu_total: int,
    *,
    sigma: float = 6.0,
    absolute_floor: float = 1e-3,
) -> None:
    """Compare two empirical rates under a two-sample binomial tolerance."""
    cpu_rate = cpu_count / cpu_total
    gpu_rate = gpu_count / gpu_total
    pooled = (cpu_count + gpu_count) / (cpu_total + gpu_total)
    tolerance = sigma * np.sqrt(pooled * (1.0 - pooled) * (1.0 / cpu_total + 1.0 / gpu_total))
    assert gpu_rate == pytest.approx(cpu_rate, abs=tolerance + absolute_floor)


def assert_distribution_matches(
    cpu_rows: npt.NDArray[np.uint8],
    gpu_rows: npt.NDArray[np.uint8],
    *,
    sigma: float = 6.0,
    absolute_floor: float = 1e-3,
) -> None:
    """Compare complete empirical row distributions with two-sample tolerances."""
    if cpu_rows.ndim != 2 or gpu_rows.ndim != 2 or cpu_rows.shape[1] != gpu_rows.shape[1]:
        raise ValueError("row arrays must be 2D with the same number of columns")
    if cpu_rows.shape[0] == 0 or gpu_rows.shape[0] == 0:
        raise ValueError("distribution comparisons require non-empty samples")

    width = cpu_rows.shape[1]
    if width > 63:
        raise ValueError("distribution helper supports at most 63 output bits")
    powers = np.left_shift(np.uint64(1), np.arange(width, dtype=np.uint64))
    cpu_keys = cpu_rows.astype(np.uint64) @ powers
    gpu_keys = gpu_rows.astype(np.uint64) @ powers
    for key in np.union1d(cpu_keys, gpu_keys):
        cpu_probability = float(np.mean(cpu_keys == key))
        gpu_probability = float(np.mean(gpu_keys == key))
        variance = (
            cpu_probability
            * (1.0 - cpu_probability)
            * (1.0 / cpu_rows.shape[0] + 1.0 / gpu_rows.shape[0])
        )
        tolerance = sigma * np.sqrt(variance) + absolute_floor
        assert gpu_probability == pytest.approx(cpu_probability, abs=tolerance)
