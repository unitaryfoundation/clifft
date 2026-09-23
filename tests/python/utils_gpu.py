"""Shared sampling cases and assertions for HIP and CUDA conformance tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

import clifft

if TYPE_CHECKING:
    from clifft.experimental import cuda, hip


# Narrow enough for CUDA to select thread-per-shot automatically, so forcing
# its cooperative tiers exercises their noise and output paths on a small case.
NARROW_NOISY_CIRCUIT = """\
H 0
T 0
H 0
CX 0 1
PAULI_CHANNEL_1(0.1, 0.2, 0.05) 1
M 0 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def assert_same_rows(left: clifft.SampleResult, right: clifft.SampleResult) -> None:
    """Compare all fixed-row outputs exactly."""
    np.testing.assert_array_equal(left.measurements, right.measurements)
    np.testing.assert_array_equal(left.detectors, right.detectors)
    np.testing.assert_array_equal(left.observables, right.observables)
    np.testing.assert_array_equal(left.exp_vals, right.exp_vals)


def assert_repeatable(sampler: cuda.Sampler | hip.Sampler, shots: int, seed: int) -> None:
    """Check that retained execution and batching preserve seeded rows."""
    assert_same_rows(sampler.sample(shots, seed=seed), sampler.sample(shots, seed=seed))


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
