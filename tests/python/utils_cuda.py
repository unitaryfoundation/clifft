"""Reusable CPU-oracle checks for experimental CUDA kernel development."""

from __future__ import annotations

import numpy as np
import pytest
from utils_gpu import (
    assert_distribution_matches as assert_distribution_matches,
)
from utils_gpu import (
    assert_repeatable as assert_repeatable,
)
from utils_gpu import (
    assert_same_rows as assert_same_rows,
)
from utils_gpu_replay import (
    assert_forced_record_probabilities as assert_forced_record_probabilities,
)

from clifft.experimental import cuda


def require_cuda_device() -> None:
    """Skip the current test when no NVIDIA device is visible."""
    if not cuda.is_available():
        pytest.skip("requires an NVIDIA GPU visible to CUDA")


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
