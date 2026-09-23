"""Reusable CPU-oracle checks for experimental HIP kernel development."""

from __future__ import annotations

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

from clifft.experimental import hip


def require_hip_device() -> None:
    """Skip the current test when no AMD device is visible."""
    if not hip.is_available():
        pytest.skip("requires an AMD GPU visible to HIP")
