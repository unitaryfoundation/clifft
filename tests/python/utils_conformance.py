"""Shared compiler configurations and independent sampling assertions."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

import clifft


@dataclass(frozen=True)
class CompilerProfile:
    name: str
    make_passes: Callable[[], Any]

    def compile(self, source: str) -> Any:
        # Pass instances may carry mutable state; never cache a pass manager.
        return clifft.compile(source, hir_passes=self.make_passes())


UNOPTIMIZED = CompilerProfile("unoptimized", lambda: None)
DEFAULT = CompilerProfile("default", lambda: clifft.default_hir_pass_manager())
COMPILER_PROFILES = (UNOPTIMIZED, DEFAULT)


@dataclass(frozen=True)
class CpuSamplingMode:
    name: str
    batch_size: int | str

    def sample(self, program: Any, shots: int, seed: int) -> Any:
        # Host-dependent worker counts can change automatic batch selection.
        return clifft.sample(program, shots, seed=seed, threads=1, batch_size=self.batch_size)


CPU_SAMPLING_MODES = (
    CpuSamplingMode("single-shot", 1),
    CpuSamplingMode("packed-65", 65),
    CpuSamplingMode("automatic", "auto"),
)


@lru_cache(maxsize=256)
def unitary_reference(source: str) -> npt.NDArray[np.complex128]:
    """Calculate once per source per worker, independently of compiler profiles."""
    state = qiskit_statevector(stim_to_qiskit_noiseless(source))
    state.setflags(write=False)
    return state


def assert_joint_distribution(measurements: npt.NDArray[np.uint8], expected: npt.ArrayLike) -> None:
    """Compare a small complete record histogram, with a family-wise error bound."""
    probabilities = np.asarray(expected, dtype=np.float64)
    assert measurements.ndim == 2 and 0 < measurements.shape[0]
    shots, width = measurements.shape
    assert width <= 10, "joint enumeration is deliberately bounded to small records"
    assert np.all((measurements == 0) | (measurements == 1))
    assert probabilities.shape == (1 << width,)
    assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-12, rtol=0)

    keys = measurements.astype(np.int64) @ (1 << np.arange(width))
    counts = np.bincount(keys, minlength=len(probabilities))
    frequencies = counts / shots
    # Bernstein's bound, unioned over all bins, keeps the test meaningful
    # without a normal approximation for rare outcomes. The 1e-7 budget is
    # per comparison, not per bin; exact zero has a stricter check below.
    log_bound = np.log(2 * len(probabilities) / 1e-7)
    variance = probabilities * np.maximum(1 - probabilities, 0)
    tolerance = np.sqrt(2 * variance * log_bound / shots) + 2 * log_bound / (3 * shots)
    np.testing.assert_array_equal(counts[probabilities == 0], 0)
    assert np.all(np.abs(frequencies - probabilities) <= tolerance), (
        f"Joint distribution differs: expected={probabilities}, "
        f"observed={frequencies}, tolerance={tolerance}"
    )
