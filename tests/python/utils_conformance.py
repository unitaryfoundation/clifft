"""Shared compiler configurations and independent sampling assertions."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt
import pytest

import clifft


@dataclass(frozen=True)
class CompilerProfile:
    name: str
    make_passes: Callable[[], Any]

    def compile(self, source: str) -> Any:
        # Pass instances may carry mutable state; never cache a pass manager.
        return clifft.compile(source, hir_passes=self.make_passes())


def fusion_squeeze_passes() -> clifft.HirPassManager:
    manager = clifft.HirPassManager()
    manager.add(clifft.PeepholeFusionPass())
    manager.add(clifft.StatevectorSqueezePass())
    return manager


def active_width_passes(
    schedule_pass: clifft.ActiveWidthSchedulePass | None = None,
) -> clifft.HirPassManager:
    manager = fusion_squeeze_passes()
    manager.add(schedule_pass if schedule_pass is not None else clifft.ActiveWidthSchedulePass())
    return manager


UNOPTIMIZED = CompilerProfile("unoptimized", lambda: None)
FUSION_SQUEEZE = CompilerProfile("fusion-squeeze", fusion_squeeze_passes)
DEFAULT = CompilerProfile("default", lambda: clifft.default_hir_pass_manager())
ACTIVE_WIDTH = CompilerProfile("active-width", lambda: active_width_passes())
COMPILER_PROFILES = (UNOPTIMIZED, DEFAULT, ACTIVE_WIDTH)


def skip_unavailable_thread_layout(error: ValueError, layout: tuple[int, int]) -> NoReturn:
    if (
        layout[1] > 1
        and str(error) == "thread_layout intra-shot workers require an OpenMP-enabled build"
    ):
        pytest.skip("Clifft was built without OpenMP")
    if min(layout) > 1 and str(error) == "hybrid thread_layout requires OMP_PROC_BIND=false":
        pytest.skip("Hybrid sampling requires OMP_PROC_BIND=false")
    raise error


@dataclass(frozen=True)
class CpuSamplingMode:
    name: str
    batch_size: int | str
    threads: int = 1
    thread_layout: tuple[int, int] | None = None
    intra_shot_min_active_width: int | None = None

    @staticmethod
    def compile(source: str, **kwargs: Any) -> Any:
        return clifft.compile(source, **kwargs)

    def sample(self, program: Any, shots: int, seed: int | None = None) -> Any:
        try:
            return clifft.sample(
                program,
                shots,
                seed=seed,
                threads=self.threads,
                batch_size=self.batch_size,
                thread_layout=self.thread_layout,
                intra_shot_min_active_width=self.intra_shot_min_active_width,
            )
        except ValueError as error:
            if self.thread_layout is not None:
                skip_unavailable_thread_layout(error, self.thread_layout)
            raise

    def sample_survivors(
        self, program: Any, shots: int, *, seed: int | None = None, keep_records: bool = False
    ) -> Any:
        try:
            return clifft.sample_survivors(
                program,
                shots,
                seed=seed,
                keep_records=keep_records,
                threads=self.threads,
                batch_size=self.batch_size,
                thread_layout=self.thread_layout,
                intra_shot_min_active_width=self.intra_shot_min_active_width,
            )
        except ValueError as error:
            if self.thread_layout is not None:
                skip_unavailable_thread_layout(error, self.thread_layout)
            raise


CPU_SAMPLING_MODES = (
    CpuSamplingMode("single-shot", 1),
    CpuSamplingMode("packed-65", 65),
    CpuSamplingMode("automatic", "auto"),
    CpuSamplingMode("scalar-2-workers", 1, threads=2),
    CpuSamplingMode("packed-65-2-workers", 65, threads=2),
    CpuSamplingMode("intra-shot-2-workers", 1, thread_layout=(1, 2), intra_shot_min_active_width=3),
    CpuSamplingMode("hybrid-2x2-workers", 1, thread_layout=(2, 2), intra_shot_min_active_width=3),
)

# Two full packed-65 batches plus a tail allow both workers to receive work.
SMALL_CIRCUIT_SHOTS = 2 * 65 + 1


@lru_cache(maxsize=256)
def unitary_reference(source: str) -> npt.NDArray[np.complex128]:
    """Calculate once per source per worker, independently of compiler profiles."""
    # Loading Aer before Clifft can change OpenMP initialization on macOS.
    # CPU sampling fixtures need no reference simulator at import time.
    from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

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
