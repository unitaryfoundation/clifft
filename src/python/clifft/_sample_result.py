"""Structured result type for Clifft sampling functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from clifft._clifft_core import BatchTuningReport


class SampleResult:
    """Structured result from Clifft sampling functions.

    Common attributes:
        measurements: uint8 array, shape (shots, num_measurements)
        detectors: uint8 array, shape (shots, num_detectors)
        observables: uint8 array, shape (shots, num_observables)
        retained_shots: number of retained rows, independent of the output selection
        bit_packed: whether uint8 rows use little-endian b8 packing
        batch_tuning: calibration report when batch_size="tune", otherwise None

    Omitted outputs have zero rows and keep their declared column width. Outputs
    with no declared columns have shape (retained_shots, 0). Bit packing changes
    each uint8 width to ceil(columns / 8); expectation values remain float64.

    Survivor-only attributes:
        total_shots: total number of shots attempted
        passed_shots: number of shots that survived post-selection
        discards: number of discarded shots (total_shots - passed_shots)
        logical_errors: number of surviving shots with at least one
            observable flipped
        observable_ones: uint64 array of per-observable error counts, shape
            (num_observables,), filled with zeros when total_shots is zero

    Supports tuple unpacking for backward compatibility:

        m, d, o = clifft.sample(prog, shots)
    """

    __slots__ = (
        "measurements",
        "detectors",
        "observables",
        "exp_vals",
        "total_shots",
        "passed_shots",
        "discards",
        "logical_errors",
        "observable_ones",
        "batch_tuning",
        "retained_shots",
        "bit_packed",
    )

    def __init__(
        self,
        measurements: npt.NDArray[np.uint8],
        detectors: npt.NDArray[np.uint8],
        observables: npt.NDArray[np.uint8],
        total_shots: int | None = None,
        passed_shots: int | None = None,
        logical_errors: int | None = None,
        observable_ones: npt.NDArray[np.uint64] | None = None,
        exp_vals: npt.NDArray[np.float64] | None = None,
        batch_tuning: BatchTuningReport | None = None,
        *,
        retained_shots: int | None = None,
        bit_packed: bool = False,
    ) -> None:
        self.retained_shots = (
            max(measurements.shape[0], detectors.shape[0], observables.shape[0])
            if retained_shots is None
            else retained_shots
        )
        self.bit_packed = bit_packed
        self.batch_tuning = batch_tuning
        self.measurements = measurements
        self.detectors = detectors
        self.observables = observables
        self.exp_vals = (
            exp_vals
            if exp_vals is not None
            else np.empty((self.retained_shots, 0), dtype=np.float64)
        )
        self.total_shots = total_shots
        self.passed_shots = passed_shots
        self.discards = (
            None if total_shots is None or passed_shots is None else total_shots - passed_shots
        )
        self.logical_errors = logical_errors
        self.observable_ones = observable_ones

    def __iter__(self) -> Iterator[npt.NDArray[np.uint8]]:
        """Yield (measurements, detectors, observables) for tuple unpacking."""
        yield self.measurements
        yield self.detectors
        yield self.observables

    def __repr__(self) -> str:
        parts = [
            f"shots={self.retained_shots}",
            f"measurements={self.measurements.shape[1]}",
        ]
        if self.bit_packed:
            parts.append("bit_packed=True")
        if self.total_shots is not None:
            parts.append(f"total_shots={self.total_shots}")
        if self.passed_shots is not None:
            parts.append(f"passed_shots={self.passed_shots}")
        return f"SampleResult({', '.join(parts)})"
