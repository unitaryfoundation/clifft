"""Structured result type for UCC sampling functions."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import numpy.typing as npt


class SampleResult:
    """Structured result from UCC sampling functions.

    Common attributes:
        measurements: uint8 array, shape (shots, num_measurements)
        detectors: uint8 array, shape (shots, num_detectors)
        observables: uint8 array, shape (shots, num_observables)

    Survivor-only attributes:
        total_shots: total number of shots attempted
        passed_shots: number of shots that survived post-selection
        discards: number of discarded shots (total_shots - passed_shots)
        logical_errors: number of surviving shots with at least one
            observable flipped
        observable_ones: uint64 array of per-observable error counts

    Supports tuple unpacking for backward compatibility:

        m, d, o = ucc.sample(prog, shots)
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
    ) -> None:
        self.measurements = measurements
        self.detectors = detectors
        self.observables = observables
        self.exp_vals = (
            exp_vals
            if exp_vals is not None
            else np.empty((measurements.shape[0], 0), dtype=np.float64)
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
            f"shots={self.measurements.shape[0]}",
            f"measurements={self.measurements.shape[1]}",
        ]
        if self.total_shots is not None:
            parts.append(f"total_shots={self.total_shots}")
        if self.passed_shots is not None:
            parts.append(f"passed_shots={self.passed_shots}")
        return f"SampleResult({', '.join(parts)})"
