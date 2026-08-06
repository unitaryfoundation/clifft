"""Explicitly selected experimental Clifft APIs.

The interfaces in this module may change between releases. They provide an
early validation surface for backends that are not yet used by the default
``clifft.compile`` and ``clifft.sample`` pipeline.
"""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from clifft import MeasurementRecords, _records_from_outcomes
from clifft._clifft_core import (
    HirPassManager,
    _compile_experimental_sampling,
    _ExperimentalSamplingProgram,
    _record_probabilities_experimental_sampling,
    _sample_experimental_sampling,
    default_hir_pass_manager,
)
from clifft._sample_result import SampleResult

Program: TypeAlias = _ExperimentalSamplingProgram


class _DefaultPasses:
    """Sentinel marker for the experimental compiler's default HIR passes."""


_DEFAULT_PASSES = _DefaultPasses()


def compile(
    stim_text: str,
    *,
    hir_passes: HirPassManager | None | _DefaultPasses = _DEFAULT_PASSES,
) -> Program:
    """Compile Stim text for the experimental scalar sampling backend.

    Only the currently implemented noiseless rotation-and-measurement subset
    is accepted. Unsupported operations fail during compilation. The default
    HIR optimization pipeline matches :func:`clifft.compile`; pass ``None`` to
    skip it.
    """
    if isinstance(hir_passes, _DefaultPasses):
        hir_passes = default_hir_pass_manager()
    return cast(Program, _compile_experimental_sampling(stim_text, hir_passes))


def sample(program: Program, shots: int, seed: int | None = None) -> SampleResult:
    """Sample a prepared experimental program without changing Clifft's default backend."""
    measurements = cast(npt.NDArray[np.uint8], _sample_experimental_sampling(program, shots, seed))
    detectors = np.empty((shots, 0), dtype=np.uint8)
    observables = np.empty((shots, 0), dtype=np.uint8)
    return SampleResult(measurements, detectors, observables)


def record_probabilities(
    program: Program,
    records: MeasurementRecords,
    *,
    return_log: bool = False,
) -> npt.NDArray[np.float64]:
    """Return exact joint probabilities of experimental-program measurement records."""
    if program.num_measurements == 0:
        raise ValueError("record_probabilities() requires at least one measurement")
    record_array = _records_from_outcomes(program, records)
    log_probabilities = cast(
        npt.NDArray[np.float64],
        _record_probabilities_experimental_sampling(program, record_array),
    )
    if return_log:
        return np.where(log_probabilities == np.finfo(np.float64).min, -np.inf, log_probabilities)
    return np.exp(log_probabilities)


__all__ = ["MeasurementRecords", "Program", "compile", "record_probabilities", "sample"]
