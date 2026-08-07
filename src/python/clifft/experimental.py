"""Explicitly selected experimental Clifft APIs.

The interfaces in this module may change between releases. They provide an
early validation surface for backends that are not yet used by the default
``clifft.compile`` and ``clifft.sample`` pipeline.
"""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from clifft import (
    BasisBitstrings,
    Circuit,
    MeasurementRecords,
    _basis_masks_from_bitstrings,
    _records_from_outcomes,
    noncomp,
)
from clifft._clifft_core import (
    HirPassManager,
    _basis_probabilities_experimental_sampling,
    _compile_experimental_sampling,
    _ExperimentalSamplingProgram,
    _get_statevector_experimental_sampling,
    _record_probabilities_experimental_sampling,
    _sample_experimental_sampling,
    _sample_survivors_experimental_sampling,
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
    postselection_mask: list[int] | None = None,
    expected_detectors: list[int] | None = None,
    expected_observables: list[int] | None = None,
    normalize_syndromes: bool = False,
    hir_passes: HirPassManager | None | _DefaultPasses = _DEFAULT_PASSES,
) -> Program:
    """Compile Stim text for the experimental scalar sampling backend.

    The default HIR optimization pipeline matches :func:`clifft.compile`;
    pass ``None`` to skip it.
    """
    if isinstance(hir_passes, _DefaultPasses):
        hir_passes = default_hir_pass_manager()
    return cast(
        Program,
        _compile_experimental_sampling(
            stim_text,
            postselection_mask if postselection_mask is not None else [],
            expected_detectors if expected_detectors is not None else [],
            expected_observables if expected_observables is not None else [],
            normalize_syndromes,
            hir_passes,
        ),
    )


def sample(program: Program, shots: int, seed: int | None = None) -> SampleResult:
    """Sample a prepared experimental program without changing Clifft's default backend."""
    measurements, detectors, observables, exp_vals = cast(
        tuple[
            npt.NDArray[np.uint8],
            npt.NDArray[np.uint8],
            npt.NDArray[np.uint8],
            npt.NDArray[np.float64],
        ],
        _sample_experimental_sampling(program, shots, seed),
    )
    return SampleResult(
        measurements,
        detectors,
        observables,
        exp_vals=exp_vals,
    )


def sample_noncomputational(
    circuit: Circuit | str,
    model: noncomp.Model,
    shots: int,
    seed: int | None = None,
    max_rank: int | None = None,
) -> noncomp.NonComputationalSample:
    """Sample leakage and loss with the experimental symbolic-coordinate backend.

    This has the same inputs and result type as :func:`clifft.noncomp.sample`,
    but does not change that API's default backend.
    """
    return noncomp._sample_experimental(circuit, model, shots, seed, max_rank)


def sample_survivors(
    program: Program,
    shots: int,
    seed: int | None = None,
    *,
    keep_records: bool = False,
) -> SampleResult:
    """Sample survivor counts and optional records from an experimental program."""
    (
        measurements,
        detectors,
        observables,
        total,
        passed,
        logical_errors,
        observable_ones,
        exp_vals,
    ) = cast(
        tuple[
            npt.NDArray[np.uint8],
            npt.NDArray[np.uint8],
            npt.NDArray[np.uint8],
            int,
            int,
            int,
            npt.NDArray[np.uint64],
            npt.NDArray[np.float64],
        ],
        _sample_survivors_experimental_sampling(program, shots, seed, keep_records),
    )
    return SampleResult(
        measurements,
        detectors,
        observables,
        total,
        passed,
        logical_errors,
        observable_ones,
        exp_vals,
    )


def record_probabilities(
    program: Program,
    records: MeasurementRecords,
    *,
    return_log: bool = False,
) -> npt.NDArray[np.float64]:
    """Return exact joint probabilities of experimental-program measurement records."""
    if program.num_measurements == 0:
        raise ValueError(
            "record_probabilities() requires a program with at least one "
            "measurement; use clifft.basis_probabilities() for unitary circuits."
        )
    record_array = _records_from_outcomes(program, records)
    log_probabilities = cast(
        npt.NDArray[np.float64],
        _record_probabilities_experimental_sampling(program, record_array),
    )
    if return_log:
        return np.where(log_probabilities == np.finfo(np.float64).min, -np.inf, log_probabilities)
    return np.exp(log_probabilities)


def basis_probabilities(
    program: Program,
    bitstrings: BasisBitstrings,
    *,
    bit_order: str = "big",
    return_log: bool = False,
) -> npt.NDArray[np.float64]:
    """Return exact Born probabilities from an experimental pure-state program."""
    probabilities = cast(
        npt.NDArray[np.float64],
        _basis_probabilities_experimental_sampling(
            program, _basis_masks_from_bitstrings(program, bitstrings, bit_order)
        ),
    )
    if return_log:
        with np.errstate(divide="ignore"):
            return np.log(probabilities)
    return probabilities


def get_statevector(program: Program) -> npt.NDArray[np.complex128]:
    """Return the dense final statevector of an experimental pure-state program."""
    return cast(npt.NDArray[np.complex128], _get_statevector_experimental_sampling(program))


__all__ = [
    "MeasurementRecords",
    "Program",
    "basis_probabilities",
    "compile",
    "get_statevector",
    "record_probabilities",
    "sample",
    "sample_noncomputational",
    "sample_survivors",
]
