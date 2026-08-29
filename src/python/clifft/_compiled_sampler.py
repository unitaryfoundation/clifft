"""Stim-shaped compiled sampler facades over Clifft's retained C++ runtime."""

from __future__ import annotations

import operator
import os
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from clifft._clifft_core import (
    Circuit,
    HirPassManager,
    _compile_fixed_sampler_circuit,
    _compile_fixed_sampler_text,
    _compile_postselected_detector_sampler_text,
    _CompiledSampler,
    default_hir_pass_manager,
)

ThreadOption = int | Literal["auto"]
BatchOption = int | Literal["auto"]
FileFormat = Literal["01", "b8"]
SampleArray = npt.NDArray[np.bool_] | npt.NDArray[np.uint8]


class _DefaultPasses:
    pass


_DEFAULT_PASSES = _DefaultPasses()


def _shot_count(shots: int) -> int:
    try:
        value = operator.index(shots)
    except TypeError as ex:
        raise TypeError("shots must be an integer") from ex
    if value < 0 or value > np.iinfo(np.uint32).max:
        raise ValueError("shots must be between 0 and 2**32 - 1")
    return value


def _path_string(path: os.PathLike[str] | str, field: str) -> str:
    value = os.fspath(path)
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or text path-like object")
    return value


def _allocate_sample_array(shots: int, columns: int, bit_packed: bool) -> SampleArray:
    if bit_packed:
        return np.empty((shots, (columns + 7) // 8), dtype=np.uint8)
    return np.empty((shots, columns), dtype=np.bool_)


class CompiledMeasurementSampler:
    """A reusable compiled sampler with Stim-compatible measurement outputs.

    Instances retain their native executor workers and RNG stream. Construct one
    with :func:`compile_sampler` instead of calling this class directly.
    """

    __slots__ = ("_native",)

    def __init__(self, native: _CompiledSampler):
        self._native = native

    @property
    def num_measurements(self) -> int:
        return int(self._native.num_measurements)

    def sample(self, shots: int, *, bit_packed: bool = False) -> SampleArray:
        """Sample measurement rows as bool values or little-endian packed bytes."""
        shot_count = _shot_count(shots)
        output = _allocate_sample_array(shot_count, self.num_measurements, bit_packed)
        self._native._sample_measurements(shot_count, bit_packed, output)
        return output

    def sample_write(
        self,
        shots: int,
        *,
        filepath: os.PathLike[str] | str,
        format: FileFormat = "01",
    ) -> None:
        """Stream measurement rows directly to a file in ``01`` or ``b8`` format."""
        self._native._sample_write_measurements(
            _shot_count(shots), _path_string(filepath, "filepath"), format
        )

    def __repr__(self) -> str:
        return f"CompiledMeasurementSampler(num_measurements={self.num_measurements})"


class CompiledDetectorSampler:
    """A reusable compiled sampler with Stim-compatible detector outputs.

    Detector and observable bits are normalized against the circuit's noiseless
    reference, matching Stim's detector sampler convention. Construct one with
    :func:`compile_detector_sampler` instead of calling this class directly.
    """

    __slots__ = ("_native",)

    def __init__(self, native: _CompiledSampler):
        self._native = native

    @property
    def num_detectors(self) -> int:
        return int(self._native.num_detectors)

    @property
    def num_observables(self) -> int:
        return int(self._native.num_observables)

    def sample(
        self,
        shots: int,
        *,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
        dets_out: SampleArray | None = None,
        obs_out: SampleArray | None = None,
    ) -> SampleArray | tuple[SampleArray, SampleArray]:
        """Sample detector rows with Stim-compatible observable placement."""
        if separate_observables and (prepend_observables or append_observables):
            raise ValueError(
                "separate_observables cannot be combined with prepending or appending observables"
            )
        shot_count = _shot_count(shots)
        main_columns = self.num_detectors + self.num_observables * (
            int(prepend_observables) + int(append_observables)
        )
        detector_output = (
            _allocate_sample_array(shot_count, main_columns, bit_packed)
            if dets_out is None
            else dets_out
        )
        observable_output = obs_out
        if separate_observables and observable_output is None:
            observable_output = _allocate_sample_array(shot_count, self.num_observables, bit_packed)

        self._native._sample_detectors(
            shot_count,
            prepend_observables,
            append_observables,
            separate_observables,
            bit_packed,
            detector_output,
            observable_output,
        )
        if separate_observables:
            return detector_output, cast(SampleArray, observable_output)
        return detector_output

    def sample_write(
        self,
        shots: int,
        *,
        filepath: os.PathLike[str] | str,
        format: FileFormat = "01",
        prepend_observables: bool = False,
        append_observables: bool = False,
        obs_out_filepath: os.PathLike[str] | str | None = None,
        obs_out_format: FileFormat = "01",
    ) -> None:
        """Stream detector and optional observable rows without matrix materialization."""
        observable_path = (
            None if obs_out_filepath is None else _path_string(obs_out_filepath, "obs_out_filepath")
        )
        self._native._sample_write_detectors(
            _shot_count(shots),
            _path_string(filepath, "filepath"),
            format,
            prepend_observables,
            append_observables,
            observable_path,
            obs_out_format,
        )

    def _sample_postselected(
        self, shots: int
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], int]:
        """Return packed survivor rows for Sinter's batch-level orchestration."""
        shot_count = _shot_count(shots)
        detectors = cast(
            npt.NDArray[np.uint8],
            _allocate_sample_array(shot_count, self.num_detectors, True),
        )
        observables = cast(
            npt.NDArray[np.uint8],
            _allocate_sample_array(shot_count, self.num_observables, True),
        )
        survivors = int(
            self._native._sample_postselected_detectors(shot_count, detectors, observables)
        )
        return detectors[:survivors], observables[:survivors], survivors

    def __repr__(self) -> str:
        return (
            "CompiledDetectorSampler("
            f"num_detectors={self.num_detectors}, num_observables={self.num_observables})"
        )


def _compile_sampler(
    circuit: object,
    *,
    detector_profile: bool,
    seed: int | None,
    threads: ThreadOption,
    batch_size: BatchOption,
    hir_passes: HirPassManager | None,
) -> _CompiledSampler:
    if isinstance(circuit, Circuit):
        return _compile_fixed_sampler_circuit(
            circuit, detector_profile, seed, threads, batch_size, hir_passes
        )
    if isinstance(circuit, os.PathLike):
        text = Path(_path_string(circuit, "circuit")).read_text(encoding="utf-8")
    elif isinstance(circuit, str):
        text = circuit
    else:
        circuit_type = type(circuit)
        is_stim_circuit = (
            circuit_type.__module__.partition(".")[0] == "stim"
            and circuit_type.__name__ == "Circuit"
        )
        if not is_stim_circuit:
            raise TypeError("circuit must be Stim text, a path, stim.Circuit, or clifft.Circuit")
        text = str(circuit)
    return _compile_fixed_sampler_text(
        text, detector_profile, seed, threads, batch_size, hir_passes
    )


def compile_sampler(
    circuit: object,
    *,
    seed: int | None = None,
    threads: ThreadOption = 1,
    batch_size: BatchOption = "auto",
    hir_passes: HirPassManager | None | _DefaultPasses = _DEFAULT_PASSES,
) -> CompiledMeasurementSampler:
    """Compile a retained measurement sampler.

    Args:
        circuit: Stim-format text, a text path-like object, ``stim.Circuit``, or
            ``clifft.Circuit``.
        seed: Optional seed for a reproducible sequence of sampler calls.
        threads: Positive native worker budget or ``"auto"``.
        batch_size: Positive packed lane capacity or ``"auto"``.
        hir_passes: HIR pass manager to run before lowering. The default applies
            Clifft's standard passes; pass ``None`` to skip optimization.

    Returns:
        A sampler whose ``sample`` and ``sample_write`` signatures match Stim's
        common compiled measurement sampler surface.
    """
    passes = default_hir_pass_manager() if isinstance(hir_passes, _DefaultPasses) else hir_passes
    native = _compile_sampler(
        circuit,
        detector_profile=False,
        seed=seed,
        threads=threads,
        batch_size=batch_size,
        hir_passes=passes,
    )
    return CompiledMeasurementSampler(native)


def compile_detector_sampler(
    circuit: object,
    *,
    seed: int | None = None,
    threads: ThreadOption = 1,
    batch_size: BatchOption = "auto",
    hir_passes: HirPassManager | None | _DefaultPasses = _DEFAULT_PASSES,
) -> CompiledDetectorSampler:
    """Compile a retained, reference-normalized detector sampler.

    Args:
        circuit: Stim-format text, a text path-like object, ``stim.Circuit``, or
            ``clifft.Circuit``.
        seed: Optional seed for a reproducible sequence of sampler calls.
        threads: Positive native worker budget or ``"auto"``.
        batch_size: Positive packed lane capacity or ``"auto"``.
        hir_passes: HIR pass manager to run before lowering. The default applies
            Clifft's standard passes; pass ``None`` to skip optimization.

    Returns:
        A detector sampler supporting observable placement, separate outputs,
        caller-owned arrays, bit packing, and native file streaming.
    """
    passes = default_hir_pass_manager() if isinstance(hir_passes, _DefaultPasses) else hir_passes
    native = _compile_sampler(
        circuit,
        detector_profile=True,
        seed=seed,
        threads=threads,
        batch_size=batch_size,
        hir_passes=passes,
    )
    return CompiledDetectorSampler(native)


def _compile_postselected_detector_sampler(
    stim_text: str,
    postselection_mask: list[int],
    *,
    threads: ThreadOption,
    batch_size: BatchOption,
) -> CompiledDetectorSampler:
    native = _compile_postselected_detector_sampler_text(
        stim_text,
        postselection_mask,
        None,
        threads,
        batch_size,
        default_hir_pass_manager(),
    )
    return CompiledDetectorSampler(native)
