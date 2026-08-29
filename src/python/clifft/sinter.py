"""Sinter sampling backend powered by Clifft's retained native sampler."""

from __future__ import annotations

import collections
import pathlib
import tempfile
import time
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

try:
    import sinter as _sinter
except ModuleNotFoundError as ex:  # pragma: no cover - exercised without the optional extra
    if ex.name != "sinter":
        raise
    raise ModuleNotFoundError(
        "clifft.sinter requires the optional Sinter dependency; "
        "install clifft[sinter] or clifft[sinter-pymatching]"
    ) from ex

from clifft._compiled_sampler import (
    BatchOption,
    CompiledDetectorSampler,
    ThreadOption,
    _compile_postselected_detector_sampler,
    _shot_count,
    compile_detector_sampler,
)

PackedArray = npt.NDArray[np.uint8]
DecoderOption: TypeAlias = str | _sinter.Decoder


class _PyMatchingCompiledDecoder(_sinter.CompiledDecoder):
    def __init__(self, matcher: Any, correlated: bool):
        self._matcher = matcher
        self._correlated = correlated

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: PackedArray
    ) -> PackedArray:
        return cast(
            PackedArray,
            self._matcher.decode_batch(
                shots=bit_packed_detection_event_data,
                bit_packed_shots=True,
                bit_packed_predictions=True,
                return_weights=False,
                enable_correlations=self._correlated,
            ),
        )


class _PyMatchingDecoder(_sinter.Decoder):
    def __init__(self, correlated: bool):
        self._correlated = correlated

    def compile_decoder_for_dem(self, *, dem: Any) -> _sinter.CompiledDecoder:
        try:
            import pymatching
        except ModuleNotFoundError as ex:
            raise ModuleNotFoundError(
                "ClifftSampler(decoder='pymatching') requires PyMatching; "
                "install clifft[sinter-pymatching]"
            ) from ex
        matcher = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=self._correlated
        )
        return _PyMatchingCompiledDecoder(matcher, self._correlated)


class _DiskCompiledDecoder(_sinter.CompiledDecoder):
    def __init__(
        self,
        *,
        decoder: _sinter.Decoder,
        dem: Any,
        tmp_dir: pathlib.Path,
    ):
        self._decoder = decoder
        self._dem = dem
        self._temporary_directory = tempfile.TemporaryDirectory(dir=tmp_dir)
        self._directory = pathlib.Path(self._temporary_directory.name)
        self._decoder_directory = self._directory / "decoder"
        self._decoder_directory.mkdir()
        self._dem_path = self._directory / "model.dem"
        self._dets_path = self._directory / "detectors.b8"
        self._predictions_path = self._directory / "predictions.b8"
        dem.to_file(self._dem_path)

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: PackedArray
    ) -> PackedArray:
        shots = bit_packed_detection_event_data.shape[0]
        bit_packed_detection_event_data.tofile(self._dets_path)
        self._decoder.decode_via_files(
            num_shots=shots,
            num_obs=self._dem.num_observables,
            num_dets=self._dem.num_detectors,
            dem_path=self._dem_path,
            dets_b8_in_path=self._dets_path,
            obs_predictions_b8_out_path=self._predictions_path,
            tmp_dir=self._decoder_directory,
        )
        observable_bytes = (self._dem.num_observables + 7) // 8
        prediction = np.fromfile(
            self._predictions_path,
            dtype=np.uint8,
            count=shots * observable_bytes,
        )
        self._dets_path.unlink(missing_ok=True)
        self._predictions_path.unlink(missing_ok=True)
        if prediction.size != shots * observable_bytes:
            raise ValueError("file decoder returned the wrong number of observable bytes")
        return prediction.reshape((shots, observable_bytes))


def _resolve_decoder(decoder: DecoderOption) -> _sinter.Decoder:
    if isinstance(decoder, _sinter.Decoder):
        return decoder
    if decoder == "pymatching":
        return _PyMatchingDecoder(False)
    if decoder == "pymatching-correlated":
        return _PyMatchingDecoder(True)
    raise ValueError("decoder must be a sinter.Decoder, 'pymatching', or 'pymatching-correlated'")


def _compile_decoder(
    decoder: _sinter.Decoder,
    *,
    dem: Any,
    decoder_name: str | None,
    tmp_dir: pathlib.Path | None,
) -> _sinter.CompiledDecoder:
    try:
        return decoder.compile_decoder_for_dem(dem=dem)
    except NotImplementedError:
        if tmp_dir is None:
            name = decoder_name if decoder_name is not None else type(decoder).__qualname__
            raise ValueError(
                f"decoder {name!r} does not implement compile_decoder_for_dem and no "
                "tmp_dir was provided for decode_via_files"
            ) from None
        return _DiskCompiledDecoder(decoder=decoder, dem=dem, tmp_dir=tmp_dir)


def _detector_error_model(task: _sinter.Task) -> Any:
    if task.detector_error_model is not None:
        return task.detector_error_model
    if task.circuit is None:
        raise ValueError("ClifftSampler requires task.circuit to be loaded")
    try:
        return task.circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
        )
    except ValueError:
        try:
            return task.circuit.detector_error_model(approximate_disjoint_errors=True)
        except ValueError:
            return task.circuit.detector_error_model(
                approximate_disjoint_errors=True,
                flatten_loops=True,
            )


def _validate_predictions(
    predictions: object,
    *,
    shots: int,
    observable_bytes: int,
) -> PackedArray:
    if not isinstance(predictions, np.ndarray):
        raise ValueError("decoder predictions must be a NumPy array")
    if predictions.dtype != np.uint8:
        raise ValueError("decoder predictions must have dtype uint8")
    if predictions.ndim != 2:
        raise ValueError("decoder predictions must be a 2D array")
    if predictions.shape[0] != shots:
        raise ValueError("decoder predictions must have one row per surviving shot")
    if predictions.shape[1] < observable_bytes:
        raise ValueError("decoder predictions have too few observable bytes")
    if predictions.shape[1] > observable_bytes + 1:
        raise ValueError("decoder predictions have too many observable bytes")
    return predictions


def _count_error_combinations(
    differences: PackedArray,
    fail_mask: npt.NDArray[np.bool_],
    *,
    num_observables: int,
    counts: collections.Counter[str],
) -> None:
    failed_differences = differences[fail_mask]
    if failed_differences.shape[0] == 0:
        return
    unique, frequencies = np.unique(failed_differences, axis=0, return_counts=True)
    for packed, frequency in zip(unique, frequencies, strict=True):
        mistakes = np.unpackbits(packed, count=num_observables, bitorder="little")
        key = "obs_mistake_mask=" + "".join("_E"[bit] for bit in mistakes)
        counts[key] += int(frequency)


def _classify_discards_and_errors(
    *,
    actual_observables: PackedArray,
    predictions: PackedArray,
    postselected_observables_mask: PackedArray | None,
    count_observable_error_combos: bool,
    num_observables: int,
    custom_counts: collections.Counter[str],
) -> tuple[int, int]:
    discards = 0
    observable_bytes = actual_observables.shape[1]

    if predictions.shape[1] == observable_bytes + 1:
        keep = predictions[:, -1] == 0
        discards += int(np.count_nonzero(~keep))
        actual_observables = actual_observables[keep]
        predictions = predictions[keep, :-1]

    differences = actual_observables ^ predictions
    if postselected_observables_mask is not None:
        keep = ~np.any(differences & postselected_observables_mask, axis=1)
        discards += int(np.count_nonzero(~keep))
        differences = differences[keep]

    fail_mask = np.any(differences, axis=1)
    if count_observable_error_combos:
        _count_error_combinations(
            differences,
            fail_mask,
            num_observables=num_observables,
            counts=custom_counts,
        )
    return discards, int(np.count_nonzero(fail_mask))


class _CompiledClifftSampler(_sinter.CompiledSampler):
    def __init__(
        self,
        *,
        sampler: CompiledDetectorSampler,
        compiled_decoder: _sinter.CompiledDecoder,
        postselection_mask: PackedArray | None,
        postselected_observables_mask: PackedArray | None,
        native_postselection: bool,
        count_observable_error_combos: bool,
        count_detection_events: bool,
    ):
        self._sampler = sampler
        self._compiled_decoder = compiled_decoder
        self._postselection_mask = postselection_mask
        self._postselected_observables_mask = postselected_observables_mask
        self._native_postselection = native_postselection
        self._count_observable_error_combos = count_observable_error_combos
        self._count_detection_events = count_detection_events

    def sample(self, suggested_shots: int) -> _sinter.AnonTaskStats:
        shots = _shot_count(suggested_shots)
        if shots == 0:
            raise ValueError("a Sinter compiled sampler must take at least one shot")

        start = time.monotonic()
        if self._native_postselection:
            detectors, actual_observables, surviving_shots = self._sampler._sample_postselected(
                shots
            )
            detector_discards = shots - surviving_shots
        else:
            sampled = self._sampler.sample(
                shots,
                separate_observables=True,
                bit_packed=True,
            )
            detectors, actual_observables = cast(tuple[PackedArray, PackedArray], sampled)
            detector_discards = 0

        custom_counts: collections.Counter[str] = collections.Counter()
        if self._count_detection_events:
            custom_counts["detectors_checked"] = self._sampler.num_detectors * shots
            unpacked = np.unpackbits(
                detectors,
                axis=1,
                count=self._sampler.num_detectors,
                bitorder="little",
            )
            custom_counts["detection_events"] = int(np.count_nonzero(unpacked))

        if self._postselection_mask is not None and not self._native_postselection:
            keep = ~np.any(detectors & self._postselection_mask, axis=1)
            detector_discards = int(np.count_nonzero(~keep))
            detectors = detectors[keep]
            actual_observables = actual_observables[keep]

        if detectors.shape[0] == 0:
            predictions = np.empty_like(actual_observables)
        else:
            predictions = _validate_predictions(
                self._compiled_decoder.decode_shots_bit_packed(
                    bit_packed_detection_event_data=detectors
                ),
                shots=detectors.shape[0],
                observable_bytes=actual_observables.shape[1],
            )
        observable_discards, errors = _classify_discards_and_errors(
            actual_observables=actual_observables,
            predictions=predictions,
            postselected_observables_mask=self._postselected_observables_mask,
            count_observable_error_combos=self._count_observable_error_combos,
            num_observables=self._sampler.num_observables,
            custom_counts=custom_counts,
        )
        elapsed = time.monotonic() - start
        return _sinter.AnonTaskStats(
            shots=shots,
            errors=errors,
            discards=detector_discards + observable_discards,
            seconds=elapsed,
            custom_counts=custom_counts,
        )


class ClifftSampler(_sinter.Sampler):
    """A Sinter sampler that simulates with Clifft and decodes packed batches.

    The object is pickle-safe and contains only configuration. Each Sinter
    worker compiles its own retained native sampler and decoder for the task.

    Args:
        decoder: A ``sinter.Decoder`` or the built-in name ``"pymatching"`` or
            ``"pymatching-correlated"``.
        threads: Native threads used by each Sinter worker. The default of one
            avoids multiplying Sinter processes by Clifft worker threads.
        batch_size: Native packed lane capacity or ``"auto"``.
        count_observable_error_combos: Populate Sinter observable mistake
            combination counts.
        count_detection_events: Populate Sinter detector counts. This requires
            complete detector rows, so detector postselection occurs after
            native sampling instead of terminating rejected shots early.
        tmp_dir: Directory for legacy Sinter decoders that only implement
            ``decode_via_files``.
    """

    def __init__(
        self,
        *,
        decoder: DecoderOption = "pymatching",
        threads: ThreadOption = 1,
        batch_size: BatchOption = "auto",
        count_observable_error_combos: bool = False,
        count_detection_events: bool = False,
        tmp_dir: str | pathlib.Path | None = None,
    ):
        _resolve_decoder(decoder)
        self.decoder = decoder
        self.threads = threads
        self.batch_size = batch_size
        self.count_observable_error_combos = bool(count_observable_error_combos)
        self.count_detection_events = bool(count_detection_events)
        self.tmp_dir = None if tmp_dir is None else pathlib.Path(tmp_dir)

    def compiled_sampler_for_task(self, task: _sinter.Task) -> _sinter.CompiledSampler:
        if task.circuit is None:
            raise ValueError("ClifftSampler requires task.circuit to be loaded")
        dem = _detector_error_model(task)

        decoder = _resolve_decoder(self.decoder)
        compiled_decoder = _compile_decoder(
            decoder,
            dem=dem,
            decoder_name=self.decoder if isinstance(self.decoder, str) else task.decoder,
            tmp_dir=self.tmp_dir,
        )

        native_postselection = (
            task.postselection_mask is not None and not self.count_detection_events
        )
        if native_postselection:
            unpacked_mask = np.unpackbits(
                task.postselection_mask,
                count=task.circuit.num_detectors,
                bitorder="little",
            )
            sampler = _compile_postselected_detector_sampler(
                str(task.circuit),
                unpacked_mask.astype(np.uint8, copy=False).tolist(),
                threads=self.threads,
                batch_size=self.batch_size,
            )
        else:
            sampler = compile_detector_sampler(
                task.circuit,
                threads=self.threads,
                batch_size=self.batch_size,
            )

        return _CompiledClifftSampler(
            sampler=sampler,
            compiled_decoder=compiled_decoder,
            postselection_mask=task.postselection_mask,
            postselected_observables_mask=task.postselected_observables_mask,
            native_postselection=native_postselection,
            count_observable_error_combos=self.count_observable_error_combos,
            count_detection_events=self.count_detection_events,
        )

    def __repr__(self) -> str:
        return (
            "ClifftSampler("
            f"decoder={self.decoder!r}, threads={self.threads!r}, "
            f"batch_size={self.batch_size!r})"
        )


__all__ = ["ClifftSampler"]
