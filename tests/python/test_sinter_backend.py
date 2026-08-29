"""Integration tests for the Clifft Sinter sampling backend."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pymatching  # noqa: F401
import pytest
import sinter
import stim

from clifft.sinter import ClifftSampler


class _ZeroCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, observable_bytes: int):
        self.observable_bytes = observable_bytes

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        return np.zeros(
            (bit_packed_detection_event_data.shape[0], self.observable_bytes),
            dtype=np.uint8,
        )


class _ZeroDecoder(sinter.Decoder):
    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _ZeroCompiledDecoder((dem.num_observables + 7) // 8)


class _FileZeroDecoder(sinter.Decoder):
    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: Path,
        dets_b8_in_path: Path,
        obs_predictions_b8_out_path: Path,
        tmp_dir: Path,
    ) -> None:
        del num_dets, dem_path, dets_b8_in_path, tmp_dir
        obs_predictions_b8_out_path.write_bytes(bytes(num_shots * ((num_obs + 7) // 8)))


def _task(
    circuit_text: str,
    *,
    postselection_mask: np.ndarray | None = None,
    postselected_observables_mask: np.ndarray | None = None,
) -> sinter.Task:
    circuit = stim.Circuit(circuit_text)
    return sinter.Task(
        circuit=circuit,
        decoder="clifft-test",
        detector_error_model=circuit.detector_error_model(),
        postselection_mask=postselection_mask,
        postselected_observables_mask=postselected_observables_mask,
    )


def test_sinter_sampler_is_pickle_safe_configuration() -> None:
    sampler = ClifftSampler(decoder=_ZeroDecoder(), threads=1, batch_size=65)
    restored = pickle.loads(pickle.dumps(sampler))
    assert isinstance(restored, ClifftSampler)
    assert restored.threads == 1
    assert restored.batch_size == 65


def test_sinter_sampler_uses_native_detector_postselection() -> None:
    task = _task(
        """
        X_ERROR(0.5) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """,
        postselection_mask=np.array([1], dtype=np.uint8),
    )
    compiled = ClifftSampler(decoder=_ZeroDecoder(), batch_size=65).compiled_sampler_for_task(task)
    stats = compiled.sample(1000)

    assert stats.shots == 1000
    assert 400 < stats.discards < 600
    assert stats.errors == 0


def test_sinter_sampler_supports_observable_postselection() -> None:
    task = _task(
        """
        X_ERROR(0.5) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        """,
        postselected_observables_mask=np.array([1], dtype=np.uint8),
    )
    stats = ClifftSampler(decoder=_ZeroDecoder()).compiled_sampler_for_task(task).sample(1000)

    assert stats.shots == 1000
    assert 400 < stats.discards < 600
    assert stats.errors == 0


def test_sinter_sampler_counts_complete_detector_rows_when_requested() -> None:
    task = _task(
        """
        X_ERROR(1) 0
        M 0
        DETECTOR rec[-1]
        """,
        postselection_mask=np.array([1], dtype=np.uint8),
    )
    compiled = ClifftSampler(
        decoder=_ZeroDecoder(),
        count_detection_events=True,
    ).compiled_sampler_for_task(task)
    stats = compiled.sample(4)

    assert stats.discards == 4
    assert stats.custom_counts["detectors_checked"] == 4
    assert stats.custom_counts["detection_events"] == 4


def test_sinter_sampler_counts_observable_error_combinations() -> None:
    task = _task(
        """
        X_ERROR(1) 0 1
        M 0 1
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )
    compiled = ClifftSampler(
        decoder=_ZeroDecoder(),
        count_observable_error_combos=True,
    ).compiled_sampler_for_task(task)
    stats = compiled.sample(5)

    assert stats.errors == 5
    assert stats.custom_counts["obs_mistake_mask=EE"] == 5


def test_sinter_sampler_integrates_with_pymatching() -> None:
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.01,
    )
    task = sinter.Task(
        circuit=circuit,
        decoder="clifft-pymatching",
    )
    stats = ClifftSampler().compiled_sampler_for_task(task).sample(256)

    assert stats.shots == 256
    assert stats.discards == 0
    assert 0 <= stats.errors <= stats.shots


def test_sinter_sampler_handles_all_shots_postselected_before_pymatching() -> None:
    task = _task(
        """
        X_ERROR(1) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """,
        postselection_mask=np.array([1], dtype=np.uint8),
    )
    stats = ClifftSampler().compiled_sampler_for_task(task).sample(4)
    assert stats.shots == stats.discards == 4
    assert stats.errors == 0


def test_sinter_sampler_supports_file_decoders(tmp_path: Path) -> None:
    task = _task("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    with pytest.raises(ValueError, match="tmp_dir"):
        ClifftSampler(decoder=_FileZeroDecoder()).compiled_sampler_for_task(task)

    compiled = ClifftSampler(
        decoder=_FileZeroDecoder(),
        tmp_dir=tmp_path,
    ).compiled_sampler_for_task(task)
    stats = compiled.sample(3)
    assert stats.shots == 3
    assert stats.errors == 0


def test_sinter_collect_uses_clifft_sampler_in_worker_processes() -> None:
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=2,
        distance=3,
        before_round_data_depolarization=0.01,
    )
    result = sinter.collect(
        num_workers=2,
        tasks=[sinter.Task(circuit=circuit, json_metadata={"backend": "clifft"})],
        decoders=["clifft-pymatching"],
        custom_decoders={"clifft-pymatching": ClifftSampler()},
        max_shots=64,
        start_batch_size=16,
        max_batch_size=16,
    )

    assert len(result) == 1
    assert result[0].shots >= 64
    assert result[0].json_metadata == {"backend": "clifft"}
