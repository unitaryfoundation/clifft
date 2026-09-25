"""Public Sinter integration and independent count/normalization checks."""

import pickle
from typing import Any

import numpy as np
import pytest
import sinter
import stim
from conftest import binomial_tolerance, cross_binomial_tolerance
from utils_conformance import CpuSamplingMode

from clifft.sinter import PerfectionistSampler


def all_detector_task(source: str) -> sinter.Task:
    circuit = stim.Circuit(source)
    return sinter.Task(
        circuit=circuit,
        postselection_mask=np.packbits(
            np.ones(circuit.num_detectors, dtype=np.uint8), bitorder="little"
        ),
    )


@pytest.fixture
def sampler(sampling_mode: CpuSamplingMode) -> PerfectionistSampler:
    if sampling_mode.threads != 1 or sampling_mode.thread_layout is not None:
        pytest.skip("The Sinter adapter deliberately uses one native thread per worker")
    return PerfectionistSampler(batch_size=sampling_mode.batch_size)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "discards", "errors"),
    [
        ("R 0\nM 0\nDETECTOR rec[-1]", 0, 0),
        ("R 0\nX_ERROR(1) 0\nM 0\nDETECTOR rec[-1]", 131, 0),
        ("R 0\nX_ERROR(1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0, 131),
        (
            "R 0\nX_ERROR(1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(8) rec[-1]",
            0,
            131,
        ),
        (
            "R 0\nX_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
            131,
            0,
        ),
        ("R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]", 0, 0),
        ("", 0, 0),
    ],
)
def test_exact_counts_match_stim(
    sampler: PerfectionistSampler, source: str, discards: int, errors: int
) -> None:
    task = all_detector_task(source)
    compiled = sampler.compiled_sampler_for_task(task)
    assert compiled.sample(0) == sinter.AnonTaskStats()
    dets, obs = task.circuit.compile_detector_sampler(seed=10).sample(
        131, separate_observables=True
    )
    rejected = np.any(dets, axis=1)
    assert np.count_nonzero(rejected) == discards
    assert np.count_nonzero(np.any(obs, axis=1) & ~rejected) == errors
    for _ in range(2):
        stats = compiled.sample(131)
        assert (stats.shots, stats.discards, stats.errors) == (131, discards, errors)
        assert stats.seconds >= 0
        assert not stats.custom_counts


def test_stochastic_counts_and_repeated_calls(sampler: PerfectionistSampler) -> None:
    # Shared error events correlate rejection with both observable flips.
    task = all_detector_task(
        "R 0 1 2\nX_ERROR(0.1) 0\nX_ERROR(0.2) 1\nX_ERROR(0.3) 2\n"
        "CORRELATED_ERROR(0.05) X0 X1 X2\nM 0 1 2\nDETECTOR rec[-3]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(8) rec[-1]"
    )
    compiled = sampler.compiled_sampler_for_task(task)
    batches = [compiled.sample(20_000) for _ in range(5)]
    assert len({(r.discards, r.errors) for r in batches}) > 1
    total = sum(batches, start=sinter.AnonTaskStats())
    assert total.shots == 100_000
    expected_discard = 0.95 * 0.1 + 0.05 * 0.9
    expected_error = 0.95 * 0.9 * (1 - 0.8 * 0.7) + 0.05 * 0.1 * (1 - 0.2 * 0.3)
    for observed, expected in [(total.discards, expected_discard), (total.errors, expected_error)]:
        assert abs(observed / total.shots - expected) < binomial_tolerance(
            expected, total.shots, sigma=6
        )
    dets, obs = task.circuit.compile_detector_sampler(seed=719).sample(
        total.shots, separate_observables=True
    )
    rejected = np.any(dets, axis=1)
    reference = [np.count_nonzero(rejected), np.count_nonzero(np.any(obs, axis=1) & ~rejected)]
    for a, b in zip([total.discards, total.errors], reference):
        pooled = (a + b) / (2 * total.shots)
        assert abs(a - b) / total.shots < cross_binomial_tolerance(pooled, total.shots, sigma=6)


@pytest.mark.parametrize("last_byte", [1, 255])
def test_mask_padding_does_not_select_extra_detectors(last_byte: int) -> None:
    task = all_detector_task("R 0\nX_ERROR(1) 0\nM 0\n" + "DETECTOR rec[-1]\n" * 9)
    task.postselection_mask = np.array([255, last_byte], dtype=np.uint8)
    assert (
        PerfectionistSampler(batch_size=65).compiled_sampler_for_task(task).sample(131).discards
        == 131
    )


@pytest.mark.parametrize(
    "mask",
    [
        None,
        np.array([0], dtype=np.uint8),
        np.array([], dtype=np.uint8),
        np.array([1], dtype=np.int64),
        np.array([[1]], dtype=np.uint8),
    ],
)
def test_reject_missing_partial_and_malformed_masks(mask: Any) -> None:
    task = all_detector_task("R 0\nM 0\nDETECTOR rec[-1]")
    task.postselection_mask = mask
    with pytest.raises(ValueError, match="postselection|postselection_mask"):
        PerfectionistSampler().compiled_sampler_for_task(task)


def test_reject_partial_mask_across_bytes() -> None:
    task = all_detector_task("R 0\nM 0\n" + "DETECTOR rec[-1]\n" * 9)
    task.postselection_mask = np.array([255, 0], dtype=np.uint8)
    with pytest.raises(ValueError, match="all-detector postselection"):
        PerfectionistSampler().compiled_sampler_for_task(task)


def test_reject_observable_postselection_and_unresolved_circuit() -> None:
    task = all_detector_task("R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    task.postselected_observables_mask = np.array([1], dtype=np.uint8)
    with pytest.raises(ValueError, match="observable postselection"):
        PerfectionistSampler().compiled_sampler_for_task(task)
    with pytest.raises(ValueError, match="resolved task.circuit"):
        PerfectionistSampler().compiled_sampler_for_task(
            sinter.Task(circuit_path="unresolved.stim")
        )


def test_no_detectors_accepts_absent_mask() -> None:
    task = sinter.Task(circuit=stim.Circuit("R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"))
    stats = PerfectionistSampler().compiled_sampler_for_task(task).sample(5)
    assert (stats.shots, stats.errors, stats.discards) == (5, 0, 0)


@pytest.mark.parametrize("value", [-1, 0, 2**32, True, 1.5, "bad"])
def test_invalid_capacity(value: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        PerfectionistSampler(batch_size=value)


@pytest.mark.parametrize("value", [-1, 2**32, True, 1.5])
def test_invalid_shot_count(value: Any) -> None:
    sampler = PerfectionistSampler().compiled_sampler_for_task(all_detector_task(""))
    with pytest.raises((ValueError, TypeError)):
        sampler.sample(value)


def test_pickling_and_multiprocess_collection() -> None:
    sampler = pickle.loads(pickle.dumps(PerfectionistSampler(batch_size=65)))
    tasks = [
        all_detector_task("R 0\nX_ERROR(1) 0\nM 0\nDETECTOR rec[-1]"),
        all_detector_task("R 0\nX_ERROR(1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"),
    ]
    stats = sinter.collect(
        num_workers=2,
        tasks=tasks,
        decoders=["clifft-perfectionist"],
        custom_decoders={"clifft-perfectionist": sampler},
        max_shots=100_000,
        max_batch_size=4096,
    )
    assert len(stats) == 2
    assert sorted((r.shots, r.discards, r.errors) for r in stats) == [
        (100_000, 0, 100_000),
        (100_000, 100_000, 0),
    ]


@pytest.mark.parametrize("option", ["count_detection_events", "count_observable_error_combos"])
def test_sinter_rejects_extra_counters(option: str) -> None:
    with pytest.raises((ValueError, RuntimeError), match=option):
        sinter.collect(
            num_workers=1,
            tasks=[all_detector_task("R 0\nM 0\nDETECTOR rec[-1]")],
            decoders=["clifft-perfectionist"],
            custom_decoders={"clifft-perfectionist": PerfectionistSampler()},
            max_shots=10,
            **{option: True},
        )
