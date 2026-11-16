"""Automatic batching preserves replay with a fixed execution configuration."""

from typing import Any

import numpy as np
import pytest
import stim

import clifft


@pytest.mark.parametrize("width", range(6))
@pytest.mark.parametrize("fixed_faults", [False, True])
@pytest.mark.parametrize("keep_records", [False, True])
def test_postselected_auto_replays_its_explicit_capacity(
    width: int, fixed_faults: bool, keep_records: bool
) -> None:
    source = (
        "X_ERROR(0.25) 6\nX_ERROR(0.375) 7\nM 6 7\n"
        "DETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    if width:
        targets = " ".join(map(str, range(width)))
        source += f"H {targets}\nT {targets}\nH {targets}\nEXP_VAL Z0\nM {targets}"
    program = clifft.compile(source, postselection_mask=[1], hir_passes=None)
    assert program.peak_active_width == width
    shots = 1025
    sample: Any = clifft.sample_k_survivors if fixed_faults else clifft.sample_survivors
    options: dict[str, Any] = {"k": 1} if fixed_faults else {}
    # Ordinary width-zero plans batch every requested shot and select their
    # noise preparation; the other plans keep the narrow postselected limit.
    if width == 0 and not fixed_faults:
        capacity = shots
    else:
        capacity = 256 if width < 5 else 1
    for threads in (1, 4):
        automatic = sample(
            program,
            shots,
            seed=927,
            threads=threads,
            keep_records=keep_records,
            **options,
        )
        assert automatic.execution.batch_size == capacity
        replay = dict(options)
        if not fixed_faults and automatic.execution.noise_preparation is not None:
            replay["noise_preparation"] = automatic.execution.noise_preparation
        explicit = sample(
            program,
            shots,
            seed=927,
            threads=1,
            batch_size=capacity,
            keep_records=keep_records,
            **replay,
        )
        assert 0 < explicit.passed_shots < shots
        assert explicit.logical_errors > 0
        for field in ("total_shots", "passed_shots", "discards", "logical_errors"):
            assert getattr(automatic, field) == getattr(explicit, field)
        for field in ("measurements", "detectors", "observables", "exp_vals", "observable_ones"):
            np.testing.assert_array_equal(getattr(automatic, field), getattr(explicit, field))
        rows = automatic.passed_shots if keep_records else 0
        assert automatic.measurements.shape == (rows, program.num_measurements)
        assert not np.any(automatic.detectors)
        if keep_records:
            np.testing.assert_array_equal(
                automatic.observable_ones, automatic.observables.sum(axis=0)
            )


def test_width_zero_auto_selects_noise_preparation_and_agrees_with_stim() -> None:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    program = clifft.compile(str(circuit))
    assert program.peak_active_width == 0

    # Every noise site fires with probability 0.01, so a 64-shot batch expects
    # fewer faults than there are sites and a 2048-shot batch expects more.
    brief = clifft.sample(program, 64, seed=5)
    assert brief.execution.noise_preparation == clifft.NoisePreparation.SPARSE
    numeric = clifft.sample(program, 4096, seed=5, batch_size=2048)
    assert numeric.execution.noise_preparation == clifft.NoisePreparation.DENSE
    shots = 200_000
    bulk = clifft.sample(program, shots, seed=5, keep_records="syndromes")
    assert bulk.execution.batch_size == 2048
    assert bulk.execution.noise_preparation == clifft.NoisePreparation.SITE_SPARSE

    def agree(sampled: Any, reference: Any, samples: int) -> bool:
        rate = (sampled + reference) / 2
        return bool(np.all(np.abs(sampled - reference) <= 6 * np.sqrt(2 * rate / samples)))

    detectors, observables = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True
    )
    assert agree(bulk.detectors.mean(axis=0), detectors.mean(axis=0), shots)
    assert agree(bulk.observables.mean(axis=0), observables.mean(axis=0), shots)

    # Postselecting every detector keeps the shots Stim reports without events.
    clean = ~detectors.any(axis=1)
    postselected = clifft.compile(str(circuit), postselection_mask=[1] * circuit.num_detectors)
    survivors = clifft.sample_survivors(postselected, shots, seed=7)
    assert survivors.execution.batch_size == 2048
    assert survivors.execution.noise_preparation == clifft.NoisePreparation.SITE_SPARSE
    assert agree(survivors.passed_shots / shots, clean.mean(), shots)
    assert agree(
        survivors.logical_errors / survivors.passed_shots,
        observables[clean].any(axis=1).mean(),
        min(survivors.passed_shots, int(clean.sum())),
    )


@pytest.mark.parametrize("width", [6, 8])
def test_terminal_postselection_batches_classical_work_with_bounded_parallel_capacity(width):
    targets = " ".join(map(str, range(width)))
    source = f"H {targets}\nT {targets}\n"
    source += f"REPEAT 256 {{\nX_ERROR(0.01) {width}\nM {width}\n}}\n"
    source += f"H {targets}\nM {targets}\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    program = clifft.compile(source, hir_passes=None, postselection_mask=[1])
    assert program.peak_active_width == width
    for threads, capacity in (
        (1, 512 if width == 6 else 128),
        (4, 256 if width == 6 else 128),
        (32, 512 if width == 6 else 128),
    ):
        options = dict(seed=48, threads=threads, keep_records=True)
        automatic = clifft.sample_survivors(program, 1025, **options)
        reported = clifft.sample_survivors(
            program, 1025, **options, batch_size="tune", tuning_budget_seconds=0
        )
        assert reported.batch_tuning.baseline_batch_size == capacity
        explicit = clifft.sample_survivors(
            program, 1025, seed=48, threads=1, keep_records=True, batch_size=capacity
        )
        repeated = clifft.sample_survivors(program, 1025, **options)
        for result in (reported, explicit, repeated):
            for name in ("measurements", "detectors", "observables", "observable_ones"):
                np.testing.assert_array_equal(getattr(automatic, name), getattr(result, name))
        assert 0 < automatic.passed_shots < automatic.total_shots
        assert not np.any(automatic.detectors)
