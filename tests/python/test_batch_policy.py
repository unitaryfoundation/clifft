"""Automatic batching preserves replay with a fixed execution configuration."""

from typing import Any

import numpy as np
import pytest

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
    options = {"k": 1} if fixed_faults else {}
    for threads in (1, 4):
        capacity = (shots // threads if width == 0 else 256) if width < 5 else 1
        explicit = sample(
            program,
            shots,
            seed=927,
            threads=1,
            batch_size=capacity,
            keep_records=keep_records,
            **options,
        )
        assert 0 < explicit.passed_shots < shots
        assert explicit.logical_errors > 0
        automatic = sample(
            program,
            shots,
            seed=927,
            threads=threads,
            keep_records=keep_records,
            **options,
        )
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
        (32, 1),
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
