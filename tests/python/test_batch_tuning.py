"""Calibration is extra work before one complete production sampling call."""

import warnings
from typing import Any, cast

import numpy as np
import pytest
import stim
from utils_conformance import assert_joint_distribution, unitary_reference

import clifft


@pytest.fixture(
    params=[
        ("sample", {}),
        ("sample_k", {"k": 1}),
        ("sample_survivors", {"keep_records": False}),
        ("sample_survivors", {"keep_records": True}),
        ("sample_k_survivors", {"k": 1, "keep_records": False}),
        ("sample_k_survivors", {"k": 1, "keep_records": True}),
    ],
    ids=[
        "rows",
        "fixed-fault-rows",
        "counts",
        "survivors",
        "fixed-fault-counts",
        "fixed-fault-survivors",
    ],
)
def sampling_call(request: pytest.FixtureRequest) -> Any:
    name, options = request.param
    survivor = "survivors" in name
    program = clifft.compile(
        "X_ERROR(0.25) 0 1\nH 2\nT 2\nH 2\nEXP_VAL Z2\nM 0 1 2\n"
        "DETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-2]",
        postselection_mask=[1] if survivor else None,
    )

    def call(shots: int, **kwargs: Any) -> clifft.SampleResult:
        return cast(clifft.SampleResult, getattr(clifft, name)(program, shots, **options, **kwargs))

    return call


@pytest.mark.parametrize("threads", [1, 2])
def test_tuning_runs_requested_shots_and_reports_reusable_configuration(
    sampling_call: Any, threads: int
) -> None:
    shots = 4097
    result = sampling_call(
        shots, seed=82, threads=threads, batch_size="tune", tuning_budget_seconds=0.03
    )
    report = result.batch_tuning
    assert isinstance(report, clifft.BatchTuningReport)
    assert report.batch_size in {1, 64, 256, 1024, 2048}
    assert report.intra_shot_workers == 1
    assert 1 <= report.shot_workers <= threads
    assert report.elapsed_seconds >= 0
    assert report.trial_shots == sum(t.shots + t.warmup_shots for t in report.trials)
    for trial in report.trials:
        assert isinstance(trial, clifft.BatchTuningTrial)
        assert trial.shots_per_second >= 0
        assert trial.setup_seconds >= 0
        assert trial.warmup_seconds >= 0
        if trial.used_warmup:
            assert trial.shots == 0
            assert trial.shots_per_second == pytest.approx(
                trial.warmup_shots / trial.warmup_seconds
            )

    pinned = sampling_call(shots, seed=82, threads=threads, batch_size=report.batch_size)
    assert pinned.batch_tuning is None
    for field in ("measurements", "detectors", "observables", "exp_vals", "observable_ones"):
        np.testing.assert_array_equal(getattr(result, field), getattr(pinned, field))
    for field in ("total_shots", "passed_shots", "discards", "logical_errors"):
        assert getattr(result, field) == getattr(pinned, field)
    if result.total_shots is None:
        assert result.measurements.shape == (shots, 3)
    else:
        assert result.total_shots == shots
        assert result.passed_shots + result.discards == shots


@pytest.mark.parametrize("shots", [0, 1, 257])
def test_zero_budget_uses_automatic_policy_without_trials(sampling_call: Any, shots: int) -> None:
    tuned = sampling_call(shots, seed=83, batch_size="tune", tuning_budget_seconds=0)
    default = sampling_call(shots, seed=83)
    report = tuned.batch_tuning
    assert report.trials == []
    assert report.trial_shots == 0
    assert report.stop_reason == ("zero_shots" if shots == 0 else "zero_budget")
    np.testing.assert_array_equal(tuned.measurements, default.measurements)
    assert tuned.passed_shots == default.passed_shots


@pytest.mark.parametrize("budget", [-1, float("inf"), float("-inf"), float("nan")])
def test_invalid_budget_is_rejected_before_sampling(sampling_call: Any, budget: float) -> None:
    with pytest.raises(ValueError, match="tuning_budget_seconds"):
        sampling_call(0, batch_size="tune", tuning_budget_seconds=budget)


@pytest.mark.parametrize("batch_size", ["auto", 1, 64])
def test_budget_requires_tuning_mode(sampling_call: Any, batch_size: str | int) -> None:
    with pytest.raises(ValueError, match="requires batch_size='tune'"):
        sampling_call(1, batch_size=batch_size, tuning_budget_seconds=0.1)


@pytest.mark.parametrize("batch_size", ["Tune", 0, -1])
def test_invalid_batch_size_lists_tuning_as_an_option(
    sampling_call: Any, batch_size: str | int
) -> None:
    with pytest.raises(ValueError, match="'auto'.*'tune'"):
        sampling_call(1, batch_size=batch_size)


def test_insufficient_tuning_warns_and_still_returns_complete_results(sampling_call: Any) -> None:
    with pytest.warns(RuntimeWarning, match="using auto.*Increase tuning_budget_seconds"):
        result = sampling_call(65, seed=87, batch_size="tune", tuning_budget_seconds=1e-300)
    baseline = sampling_call(65, seed=87)
    report = result.batch_tuning
    assert not report.sufficient_measurements
    assert report.stop_reason == "budget_exhausted"
    assert report.batch_size == report.baseline_batch_size
    assert "sufficient_measurements=False" in repr(report)
    for field in ("measurements", "detectors", "observables", "exp_vals", "observable_ones"):
        np.testing.assert_array_equal(getattr(result, field), getattr(baseline, field))
    for field in ("total_shots", "passed_shots", "discards", "logical_errors"):
        assert getattr(result, field) == getattr(baseline, field)


def test_wide_circuit_skips_calibration_without_a_budget_warning() -> None:
    qubits = " ".join(map(str, range(14)))
    program = clifft.compile(f"H {qubits}\nT {qubits}\nM 0", hir_passes=None)
    assert program.peak_active_width == 14
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = clifft.sample(program, 65, batch_size="tune")
    report = result.batch_tuning
    assert report.stop_reason == "single_candidate"
    assert report.trial_shots == 0
    assert report.trials == []
    assert report.batch_size == 1
    assert result.measurements.shape == (65, 1)


def test_tuning_uses_default_budget_and_entropy() -> None:
    program = clifft.compile("X 0\nM 0")
    result = clifft.sample(program, 65, batch_size="tune")
    np.testing.assert_array_equal(result.measurements, np.ones((65, 1), dtype=np.uint8))
    assert result.batch_tuning.batch_size in {1, 64, 65}


def test_tuning_preserves_explicit_intra_shot_layout() -> None:
    program = clifft.compile("H 0\nT 0\nH 0\nM 0", hir_passes=None)
    try:
        result = clifft.sample(
            program,
            65,
            thread_layout=(1, 2),
            intra_shot_min_active_width=0,
            batch_size="tune",
            tuning_budget_seconds=0.01,
        )
    except ValueError as error:
        if str(error) == "thread_layout intra-shot workers require an OpenMP-enabled build":
            pytest.skip(str(error))
        raise
    report = result.batch_tuning
    assert report.batch_size == 1
    assert report.intra_shot_workers == 2
    assert report.stop_reason == "single_candidate"
    assert report.trial_shots == 0
    assert result.measurements.shape == (65, 1)


def test_tuned_sampling_matches_aer_joint_probabilities() -> None:
    source = "H 0 1\nT 0\nCX 0 1\nR_Y(0.17) 1\nH 0"
    expected = np.abs(unitary_reference(source)) ** 2
    result = clifft.sample(
        clifft.compile(source + "\nM 0 1"),
        20_000,
        seed=84,
        threads=2,
        batch_size="tune",
        tuning_budget_seconds=0.03,
    )
    assert_joint_distribution(result.measurements, expected)


def test_tuned_noise_matches_stim_joint_distribution() -> None:
    circuit = stim.Circuit("H 0\nCX 0 1\nX_ERROR(0.15) 1\nM 0 1")
    shots = 30_000
    reference = circuit.compile_sampler(seed=85).sample(shots).astype(np.uint8)
    outcomes = reference[:, 0] + 2 * reference[:, 1]
    expected = np.bincount(outcomes, minlength=4) / shots
    result = clifft.sample(
        clifft.compile(str(circuit)),
        shots,
        seed=86,
        threads=2,
        batch_size="tune",
        tuning_budget_seconds=0.03,
    )
    assert_joint_distribution(result.measurements, expected)
