"""Calibration measures complete fixed samples without consuming production randomness."""

import warnings

import numpy as np
import pytest
import stim
from utils_conformance import assert_joint_distribution, unitary_reference

import clifft
from clifft import noncomp


@pytest.fixture(params=["prepared", "convenience"])
def sampling_call(request):
    def call(text, model, *, final_status=False, **kwargs):
        if request.param == "prepared":
            execution = kwargs.pop("execution", "auto")
            return noncomp.compile(
                text, model, final_status=final_status, execution=execution
            ).sample(**kwargs)
        return noncomp.sample(text, model, final_status=final_status, **kwargs)

    return call


def model():
    return noncomp.Model(
        classifier=noncomp.Classifier(
            [[0.9, 0.1, 0.5, 0.5, 0.2], [0.1, 0.9, 0.5, 0.5, 0.3], [0, 0, 0, 0, 0.5]]
        )
    )


def same_rows(left, right):
    for field in ("measurements", "heralds", "detectors", "observables", "final_status"):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field), err_msg=field)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("final_status", [False, True])
@pytest.mark.parametrize("threads", [1, 3])
def test_tuning_replays_all_production_outputs_at_the_selected_capacity(
    sampling_call, dynamic, final_status, threads
):
    text = (
        f"H 0\nT 0\nCX 0 1\nLOSS({0.3 if dynamic else 0}) 1\n"
        "DEPOLARIZE1(0.2) 0\nMR !0\nREADOUT_NOISE(0.2,0.4) rec[-1]\n"
        "CX rec[-1] 2\nM 1 2\nHERALD_LOSS_EVENT 1\n"
        "DETECTOR rec[-4] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-1]"
    )
    options = dict(
        shots=4097, seed=82, threads=threads, final_status=final_status, collect_stats=True
    )
    result = sampling_call(text, model(), **options, batch_size="tune", tuning_budget_seconds=0.1)
    report = result.batch_tuning
    assert isinstance(report, clifft.BatchTuningReport)
    assert report.trial_shots > 0
    assert report.trial_shots == sum(t.shots + t.warmup_shots for t in report.trials)
    assert report.batch_size == result.stats.batch_size
    assert report.batch_size in {1, 64, 256, 1024, 2048}
    assert result.stats.fixed_shots == 4097
    assert result.stats.scalar_shots + result.stats.packed_shots == 4097
    assert result.stats.packed_classical_shots == (result.stats.packed_shots if dynamic else 0)
    assert result.measurements.shape == result.heralds.shape == (4097, 4)
    assert result.stats.sampling_seconds >= report.elapsed_seconds
    pinned = sampling_call(text, model(), **options, batch_size=report.batch_size)
    assert pinned.batch_tuning is None
    same_rows(result, pinned)
    options["threads"] = 1
    same_rows(result, sampling_call(text, model(), **options, batch_size=report.batch_size))


@pytest.mark.parametrize("shots", [0, 1, 65])
def test_zero_budget_uses_auto_without_calibration(sampling_call, shots):
    text = "LOSS(0.4) 0\nM 0"
    tuned = sampling_call(
        text, model(), shots=shots, seed=83, batch_size="tune", tuning_budget_seconds=0
    )
    baseline = sampling_call(text, model(), shots=shots, seed=83)
    assert tuned.stats is None
    assert tuned.batch_tuning.trial_shots == 0
    assert tuned.batch_tuning.trials == []
    assert tuned.batch_tuning.stop_reason == ("zero_shots" if shots == 0 else "zero_budget")
    same_rows(tuned, baseline)


@pytest.mark.parametrize("budget", [-1, float("inf"), float("nan")])
def test_invalid_tuning_budget_is_rejected_for_empty_calls(sampling_call, budget):
    with pytest.raises(ValueError, match="tuning_budget_seconds"):
        sampling_call("M 0", model(), shots=0, batch_size="tune", tuning_budget_seconds=budget)


@pytest.mark.parametrize("batch_size", ["auto", 1, 64])
def test_budget_requires_tuning_mode(sampling_call, batch_size):
    with pytest.raises(ValueError, match="requires batch_size='tune'"):
        sampling_call("M 0", model(), shots=0, batch_size=batch_size, tuning_budget_seconds=0.1)


@pytest.mark.parametrize("shots", [0, 65])
@pytest.mark.parametrize("execution", ["auto", "trajectory"])
def test_tuning_rejects_trajectory_programs_before_sampling(sampling_call, shots, execution):
    with pytest.raises(ValueError, match="selected strategy is trajectory"):
        sampling_call(
            "LOSS(1) 0\nCX 0 1\nM 1",
            model(),
            shots=shots,
            execution=execution,
            batch_size="tune",
            tuning_budget_seconds=0,
        )


def test_inconclusive_tuning_reports_and_replays_auto(sampling_call):
    with pytest.warns(RuntimeWarning, match="using auto"):
        result = sampling_call(
            "LOSS(0.4) 0\nM 0",
            model(),
            shots=65,
            seed=87,
            batch_size="tune",
            tuning_budget_seconds=1e-300,
        )
    assert result.batch_tuning.stop_reason == "budget_exhausted"
    assert not result.batch_tuning.sufficient_measurements
    same_rows(result, sampling_call("LOSS(0.4) 0\nM 0", model(), shots=65, seed=87))


def test_wide_plan_with_no_packed_candidate_skips_calibration():
    qubits = " ".join(map(str, range(14)))
    program = noncomp.compile(f"H {qubits}\nT {qubits}\nLOSS(0.1) 14\nM 0", model())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = program.sample(65, seed=9, batch_size="tune")
    assert result.batch_tuning is not None
    assert result.batch_tuning.stop_reason == "single_candidate"
    assert result.batch_tuning.trial_shots == 0
    assert result.batch_tuning.batch_size == 1
    same_rows(result, program.sample(65, seed=9))


def test_tuning_uses_default_budget_without_mutating_preparation():
    program = noncomp.compile("LOSS(0.4) 0\nM 0", model())
    explanation = program.explain()
    result = program.sample(65, batch_size="tune")
    assert result.measurements.shape == (65, 1)
    assert result.batch_tuning is not None
    assert result.batch_tuning.trial_shots > 0
    assert result.batch_tuning.batch_size in {1, 64, 65}
    assert result.stats is None
    assert program.explain() == explanation


def test_tuned_non_clifford_loss_matches_aer_and_independent_classical_status():
    quantum = "H 0\nT 0\nCX 0 1\nR_Y(0.17) 1\nH 0"
    expected = np.abs(unitary_reference(quantum)) ** 2
    text = quantum + "\nLOSS(0.3) 2\nHERALD_LOSS_EVENT 2\nM 0 1"
    program = noncomp.compile(
        text,
        noncomp.Model(
            classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]])
        ),
    )
    result = program.sample(20000, seed=84, batch_size="tune", tuning_budget_seconds=0.03)
    assert_joint_distribution(result.measurements, np.repeat(expected, 2) * np.tile([0.7, 0.3], 4))


def test_tuned_clifford_partner_noise_matches_stim():
    effect = noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25))
    program = noncomp.compile(
        "H 1\nCX 1 2\nLEAKAGE(1) 0\nCX 0 1\nH 1 2\nM 1 2",
        noncomp.Model(
            classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]]),
            gate_partner_effects={"leaked": effect},
        ),
    )
    result = program.sample(20000, seed=85, batch_size="tune", tuning_budget_seconds=0.03)
    reference = (
        stim.Circuit("H 1\nCX 1 2\nDEPOLARIZE1(0.75) 1\nH 1 2\nM 1 2")
        .compile_sampler(seed=86)
        .sample(20000)
    )
    expected = np.bincount(reference[:, 0] + 2 * reference[:, 1], minlength=4) / 20000
    assert_joint_distribution(result.measurements, expected)
