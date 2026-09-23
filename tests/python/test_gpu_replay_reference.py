"""Exercise the GPU tests' CPU reference and comparison without a device."""

import itertools
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import stim
from utils_gpu_replay import (
    PAULI_REPLAY_CIRCUIT,
    REPLAY_CASES,
    ReplayCase,
    assert_forced_record_probabilities,
    cpu_replay_branches,
)

import clifft
from clifft._clifft_core import _replay_record


@pytest.mark.parametrize("case", REPLAY_CASES, ids=lambda case: case.name)
def test_cpu_replay_reference_matches_analytic_branches(case: ReplayCase) -> None:
    program = clifft.compile(case.circuit)
    assert program.num_measurements == case.visible
    assert program.num_hidden_measurements == case.hidden
    assert program.peak_active_width >= case.min_active_width
    branches = cpu_replay_branches(program)
    assert len(branches) == len(case.probabilities)
    for branch, probability in zip(branches, case.probabilities, strict=True):
        assert branch.reachable == (probability > 0)
        if branch.reachable:
            assert branch.log_probability == pytest.approx(math.log(probability), abs=1e-12)
    assert sum(math.exp(b.log_probability) for b in branches if b.reachable) == pytest.approx(1)


def _analytic_sampler(case: ReplayCase) -> SimpleNamespace:
    # A known distribution exercises the actual GPU comparison helper on CPU
    # CI, including impossible branches and visible-only output arrays.
    probabilities = dict(
        zip(
            itertools.product((0, 1), repeat=case.visible + case.hidden),
            case.probabilities,
            strict=True,
        )
    )

    def replay_shot(record: list[int]) -> SimpleNamespace:
        probability = probabilities[tuple(record)]
        return SimpleNamespace(
            reachable=probability > 0,
            survived=True,
            log_probability=math.log(probability) if probability else 0,
            outputs=SimpleNamespace(
                measurements=np.asarray([record[: case.visible]], dtype=np.uint8)
            ),
        )

    return SimpleNamespace(
        program=SimpleNamespace(
            num_measurements=case.visible, num_records=case.visible + case.hidden
        ),
        replay_shot=replay_shot,
    )


@pytest.mark.parametrize("case", REPLAY_CASES, ids=lambda case: case.name)
def test_forced_record_comparison_accepts_analytic_backend_results(case: ReplayCase) -> None:
    assert_forced_record_probabilities(
        clifft.compile(case.circuit), cast(Any, _analytic_sampler(case)), absolute_tolerance=1e-12
    )


@pytest.mark.parametrize("fault", ["probability", "reachability", "hidden-measurements"])
def test_forced_record_comparison_rejects_incorrect_backend_results(fault: str) -> None:
    case = next(case for case in REPLAY_CASES if case.name == "deterministic-reset")
    sampler = _analytic_sampler(case)
    original_replay = sampler.replay_shot

    def replay_shot(record: list[int]) -> SimpleNamespace:
        result = original_replay(record)
        if fault == "probability":
            result.log_probability += 1e-3
        elif fault == "reachability":
            result.reachable = not result.reachable
        else:
            result.outputs.measurements = np.asarray([record], dtype=np.uint8)
        return cast(SimpleNamespace, result)

    sampler.replay_shot = replay_shot
    with pytest.raises(AssertionError):
        assert_forced_record_probabilities(
            clifft.compile(case.circuit), cast(Any, sampler), absolute_tolerance=1e-12
        )


@pytest.mark.parametrize("preparation", ["H", "X"])
def test_cpu_reset_reference_matches_stim_measure_and_reset(preparation: str) -> None:
    program = clifft.compile(f"{preparation} 0\nR 0\nH 0\nM 0")
    # MR exposes the hidden reset outcome as the first Stim measurement.
    rows = stim.Circuit(f"{preparation} 0\nMR 0\nH 0\nM 0").compile_sampler(seed=91).sample(20000)
    rows = rows[:, [1, 0]]
    for branch in cpu_replay_branches(program):
        observed = np.mean(np.all(rows == branch.record, axis=1))
        expected = math.exp(branch.log_probability) if branch.reachable else 0
        if expected == 0:
            assert observed == 0
        else:
            tolerance = 6 * math.sqrt(expected * (1 - expected) / len(rows))
            assert observed == pytest.approx(expected, abs=tolerance)


def test_cpu_multi_pauli_replay_reference_normalizes() -> None:
    program = clifft.compile(PAULI_REPLAY_CIRCUIT)
    assert program.num_measurements == 2
    assert program.num_hidden_measurements == 0
    branches = cpu_replay_branches(program)
    assert len(branches) == 4
    assert all(branch.reachable for branch in branches)
    assert sum(math.exp(branch.log_probability) for branch in branches) == pytest.approx(1)


@pytest.mark.parametrize("record", [[], [0], [0, 0, 0]])
def test_cpu_replay_rejects_incomplete_or_extra_records(record: list[int]) -> None:
    program = clifft.compile("H 0\nR 0\nH 0\nM 0")
    with pytest.raises(ValueError, match="visible followed by hidden; expected 2"):
        _replay_record(program, record)


@pytest.mark.parametrize("value", [-1, 2, 256, 0.5, -0.5, 1.5, "0"])
def test_cpu_replay_rejects_non_boolean_values(value: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        _replay_record(clifft.compile("M 0"), [value])


@pytest.mark.parametrize("source", ["X_ERROR(0.25) 0\nM 0", "M(0.25) 0"])
def test_cpu_replay_rejects_noise(source: str) -> None:
    with pytest.raises(ValueError, match="noiseless program"):
        _replay_record(clifft.compile(source), [0])


def test_cpu_replay_rejects_postselection_before_partial_execution() -> None:
    program = clifft.compile(
        "H 0\nM 0\nDETECTOR rec[-1]\nCX rec[-1] 1\nM 1", postselection_mask=[1]
    )
    with pytest.raises(ValueError, match="postselection"):
        _replay_record(program, [1, 0])


def test_cpu_replay_of_an_empty_record_has_unit_probability() -> None:
    result = _replay_record(clifft.compile("H 0"), [])
    assert result["reachable"]
    assert result["log_probability"] == 0
