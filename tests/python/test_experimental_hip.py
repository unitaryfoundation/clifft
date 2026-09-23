"""Developer-facing examples and boundary tests for the experimental HIP API."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from utils_conformance import assert_joint_distribution
from utils_gpu import NARROW_NOISY_CIRCUIT
from utils_gpu_replay import REPLAY_CASES, ReplayCase
from utils_hip import (
    assert_distribution_matches,
    assert_forced_record_probabilities,
    assert_repeatable,
    require_hip_device,
)

import clifft
from clifft.experimental import hip


def test_hip_facade_explains_when_native_extension_is_absent() -> None:
    if hip.is_built():
        program = hip.compile("H 0\nT 0\nM 0")
        assert program.num_actions > 0
        assert "HIP executable" in program.inspect()
        return

    assert not hip.is_available()
    assert "CLIFFT_ENABLE_HIP=ON" in hip.backend_info()
    with pytest.raises(RuntimeError, match="CLIFFT_ENABLE_HIP"):
        hip.compile("M 0")


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_hip_python_sampler_reuses_bounded_workspace(precision: hip.Precision) -> None:
    require_hip_device()
    program = hip.compile("H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    sampler = hip.Sampler(program, precision=precision, max_batch_shots=7)

    assert sampler.precision == precision
    assert sampler.max_batch_shots == 7
    assert sampler.allocated_device_bytes > 0
    assert_repeatable(sampler, 257, 1234)


@pytest.mark.parametrize(
    ("precision", "tolerance"),
    [("fp64", 1e-12), ("fp32", 2e-5)],
)
@pytest.mark.parametrize("case", REPLAY_CASES, ids=lambda case: case.name)
def test_hip_python_forced_replay_probes_each_branch(
    case: ReplayCase,
    precision: hip.Precision,
    tolerance: float,
) -> None:
    if not hip.is_built():
        pytest.skip("requires the HIP extension")
    circuit = case.circuit
    cpu_program = clifft.compile(circuit)
    hip_program = hip.compile(circuit)
    assert hip_program.num_measurements == case.visible
    assert hip_program.num_records == case.visible + case.hidden
    assert hip_program.peak_active_width >= case.min_active_width

    require_hip_device()
    sampler = hip.Sampler(hip_program, precision=precision, max_batch_shots=1)

    assert_forced_record_probabilities(
        cpu_program,
        sampler,
        absolute_tolerance=tolerance,
    )


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_hip_python_matches_cpu_joint_distribution(
    precision: hip.Precision, hip_cpu_distribution: clifft.SampleResult
) -> None:
    require_hip_device()
    circuit = NARROW_NOISY_CIRCUIT
    shots = 20_000
    cpu = hip_cpu_distribution
    gpu = hip.Sampler(hip.compile(circuit), precision=precision).sample(shots, seed=42)
    cpu_rows = np.concatenate((cpu.measurements, cpu.detectors, cpu.observables), axis=1)
    gpu_rows = np.concatenate((gpu.measurements, gpu.detectors, gpu.observables), axis=1)

    assert_distribution_matches(cpu_rows, gpu_rows)


@pytest.fixture(scope="module")
def hip_cpu_distribution() -> clifft.SampleResult:
    return cast(
        clifft.SampleResult, clifft.sample(clifft.compile(NARROW_NOISY_CIRCUIT), 20_000, seed=41)
    )


def test_hip_cpu_distribution_reference(hip_cpu_distribution: clifft.SampleResult) -> None:
    cpu = hip_cpu_distribution
    p0 = (2 + np.sqrt(2)) / 4
    # X and Y faults flip the second measurement with total probability 0.3.
    assert_joint_distribution(
        cpu.measurements, [0.7 * p0, 0.3 * (1 - p0), 0.3 * p0, 0.7 * (1 - p0)]
    )
    assert cpu.detectors.shape == (20_000, 1)
    assert cpu.observables.shape == (20_000, 1)
    np.testing.assert_array_equal(
        cpu.detectors[:, 0], cpu.measurements[:, 0] ^ cpu.measurements[:, 1]
    )
    np.testing.assert_array_equal(cpu.observables[:, 0], cpu.measurements[:, 1])
