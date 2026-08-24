"""Developer-facing examples and boundary tests for the experimental HIP API."""

from __future__ import annotations

import numpy as np
import pytest
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
def test_hip_python_forced_replay_probes_each_branch(
    precision: hip.Precision,
    tolerance: float,
) -> None:
    require_hip_device()
    circuit = "H 0\nT 0\nM 0\nEXP_VAL Z0\nOBSERVABLE_INCLUDE(0) rec[-1]"
    cpu_program = clifft.compile(circuit)
    sampler = hip.Sampler(hip.compile(circuit), precision=precision, max_batch_shots=1)

    assert_forced_record_probabilities(
        cpu_program,
        sampler,
        1,
        absolute_tolerance=tolerance,
    )


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_hip_python_matches_cpu_joint_distribution(precision: hip.Precision) -> None:
    require_hip_device()
    circuit = """\
H 0
T 0
H 0
CX 0 1
PAULI_CHANNEL_1(0.1, 0.2, 0.05) 1
M 0 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
    shots = 20_000
    cpu = clifft.sample(clifft.compile(circuit), shots, seed=41)
    gpu = hip.Sampler(hip.compile(circuit), precision=precision).sample(shots, seed=42)
    cpu_rows = np.concatenate((cpu.measurements, cpu.detectors, cpu.observables), axis=1)
    gpu_rows = np.concatenate((gpu.measurements, gpu.detectors, gpu.observables), axis=1)

    assert_distribution_matches(cpu_rows, gpu_rows)
