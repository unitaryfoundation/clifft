"""Common HIP/CUDA API, retained-sampler, and CPU-reference checks."""

from typing import cast

import numpy as np
import pytest
from utils_conformance import assert_joint_distribution, unitary_reference
from utils_gpu import (
    NARROW_NOISY_CIRCUIT,
    NARROW_UNITARY_CIRCUIT,
    GpuApi,
    assert_distribution_matches,
    assert_repeatable,
    require_gpu_device,
)
from utils_gpu_replay import REPLAY_CASES, ReplayCase, assert_forced_record_probabilities

import clifft
from clifft.experimental import cuda, hip

_HIP: GpuApi = hip
_CUDA: GpuApi = cuda
_GPU_EXECUTIONS = [
    pytest.param(_HIP, "auto", id="hip-auto"),
    pytest.param(_CUDA, "thread_per_shot", id="cuda-thread-per-shot"),
    pytest.param(_CUDA, "block_shared", id="cuda-block-shared"),
    pytest.param(_CUDA, "block_global", id="cuda-block-global"),
]


@pytest.fixture(params=[_HIP, _CUDA], ids=["hip", "cuda"])
def gpu_api(request: pytest.FixtureRequest) -> GpuApi:
    return cast(GpuApi, request.param)


def test_facade_explains_when_native_extension_is_absent(gpu_api: GpuApi) -> None:
    backend = gpu_api.__name__.rsplit(".", 1)[-1].upper()
    if gpu_api.is_built():
        program = gpu_api.compile("H 0\nT 0\nM 0")
        assert program.num_actions > 0
        assert f"{backend} executable" in program.inspect()
        return

    assert not gpu_api.is_available()
    assert f"CLIFFT_ENABLE_{backend}=ON" in gpu_api.backend_info()
    with pytest.raises(RuntimeError, match=f"CLIFFT_ENABLE_{backend}"):
        gpu_api.compile("M 0")


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_sampler_reuses_bounded_workspace(gpu_api: GpuApi, precision: hip.Precision) -> None:
    require_gpu_device(gpu_api)
    program = gpu_api.compile("H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    sampler = gpu_api.Sampler(program, precision=precision, max_batch_shots=7)

    assert sampler.precision == precision
    assert sampler.tier == "thread_per_shot"
    assert sampler.max_batch_shots == 7
    assert sampler.allocated_device_bytes > 0
    assert_repeatable(sampler, 257, 1234)


@pytest.mark.parametrize(("gpu_api", "tier"), _GPU_EXECUTIONS)
@pytest.mark.parametrize(("precision", "tolerance"), [("fp64", 1e-12), ("fp32", 2e-5)])
@pytest.mark.parametrize("case", REPLAY_CASES, ids=lambda case: case.name)
def test_forced_replay_probes_each_branch(
    gpu_api: GpuApi,
    tier: hip.Tier,
    case: ReplayCase,
    precision: hip.Precision,
    tolerance: float,
) -> None:
    if not gpu_api.is_built():
        pytest.skip(gpu_api.backend_info())
    cpu_program = clifft.compile(case.circuit)
    gpu_program = gpu_api.compile(case.circuit)
    assert gpu_program.num_measurements == case.visible
    assert gpu_program.num_records == case.visible + case.hidden
    assert gpu_program.peak_active_width >= case.min_active_width

    require_gpu_device(gpu_api)
    sampler = gpu_api.Sampler(gpu_program, precision=precision, max_batch_shots=1, tier=tier)
    assert sampler.precision == precision
    if tier != "auto":
        assert sampler.tier == tier
    assert_forced_record_probabilities(cpu_program, sampler, absolute_tolerance=tolerance)


@pytest.mark.parametrize(("gpu_api", "tier"), _GPU_EXECUTIONS)
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_matches_cpu_narrow_joint_distribution(
    gpu_api: GpuApi,
    tier: hip.Tier,
    precision: hip.Precision,
    cpu_narrow_distribution: clifft.SampleResult,
) -> None:
    require_gpu_device(gpu_api)
    sampler = gpu_api.Sampler(gpu_api.compile(NARROW_NOISY_CIRCUIT), precision=precision, tier=tier)
    assert sampler.precision == precision
    if tier != "auto":
        assert sampler.tier == tier
    gpu = sampler.sample(20_000, seed=42)
    cpu = cpu_narrow_distribution
    cpu_rows = np.concatenate((cpu.measurements, cpu.detectors, cpu.observables), axis=1)
    gpu_rows = np.concatenate((gpu.measurements, gpu.detectors, gpu.observables), axis=1)
    assert_distribution_matches(cpu_rows, gpu_rows)


@pytest.fixture(scope="module")
def cpu_narrow_distribution() -> clifft.SampleResult:
    return cast(
        clifft.SampleResult, clifft.sample(clifft.compile(NARROW_NOISY_CIRCUIT), 20_000, seed=41)
    )


def test_gpu_reference_narrow_distribution(cpu_narrow_distribution: clifft.SampleResult) -> None:
    cpu = cpu_narrow_distribution
    probabilities = np.abs(unitary_reference(NARROW_UNITARY_CIRCUIT)) ** 2
    # X and Y faults flip the second measurement with total probability 0.3.
    probabilities = 0.7 * probabilities + 0.3 * probabilities[np.arange(4) ^ 2]
    p0 = (2 + np.sqrt(2)) / 4
    np.testing.assert_allclose(
        probabilities, [0.7 * p0, 0.3 * (1 - p0), 0.3 * p0, 0.7 * (1 - p0)], atol=1e-12
    )
    assert_joint_distribution(cpu.measurements, probabilities)
    assert cpu.detectors.shape == (20_000, 1)
    assert cpu.observables.shape == (20_000, 1)
    np.testing.assert_array_equal(
        cpu.detectors[:, 0], cpu.measurements[:, 0] ^ cpu.measurements[:, 1]
    )
    np.testing.assert_array_equal(cpu.observables[:, 0], cpu.measurements[:, 1])
