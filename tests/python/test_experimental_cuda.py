"""Developer-facing examples and boundary tests for the experimental CUDA API."""

from __future__ import annotations

import numpy as np
import pytest
from utils_cuda import (
    assert_distribution_matches,
    assert_forced_record_probabilities,
    assert_repeatable,
    assert_same_rows,
    require_cuda_device,
)

import clifft
from clifft.experimental import cuda

# Six promoted coordinates exercise the cooperative tiers with real lane
# striding while staying small enough for thread-per-shot to cross-check.
_WIDE_CIRCUIT = """\
H 0
H 1
H 2
H 3
H 4
H 5
T 0
T 1
T 2
T 3
T 4
T 5
CX 0 1
CX 2 3
CX 4 5
CX 1 2
CX 3 4
R_PAULI(0.21) X0*Y3
EXP_VAL X0
EXP_VAL Z1*Z2
EXP_VAL X0*Y3
M 0 1 2 3 4 5
"""


def test_cuda_facade_explains_when_native_extension_is_absent() -> None:
    if cuda.is_built():
        program = cuda.compile("H 0\nT 0\nM 0")
        assert program.num_actions > 0
        assert "CUDA executable" in program.inspect()
        return

    assert not cuda.is_available()
    assert "CLIFFT_ENABLE_CUDA=ON" in cuda.backend_info()
    with pytest.raises(RuntimeError, match="CLIFFT_ENABLE_CUDA"):
        cuda.compile("M 0")


def test_cuda_facade_rejects_unknown_tier_names() -> None:
    if not cuda.is_built():
        pytest.skip("requires the CUDA extension")
    program = cuda.compile("H 0\nT 0\nM 0")
    require_cuda_device()
    with pytest.raises(ValueError, match="tier must be"):
        cuda.Sampler(program, tier="warp_per_shot")  # type: ignore[arg-type]


def test_cuda_python_tier_selection_follows_width() -> None:
    require_cuda_device()
    narrow = cuda.compile("H 0\nT 0\nH 0\nM 0")
    wide = cuda.compile(_WIDE_CIRCUIT)

    assert narrow.peak_active_width <= 4
    assert wide.peak_active_width == 6
    assert cuda.selected_tier(narrow) == "thread_per_shot"
    assert cuda.selected_tier(wide) == "block_shared"
    assert cuda.selected_tier(wide, "fp32") == "block_shared"

    auto = cuda.Sampler(wide, max_batch_shots=64)
    assert auto.tier == "block_shared"
    forced = cuda.Sampler(wide, max_batch_shots=64, tier="block_global", max_concurrent_shots=3)
    assert forced.tier == "block_global"
    assert forced.max_concurrent_shots == 3
    assert forced.max_batch_shots == 64


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_cuda_python_sampler_reuses_bounded_workspace(precision: cuda.Precision) -> None:
    require_cuda_device()
    program = cuda.compile("H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    sampler = cuda.Sampler(program, precision=precision, max_batch_shots=7)

    assert sampler.precision == precision
    assert sampler.tier == "thread_per_shot"
    assert sampler.max_batch_shots == 7
    assert sampler.allocated_device_bytes > 0
    assert_repeatable(sampler, 257, 1234)


@pytest.mark.parametrize(
    ("precision", "tolerance"),
    [("fp64", 1e-12), ("fp32", 2e-5)],
)
@pytest.mark.parametrize("tier", ["thread_per_shot", "block_shared", "block_global"])
def test_cuda_python_forced_replay_probes_each_branch(
    precision: cuda.Precision,
    tolerance: float,
    tier: cuda.Tier,
) -> None:
    require_cuda_device()
    # Two visible measurements with non-Clifford branching and nothing else:
    # the CPU record-probability oracle accepts only pure measurement
    # programs without hidden records, detectors, or observables.
    circuit = """\
H 0
H 1
T 0
T 1
CX 0 1
MPP Y0*Z1
R_PAULI(0.17) X0*Y1
M 0
"""
    cpu_program = clifft.compile(circuit)
    cuda_program = cuda.compile(circuit)
    sampler = cuda.Sampler(cuda_program, precision=precision, max_batch_shots=1, tier=tier)

    assert cuda_program.num_measurements == 2
    assert cuda_program.num_records == 2
    assert sampler.tier == tier

    assert_forced_record_probabilities(
        cpu_program,
        sampler,
        absolute_tolerance=tolerance,
    )


@pytest.mark.parametrize("tier", ["thread_per_shot", "block_shared", "block_global"])
def test_cuda_python_tiers_agree_on_a_wide_program(tier: cuda.Tier) -> None:
    require_cuda_device()
    shots = 20_000
    cpu = clifft.sample(clifft.compile(_WIDE_CIRCUIT), shots, seed=41)
    sampler = cuda.Sampler(cuda.compile(_WIDE_CIRCUIT), tier=tier)
    gpu = sampler.sample(shots, seed=42)

    # Expectation values precede the measurements, so every shot carries the
    # same deterministic values and the strided kernels are checked exactly.
    assert gpu.exp_vals.shape == cpu.exp_vals.shape
    np.testing.assert_allclose(gpu.exp_vals, cpu.exp_vals, atol=1e-12)
    cpu_marginals = cpu.measurements.mean(axis=0)
    gpu_marginals = gpu.measurements.mean(axis=0)
    tolerance = 6.0 * np.sqrt(cpu_marginals * (1.0 - cpu_marginals) * 2.0 / shots) + 1e-3
    assert np.all(np.abs(gpu_marginals - cpu_marginals) <= tolerance)

    # The concurrency cap changes the launch grid but never the seeded rows.
    capped = cuda.Sampler(cuda.compile(_WIDE_CIRCUIT), tier=tier, max_concurrent_shots=5)
    assert_same_rows(capped.sample(301, seed=7), sampler.sample(301, seed=7))


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_cuda_python_matches_cpu_joint_distribution(precision: cuda.Precision) -> None:
    require_cuda_device()
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
    gpu = cuda.Sampler(cuda.compile(circuit), precision=precision).sample(shots, seed=42)
    cpu_rows = np.concatenate((cpu.measurements, cpu.detectors, cpu.observables), axis=1)
    gpu_rows = np.concatenate((gpu.measurements, gpu.detectors, gpu.observables), axis=1)

    assert_distribution_matches(cpu_rows, gpu_rows)
