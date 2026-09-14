"""Developer-facing examples and boundary tests for the experimental CUDA API."""

from __future__ import annotations

import numpy as np
import pytest
from utils_cuda import (
    assert_distribution_matches,
    assert_forced_record_probabilities,
    assert_rate_matches,
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

_EXPLICIT_TIERS: list[cuda.Tier] = ["thread_per_shot", "block_shared", "block_global"]

# Narrow enough for automatic selection to pick thread-per-shot, so forcing
# the cooperative tiers on it runs their noise and output paths on the same
# distribution.
_NARROW_NOISY_CIRCUIT = """\
H 0
T 0
H 0
CX 0 1
PAULI_CHANNEL_1(0.1, 0.2, 0.05) 1
M 0 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""

# Six promoted coordinates with noise, so the cooperative tiers run lane-strided
# sweeps between the noise draws; eight output bits keep the complete-row
# distribution dense enough to compare.
_WIDE_NOISY_CIRCUIT = """\
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
PAULI_CHANNEL_1(0.1, 0.2, 0.05) 1
X_ERROR(0.1) 3
M 0 1 2 3 4 5
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-3]
"""

# The detector reads the measurement of qubit 1, the observable is the parity
# of the last two records, and the trailing EXP_VAL observes the collapsed
# qubit 0, so every retained row can be checked for alignment against itself.
_POSTSELECTED_CIRCUIT = """\
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
X_ERROR(0.2) 1
M 0 1 2 3 4 5
DETECTOR rec[-5]
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
EXP_VAL Z0
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


def test_cuda_facade_rejects_unknown_tier_and_precision_names() -> None:
    if not cuda.is_built():
        pytest.skip("requires the CUDA extension")
    # Lowering and argument validation need the extension but no device: the
    # names are rejected before any native sampler or device query exists.
    program = cuda.compile("H 0\nT 0\nM 0")
    with pytest.raises(ValueError, match="tier must be"):
        cuda.Sampler(program, tier="warp_per_shot")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="precision must be"):
        cuda.Sampler(program, precision="fp16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="precision must be"):
        cuda.selected_tier(program, "fp16")  # type: ignore[arg-type]


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
@pytest.mark.parametrize("tier", _EXPLICIT_TIERS)
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


@pytest.mark.parametrize("tier", _EXPLICIT_TIERS)
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
@pytest.mark.parametrize("tier", _EXPLICIT_TIERS)
@pytest.mark.parametrize(
    "circuit", [_NARROW_NOISY_CIRCUIT, _WIDE_NOISY_CIRCUIT], ids=["narrow", "wide"]
)
def test_cuda_python_matches_cpu_joint_distribution(
    circuit: str, tier: cuda.Tier, precision: cuda.Precision
) -> None:
    require_cuda_device()
    shots = 20_000
    cpu = clifft.sample(clifft.compile(circuit), shots, seed=41)
    sampler = cuda.Sampler(cuda.compile(circuit), precision=precision, tier=tier)
    assert sampler.tier == tier
    gpu = sampler.sample(shots, seed=42)
    cpu_rows = np.concatenate((cpu.measurements, cpu.detectors, cpu.observables), axis=1)
    gpu_rows = np.concatenate((gpu.measurements, gpu.detectors, gpu.observables), axis=1)

    # Complete rows, not marginals: a tier that mishandles correlations
    # between records, detectors, and observables fails here.
    assert_distribution_matches(cpu_rows, gpu_rows)


@pytest.mark.parametrize("keep_records", [False, True], ids=["counts", "records"])
@pytest.mark.parametrize("tier", _EXPLICIT_TIERS)
def test_cuda_python_survivor_sampling_counts_and_rows(tier: cuda.Tier, keep_records: bool) -> None:
    require_cuda_device()
    shots = 8192
    cpu_program = clifft.compile(_POSTSELECTED_CIRCUIT, postselection_mask=[1])
    gpu_program = cuda.compile(_POSTSELECTED_CIRCUIT, postselection_mask=[1])
    assert gpu_program.has_postselection
    assert gpu_program.num_detectors == 1
    assert gpu_program.num_observables == 1
    assert gpu_program.num_exp_vals == 1
    sampler = cuda.Sampler(gpu_program, tier=tier)
    with pytest.raises(ValueError, match="sample_survivors"):
        sampler.sample(4, seed=1)

    cpu = clifft.sample_survivors(cpu_program, shots, seed=41, keep_records=True)
    gpu = sampler.sample_survivors(shots, keep_records=keep_records, seed=42)

    assert gpu.total_shots == shots
    assert gpu.passed_shots is not None
    assert 0 < gpu.passed_shots < shots
    assert gpu.discards == shots - gpu.passed_shots
    rows = gpu.passed_shots if keep_records else 0
    assert gpu.measurements.shape == (rows, 6)
    assert gpu.detectors.shape == (rows, 1)
    assert gpu.observables.shape == (rows, 1)
    assert gpu.exp_vals.shape == (rows, 1)
    assert gpu.observable_ones is not None
    assert gpu.observable_ones.shape == (1,)
    # One observable, so a logical error is exactly an observable one.
    assert gpu.logical_errors == gpu.observable_ones[0]

    assert cpu.passed_shots is not None
    assert cpu.observable_ones is not None
    assert_rate_matches(cpu.passed_shots, shots, gpu.passed_shots, shots)
    # Aggregate counts are kept in both modes.
    assert_rate_matches(
        int(cpu.observable_ones[0]), cpu.passed_shots, int(gpu.observable_ones[0]), gpu.passed_shots
    )
    if not keep_records:
        return

    # Every retained row is complete and aligned with itself: the postselected
    # detector reads qubit 1, so that column is zero; the observable is the
    # parity of the last two records; and the trailing EXP_VAL saw the
    # collapsed qubit 0, so it equals 1 - 2 * m0 row by row.
    assert np.all(gpu.detectors == 0)
    assert np.all(gpu.measurements[:, 1] == 0)
    np.testing.assert_array_equal(
        gpu.observables[:, 0], gpu.measurements[:, 4] ^ gpu.measurements[:, 5]
    )
    np.testing.assert_allclose(gpu.exp_vals[:, 0], 1.0 - 2.0 * gpu.measurements[:, 0], atol=1e-9)
    assert gpu.observable_ones[0] == gpu.observables.sum()

    cpu_rows = np.concatenate((cpu.measurements, cpu.observables), axis=1)
    gpu_rows = np.concatenate((gpu.measurements, gpu.observables), axis=1)
    assert_distribution_matches(cpu_rows, gpu_rows)
