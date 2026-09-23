"""Developer-facing examples and boundary tests for the experimental CUDA API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
from utils_conformance import assert_joint_distribution, unitary_reference
from utils_cuda import (
    assert_distribution_matches,
    assert_forced_record_probabilities,
    assert_rate_matches,
    assert_repeatable,
    assert_same_rows,
    require_cuda_device,
)
from utils_gpu import NARROW_NOISY_CIRCUIT
from utils_gpu_replay import PAULI_REPLAY_CIRCUIT, REPLAY_CASES, ReplayCase

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


@dataclass(frozen=True)
class _NoisyCase:
    circuit: str
    bit_flips: tuple[tuple[int, float], ...]
    observable_column: int


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
@pytest.mark.parametrize("case", REPLAY_CASES, ids=lambda case: case.name)
def test_cuda_python_forced_replay_probes_each_branch(
    case: ReplayCase,
    precision: cuda.Precision,
    tolerance: float,
    tier: cuda.Tier,
) -> None:
    require_cuda_device()
    circuit = case.circuit
    cpu_program = clifft.compile(circuit)
    cuda_program = cuda.compile(circuit)
    sampler = cuda.Sampler(cuda_program, precision=precision, max_batch_shots=1, tier=tier)

    assert cuda_program.num_measurements == case.visible
    assert cuda_program.num_records == case.visible + case.hidden
    assert sampler.tier == tier

    assert_forced_record_probabilities(
        cpu_program,
        sampler,
        absolute_tolerance=tolerance,
    )


@pytest.mark.parametrize("tier", _EXPLICIT_TIERS)
@pytest.mark.parametrize(("precision", "tolerance"), [("fp64", 1e-12), ("fp32", 2e-5)])
def test_cuda_python_replay_with_multiple_pauli_measurements(
    tier: cuda.Tier, precision: cuda.Precision, tolerance: float
) -> None:
    require_cuda_device()
    cpu_program = clifft.compile(PAULI_REPLAY_CIRCUIT)
    sampler = cuda.Sampler(
        cuda.compile(PAULI_REPLAY_CIRCUIT), precision=precision, max_batch_shots=1, tier=tier
    )
    assert_forced_record_probabilities(cpu_program, sampler, absolute_tolerance=tolerance)


@pytest.mark.parametrize("tier", _EXPLICIT_TIERS)
def test_cuda_python_tiers_agree_on_a_wide_program(
    tier: cuda.Tier, cuda_cpu_wide: clifft.SampleResult
) -> None:
    require_cuda_device()
    shots = 20_000
    cpu = cuda_cpu_wide
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
def test_cuda_python_matches_cpu_joint_distribution(
    tier: cuda.Tier,
    precision: cuda.Precision,
    cuda_cpu_distribution: tuple[_NoisyCase, clifft.SampleResult],
) -> None:
    require_cuda_device()
    shots = 20_000
    case, cpu = cuda_cpu_distribution
    sampler = cuda.Sampler(cuda.compile(case.circuit), precision=precision, tier=tier)
    assert sampler.tier == tier
    gpu = sampler.sample(shots, seed=42)
    cpu_rows = np.concatenate((cpu.measurements, cpu.detectors, cpu.observables), axis=1)
    gpu_rows = np.concatenate((gpu.measurements, gpu.detectors, gpu.observables), axis=1)

    # Complete rows, not marginals: a tier that mishandles correlations
    # between records, detectors, and observables fails here.
    assert_distribution_matches(cpu_rows, gpu_rows)


@pytest.mark.parametrize("keep_records", [False, True], ids=["counts", "records"])
@pytest.mark.parametrize("tier", _EXPLICIT_TIERS)
def test_cuda_python_survivor_sampling_counts_and_rows(
    tier: cuda.Tier, keep_records: bool, cuda_cpu_survivors: clifft.SampleResult
) -> None:
    require_cuda_device()
    shots = 8192
    gpu_program = cuda.compile(_POSTSELECTED_CIRCUIT, postselection_mask=[1])
    assert gpu_program.has_postselection
    assert gpu_program.num_detectors == 1
    assert gpu_program.num_observables == 1
    assert gpu_program.num_exp_vals == 1
    sampler = cuda.Sampler(gpu_program, tier=tier)
    with pytest.raises(ValueError, match="sample_survivors"):
        sampler.sample(4, seed=1)

    cpu = cuda_cpu_survivors
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


@pytest.fixture(scope="module")
def cuda_cpu_wide() -> clifft.SampleResult:
    return cast(clifft.SampleResult, clifft.sample(clifft.compile(_WIDE_CIRCUIT), 20_000, seed=41))


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            _NoisyCase(NARROW_NOISY_CIRCUIT, bit_flips=((1, 0.3),), observable_column=1),
            id="narrow",
        ),
        pytest.param(
            _NoisyCase(_WIDE_NOISY_CIRCUIT, bit_flips=((1, 0.3), (3, 0.1)), observable_column=3),
            id="wide",
        ),
    ],
)
def cuda_cpu_distribution(request: pytest.FixtureRequest) -> tuple[_NoisyCase, clifft.SampleResult]:
    case = cast(_NoisyCase, request.param)
    return case, clifft.sample(clifft.compile(case.circuit), 20_000, seed=41)


@pytest.fixture(scope="module")
def cuda_cpu_survivors() -> clifft.SampleResult:
    program = clifft.compile(_POSTSELECTED_CIRCUIT, postselection_mask=[1])
    return cast(
        clifft.SampleResult, clifft.sample_survivors(program, 8192, seed=41, keep_records=True)
    )


def _unitary_prefix(circuit: str) -> str:
    return "\n".join(
        line
        for line in circuit.splitlines()
        if not line.startswith(
            ("M ", "EXP_VAL", "DETECTOR", "OBSERVABLE_INCLUDE", "PAULI_CHANNEL", "X_ERROR")
        )
    )


def test_cuda_wide_cpu_reference(cuda_cpu_wide: clifft.SampleResult) -> None:
    from qiskit.quantum_info import Pauli, Statevector

    state = unitary_reference(_unitary_prefix(_WIDE_CIRCUIT))
    assert_joint_distribution(cuda_cpu_wide.measurements, np.abs(state) ** 2)
    expected = [
        Statevector(state).expectation_value(Pauli(pauli)).real
        for pauli in ("IIIIIX", "IIIZZI", "IIYIIX")
    ]
    np.testing.assert_allclose(
        cuda_cpu_wide.exp_vals, np.broadcast_to(expected, (20_000, 3)), atol=1e-12
    )


def test_cuda_cpu_distribution_reference(
    cuda_cpu_distribution: tuple[_NoisyCase, clifft.SampleResult],
) -> None:
    case, cpu = cuda_cpu_distribution
    probabilities = np.abs(unitary_reference(_unitary_prefix(case.circuit))) ** 2
    indices = np.arange(len(probabilities))
    # Noise follows all unitaries, so X/Y faults permute final Z-basis outcomes.
    for qubit, probability in case.bit_flips:
        probabilities = (1 - probability) * probabilities + probability * probabilities[
            indices ^ (1 << qubit)
        ]
    assert_joint_distribution(cpu.measurements, probabilities)
    assert cpu.detectors.shape == (20_000, 1)
    assert cpu.observables.shape == (20_000, 1)
    np.testing.assert_array_equal(
        cpu.detectors[:, 0], cpu.measurements[:, -1] ^ cpu.measurements[:, -2]
    )
    np.testing.assert_array_equal(
        cpu.observables[:, 0], cpu.measurements[:, case.observable_column]
    )


def test_cuda_cpu_survivor_reference(cuda_cpu_survivors: clifft.SampleResult) -> None:
    cpu = cuda_cpu_survivors
    assert cpu.total_shots == 8192
    assert cpu.passed_shots is not None and 0 < cpu.passed_shots < 8192
    assert cpu.discards == 8192 - cpu.passed_shots
    assert cpu.measurements.shape == (cpu.passed_shots, 6)
    assert cpu.detectors.shape == (cpu.passed_shots, 1)
    assert cpu.observables.shape == (cpu.passed_shots, 1)
    assert cpu.exp_vals.shape == (cpu.passed_shots, 1)
    assert np.all(cpu.detectors == 0)
    assert np.all(cpu.measurements[:, 1] == 0)
    np.testing.assert_array_equal(
        cpu.observables[:, 0], cpu.measurements[:, 4] ^ cpu.measurements[:, 5]
    )
    np.testing.assert_allclose(cpu.exp_vals[:, 0], 1.0 - 2.0 * cpu.measurements[:, 0], atol=1e-9)
    assert cpu.observable_ones is not None
    assert cpu.logical_errors == cpu.observable_ones[0] == cpu.observables.sum()
