"""Independent quantum-output checks shared across execution backends.

The unitary corpus runs across CPU compiler profiles, reconstruction, and sampling
modes, plus noncomputational fixed, trajectory, and online execution with
leakage/loss off. Categorical noise histories drive non-Clifford feedback and
postselection on both original and reconstructed plans. Small reset/reuse
circuits check joint record laws and earlier-record feedback. Compiler rewrite
witnesses and the larger batch-boundary matrix remain on the ordinary CPU route.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, cast

import numpy as np
import pytest
from conftest import binomial_tolerance
from utils_conformance import (
    ACTIVE_WIDTH,
    CATEGORICAL_NOISE_SOURCE,
    COMPILER_PROFILES,
    CPU_SAMPLING_MODES,
    CPU_TUNING_MODES,
    DEFAULT,
    DEFAULT_SCHEDULED,
    FUSION_SQUEEZE,
    NONCOMP_SAMPLING_MODES,
    UNOPTIMIZED,
    CompilerProfile,
    CpuSamplingMode,
    NoncompSamplingMode,
    assert_joint_distribution,
    categorical_noise_reference,
    unitary_reference,
)
from utils_fuzzing import generate_active_state_lifecycle
from utils_qiskit import stim_to_qiskit_noiseless

import clifft


@dataclass(frozen=True)
class UnitaryCase:
    name: str
    source: str
    num_qubits: int
    witness_for: str | None = None
    reconstructed_width: int | None = None

    @property
    def measured_source(self) -> str:
        return self.source + "\nM " + " ".join(map(str, range(self.num_qubits)))

    @property
    def annotated_source(self) -> str:
        return (
            self.measured_source
            + f"\nDETECTOR rec[-{self.num_qubits}] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
        )


PHASE_SOURCE = (
    "H 0 1 2 3\n"
    + "\n".join(
        "R_PAULI(0.25) " + "*".join(f"Z{q}" for q in range(4) if parity & (1 << q))
        for parity in [*range(1, 16), 15]
    )
    + "\nH 0 1 2 3"
)

INTERFERENCE_SOURCE = "H 0 1\nT 0 1\nH 0 1"

CASES = (
    # Neither pass alone can satisfy the other pass's width-reduction witness.
    UnitaryCase("fusion", "H 0\nT 0\nT_DAG 0\nH 0", 1, "PeepholeFusionPass"),
    UnitaryCase("squeeze", INTERFERENCE_SOURCE, 2, "StatevectorSqueezePass", reconstructed_width=1),
    UnitaryCase("bell", "H 0\nCX 0 1\nT 0", 2, reconstructed_width=0),
    UnitaryCase("phase", PHASE_SOURCE, 4),
    UnitaryCase(
        "mixed-pauli", "R_X(0.3) 0\nR_Y(0.2) 1\nH 2\nCX 1 2\nR_PAULI(0.17) X0*Y1*Z2\nH 0 1", 3
    ),
    UnitaryCase(
        "mirror",
        "H 0\nT 0\nCX 0 1\nH 1\nT 1\nCX 1 2\nR_X(0.3) 2\n"
        "R_X(-0.3) 2\nCX 1 2\nT_DAG 1\nH 1\nCX 0 1\nT_DAG 0\nH 0",
        3,
    ),
)

# Nonzero records on both sides of a word boundary expose unwritten output
# tails that an aggregate statistical check could tolerate.
BOUNDARY_SOURCE = (
    "X 0 2 63 64\nM "
    + " ".join(map(str, range(65)))
    + "\nDETECTOR rec[-65] rec[-64]\nDETECTOR rec[-2] rec[-1]"
    + "\nOBSERVABLE_INCLUDE(0) rec[-64]\nOBSERVABLE_INCLUDE(1) rec[-1]"
)

# These choices exercise scalar, packed worker/tail handling, and automatic
# selection without repeating the compiler-profile and thread-layout matrix.
RECONSTRUCTION_MODES = tuple(
    next(mode for mode in CPU_SAMPLING_MODES if mode.name == name)
    for name in ("single-shot", "packed-65-2-workers", "automatic")
)


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case.name)
def case(request: pytest.FixtureRequest) -> UnitaryCase:
    return cast(UnitaryCase, request.param)


@pytest.fixture(scope="module", params=COMPILER_PROFILES, ids=lambda profile: profile.name)
def compiler(request: pytest.FixtureRequest) -> CompilerProfile:
    return cast(CompilerProfile, request.param)


@pytest.fixture(scope="module")
def probabilities(case: UnitaryCase) -> np.ndarray:
    return np.abs(unitary_reference(case.source)) ** 2


@pytest.fixture(scope="module")
def program(case: UnitaryCase, compiler: CompilerProfile) -> Any:
    return compiler.compile(case.measured_source)


@pytest.fixture(scope="module")
def annotated_program(case: UnitaryCase, compiler: CompilerProfile) -> Any:
    return compiler.compile(case.annotated_source)


def test_exact_records_match_independent_oracle(
    case: UnitaryCase, program: Any, probabilities: np.ndarray
) -> None:
    assert program.num_measurements == case.num_qubits
    records = [format(i, f"0{case.num_qubits}b")[::-1] for i in range(1 << case.num_qubits)]
    actual = clifft.record_probabilities(program, records)
    np.testing.assert_allclose(actual, probabilities, atol=1e-12, rtol=0)
    if case.name == "mirror":
        np.testing.assert_allclose(probabilities, [1] + [0] * (len(probabilities) - 1), atol=1e-12)


@pytest.mark.parametrize("mode", CPU_SAMPLING_MODES + CPU_TUNING_MODES, ids=lambda mode: mode.name)
def test_samples_and_annotations_match_independent_oracle(
    case: UnitaryCase, annotated_program: Any, probabilities: np.ndarray, mode: CpuSamplingMode
) -> None:
    # 8193 shots leave a partial final batch with capacity 65. Whole-record
    # comparisons distinguish Bell correlations that marginals cannot.
    result = mode.sample(annotated_program, 8193, seed=1907)
    _assert_annotated_distribution(result, probabilities, 8193, case.num_qubits)


@pytest.mark.parametrize("mode", RECONSTRUCTION_MODES, ids=lambda mode: mode.name)
@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.reconstructed_width is not None],
    ids=lambda case: case.name,
)
def test_reconstructed_samples_share_unitary_oracles(
    case: UnitaryCase, mode: CpuSamplingMode
) -> None:
    program = mode.compile(case.annotated_source, hir_passes=None, reconstruction=True)
    for keep_records in (False, True):
        plan = program.sampling_plan(keep_records=keep_records)
        assert plan.reconstruction_selected and not plan.fallback_reason
        assert plan.peak_active_width == case.reconstructed_width
        assert plan.peak_active_width < program.original_plan.peak_active_width
    result = mode.sample(program, 4097, seed=1907)
    probabilities = np.abs(unitary_reference(case.source)) ** 2
    _assert_annotated_distribution(result, probabilities, 4097, case.num_qubits)


def test_reconstruction_fallback_preserves_original_plan_width() -> None:
    for case in (case for case in CASES if case.reconstructed_width is None):
        program = clifft.compile(case.annotated_source, hir_passes=None, reconstruction=True)
        reason = (
            "does not reduce peak active width"
            if case.name == "fusion"
            else "unsupported source operation"
        )
        for keep_records in (False, True):
            plan = program.sampling_plan(keep_records=keep_records)
            assert not plan.reconstruction_selected
            assert reason in plan.fallback_reason
            assert plan.peak_active_width == program.original_plan.peak_active_width


def _assert_annotated_distribution(
    result: Any, probabilities: np.ndarray, shots: int, width: int
) -> None:
    assert result.measurements.shape == (shots, width)
    assert_joint_distribution(result.measurements, probabilities)
    np.testing.assert_array_equal(
        result.detectors, (result.measurements[:, 0] ^ result.measurements[:, -1])[:, None]
    )
    np.testing.assert_array_equal(result.observables, result.measurements[:, -1:])


NOISY_INTERFERENCE_SOURCE = (
    "H 4 5\nT 4 5\n" + CATEGORICAL_NOISE_SOURCE + "\n"
    "CZ rec[-4] 4\nCZ rec[-3] 5\nH 4 5\nM 4 5\n"
    "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-2] rec[-1] rec[-3]"
)


@pytest.fixture(scope="module")
def noisy_joint_probabilities() -> np.ndarray:
    """Combine categorical histories with independent Aer interference outcomes.

    The four Bell/readout records control Z feedback onto two |T> states.
    Their final X measurements therefore XOR the controlling records into
    Aer-distributed bits. Detectors and observables use explicit nonzero
    reference parities. Every history is retained, including impossible bins.
    """
    categorical = categorical_noise_reference().sum(axis=0)
    interference = np.abs(unitary_reference(INTERFERENCE_SOURCE)) ** 2
    joint = np.zeros(1024)
    for record, quantum in product(range(16), range(4)):
        bits = [(record >> bit) & 1 for bit in range(4)]
        bits += [(quantum & 1) ^ bits[0], (quantum >> 1) ^ bits[1]]
        detectors = [bits[0] ^ bits[2] ^ 1, bits[4] ^ bits[5] ^ 1]
        observables = [bits[1] ^ 1, bits[4] ^ bits[5] ^ bits[3]]
        key = sum(value << bit for bit, value in enumerate(bits + detectors + observables))
        joint[key] += categorical[record] * interference[quantum]
    np.testing.assert_allclose(joint.sum(), 1, rtol=0, atol=1e-14)
    return joint


def _assert_noisy_survivors(
    result: clifft.SampleResult, expected: np.ndarray, shots: int, keep_records: bool
) -> None:
    """Check accepted joint outputs and aggregate event masses per attempt."""
    keys = np.arange(len(expected))
    masses = [
        expected.sum(),
        expected[(keys >> 8) != 0].sum(),
        expected[((keys >> 8) & 1) != 0].sum(),
        expected[((keys >> 9) & 1) != 0].sum(),
    ]
    assert result.total_shots == shots
    assert result.passed_shots is not None
    assert result.logical_errors is not None
    assert result.observable_ones is not None
    assert result.discards == shots - result.passed_shots
    counts = [result.passed_shots, result.logical_errors, *result.observable_ones]
    assert len(counts) == len(masses)
    for label, count, mass in zip(
        ("acceptance", "logical error", "observable 0", "observable 1"), counts, masses, strict=True
    ):
        # Aer and the finite weighted sum have a 1e-14 absolute error budget.
        # Clamp only the variance input when a unit mass rounds above one.
        assert 0 <= mass <= 1 + 1e-14
        assert (
            abs(count / shots - mass) < binomial_tolerance(min(float(mass), 1), shots) + 1e-14
        ), label
    rows = result.passed_shots if keep_records else 0
    assert result.measurements.shape == (rows, 6), "measurement retention"
    assert result.detectors.shape == (rows, 2), "detector retention"
    assert result.observables.shape == (rows, 2), "observable retention"
    assert result.exp_vals.shape == (rows, 0)
    if keep_records:
        assert_joint_distribution(
            np.column_stack((result.measurements, result.detectors, result.observables)),
            expected / masses[0],
        )
        np.testing.assert_array_equal(result.observable_ones, result.observables.sum(axis=0))
        assert result.logical_errors == np.count_nonzero(np.any(result.observables, axis=1))


@pytest.mark.parametrize("mode", RECONSTRUCTION_MODES, ids=lambda mode: mode.name)
@pytest.mark.parametrize("reconstruction", [False, True], ids=["original", "reconstructed"])
@pytest.mark.parametrize("postselect", [False, True])
def test_noisy_joint_outputs_and_survivor_reuse(
    mode: CpuSamplingMode,
    reconstruction: bool,
    postselect: bool,
    noisy_joint_probabilities: np.ndarray,
) -> None:
    program = mode.compile(
        NOISY_INTERFERENCE_SOURCE,
        hir_passes=None,
        reconstruction=reconstruction,
        postselection_mask=[1, 0] if postselect else None,
        expected_detectors=[1, 1],
        expected_observables=[1, 0],
    )
    assert program.original_plan.peak_active_width == 2
    expected = noisy_joint_probabilities.copy()
    shots = 4097
    if not postselect:
        result = mode.sample(program, shots, seed=803)
        assert_joint_distribution(
            np.column_stack((result.measurements, result.detectors, result.observables)), expected
        )
    else:
        # Only the first detector rejects. The second remains a random,
        # correlated output, even though both use nonzero reference parities.
        expected[((np.arange(1024) >> 6) & 1) != 0] = 0
    for keep_records in (False, True, False):
        plan = program.sampling_plan(keep_records=keep_records)
        assert plan.reconstruction_selected == reconstruction
        assert bool(plan.fallback_reason) != reconstruction
        assert plan.peak_active_width == (1 if reconstruction else 2)
        result = mode.sample_survivors(program, shots, seed=803, keep_records=keep_records)
        _assert_noisy_survivors(result, expected, shots, keep_records)


def test_joint_output_check_detects_unwritten_observable(
    noisy_joint_probabilities: np.ndarray,
) -> None:
    program = clifft.compile(
        NOISY_INTERFERENCE_SOURCE,
        hir_passes=None,
        reconstruction=True,
        expected_detectors=[1, 1],
        expected_observables=[1, 0],
    )
    result = clifft.sample_survivors(program, 4097, seed=803, keep_records=True, batch_size=65)
    expected = noisy_joint_probabilities
    _assert_noisy_survivors(result, expected, 4097, True)
    # An unwritten observable keeps its expected shape but loses its joint law.
    result.observables[:, 1] = 0
    with pytest.raises(AssertionError):
        assert_joint_distribution(
            np.column_stack((result.measurements, result.detectors, result.observables)), expected
        )


@pytest.mark.parametrize("mode", NONCOMP_SAMPLING_MODES, ids=lambda mode: mode.name)
def test_noncomp_backends_share_unitary_oracles(
    case: UnitaryCase, probabilities: np.ndarray, mode: NoncompSamplingMode
) -> None:
    result = mode.sample(mode.compile(case.annotated_source), 4097, seed=1907)
    _assert_annotated_distribution(result, probabilities, 4097, case.num_qubits)
    if mode.execution == "online":
        # Unlike optimized mirror/fusion circuits, online must execute the
        # non-Clifford prefix before its inverse, including active-state growth.
        assert result.stats.peak_active_width > 0
        assert result.stats.online_state_growths > 0
    if case.name == "mirror":
        np.testing.assert_array_equal(result.measurements, 0)


@pytest.mark.parametrize(
    "mode",
    (
        next(mode for mode in CPU_SAMPLING_MODES if mode.name == "single-shot"),
        next(mode for mode in CPU_SAMPLING_MODES if mode.name == "packed-65"),
        *NONCOMP_SAMPLING_MODES,
    ),
    ids=lambda m: m.name,
)
@pytest.mark.parametrize("rounds", [3, 4])
def test_shared_lifecycle_preserves_joint_records(
    mode: CpuSamplingMode | NoncompSamplingMode, rounds: int
) -> None:
    source = generate_active_state_lifecycle(rounds) + (
        f"\nCZ rec[-{rounds}] 0\nS_DAG 0\nH 0\nM 0"
        f"\nDETECTOR rec[-{rounds + 1}] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    # Defer measurements in the independent Aer circuit, using a fresh ancilla
    # each round instead of reset/reuse. The first record controls the final
    # feedback across all intervening measurements and coordinate removals.
    reference = ["H 0", "T 0"]
    for q in range(1, rounds + 1):
        reference.extend([f"H {q}", f"T {q}", f"H {q}", f"CX {q} 0"])
    reference.extend(["CZ 1 0", "S_DAG 0", "H 0"])
    state = unitary_reference("\n".join(reference))
    # Aer indexes qubit 0 first; the reused-ancilla records precede its readout.
    keys = np.arange(1 << (rounds + 1))
    indices = (keys >> rounds) | ((keys & ((1 << rounds) - 1)) << 1)
    probabilities = np.abs(state[indices]) ** 2

    if isinstance(mode, NoncompSamplingMode):
        program = mode.compile(source, max_active_width=2)
    else:
        program = mode.compile(source, hir_passes=None)
        assert program.peak_active_width == 2
    result = mode.sample(program, 4097, seed=317)
    _assert_annotated_distribution(result, probabilities, 4097, rounds + 1)
    if isinstance(mode, NoncompSamplingMode):
        assert result.stats.peak_active_width == 2
    if isinstance(mode, NoncompSamplingMode) and mode.execution == "online":
        # Storage grows only to the peak and is retained across later rounds
        # and shots. Workers need not all receive work under dynamic scheduling.
        assert 2 <= result.stats.online_state_growths <= 2 * result.stats.workers
        assert result.stats.online_state_growths % 2 == 0


def _assert_boundary_outputs(result: clifft.SampleResult, shots: int) -> None:
    expected_record = np.zeros(65, dtype=np.uint8)
    expected_record[[0, 2, 63, 64]] = 1
    np.testing.assert_array_equal(
        result.measurements, np.broadcast_to(expected_record, (shots, 65))
    )
    np.testing.assert_array_equal(result.detectors, np.broadcast_to([1, 0], (shots, 2)))
    np.testing.assert_array_equal(result.observables, np.broadcast_to([0, 1], (shots, 2)))


@pytest.fixture(scope="module")
def boundary_program(compiler: CompilerProfile) -> Any:
    return compiler.compile(BOUNDARY_SOURCE)


@pytest.mark.parametrize("mode", CPU_SAMPLING_MODES + CPU_TUNING_MODES, ids=lambda mode: mode.name)
@pytest.mark.parametrize("shots", [63, 64, 65, 66, 129, 130, 131, 2049])
def test_deterministic_outputs_cross_word_and_batch_boundaries(
    boundary_program: Any, mode: CpuSamplingMode, shots: int
) -> None:
    # The automatic policy currently uses 2048 lanes for this narrow plan;
    # 2049 therefore checks its partial batch as well as capacity 65 tails.
    result = mode.sample(boundary_program, shots, seed=1907)
    _assert_boundary_outputs(result, shots)


@pytest.mark.parametrize("shots", [0, 1, 131])
def test_reconstructed_outputs_cross_word_and_batch_boundaries(shots: int) -> None:
    # The unobserved |T> factors keep the original plan active; reconstruction
    # eliminates them while retaining all 65 visible records and annotations.
    program = clifft.compile(
        "H 65 66\nT 65 66\n" + BOUNDARY_SOURCE, hir_passes=None, reconstruction=True
    )
    assert program.original_plan.peak_active_width == 2
    for keep_records in (False, True):
        plan = program.sampling_plan(keep_records=keep_records)
        assert plan.reconstruction_selected and plan.peak_active_width == 0
        result = clifft.sample_survivors(
            program, shots, seed=1907, keep_records=keep_records, batch_size=65, threads=2
        )
        assert result.total_shots == result.passed_shots == result.logical_errors == shots
        assert result.discards == 0
        np.testing.assert_array_equal(result.observable_ones, [0, shots])
        _assert_boundary_outputs(result, shots if keep_records else 0)
    _assert_boundary_outputs(
        clifft.sample(program, shots, seed=1907, batch_size=65, threads=2), shots
    )


@pytest.mark.parametrize(
    "witness", [case for case in CASES if case.witness_for], ids=lambda c: c.name
)
def test_default_pipeline_really_transforms_witness(witness: UnitaryCase) -> None:
    baseline = UNOPTIMIZED.compile(witness.measured_source)
    optimized = DEFAULT.compile(witness.measured_source)
    assert optimized.peak_active_width < baseline.peak_active_width, witness.witness_for


def test_active_width_profile_really_transforms_witness() -> None:
    source = "R_PAULI(0.3) X0*X1\nR_PAULI(0.3) Z0*Y1\nMPP Y0*Y1\nMPP Y0"
    baseline = FUSION_SQUEEZE.compile(source)
    scheduled = ACTIVE_WIDTH.compile(source)
    assert scheduled.peak_active_width < baseline.peak_active_width, "ActiveWidthSchedulePass"


def test_phase_profile_really_transforms_witness() -> None:
    source = PHASE_SOURCE + "\nM 0 1 2 3"
    baseline = ACTIVE_WIDTH.compile(source)
    reduced = DEFAULT_SCHEDULED.compile(source)
    assert reduced.peak_active_width == 1 < baseline.peak_active_width


def test_joint_check_detects_reversed_record_bit_order() -> None:
    # Unequal probabilities for 01 and 10 expose reversed bit significance.
    records = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    measurements = np.repeat(records, [1024, 2048, 3072, 4096], axis=0)
    expected = [0.1, 0.2, 0.3, 0.4]
    assert_joint_distribution(measurements, expected)
    with pytest.raises(AssertionError, match="Joint distribution differs"):
        assert_joint_distribution(measurements[:, ::-1], expected)


def test_joint_check_detects_wrong_correlations_with_correct_marginals() -> None:
    correlated = np.tile(np.array([[0, 0], [1, 1]], dtype=np.uint8), (4096, 1))
    anticorrelated = correlated.copy()
    anticorrelated[:, 1] ^= 1
    np.testing.assert_array_equal(correlated.mean(axis=0), anticorrelated.mean(axis=0))
    assert_joint_distribution(correlated, [0.5, 0, 0, 0.5])
    with pytest.raises(AssertionError):
        assert_joint_distribution(anticorrelated, [0.5, 0, 0, 0.5])


def test_joint_check_detects_wrong_correlations_with_all_outcomes_legal() -> None:
    records = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    independent = np.repeat(records, 2048, axis=0)
    correlated = np.repeat(records, [3072, 1024, 1024, 3072], axis=0)
    np.testing.assert_array_equal(independent.mean(axis=0), correlated.mean(axis=0))
    assert_joint_distribution(independent, [0.25, 0.25, 0.25, 0.25])
    # Every bin has positive reference probability, so only the statistical
    # check can reject the changed correlations.
    with pytest.raises(AssertionError, match="Joint distribution differs"):
        assert_joint_distribution(correlated, [0.25, 0.25, 0.25, 0.25])


def test_joint_check_detects_bias_without_impossible_outcomes() -> None:
    # Both outcomes are legal, so this must fail the statistical bound rather
    # than the exact-zero support check used by the correlation control.
    biased = np.zeros((8192, 1), dtype=np.uint8)
    biased[:2048] = 1
    with pytest.raises(AssertionError, match="Joint distribution differs"):
        assert_joint_distribution(biased, [0.5, 0.5])


@pytest.mark.parametrize(
    "instruction",
    [
        "X_ERROR(0.1) 0",
        "Y_ERROR(0.1) 0",
        "Z_ERROR(0.1) 0",
        "DEPOLARIZE1(0.1) 0",
        "DEPOLARIZE2(0.1) 0 1",
        "PAULI_CHANNEL_1(0.1,0,0) 0",
        "CORRELATED_ERROR(0.1) X0",
        "ELSE_CORRELATED_ERROR(0.1) X0",
        "M 0",
        "MPP X0",
        "R 0",
        "MR 0",
        "CX rec[-1] 0",
        "READOUT_NOISE(0.1) rec[-1]",
        "LEVEL_TRANSITION[jump] 0",
        "LEAKAGE(0.1) 0",
        "LOSS(0.1) 0",
        "REPEAT 2 {",
        "}",
        "UNKNOWN 0",
        "H !0",
        "H 0 garbage",
        "H",
        "CX 0",
        "T(0.25) 0",
        "R_Z 0",
        "R_Z(nan) 0",
        "U3(0.1) 0",
        "R_PAULI(0.3) X0 Y1",
        "R_PAULI(0.3) X0*Y0",
        "R_PAULI(0.3) X0*Y1garbage",
    ],
)
def test_noiseless_oracle_rejects_unsupported_input(instruction: str) -> None:
    with pytest.raises(ValueError):
        stim_to_qiskit_noiseless("H 0\n" + instruction)


def test_noiseless_oracle_accepts_comments_and_ticks() -> None:
    reference = unitary_reference("H 0\nT 0")
    actual = unitary_reference("# preparation\nH 0 # comment\nTICK\nT 0\n")
    np.testing.assert_allclose(actual, reference, atol=1e-12)


def test_default_profile_uses_fresh_production_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    managers = []
    received = []

    def factory() -> Any:
        manager = clifft.HirPassManager()
        managers.append(manager)
        return manager

    monkeypatch.setattr(clifft, "default_hir_pass_manager", factory)
    monkeypatch.setattr(
        clifft, "compile", lambda source, *, hir_passes: received.append(hir_passes)
    )
    DEFAULT.compile("H 0")
    DEFAULT.compile("H 0")
    assert len(managers) == 2 and managers[0] is not managers[1]
    assert received == managers


def test_cpu_sampling_modes_include_batch_and_worker_configurations() -> None:
    assert {(1, 1), (65, 1), ("auto", 1), (1, 2), (65, 2)} <= {
        (mode.batch_size, mode.threads) for mode in CPU_SAMPLING_MODES
    }
    for layout in ((1, 2), (2, 2)):
        assert any(
            mode.batch_size == 1
            and mode.thread_layout == layout
            and mode.intra_shot_min_active_width == 3
            for mode in CPU_SAMPLING_MODES
        )


@pytest.mark.parametrize("mode", CPU_SAMPLING_MODES + CPU_TUNING_MODES, ids=lambda mode: mode.name)
@pytest.mark.parametrize(
    ("sampler", "options"),
    [
        ("sample", {}),
        ("sample_survivors", {"keep_records": False}),
        ("sample_survivors", {"keep_records": True}),
        ("sample_k", {"k": 2}),
        ("sample_k_survivors", {"k": 2, "keep_records": False}),
        ("sample_k_survivors", {"k": 2, "keep_records": True}),
    ],
)
def test_sampling_mode_forwards_its_configuration(
    mode: CpuSamplingMode, sampler: str, options: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    program = object()
    monkeypatch.setattr(clifft, sampler, lambda *args, **kwargs: calls.append((args, kwargs)))
    getattr(mode, sampler)(program, 257, seed=41, **options)
    assert calls == [
        (
            (program, 257),
            {
                **options,
                "seed": 41,
                "threads": mode.threads,
                "batch_size": mode.batch_size,
                "tuning_budget_seconds": mode.tuning_budget_seconds,
                "thread_layout": mode.thread_layout,
                "intra_shot_min_active_width": mode.intra_shot_min_active_width,
            },
        )
    ]


_MISSING_OPENMP = "thread_layout intra-shot workers require an OpenMP-enabled build"
_PROCESSOR_BINDING = "hybrid thread_layout requires OMP_PROC_BIND=false"


@pytest.mark.parametrize(
    "sampler", ["sample", "sample_survivors", "sample_k", "sample_k_survivors"]
)
@pytest.mark.parametrize(
    ("layout", "message", "skips"),
    [
        pytest.param(None, _MISSING_OPENMP, False, id="implicit"),
        pytest.param((1, 1), _MISSING_OPENMP, False, id="serial"),
        pytest.param((2, 1), _MISSING_OPENMP, False, id="cross-shot"),
        pytest.param((1, 2), _MISSING_OPENMP, True, id="intra-shot-without-openmp"),
        pytest.param((2, 2), _MISSING_OPENMP, True, id="hybrid-without-openmp"),
        pytest.param((1, 2), _PROCESSOR_BINDING, False, id="bound-intra-shot"),
        pytest.param((2, 2), _PROCESSOR_BINDING, True, id="bound-hybrid"),
        pytest.param((2, 2), "invalid program", False, id="unrelated"),
        pytest.param((2, 2), f"unexpected error: {_MISSING_OPENMP}", False, id="openmp-substring"),
        pytest.param(
            (2, 2), f"unexpected error: {_PROCESSOR_BINDING}", False, id="binding-substring"
        ),
    ],
)
def test_sampling_mode_only_skips_unavailable_layouts(
    sampler: str,
    layout: tuple[int, int] | None,
    message: str,
    skips: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ValueError(message)

    def reject(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(clifft, sampler, reject)
    mode = CpuSamplingMode("threaded", 1, thread_layout=layout)
    options = {"k": 1} if sampler in ("sample_k", "sample_k_survivors") else {}
    if skips:
        with pytest.raises(pytest.skip.Exception):
            getattr(mode, sampler)(object(), 2, **options)
    else:
        with pytest.raises(ValueError) as raised:
            getattr(mode, sampler)(object(), 2, **options)
        assert raised.value is error


@pytest.mark.parametrize("reconstruction", [False, True])
@pytest.mark.parametrize("mask", [None, [1, 0], [1, 1]])
@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize(
    "preparation", [clifft.NoisePreparation.DENSE, clifft.NoisePreparation.SITE_ROWS]
)
def test_reusable_sharing_preserves_noisy_aggregate_masses(
    reconstruction: bool,
    mask: list[int] | None,
    group: bool,
    preparation: clifft.NoisePreparation,
    noisy_joint_probabilities: np.ndarray,
) -> None:
    program = clifft.compile(
        NOISY_INTERFERENCE_SOURCE,
        hir_passes=None,
        reconstruction=reconstruction,
        postselection_mask=mask,
        expected_detectors=[1, 1],
        expected_observables=[1, 0],
    )
    if reconstruction:
        assert program.sampling_plan(keep_records=False).reconstruction_selected
    expected = noisy_joint_probabilities.copy()
    for detector, selected in enumerate(mask or []):
        if selected:
            expected[((np.arange(1024) >> (6 + detector)) & 1) != 0] = 0
    sampler = clifft.SurvivorSampler(
        program,
        batch_size=65,
        initial_states=1,
        threads=2,
        noise_preparation=preparation,
        group_by_noise=group,
    )
    result = sampler.sample(8193, seed=803)
    _assert_noisy_survivors(result, expected, 8193, keep_records=False)
