"""Small independent-oracle corpus shared across compiler and sampling modes."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from utils_conformance import (
    COMPILER_PROFILES,
    CPU_SAMPLING_MODES,
    DEFAULT,
    UNOPTIMIZED,
    CompilerProfile,
    CpuSamplingMode,
    assert_joint_distribution,
    unitary_reference,
)
from utils_pass_registry import registered_hir_passes
from utils_qiskit import stim_to_qiskit_noiseless

import clifft


@dataclass(frozen=True)
class UnitaryCase:
    name: str
    source: str
    num_qubits: int
    witness_for: str | None = None

    @property
    def measured_source(self) -> str:
        return self.source + "\nM " + " ".join(map(str, range(self.num_qubits)))


CASES = (
    # Neither pass alone can satisfy the other pass's width-reduction witness.
    UnitaryCase("fusion", "H 0\nT 0\nT_DAG 0\nH 0", 1, "PeepholeFusionPass"),
    UnitaryCase("squeeze", "H 0 1\nT 0 1\nH 0 1", 2, "StatevectorSqueezePass"),
    UnitaryCase("bell", "H 0\nCX 0 1\nT 0", 2),
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

# These passes deliberately change the reference distribution; they need
# their own contract tests instead of this original-circuit equivalence test.
EXCLUDED_PASSES = {
    "RemoveNoisePass": "Removes noise, changing the noisy circuit's distribution.",
    "DropNonUnitaryPass": "Removes measurements and other nonunitary operations.",
}


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
    return compiler.compile(
        case.measured_source
        + f"\nDETECTOR rec[-{case.num_qubits}] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )


def test_exact_records_match_independent_oracle(
    case: UnitaryCase, program: Any, probabilities: np.ndarray
) -> None:
    assert program.num_measurements == case.num_qubits
    records = [format(i, f"0{case.num_qubits}b")[::-1] for i in range(1 << case.num_qubits)]
    actual = clifft.record_probabilities(program, records)
    np.testing.assert_allclose(actual, probabilities, atol=1e-12, rtol=0)
    if case.name == "mirror":
        np.testing.assert_allclose(probabilities, [1] + [0] * (len(probabilities) - 1), atol=1e-12)


@pytest.mark.parametrize("mode", CPU_SAMPLING_MODES, ids=lambda mode: mode.name)
def test_samples_and_annotations_match_independent_oracle(
    case: UnitaryCase, annotated_program: Any, probabilities: np.ndarray, mode: CpuSamplingMode
) -> None:
    # 8193 shots leave a partial final batch with capacity 65. Whole-record
    # comparisons distinguish Bell correlations that marginals cannot.
    result = mode.sample(annotated_program, 8193, seed=1907)
    assert result.measurements.shape == (8193, case.num_qubits)
    assert_joint_distribution(result.measurements, probabilities)
    np.testing.assert_array_equal(
        result.detectors, (result.measurements[:, 0] ^ result.measurements[:, -1])[:, None]
    )
    np.testing.assert_array_equal(result.observables, result.measurements[:, -1:])


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


@pytest.mark.parametrize("mode", CPU_SAMPLING_MODES, ids=lambda mode: mode.name)
@pytest.mark.parametrize("shots", [63, 64, 65, 66, 129, 130, 131, 2049])
def test_deterministic_outputs_cross_word_and_batch_boundaries(
    boundary_program: Any, mode: CpuSamplingMode, shots: int
) -> None:
    # The automatic policy currently uses 2048 lanes for this narrow plan;
    # 2049 therefore checks its partial batch as well as capacity 65 tails.
    result = mode.sample(boundary_program, shots, seed=1907)
    _assert_boundary_outputs(result, shots)


@pytest.mark.parametrize("output", ["measurements", "detectors", "observables"])
def test_boundary_check_rejects_an_unwritten_final_row(output: str) -> None:
    mode = next(mode for mode in CPU_SAMPLING_MODES if mode.name == "packed-65")
    result = mode.sample(DEFAULT.compile(BOUNDARY_SOURCE), 131, seed=1907)
    _assert_boundary_outputs(result, 131)
    getattr(result, output)[-1] = 0
    with pytest.raises(AssertionError):
        _assert_boundary_outputs(result, 131)


@pytest.mark.parametrize(
    "witness", [case for case in CASES if case.witness_for], ids=lambda c: c.name
)
def test_default_pipeline_really_transforms_witness(witness: UnitaryCase) -> None:
    baseline = UNOPTIMIZED.compile(witness.measured_source)
    optimized = DEFAULT.compile(witness.measured_source)
    assert optimized.peak_active_width < baseline.peak_active_width, witness.witness_for


@pytest.mark.parametrize(
    "witness", [case for case in CASES if case.witness_for], ids=lambda c: c.name
)
def test_pass_witness_rejects_a_missing_transformation(
    witness: UnitaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    remaining_pass = (
        clifft.StatevectorSqueezePass
        if witness.witness_for == "PeepholeFusionPass"
        else clifft.PeepholeFusionPass
    )

    def remaining_pipeline() -> Any:
        manager = clifft.HirPassManager()
        manager.add(remaining_pass())
        return manager

    monkeypatch.setattr(clifft, "default_hir_pass_manager", remaining_pipeline)
    with pytest.raises(AssertionError, match=witness.witness_for):
        test_default_pipeline_really_transforms_witness(witness)


def _assert_pass_inventory(registry: dict[str, dict[str, object]]) -> None:
    witnesses = {case.witness_for for case in CASES if case.witness_for}
    declared = witnesses | EXCLUDED_PASSES.keys()
    assert set(registry) == declared, (
        f"Pass coverage decision required: missing={set(registry) - declared}, "
        f"stale={declared - set(registry)}"
    )
    assert all(EXCLUDED_PASSES.values())
    for name in witnesses:
        assert registry[name]["default_enabled"], f"{name} needs an explicit opt-in profile"


def test_every_registered_pass_has_a_coverage_decision() -> None:
    _assert_pass_inventory(registered_hir_passes())


def test_coverage_guard_rejects_an_unaccounted_pass() -> None:
    registry = registered_hir_passes()
    registry["UnaccountedPass"] = {"default_enabled": True}
    with pytest.raises(AssertionError, match="Pass coverage decision required"):
        _assert_pass_inventory(registry)


def test_joint_check_detects_wrong_correlations_with_correct_marginals() -> None:
    correlated = np.tile(np.array([[0, 0], [1, 1]], dtype=np.uint8), (4096, 1))
    anticorrelated = correlated.copy()
    anticorrelated[:, 1] ^= 1
    np.testing.assert_array_equal(correlated.mean(axis=0), anticorrelated.mean(axis=0))
    assert_joint_distribution(correlated, [0.5, 0, 0, 0.5])
    with pytest.raises(AssertionError):
        assert_joint_distribution(anticorrelated, [0.5, 0, 0, 0.5])


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


@pytest.mark.parametrize("mode", CPU_SAMPLING_MODES, ids=lambda mode: mode.name)
def test_sampling_mode_forwards_its_configuration(
    mode: CpuSamplingMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    program = object()
    monkeypatch.setattr(clifft, "sample", lambda *args, **kwargs: calls.append((args, kwargs)))
    mode.sample(program, 8193, 1907)
    assert calls == [((program, 8193), {"seed": 1907, "threads": 1, "batch_size": mode.batch_size})]
