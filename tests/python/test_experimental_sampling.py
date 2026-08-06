"""Conformance and boundary tests for the explicitly selected sampling backend."""

from pathlib import Path

import numpy as np
import pytest
import stim

import clifft
import clifft.experimental as experimental


def test_program_is_separate_from_legacy_bytecode() -> None:
    legacy = clifft.compile("H 0\nT 0\nM 0")
    program = experimental.compile("H 0\nT 0\nM 0")

    assert isinstance(legacy, clifft.Program)
    assert isinstance(program, experimental.Program)
    assert not isinstance(program, clifft.Program)
    assert program.num_qubits == 1
    assert program.num_measurements == 1
    assert program.num_hidden_measurements == 0
    assert program.num_actions > 0


@pytest.mark.parametrize(
    "circuit",
    [
        "H 0\nM 0\nM 0",
        "H 0\nCX 0 1\nM 0 1",
        "H 0\nT 0\nS 0\nMX 0",
        "H 0\nT 0\nR_Z(0.3) 0\nM 0",
        "H 0\nH 1\nT 0\nT 1\nCX 0 1\nMPP Y0*Z1",
    ],
)
def test_seeded_samples_match_legacy_for_curated_circuits(circuit: str) -> None:
    legacy = clifft.sample(clifft.compile(circuit), shots=128, seed=1234).measurements
    actual = experimental.sample(experimental.compile(circuit), shots=128, seed=1234).measurements
    np.testing.assert_array_equal(actual, legacy)


@pytest.mark.parametrize(
    "circuit,operation",
    [
        ("EXP_VAL Z0", "EXP_VAL"),
    ],
)
def test_unsupported_capabilities_fail_during_compile(circuit: str, operation: str) -> None:
    with pytest.raises(ValueError, match=operation):
        experimental.compile(circuit)


def test_noise_readout_feedback_and_syndrome_share_one_symbolic_record() -> None:
    circuit = """
        X_ERROR(1) 0
        M 0
        READOUT_NOISE(1) rec[-1]
        CX rec[-1] 1
        M 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    result = experimental.sample(experimental.compile(circuit), shots=32, seed=7)

    np.testing.assert_array_equal(result.measurements, np.zeros((32, 2), dtype=np.uint8))
    np.testing.assert_array_equal(result.detectors, np.zeros((32, 1), dtype=np.uint8))
    np.testing.assert_array_equal(result.observables, np.zeros((32, 1), dtype=np.uint8))


def test_asymmetric_readout_noise_uses_the_pre_flip_record() -> None:
    zero = experimental.sample(experimental.compile("M 0\nREADOUT_NOISE(1, 0) rec[-1]"), 16, seed=1)
    one = experimental.sample(
        experimental.compile("X 0\nM 0\nREADOUT_NOISE(0, 1) rec[-1]"), 16, seed=1
    )

    assert np.all(zero.measurements == 1)
    assert np.all(one.measurements == 0)


def test_postselection_survivor_metadata_and_records() -> None:
    program = experimental.compile(
        "H 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        postselection_mask=[1],
    )
    with pytest.raises(ValueError, match="sample_survivors"):
        experimental.sample(program, 10, seed=1)

    result = experimental.sample_survivors(program, 1000, seed=1, keep_records=True)
    assert result.passed_shots is not None
    assert result.total_shots is not None
    assert result.discards is not None
    assert 0 < result.passed_shots < result.total_shots
    assert result.discards == result.total_shots - result.passed_shots
    assert result.logical_errors == 0
    assert np.all(result.measurements == 0)
    assert np.all(result.detectors == 0)
    assert np.all(result.observables == 0)


def test_noisy_record_probabilities_remain_explicitly_unsupported() -> None:
    program = experimental.compile("X_ERROR(0.1) 0\nM 0")
    with pytest.raises(ValueError, match="presampled symbols"):
        experimental.record_probabilities(program, ["0"])


@pytest.mark.parametrize(
    "fixture",
    [
        "tests/fixtures/target_qec.stim",
        "tests/fixtures/cultivation_d5.stim",
    ],
)
def test_representative_qec_fixtures_execute(fixture: str) -> None:
    program = experimental.compile(Path(fixture).read_text())
    result = experimental.sample(program, shots=1, seed=3)

    assert result.measurements.shape == (1, program.num_measurements)
    assert result.detectors.shape == (1, program.num_detectors)
    assert result.observables.shape == (1, program.num_observables)


def test_generated_surface_code_executes_with_reference_normalization() -> None:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
    )
    program = experimental.compile(str(circuit), normalize_syndromes=True)
    result = experimental.sample(program, shots=2, seed=5)

    assert result.detectors.shape == (2, circuit.num_detectors)
    assert result.observables.shape == (2, circuit.num_observables)
