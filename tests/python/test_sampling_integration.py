"""Integration and boundary tests for the public sampling API."""

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import stim

import clifft

_MACHINE = platform.machine().lower()
_RUNTIME_DISPATCH_BUILD = (
    _MACHINE in {"amd64", "x86_64"} and not platform.python_compiler().startswith("MSC")
) or (sys.platform == "darwin" and _MACHINE in {"aarch64", "arm64"})


def test_experimental_namespace_reports_optional_hip_build() -> None:
    built = clifft.experimental.hip.is_built()
    info = clifft.experimental.hip.backend_info()

    if built:
        assert info.startswith("HIP ")
    else:
        assert info == "HIP backend not built; rebuild Clifft with CLIFFT_ENABLE_HIP=ON"


def test_import_clifft_does_not_eagerly_load_experimental_hip() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import clifft; "
            "print('clifft.experimental' in sys.modules); "
            "print('clifft._clifft_hip' in sys.modules)",
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["False", "False"]


@pytest.mark.skipif(
    not _RUNTIME_DISPATCH_BUILD,
    reason="CLIFFT_FORCE_ISA is ignored when runtime dispatch is not compiled",
)
def test_unknown_forced_isa_is_rejected_by_compile() -> None:
    environment = os.environ.copy()
    environment["CLIFFT_FORCE_ISA"] = "not-an-isa"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import clifft; clifft.compile('M 0')",
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode != 0
    assert "CLIFFT_FORCE_ISA" in completed.stderr
    assert "unrecognized value" in completed.stderr


def test_compile_returns_public_program() -> None:
    program = clifft.compile("H 0\nT 0\nM 0")

    assert isinstance(program, clifft.Program)
    assert program.num_qubits == 1
    assert program.num_measurements == 1
    assert program.num_hidden_measurements == 0
    assert program.num_actions > 0


def test_expectation_probes_are_available_through_public_api() -> None:
    program = clifft.compile("EXP_VAL X0 Z0")
    result = clifft.sample(program, 3, seed=1)

    assert program.num_exp_vals == 2
    np.testing.assert_allclose(result.exp_vals, [[0.0, 1.0]] * 3, atol=1e-12)


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
    result = clifft.sample(clifft.compile(circuit), shots=32, seed=7)

    np.testing.assert_array_equal(result.measurements, np.zeros((32, 2), dtype=np.uint8))
    np.testing.assert_array_equal(result.detectors, np.zeros((32, 1), dtype=np.uint8))
    np.testing.assert_array_equal(result.observables, np.zeros((32, 1), dtype=np.uint8))


def test_asymmetric_readout_noise_uses_the_pre_flip_record() -> None:
    zero = clifft.sample(clifft.compile("M 0\nREADOUT_NOISE(1, 0) rec[-1]"), 16, seed=1)
    one = clifft.sample(clifft.compile("X 0\nM 0\nREADOUT_NOISE(0, 1) rec[-1]"), 16, seed=1)

    assert np.all(zero.measurements == 1)
    assert np.all(one.measurements == 0)


def test_postselection_survivor_metadata_and_records() -> None:
    program = clifft.compile(
        "H 0\nEXP_VAL X0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        postselection_mask=[1],
    )
    with pytest.raises(ValueError, match="sample_survivors"):
        clifft.sample(program, 10, seed=1)

    result = clifft.sample_survivors(program, 1000, seed=1, keep_records=True)
    assert result.passed_shots is not None
    assert result.total_shots is not None
    assert result.discards is not None
    assert 0 < result.passed_shots < result.total_shots
    assert result.discards == result.total_shots - result.passed_shots
    assert result.logical_errors == 0
    assert np.all(result.measurements == 0)
    assert np.all(result.detectors == 0)
    assert np.all(result.observables == 0)
    np.testing.assert_allclose(result.exp_vals, 1.0, atol=1e-12)


def test_noisy_record_probabilities_remain_explicitly_unsupported() -> None:
    program = clifft.compile("X_ERROR(0.1) 0\nM 0")
    with pytest.raises(ValueError, match="pure-state evolution"):
        clifft.record_probabilities(program, ["0"])


@pytest.mark.parametrize(
    "fixture",
    [
        "tests/fixtures/target_qec.stim",
        "tests/fixtures/cultivation_d5.stim",
    ],
)
def test_representative_qec_fixtures_execute(fixture: str) -> None:
    program = clifft.compile(Path(fixture).read_text())
    result = clifft.sample(program, shots=1, seed=3)

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
    program = clifft.compile(str(circuit), normalize_syndromes=True)
    result = clifft.sample(program, shots=2, seed=5)

    assert result.detectors.shape == (2, circuit.num_detectors)
    assert result.observables.shape == (2, circuit.num_observables)
