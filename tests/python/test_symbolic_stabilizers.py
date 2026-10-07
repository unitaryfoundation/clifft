"""Measured-state rewrites checked against conditional Aer and Stim references."""

from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from utils_conformance import SamplingMode, assert_joint_distribution

import clifft


@lru_cache(maxsize=None)
def _entangled_reference(kind: str, basis: str, angle: float = 0.137) -> tuple[str, np.ndarray]:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import pauli_error

    source = ["H 2", "CX 2 0", "R_X(0.173) 1", "MPP Z0*Z1"]
    circuit = QuantumCircuit(5)
    circuit.h(2)
    circuit.cx(2, 0)
    circuit.rx(np.pi * 0.173, 1)
    # Keep both the physical outcome and reported record, including correlations
    # with the spectator, through deferred measurement.
    circuit.cx(0, 3)
    circuit.cx(1, 3)
    circuit.cx(3, 4)
    if kind in ("readout", "asymmetric", "readout_one"):
        p0, p1 = {"readout": (0.2, 0.2), "asymmetric": (0.1, 0.3), "readout_one": (1.0, 1.0)}[kind]
        source.append(f"READOUT_NOISE({p0},{p1}) rec[-1]")
        circuit.append(
            Kraus(
                [np.diag(np.sqrt([1 - p0, 1 - p1])), np.array([[0, np.sqrt(p1)], [np.sqrt(p0), 0]])]
            ),
            [4],
        )
    source.append("CX rec[-1] 1")
    circuit.cx(4, 1)
    if kind in ("x_fault", "x_one"):
        probability = 1.0 if kind == "x_one" else 0.2
        source.append(f"X_ERROR({probability}) 1")
        circuit.append(
            pauli_error([("X", probability), ("I", 1 - probability)]).to_instruction(), [1]
        )
    elif kind == "shared_fault":
        source.append("E(0.2) X0 X1")
        circuit.append(pauli_error([("XX", 0.2), ("II", 0.8)]).to_instruction(), [0, 1])
    source += [f"R_Z({angle}) 0", f"R_Z({-angle}) 1"]
    circuit.rz(np.pi * angle, 0)
    circuit.rz(-np.pi * angle, 1)
    for q, axis in enumerate(basis):
        source.append(f"M{axis} {q}")
        if axis == "Y":
            circuit.sdg(q)
        if axis in "XY":
            circuit.h(q)
    circuit.save_probabilities([4, 0, 1, 2])
    result = AerSimulator(method="density_matrix", max_parallel_threads=1).run(circuit).result()
    assert result.success
    expected = np.asarray(result.data()["probabilities"])
    expected[np.abs(expected) < 1e-14] = 0
    return "\n".join(source), expected


@pytest.mark.parametrize("angle", [0.137, 0.25])
@pytest.mark.parametrize("axes", product("XYZ", repeat=3))
def test_measured_rewrites_preserve_full_conditional_state(
    axes: tuple[str, ...], angle: float
) -> None:
    source, expected = _entangled_reference("ideal", "".join(axes), angle)
    records = np.array([[(i >> j) & 1 for j in range(4)] for i in range(16)], dtype=np.uint8)
    program = clifft.compile(source)
    actual = clifft.record_probabilities(program, records)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize(
    "kind", ["ideal", "readout", "asymmetric", "readout_one", "x_fault", "x_one", "shared_fault"]
)
@pytest.mark.parametrize("basis", ["XXX", "YXY", "ZZZ"])
def test_noisy_measured_rewrites_preserve_records_and_spectators(
    sampling_mode: SamplingMode, kind: str, basis: str
) -> None:
    source, expected = _entangled_reference(kind, basis)
    source += "\nDETECTOR rec[-4]\nOBSERVABLE_INCLUDE(0) rec[-4] rec[-1]"
    program = sampling_mode.compile(source)
    result = sampling_mode.sample(program, shots=8192, seed=751)
    assert_joint_distribution(result.measurements, expected)
    np.testing.assert_array_equal(result.detectors[:, 0], result.measurements[:, 0])
    np.testing.assert_array_equal(
        result.observables[:, 0], result.measurements[:, 0] ^ result.measurements[:, 3]
    )


def test_default_pipeline_discovers_measured_reed_muller_reduction() -> None:
    fixture = Path(__file__).parents[1] / "fixtures/reed_muller_measured.stim"

    for noise, width, count in (("", 1, 1), ("Z_ERROR(0.2) 4", 1, 1), ("X_ERROR(0.2) 4", 2, 3)):
        source = fixture.read_text().replace("\nT 0", f"\n{noise}\nT 0")
        hir = clifft.trace(clifft.parse(source))
        clifft.default_hir_pass_manager().run(hir)
        assert hir.num_t_gates == count
        assert clifft.compile(source).peak_active_width == width


def test_later_measurements_and_hidden_resets_restore_optimizer_knowledge() -> None:
    for preparation in ("M 0\nCX rec[-1] 0", "R 0", "MR(0.2) 0"):
        source = "R_X(0.173) 0\n" + preparation + "\nR_Z(0.137) 0\n"
        source += "R_X(0.213) 0\nM 0\nCX rec[-1] 0\nR_Z(0.217) 0\nM 0"
        simplifier = clifft.RotationSimplificationPass()
        manager = clifft.HirPassManager()
        manager.add(simplifier)
        program = clifft.compile(source, hir_passes=manager)
        assert simplifier.rotations_removed == 2
        result = clifft.sample(program, shots=64, seed=26)
        np.testing.assert_array_equal(result.measurements[:, -1], 0)


@pytest.mark.parametrize("readout", [0, 0.2])
def test_clifford_preparation_with_shared_faults_matches_stim(
    sampling_mode: SamplingMode, readout: float
) -> None:
    import stim

    source = f"""H 0 1
MPP({readout}) Z0*Z1
CX rec[-1] 1
E(0.17) X0 X1
MPP Z0*Z1
CX rec[-1] 1
MPP X0*X1 Z0*Z1
"""
    reference = stim.Circuit(source).compile_sampler(seed=693).sample(shots=32768)
    indices = reference.astype(np.uint64) @ (1 << np.arange(4, dtype=np.uint64))
    expected = np.bincount(indices.astype(np.int64), minlength=16) / len(reference)
    actual = sampling_mode.sample(sampling_mode.compile(source), shots=32768, seed=972)
    assert_joint_distribution(actual.measurements, expected)
