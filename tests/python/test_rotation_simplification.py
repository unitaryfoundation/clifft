"""State-dependent rotation simplification checked with independent references."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import assert_statevectors_equiv, binomial_tolerance
from utils_conformance import (
    CpuSamplingMode,
    SamplingMode,
    assert_joint_distribution,
    unitary_reference,
)

import clifft


def _manager(
    region: int = 256,
    scheduled: bool = False,
    simplifier: clifft.RotationSimplificationPass | None = None,
) -> clifft.HirPassManager:
    passes = clifft.HirPassManager()
    passes.add(clifft.PeepholeFusionPass())
    passes.add(clifft.PhasePolynomialPass())
    if simplifier is None:
        simplifier = clifft.RotationSimplificationPass(max_region_ops=region)
    passes.add(simplifier)
    passes.add(clifft.StatevectorSqueezePass())
    if scheduled:
        passes.add(clifft.ActiveWidthSchedulePass())
    return passes


@pytest.mark.parametrize("preparation", ["", "X 1\n", "H 1\n", "H 1\nCX 1 0\n"])
@pytest.mark.parametrize("region", [1, 2, 256])
def test_signed_constraints_match_aer(preparation: str, region: int) -> None:
    source = preparation + "H 0\nR_Z(0.137) 0\nR_PAULI(-0.137) Z0*Z1"
    program = clifft.compile(source, hir_passes=_manager(region))
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))
    if preparation == "" and region > 1:
        assert program.peak_active_width == 0


@pytest.mark.parametrize("seed", range(64))
def test_mixed_axes_and_angles_match_aer(seed: int) -> None:
    rng = np.random.default_rng(seed)
    lines = ["H 3", "H 3"]
    for _ in range(6):
        q = int(rng.integers(4))
        lines.append(f"{rng.choice(['H', 'S', 'X'])} {q}")
        angle = float(rng.choice([0.137, -0.213, 0.25, -0.25, 0.5]))
        a = int(rng.integers(4))
        b = (a + int(rng.integers(1, 4))) % 4
        axis = rng.choice(["X", "Y", "Z"])
        lines.extend(
            [
                f"R_PAULI({angle}) {axis}{a}",
                f"R_PAULI({-angle}) {axis}{a}*Z{b}",
                f"CX {a} {b}",
            ]
        )
    source = "\n".join(lines)
    expected = unitary_reference(source)
    for region in (1, 2, 256):
        program = clifft.compile(source, hir_passes=_manager(region))
        assert_statevectors_equiv(clifft.get_statevector(program), expected)


@pytest.mark.parametrize(
    "family,bits",
    [("cdkm", 2), ("cdkm", 3), ("vbe", 2), ("draper", 2), ("draper", 3), ("draper", 4)],
)
@pytest.mark.parametrize("preparation", ["zero", "basis", "entangled"])
def test_quantum_adders_match_aer(family: str, bits: int, preparation: str) -> None:
    from qiskit import QuantumCircuit, qasm2, transpile
    from qiskit.synthesis import adder_qft_d00, adder_ripple_c04, adder_ripple_v95
    from utils_qiskit import qiskit_statevector

    constructors = {"cdkm": adder_ripple_c04, "vbe": adder_ripple_v95, "draper": adder_qft_d00}
    adder = constructors[family](bits, kind="fixed" if family == "draper" else "full")
    circuit = QuantumCircuit(adder.num_qubits + (preparation == "entangled"))
    operands = [
        q
        for q in range(adder.num_qubits)
        if adder.find_bit(adder.qubits[q]).registers[0][0].name in ("a", "b")
    ]
    if preparation == "basis":
        for q in operands:
            if q % 3 == 0:
                circuit.x(q)
    elif preparation == "entangled":
        reference = adder.num_qubits
        circuit.h(reference)
        circuit.cx(reference, operands[0])
        circuit.ry(0.317, operands[-1])
    circuit.compose(adder, qubits=range(adder.num_qubits), inplace=True)
    lowered = transpile(
        circuit,
        basis_gates=["h", "s", "sdg", "x", "cx", "t", "tdg", "rx", "ry", "rz"],
        optimization_level=0,
    )
    expected = qiskit_statevector(lowered)
    program = clifft.compile(qasm2.dumps(lowered), input_format="qasm2", hir_passes=_manager())
    assert_statevectors_equiv(clifft.get_statevector(program), expected)
    if preparation != "entangled":
        assert program.peak_active_width == 0


def _clifford_suffix_reference(axis: str, force: bool | None = None) -> tuple[str, np.ndarray]:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import pauli_error

    source = (
        "H 0\nR_Z(0.137) 0\nR_PAULI(0.363) Z0*Z1\n"
        f"{axis}_ERROR(0.13) 0\n"
        "R_Y(0.217) 2\nM 2\nCX rec[-1] 0\nMY 0\nM 1\n"
        "DETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-3]"
    )
    circuit = QuantumCircuit(6)
    circuit.h(0)
    circuit.rz(0.137 * np.pi, 0)
    circuit.rzz(0.363 * np.pi, 0, 1)
    probability = 0.13 if force is None else float(force)
    circuit.append(pauli_error([(axis, probability), ("I", 1 - probability)]).to_instruction(), [0])
    circuit.ry(0.217 * np.pi, 2)
    # Deferred measurements retain the full joint record and its feedback correlations.
    circuit.cx(2, 3)
    circuit.cx(3, 0)
    circuit.sdg(0)
    circuit.h(0)
    circuit.cx(0, 4)
    circuit.cx(1, 5)
    circuit.save_probabilities([3, 4, 5])
    expected = np.asarray(
        AerSimulator(method="density_matrix").run(circuit).result().data()["probabilities"]
    )
    expected[np.abs(expected) < 1e-14] = 0
    return source, expected


def _assert_records(result: Any, expected: np.ndarray) -> None:
    assert_joint_distribution(result.measurements, expected)
    np.testing.assert_array_equal(result.detectors[:, 0], result.measurements[:, 0])
    np.testing.assert_array_equal(
        result.observables[:, 0], result.measurements[:, 0] ^ result.measurements[:, 1]
    )


def _assert_survivors(result: Any, expected: np.ndarray) -> None:
    conditional = expected.copy()
    conditional[1::2] = 0
    probability = float(conditional.sum())
    assert abs(result.passed_shots / result.total_shots - probability) < binomial_tolerance(
        probability, result.total_shots
    )
    np.testing.assert_array_equal(result.measurements[:, 0], 0)
    _assert_records(result, conditional / probability)


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
@pytest.mark.parametrize("scheduled", [False, True])
def test_absorbed_clifford_preserves_noisy_records_and_postselection(
    sampling_mode: SamplingMode, axis: str, scheduled: bool
) -> None:
    source, expected = _clifford_suffix_reference(axis)
    for postselect in (False, True):
        simplifier = clifft.RotationSimplificationPass()
        program = sampling_mode.compile(
            source,
            hir_passes=_manager(scheduled=scheduled, simplifier=simplifier),
            normalize_syndromes=False,
            postselection_mask=[1] if postselect else None,
        )
        assert simplifier.applied
        assert simplifier.rotations_removed == 2
        assert program.num_measurements == 3
        assert program.peak_active_width == 1
        assert len(program.noise_site_probabilities) == 1
        if postselect:
            result = sampling_mode.sample_survivors(
                program, shots=16384, seed=621, keep_records=True
            )
            _assert_survivors(result, expected)
        else:
            _assert_records(sampling_mode.sample(program, shots=16384, seed=620), expected)


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
@pytest.mark.parametrize("k", [0, 1])
def test_absorbed_clifford_preserves_forced_faults(
    importance_sampling_mode: CpuSamplingMode, axis: str, k: int
) -> None:
    source, expected = _clifford_suffix_reference(axis, force=bool(k))
    for postselect in (False, True):
        simplifier = clifft.RotationSimplificationPass()
        program = importance_sampling_mode.compile(
            source,
            hir_passes=_manager(simplifier=simplifier),
            normalize_syndromes=False,
            postselection_mask=[1] if postselect else None,
        )
        assert simplifier.applied
        assert simplifier.rotations_removed == 2
        assert len(program.noise_site_probabilities) == 1
        if postselect:
            result = importance_sampling_mode.sample_k_survivors(
                program, shots=16384, k=k, seed=642, keep_records=True
            )
            _assert_survivors(result, expected)
        else:
            result = importance_sampling_mode.sample_k(program, shots=16384, k=k, seed=641)
            _assert_records(result, expected)


def test_absorbed_clifford_matches_stim(sampling_mode: SamplingMode) -> None:
    import stim

    suffix = "DEPOLARIZE1(0.13) 0\nMY 0\nM 1"
    source = "H 0\nR_Z(0.137) 0\nR_PAULI(0.363) Z0*Z1\n" + suffix
    reference = stim.Circuit("H 0\nS 0\n" + suffix)
    expected = np.array([1 - 2 * 0.13 / 3, 2 * 0.13 / 3, 0, 0])
    assert_joint_distribution(reference.compile_sampler(seed=812).sample(16384), expected)
    simplifier = clifft.RotationSimplificationPass()
    program = sampling_mode.compile(source, hir_passes=_manager(simplifier=simplifier))
    assert simplifier.applied
    assert simplifier.rotations_removed == 2
    assert_joint_distribution(sampling_mode.sample(program, 16384, seed=813).measurements, expected)


def test_expectation_remains_between_rotations(sampling_mode: SamplingMode) -> None:
    source = "H 0\nR_Z(0.137) 0\nEXP_VAL X0\nR_PAULI(-0.137) Z0*Z1\nEXP_VAL X0"
    program = sampling_mode.compile(source, hir_passes=_manager())
    result = sampling_mode.sample(program, 131, seed=33)
    np.testing.assert_allclose(result.exp_vals[:, 0], np.cos(0.137 * np.pi), atol=1e-6)
    np.testing.assert_allclose(result.exp_vals[:, 1], 1, atol=1e-6)


def test_benchmark_adder_matches_classical_addition(sampling_mode: SamplingMode) -> None:
    path = Path(__file__).resolve().parents[2] / "tools/bench/fixtures/draper_adder_16_basis.stim"
    program = sampling_mode.compile(path.read_text())
    assert program.peak_active_width == 0
    expected = np.array(
        [(37449 >> bit) & 1 for bit in range(16)] + [(56173 >> bit) & 1 for bit in range(16)],
        dtype=np.uint8,
    )
    result = sampling_mode.sample(program, 131, seed=42)
    np.testing.assert_array_equal(result.measurements, np.broadcast_to(expected, (131, 32)))
    np.testing.assert_array_equal(result.observables, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_region_ops": -1},
        {"max_region_ops": 4097},
        {"max_region_ops": 2**32},
        {"max_region_passes": 0},
        {"max_region_passes": 65},
        {"max_region_passes": -1},
    ],
)
def test_invalid_limits_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        clifft.RotationSimplificationPass(**kwargs)


@pytest.mark.parametrize("position", ["before", "between", "after"])
@pytest.mark.parametrize("probability", [0.0, 0.13, 1.0])
@pytest.mark.parametrize("preparation", ["zero", "negative", "unknown", "feedback"])
def test_arbitrary_angles_with_noise_and_feedback(
    sampling_mode: SamplingMode,
    position: str,
    probability: float,
    preparation: str,
) -> None:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import pauli_error

    circuit = QuantumCircuit(5)
    source = ["H 0"]
    circuit.h(0)
    if preparation == "negative":
        source.append("X 1")
        circuit.x(1)
    elif preparation == "unknown":
        source.append("H 1")
        circuit.h(1)
    source.append("M 1")
    circuit.cx(1, 2)
    if preparation == "feedback":
        # A random recorded outcome controls the second rotation's sign.
        source = ["H 0", "H 1", "M 1", "CX rec[-1] 0"]
        circuit = QuantumCircuit(5)
        circuit.h(0)
        circuit.h(1)
        circuit.cx(1, 2)
        circuit.cx(2, 0)

    def noise():
        source.append(f"X_ERROR({probability}) 1")
        circuit.append(
            pauli_error([("X", probability), ("I", 1 - probability)]).to_instruction(),
            [1],
        )

    if position == "before":
        noise()
    source.append("R_Z(0.137) 0")
    circuit.rz(0.137 * np.pi, 0)
    if position == "between":
        noise()
    source.append("R_PAULI(-0.137) Z0*Z1")
    circuit.rzz(-0.137 * np.pi, 0, 1)
    if position == "after":
        noise()
    source.extend(["MX 0", "M 1", "DETECTOR rec[-1] rec[-3]", "OBSERVABLE_INCLUDE(0) rec[-2]"])
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.cx(1, 4)
    circuit.save_probabilities([2, 3, 4])
    expected = np.asarray(
        AerSimulator(method="density_matrix").run(circuit).result().data()["probabilities"]
    )
    expected[np.abs(expected) < 1e-14] = 0
    for scheduled in (False, True):
        simplifier = clifft.RotationSimplificationPass()
        program = sampling_mode.compile(
            "\n".join(source),
            hir_passes=_manager(scheduled=scheduled, simplifier=simplifier),
        )
        assert simplifier.applied == (
            preparation in ("zero", "negative") and (probability == 0 or position == "after")
        )
        result = sampling_mode.sample(program, 16384, seed=721)
        assert_joint_distribution(result.measurements, expected)
        np.testing.assert_array_equal(
            result.detectors[:, 0],
            result.measurements[:, 0] ^ result.measurements[:, 2],
        )
        np.testing.assert_array_equal(result.observables[:, 0], result.measurements[:, 1])
