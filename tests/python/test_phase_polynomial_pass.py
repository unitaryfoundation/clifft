"""Native phase reduction checked against independent Aer and Stim references."""

from functools import lru_cache
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


def _manager(pass_: Any | None = None) -> Any:
    manager = clifft.HirPassManager()
    manager.add(clifft.PeepholeFusionPass())
    manager.add(pass_ if pass_ is not None else clifft.PhasePolynomialPass())
    manager.add(clifft.StatevectorSqueezePass())
    return manager


def _parity_instructions(mask: int, gate: str = "T") -> list[str]:
    bits = [q for q in range(mask.bit_length()) if mask & (1 << q)]
    target = bits[-1]
    compute = [f"CX {q} {target}" for q in bits[:-1]]
    return compute + [f"{gate} {target}"] + compute[::-1]


def _identity_parities() -> list[str]:
    return [line for mask in range(1, 16) for line in _parity_instructions(mask)]


def test_native_phase_reduction_is_opt_in() -> None:
    source = "\n".join(["H 0 1 2 3", *_identity_parities(), *_parity_instructions(15)])
    baseline = clifft.compile(source)
    phase = clifft.PhasePolynomialPass()
    program = clifft.compile(source, hir_passes=_manager(phase))
    assert baseline.peak_active_width == 4
    assert program.peak_active_width == 1
    assert phase.applied
    assert phase.output_t_count == 1
    assert "PhasePolynomialPass" in repr(phase)
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))


@pytest.mark.parametrize("seed", range(12))
def test_native_phase_reduction_on_general_clifford_layouts(seed: int) -> None:
    rng = np.random.default_rng(seed)
    prep = ["H 0 1 2 3", "S 0 2", "CX 0 1", "H 1", "CX 1 3", "S_DAG 3"]
    core = _identity_parities()
    for _ in range(6):
        core.extend(
            _parity_instructions(int(rng.integers(1, 16)), "T" if rng.integers(2) else "T_DAG")
        )
    tail = ["H 0", "S 1", "CX 0 2", "T_DAG 2", "H 3", "T 3", "CZ 1 3"]
    source = "\n".join(prep + core + tail)
    program = clifft.compile(source, hir_passes=_manager())
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))


@pytest.mark.parametrize(
    "core,expected_width",
    [
        ([(1, "T"), (2, "T"), (3, "T_DAG")], 2),
        ([(mask, "T" if mask.bit_count() % 2 else "T_DAG") for mask in range(1, 8)], 3),
    ],
)
def test_native_phase_reduction_handles_quadratic_and_cubic_cores(
    core: list[tuple[int, str]], expected_width: int
) -> None:
    lines = ["H 0 1 2 3", *_identity_parities()]
    for mask, gate in core:
        lines.extend(_parity_instructions(mask, gate))
    source = "\n".join(lines)
    phase = clifft.PhasePolynomialPass()
    program = clifft.compile(source, hir_passes=_manager(phase))
    assert phase.applied
    assert program.peak_active_width == expected_width
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))


@pytest.mark.parametrize("prefix", ["", "X 1", "Y 1", "R_Y(0.137) 1"])
def test_phase_reduction_does_not_assume_entry_relations(prefix: str) -> None:
    source = prefix + "\nH 0\nCX 0 1\nT 1\nCX 0 1\nT 0"
    phase = clifft.PhasePolynomialPass()
    program = clifft.compile(source, hir_passes=_manager(phase))
    assert not phase.applied
    assert phase.output_t_count == 2
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))


@pytest.mark.parametrize("cap", [0, 1, 2, 3, 4, 32, 64])
def test_phase_reduction_variable_limits_preserve_unitaries(cap: int) -> None:
    source = "\n".join(["H 0 1 2 3", *_identity_parities(), *_parity_instructions(15)])
    phase = clifft.PhasePolynomialPass(max_variables=cap)
    program = clifft.compile(source, hir_passes=_manager(phase))
    assert phase.output_t_count <= phase.input_t_count
    if cap == 0:
        assert not phase.applied
        assert program.peak_active_width == 4
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))


@pytest.mark.parametrize("cap", [-1, 65, 2**32])
def test_phase_reduction_rejects_out_of_range_variable_limits(cap: int) -> None:
    with pytest.raises(ValueError, match="max_variables"):
        clifft.PhasePolynomialPass(max_variables=cap)


@lru_cache(maxsize=8)
def _noisy_reference(kind: str, force: bool | None = None) -> tuple[str, np.ndarray]:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import pauli_error

    source: list[str] = []
    circuit = QuantumCircuit(7)
    next_record = 4

    def emit(line: str) -> None:
        source.append(line)
        gate, *args = line.split()
        targets = list(map(int, args))
        if gate in ("H", "S", "S_DAG", "T", "T_DAG"):
            method = {"S_DAG": "sdg", "T_DAG": "tdg"}.get(gate, gate.lower())
            getattr(circuit, method)(targets)
        else:
            assert gate in ("CX", "CZ")
            getattr(circuit, gate.lower())(*targets)

    def measure(product: str) -> None:
        nonlocal next_record
        source.append("MPP " + product)
        factors = [(term[0], int(term[1:])) for term in product.split("*")]
        for axis, q in factors:
            if axis == "Y":
                circuit.sdg(q)
            if axis != "Z":
                circuit.h(q)
            circuit.cx(q, next_record)
            if axis != "Z":
                circuit.h(q)
            if axis == "Y":
                circuit.s(q)
        next_record += 1

    def noise(position: int) -> None:
        if kind == "late_axis":
            source.append("X_ERROR(0.13) 3")
            probability = 0.13 if force is None else float(force)
            circuit.append(
                pauli_error([("X", probability), ("I", 1 - probability)]).to_instruction(), [3]
            )
        elif kind in ("commuting", "hoisted"):
            target = 3 if kind == "hoisted" else 2
            source.append(f"Z_ERROR(0.13) {target}")
            probability = 0.13 if force is None else float(force)
            circuit.append(
                pauli_error([("Z", probability), ("I", 1 - probability)]).to_instruction(), [target]
            )
        elif position == 0 or kind == "interleaved":
            source.append("PAULI_CHANNEL_1(0.07,0.11,0.13) 2")
            circuit.append(
                pauli_error([("X", 0.07), ("Y", 0.11), ("Z", 0.13), ("I", 0.69)]).to_instruction(),
                [2],
            )
        else:
            probabilities = [0.0] * 15
            probabilities[4], probabilities[9], probabilities[14] = 0.07, 0.11, 0.13
            source.append("PAULI_CHANNEL_2(" + ",".join(map(str, probabilities)) + ") 0 2")
            circuit.append(
                pauli_error(
                    [("XX", 0.07), ("YY", 0.11), ("ZZ", 0.13), ("II", 0.69)]
                ).to_instruction(),
                [0, 2],
            )

    emit("H 0 1 2 3")
    if kind == "late_axis":
        emit("S 3")
    if kind not in ("pulled", "pulled_feedback", "hoisted"):
        measure("Z0" if kind == "feedback" else "X0*X1")
    if kind == "feedback":
        source.append("CZ rec[-1] 1")
        circuit.cz(4, 1)
    if kind not in ("commuting", "hoisted", "late_axis"):
        noise(0)
    masks = list(range(8, 16)) + list(range(1, 8)) if kind == "hoisted" else list(range(1, 16))
    for mask in masks:
        gate = "T_DAG" if kind in ("pulled", "pulled_feedback") and mask == 1 else "T"
        for line in _parity_instructions(mask, gate):
            emit(line)
        if kind in ("pulled", "pulled_feedback", "hoisted") and mask == 15:
            if kind == "hoisted":
                noise(0)
            if kind == "pulled_feedback":
                measure("Z3")
                source.append("CX rec[-1] 0")
                circuit.cx(4, 0)
            else:
                measure("X3" if kind == "hoisted" else "X0")
        if mask == 7 and kind in ("commuting", "interleaved", "late_axis"):
            noise(0)
    for line in _parity_instructions(15):
        emit(line)
    if kind not in ("commuting", "hoisted", "late_axis"):
        noise(1)
    if kind in ("pulled", "pulled_feedback", "hoisted"):
        emit("H 3")
        emit("T 3")
    emit("S 1")
    emit("CZ 0 2")
    measure("Y0*Z2")
    measure("X3")
    source.extend(["DETECTOR rec[-1] rec[-2]", "OBSERVABLE_INCLUDE(0) rec[-3] rec[-1]"])
    circuit.save_probabilities([4, 5, 6])
    result = AerSimulator(method="density_matrix").run(circuit, shots=1).result()
    assert result.success
    probabilities = np.asarray(result.data()["probabilities"])
    assert np.all(probabilities >= -1e-14)
    np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-12, rtol=0)
    # Aer can put signed roundoff in exactly unreachable density-matrix bins.
    probabilities[np.abs(probabilities) < 1e-14] = 0
    return "\n".join(source), probabilities


@pytest.mark.parametrize(
    "kind",
    [
        "commuting",
        "prefix_suffix",
        "interleaved",
        "feedback",
        "pulled",
        "pulled_feedback",
        "hoisted",
        "late_axis",
    ],
)
def test_native_phase_reduction_preserves_noisy_joint_records(
    sampling_mode: SamplingMode, kind: str
) -> None:
    source, expected = _noisy_reference(kind)
    phase = clifft.PhasePolynomialPass()
    program = sampling_mode.compile(source, hir_passes=_manager(phase), normalize_syndromes=False)
    assert program.num_measurements == 3
    if kind != "interleaved":
        assert phase.applied
        assert program.peak_active_width == 1
    if kind in ("pulled", "pulled_feedback", "hoisted"):
        assert phase.pauli_pullbacks > 0
    result = sampling_mode.sample(program, shots=16384, seed=620)
    assert_joint_distribution(result.measurements, expected)
    np.testing.assert_array_equal(
        result.detectors[:, 0], result.measurements[:, 1] ^ result.measurements[:, 2]
    )
    np.testing.assert_array_equal(
        result.observables[:, 0], result.measurements[:, 0] ^ result.measurements[:, 2]
    )

    selected = sampling_mode.compile(
        source, hir_passes=_manager(), normalize_syndromes=False, postselection_mask=[1]
    )
    survivors = sampling_mode.sample_survivors(selected, shots=16384, seed=621, keep_records=True)
    probability = sum(
        p for record, p in enumerate(expected) if ((record >> 1) ^ (record >> 2)) & 1 == 0
    )
    assert abs(survivors.passed_shots / survivors.total_shots - probability) < binomial_tolerance(
        probability, survivors.total_shots
    )


@pytest.mark.parametrize("k", [0, 1])
@pytest.mark.parametrize("kind", ["commuting", "hoisted", "late_axis"])
def test_native_phase_reduction_preserves_forced_fault_sampling(
    importance_sampling_mode: CpuSamplingMode, k: int, kind: str
) -> None:
    source, expected = _noisy_reference(kind, force=bool(k))
    phase = clifft.PhasePolynomialPass()
    program = importance_sampling_mode.compile(source, hir_passes=_manager(phase))
    assert phase.applied
    assert len(program.noise_site_probabilities) == 1
    result = importance_sampling_mode.sample_k(program, shots=16384, k=k, seed=641)
    assert_joint_distribution(result.measurements, expected)


def test_native_phase_reduction_preserves_exact_measurement_records() -> None:
    source, _ = _noisy_reference("commuting")
    source = "\n".join(
        line.replace("Z_ERROR(0.13) 2", "Z 2")
        for line in source.splitlines()
        if not line.startswith(("DETECTOR", "OBSERVABLE_INCLUDE"))
    )
    program = clifft.compile(source, hir_passes=_manager())
    baseline = clifft.compile(source, hir_passes=None)
    records = ["".join(str((key >> j) & 1) for j in range(3)) for key in range(8)]
    np.testing.assert_allclose(
        clifft.record_probabilities(program, records),
        clifft.record_probabilities(baseline, records),
        atol=1e-12,
    )


def test_native_phase_reduction_of_clifford_block_matches_stim(
    sampling_mode: SamplingMode,
) -> None:
    import stim

    source = "\n".join(
        [
            "H 0 1 2 3",
            *_identity_parities(),
            "T 0",
            "T 0",
            "DEPOLARIZE1(0.17) 0",
            "MPP Y0*Z1",
            "MY 0",
        ]
    )
    phase = clifft.PhasePolynomialPass()
    program = sampling_mode.compile(source, hir_passes=_manager(phase))
    assert phase.applied
    assert program.peak_active_width == 0
    reference = stim.Circuit("H 0 1 2 3\nS 0\nDEPOLARIZE1(0.17) 0\nMPP Y0*Z1\nMY 0")
    samples = reference.compile_sampler(seed=719).sample(16384).astype(np.uint8)
    expected = np.bincount(samples.astype(np.int64) @ np.array([1, 2]), minlength=4) / len(samples)
    result = sampling_mode.sample(program, shots=16384, seed=720)
    assert_joint_distribution(result.measurements, expected)
