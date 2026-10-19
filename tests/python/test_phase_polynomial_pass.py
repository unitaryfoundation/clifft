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


def test_default_phase_reduction_can_be_disabled() -> None:
    source = "\n".join(["H 0 1 2 3", *_identity_parities(), *_parity_instructions(15)])
    without_phase = clifft.HirPassManager()
    without_phase.add(clifft.PeepholeFusionPass())
    without_phase.add(clifft.StatevectorSqueezePass())
    baseline = clifft.compile(source, hir_passes=without_phase)
    default = clifft.compile(source)
    assert default.peak_active_width == 1
    phase = clifft.PhasePolynomialPass()
    program = clifft.compile(source, hir_passes=_manager(phase))
    assert baseline.peak_active_width == 4
    assert program.peak_active_width == 1
    assert phase.applied
    assert phase.output_t_count == 1
    assert "PhasePolynomialPass" in repr(phase)
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))


# These phases cancel after the fault, but flip qubit 0 without it. Folding
# the probability-one fault into their signs would change the reference parity.
_REFERENCE_NOISE_SOURCE = """X_ERROR(1) 1
H 0
CX 2 0
T 0
CX 2 0
CX 3 0
T 0
CX 3 0
CX 1 0
T 0
CX 3 0
T 0
CX 3 0
CX 1 0
H 0
M 0
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.mark.parametrize("normalize", [False, True])
def test_probability_one_noise_preserves_normalized_annotations(
    sampling_mode: SamplingMode, normalize: bool
) -> None:
    reference = sampling_mode.compile(
        _REFERENCE_NOISE_SOURCE, hir_passes=None, normalize_syndromes=normalize
    )
    optimized = sampling_mode.compile(_REFERENCE_NOISE_SOURCE, normalize_syndromes=normalize)
    for program in (reference, optimized):
        result = sampling_mode.sample(program, 129, seed=1)
        np.testing.assert_array_equal(result.measurements, 0)
        np.testing.assert_array_equal(result.detectors, int(normalize))
        np.testing.assert_array_equal(result.observables, int(normalize))


def test_probability_one_noise_preserves_public_reference_syndrome() -> None:
    hir = clifft.trace(clifft.parse(_REFERENCE_NOISE_SOURCE))
    expected = clifft.compute_reference_syndrome(hir)
    assert expected == {"detectors": [1], "observables": [1]}
    clifft.default_hir_pass_manager().run(hir)
    assert clifft.compute_reference_syndrome(hir) == expected


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
    program = clifft.compile(
        source,
        hir_passes=_manager(clifft.PhasePolynomialPass()),
    )
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


@pytest.mark.parametrize("cap", [0, 1, 2, 3, 4, 32, 64, 65, 128])
def test_phase_reduction_variable_limits_preserve_unitaries(cap: int) -> None:
    source = "\n".join(["H 0 1 2 3", *_identity_parities(), *_parity_instructions(15)])
    phase = clifft.PhasePolynomialPass(max_variables=cap)
    program = clifft.compile(source, hir_passes=_manager(phase))
    assert phase.output_t_count <= phase.input_t_count
    if cap == 0:
        assert not phase.applied
        assert program.peak_active_width == 4
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))


@pytest.mark.parametrize("cap", [-1, 129, 2**32])
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
    sampling_mode: SamplingMode,
    kind: str,
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
        source,
        hir_passes=_manager(clifft.PhasePolynomialPass()),
        normalize_syndromes=False,
        postselection_mask=[1],
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
    importance_sampling_mode: CpuSamplingMode,
    k: int,
    kind: str,
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


@pytest.mark.parametrize("prefix", ["", "X 1", "Y 1", "R_Y(0.137) 1", "H 1\nT 1"])
@pytest.mark.parametrize("suffix", ["", "R_PAULI(0.25) X1", "H 0\nT 0\nS 1\nCX 1 0\nT 1"])
def test_known_phase_relations_preserve_prepared_states(prefix: str, suffix: str) -> None:
    source = prefix + "\nR_PAULI(0.25) X0\nR_PAULI(0.25) X0*Z1\n" + suffix
    phase = clifft.PhasePolynomialPass()
    program = clifft.compile(source, hir_passes=_manager(phase))
    if prefix in ("", "X 1", "Y 1"):
        assert phase.applied
    assert_statevectors_equiv(clifft.get_statevector(program), unitary_reference(source))


@lru_cache(maxsize=16)
def _entry_reference(kind: str, force: bool | None = None) -> tuple[str, np.ndarray]:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import pauli_error

    circuit = QuantumCircuit(5)
    source: list[str] = []

    def noise() -> None:
        probability = 1.0 if kind == "deterministic" else 0.23
        source.append(f"X_ERROR({probability}) 1")
        if force is not None:
            probability = float(force)
        circuit.append(
            pauli_error([("X", probability), ("I", 1 - probability)]).to_instruction(), [1]
        )

    def measure(q: int, axis: str, record: int) -> None:
        source.append(f"M{axis if axis != 'Z' else ''} {q}")
        if axis == "Y":
            circuit.sdg(q)
        if axis != "Z":
            circuit.h(q)
        circuit.cx(q, record)
        if axis != "Z":
            circuit.h(q)
        if axis == "Y":
            circuit.s(q)

    if kind == "before":
        noise()
    source.append("R_PAULI(0.25) X0")
    circuit.rx(np.pi / 4, 0)
    if kind in ("hoisted", "deterministic"):
        noise()
    elif kind in ("measure", "feedback"):
        measure(1, "X", 2)
        if kind == "feedback":
            source.append("CX rec[-1] 0")
            circuit.cx(2, 0)
    source.append("R_PAULI(0.25) X0*Z1")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(np.pi / 4, 1)
    circuit.cx(0, 1)
    circuit.h(0)
    if kind == "after":
        noise()
    source.append("R_PAULI(0.25) X1")
    circuit.rx(np.pi / 4, 1)
    if kind not in ("measure", "feedback"):
        measure(1, "Z", 2)
    measure(0, "Y", 3)
    measure(1, "X", 4)
    source.extend(["DETECTOR rec[-3] rec[-2]", "OBSERVABLE_INCLUDE(0) rec[-1]"])
    circuit.save_probabilities([2, 3, 4])
    result = AerSimulator(method="density_matrix").run(circuit, shots=1).result()
    assert result.success
    expected = np.asarray(result.data()["probabilities"])
    expected[np.abs(expected) < 1e-14] = 0
    np.testing.assert_allclose(expected.sum(), 1, atol=1e-12)
    return "\n".join(source), expected


@pytest.mark.parametrize(
    "kind", ["ideal", "before", "hoisted", "deterministic", "after", "measure", "feedback"]
)
def test_known_phase_constraints_preserve_joint_records(
    sampling_mode: SamplingMode, kind: str
) -> None:
    source, expected = _entry_reference(kind)
    phase = clifft.PhasePolynomialPass()
    program = sampling_mode.compile(source, hir_passes=_manager(phase), normalize_syndromes=False)
    assert phase.applied == (kind in ("ideal", "after"))
    samples = sampling_mode.sample(program, shots=16384, seed=891)
    assert_joint_distribution(samples.measurements, expected)
    selected = sampling_mode.compile(
        source,
        hir_passes=_manager(clifft.PhasePolynomialPass()),
        postselection_mask=[1],
        normalize_syndromes=False,
    )
    survivors = sampling_mode.sample_survivors(selected, shots=16384, seed=892)
    probability = sum(p for record, p in enumerate(expected) if ((record >> 1) ^ record) & 1 == 0)
    assert abs(survivors.passed_shots / survivors.total_shots - probability) < binomial_tolerance(
        probability, survivors.total_shots
    )


@pytest.mark.parametrize("kind", ["before", "hoisted", "after"])
@pytest.mark.parametrize("k", [0, 1])
def test_known_phase_constraints_preserve_forced_faults(
    importance_sampling_mode: CpuSamplingMode, kind: str, k: int
) -> None:
    source, expected = _entry_reference(kind, force=bool(k))
    program = importance_sampling_mode.compile(
        source, hir_passes=_manager(clifft.PhasePolynomialPass())
    )
    samples = importance_sampling_mode.sample_k(program, shots=16384, k=k, seed=893)
    assert_joint_distribution(samples.measurements, expected)


def test_postselected_single_record_detector_recovers_phase_constraint(
    sampling_mode: SamplingMode,
) -> None:
    source = "\n".join(
        [
            "H 0",
            "DEPOLARIZE1(0.05) 0 1",
            "T 0",
            "CX 1 0",
            "T_DAG 0",
            "CX 1 0",
            "M 1",
            "DETECTOR rec[-1]",
            "H 0",
            "M 0",
            "OBSERVABLE_INCLUDE(0) rec[-1]",
        ]
    )
    unselected_phase = clifft.PhasePolynomialPass()
    unselected = sampling_mode.compile(source, hir_passes=_manager(unselected_phase))
    assert not unselected_phase.applied
    assert unselected.peak_active_width == 1

    selected_phase = clifft.PhasePolynomialPass()
    selected = sampling_mode.compile(
        source,
        hir_passes=_manager(selected_phase),
        postselection_mask=[1],
    )
    assert selected_phase.applied
    assert selected_phase.output_t_count == 0
    assert selected.peak_active_width == 0

    reference = sampling_mode.compile(source, hir_passes=None, postselection_mask=[1])
    opt_survivors = sampling_mode.sample_survivors(selected, shots=8192, seed=901)
    ref_survivors = sampling_mode.sample_survivors(reference, shots=8192, seed=901)
    assert opt_survivors.passed_shots == ref_survivors.passed_shots
    np.testing.assert_array_equal(opt_survivors.observables, ref_survivors.observables)


def test_postselected_upstream_and_difference_detectors_recover_phase_constraints(
    sampling_mode: SamplingMode,
) -> None:
    source = "\n".join(
        [
            "RX 0",
            "R 1 2 3",
            "X 1",
            "CX 0 1",
            "DEPOLARIZE1(0.05) 0 1",
            "CX 0 2 1 2",
            "M 2",
            "DETECTOR rec[-1]",
            "DEPOLARIZE1(0.05) 3",
            "CX 0 3",
            "T 0 1",
            "T_DAG 0",
            "T 3",
            "CX 0 3 0 1",
            "MX 0",
            "M 3",
            "DETECTOR rec[-3] rec[-1]",
            "OBSERVABLE_INCLUDE(0) rec[-2]",
        ]
    )
    selected_phase = clifft.PhasePolynomialPass()
    selected = sampling_mode.compile(
        source,
        hir_passes=_manager(selected_phase),
        postselection_mask=[1, 1],
        normalize_syndromes=True,
    )
    assert selected_phase.applied
    assert selected_phase.output_t_count == 0
    assert selected.peak_active_width == 0

    reference = sampling_mode.compile(
        source,
        hir_passes=None,
        postselection_mask=[1, 1],
        normalize_syndromes=True,
    )
    opt_survivors = sampling_mode.sample_survivors(selected, shots=8192, seed=902)
    ref_survivors = sampling_mode.sample_survivors(reference, shots=8192, seed=902)
    assert opt_survivors.passed_shots == ref_survivors.passed_shots
    np.testing.assert_array_equal(opt_survivors.observables, ref_survivors.observables)


@pytest.mark.parametrize(
    "probe,expect_reduced,expected_ev", [("X0", False, 0.5), ("Z0", True, 0.0)]
)
def test_postselected_lookahead_preserves_intervening_exp_val(
    sampling_mode: SamplingMode, probe: str, expect_reduced: bool, expected_ev: float
) -> None:
    source = "\n".join(
        [
            "H 0 1",
            "DEPOLARIZE1(0.05) 1",
            "T 0",
            "CX 1 0",
            "T_DAG 0",
            "CX 1 0",
            f"EXP_VAL {probe}",
            "M 1",
            "DETECTOR rec[-1]",
        ]
    )
    phase = clifft.PhasePolynomialPass()
    optimized = sampling_mode.compile(
        source,
        hir_passes=_manager(phase),
        postselection_mask=[1],
    )
    assert phase.applied == expect_reduced
    assert phase.output_t_count == (0 if expect_reduced else 2)
    reference = sampling_mode.compile(source, hir_passes=None, postselection_mask=[1])
    opt_survivors = sampling_mode.sample_survivors(
        optimized, shots=512, seed=903, keep_records=True
    )
    ref_survivors = sampling_mode.sample_survivors(
        reference, shots=512, seed=903, keep_records=True
    )
    assert opt_survivors.passed_shots > 0
    np.testing.assert_allclose(opt_survivors.exp_vals, expected_ev, atol=1e-12)
    np.testing.assert_allclose(ref_survivors.exp_vals, expected_ev, atol=1e-12)


def test_postselected_circuit_honors_explicit_max_variables_cap() -> None:
    targets = " ".join(str(q) for q in range(66))
    source = f"MX {targets}\nM {targets}\nT {targets}\nT {targets}\nM 66\nDETECTOR rec[-1]\n"
    capped = clifft.PhasePolynomialPass(max_variables=65)
    capped_manager = clifft.HirPassManager()
    capped_manager.add(capped)
    clifft.compile(source, hir_passes=capped_manager, postselection_mask=[1])
    assert capped.wide_rewrites == 0
    assert capped.output_t_count == 132

    default_phase = clifft.PhasePolynomialPass()
    default_manager = clifft.HirPassManager()
    default_manager.add(default_phase)
    clifft.compile(source, hir_passes=default_manager, postselection_mask=[1])
    assert default_phase.wide_rewrites == 1
    assert default_phase.output_t_count == 0


def test_trace_and_lower_propagate_postselection_contract() -> None:
    source = "\n".join(
        [
            "H 0 2",
            "X 1",
            "DEPOLARIZE1(0.05) 1",
            "T 0",
            "CX 1 0",
            "T_DAG 0",
            "CX 1 0",
            "S_DAG 0",
            "M 1",
            "DETECTOR rec[-1]",
            "MX 0 2",
            "OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]",
        ]
    )
    hir = clifft.trace(
        clifft.parse(source),
        postselection_mask=[1],
        normalize_syndromes=True,
    )
    assert hir.postselection_mask == [1]
    assert hir.expected_detectors == [1]
    assert hir.expected_observables == [0]

    phase = clifft.PhasePolynomialPass()
    _manager(phase).run(hir)
    assert phase.applied
    assert phase.output_t_count == 0

    with pytest.raises(ValueError, match="postselection_mask omits detectors"):
        clifft.lower(hir, postselection_mask=[])
    with pytest.raises(ValueError, match="postselection_mask omits detectors"):
        clifft.lower(hir, postselection_mask=[0])
    with pytest.raises(ValueError, match="expected_detectors conflicts"):
        clifft.lower(hir, expected_detectors=[0])

    lowered = clifft.lower(hir)
    compiled = clifft.compile(
        source,
        postselection_mask=[1],
        normalize_syndromes=True,
    )
    low_survivors = clifft.sample_survivors(lowered, shots=2048, seed=904)
    cmp_survivors = clifft.sample_survivors(compiled, shots=2048, seed=904)
    assert low_survivors.passed_shots == cmp_survivors.passed_shots
    np.testing.assert_array_equal(low_survivors.observables, cmp_survivors.observables)


@pytest.mark.parametrize(
    "intervening_meas,expect_reduced,expected_ev",
    [("MX 0", False, 1.0 / 3.0), ("M 1", True, 1.0)],
)
def test_postselected_lookahead_preserves_measurement_conditioned_exp_val(
    sampling_mode: SamplingMode,
    intervening_meas: str,
    expect_reduced: bool,
    expected_ev: float,
) -> None:
    source = "\n".join(
        [
            "H 0 1",
            "Z_ERROR(0.1) 1",
            "T 0",
            "CX 1 0",
            "T_DAG 0",
            "CX 1 0",
            intervening_meas,
            "EXP_VAL Z1",
            "M 1",
            "DETECTOR rec[-1]",
        ]
    )
    phase = clifft.PhasePolynomialPass()
    optimized = sampling_mode.compile(
        source,
        hir_passes=_manager(phase),
        postselection_mask=[1],
        normalize_syndromes=False,
    )
    assert phase.applied == expect_reduced
    assert phase.output_t_count == (0 if expect_reduced else 2)
    default_prog = sampling_mode.compile(
        source,
        postselection_mask=[1],
        normalize_syndromes=False,
    )
    reference = sampling_mode.compile(
        source,
        hir_passes=None,
        postselection_mask=[1],
        normalize_syndromes=False,
    )
    for prog in (optimized, default_prog, reference):
        survivors = sampling_mode.sample_survivors(prog, shots=512, seed=905, keep_records=True)
        assert survivors.passed_shots > 0
        np.testing.assert_allclose(survivors.exp_vals, expected_ev, atol=1e-12)


def test_postselected_specialization_preserves_public_reference_syndrome() -> None:
    source = "\n".join(
        [
            "H 0",
            "X 1",
            "X_ERROR(0.1) 1",
            "T 0",
            "CX 1 0",
            "T_DAG 0",
            "CX 1 0",
            "T 0",
            "CX 1 0",
            "T_DAG 0",
            "CX 1 0",
            "M 1",
            "DETECTOR rec[-1]",
            "MX 0",
            "OBSERVABLE_INCLUDE(0) rec[-1]",
        ]
    )
    for pass_ in (clifft.PhasePolynomialPass(), clifft.RotationSimplificationPass()):
        hir = clifft.trace(clifft.parse(source), postselection_mask=[1])
        expected = clifft.compute_reference_syndrome(hir)
        assert expected == {"detectors": [1], "observables": [1]}
        assert hir.num_t_gates == 4
        manager = clifft.HirPassManager()
        manager.add(pass_)
        manager.run(hir)
        assert hir.num_t_gates == 0
        assert clifft.compute_reference_syndrome(hir) == expected
        hir.postselection_mask = []
        with pytest.raises(ValueError, match="postselection_mask omits detectors"):
            clifft.lower(hir)


@pytest.mark.parametrize("second_mask", [[1, 1], [0, 1]])
@pytest.mark.parametrize(
    "pass_factory,downstream_detector",
    [
        (clifft.PhasePolynomialPass, True),
        (clifft.PhasePolynomialPass, False),
        (clifft.RotationSimplificationPass, False),
    ],
)
def test_successive_postselected_specializations_retain_cumulative_assumptions(
    second_mask: list[int],
    pass_factory: type,
    downstream_detector: bool,
) -> None:
    if downstream_detector:
        source = "\n".join(
            [
                "H 0 2",
                "X_ERROR(0.2) 1 3",
                "T 0",
                "CX 1 0",
                "T_DAG 0",
                "CX 1 0",
                "M 1",
                "DETECTOR rec[-1]",
                "T 2",
                "CX 3 2",
                "T_DAG 2",
                "CX 3 2",
                "M 3",
                "DETECTOR rec[-1]",
                "MX 0 2",
            ]
        )
    else:
        source = "\n".join(
            [
                "H 0 2",
                "X_ERROR(0.2) 1 3",
                "M 1",
                "DETECTOR rec[-1]",
                "T 0",
                "CX 1 0",
                "T_DAG 0",
                "CX 1 0",
                "M 3",
                "DETECTOR rec[-1]",
                "T 2",
                "CX 3 2",
                "T_DAG 2",
                "CX 3 2",
                "MX 0 2",
            ]
        )
    hir = clifft.trace(clifft.parse(source), postselection_mask=[1, 0])
    for mask, expected_t_count in (([1, 0], 2), (second_mask, 0)):
        hir.postselection_mask = mask
        manager = clifft.HirPassManager()
        manager.add(pass_factory())
        manager.run(hir)
        assert hir.num_t_gates == expected_t_count

    for weakened_mask in ([], [1, 0], [0, 1]):
        hir.postselection_mask = weakened_mask
        with pytest.raises(ValueError, match="postselection_mask omits detectors"):
            clifft.lower(hir)

    hir.postselection_mask = [1, 1]
    hir.expected_detectors = [1, 0]
    with pytest.raises(ValueError, match="expected_detectors conflicts"):
        _manager(pass_factory()).run(hir)
    with pytest.raises(ValueError, match="expected_detectors conflicts"):
        clifft.lower(hir)

    hir.expected_detectors = []
    lowered = clifft.lower(hir)
    reference = clifft.compile(source, postselection_mask=[1, 1], hir_passes=None)
    low_survivors = clifft.sample_survivors(lowered, shots=1024, seed=906, keep_records=True)
    ref_survivors = clifft.sample_survivors(reference, shots=1024, seed=906, keep_records=True)
    assert low_survivors.passed_shots == ref_survivors.passed_shots
    np.testing.assert_array_equal(low_survivors.measurements, ref_survivors.measurements)
