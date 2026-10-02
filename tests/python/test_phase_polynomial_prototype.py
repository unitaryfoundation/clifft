"""Independent oracles and record contracts for the opt-in source prototype."""

import itertools
import random

import numpy as np
import pytest
from conftest import assert_statevectors_equiv
from utils_conformance import SamplingMode, assert_joint_distribution, unitary_reference

import clifft
from tools.prototypes.phase_polynomial import reduce_phase_polynomial


def _oracle_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        gate, *args = line.split()
        if gate == "I":
            lines.extend([f"Z {args[0]}"] * 2)
        elif gate == "SWAP":
            a, b = args
            lines.extend([f"CX {a} {b}", f"CX {b} {a}", f"CX {a} {b}"])
        else:
            lines.append(line)
    return "\n".join(lines)


def _without_annotations(source: str) -> str:
    return "\n".join(
        line
        for line in source.splitlines()
        if not line.startswith(("DETECTOR", "OBSERVABLE_INCLUDE"))
    )


def _all_parities(width: int) -> str:
    lines = ["H " + " ".join(map(str, range(width)))]
    for mask in range(1, 1 << width):
        qubits = [q for q in range(width) if mask & (1 << q)]
        target = qubits[-1]
        compute = [f"CX {q} {target}" for q in qubits[:-1]]
        lines.extend(compute + [f"T {target}"] + compute[::-1])
    return "\n".join(lines)


def test_collective_distinct_parities_cancel() -> None:
    source = _all_parities(4)
    result = reduce_phase_polynomial(source)
    assert result.applied
    assert result.input_t_count == 15
    assert result.output_t_count == 0
    program = clifft.compile(result.circuit)
    assert program.peak_active_width == 0
    assert_statevectors_equiv(
        clifft.get_statevector(program), unitary_reference(_oracle_source(source))
    )


def test_random_prepared_states_match_aer() -> None:
    rng = random.Random(104731)
    for _ in range(32):
        n = rng.randrange(3, 8)
        prepared = rng.sample(range(n), rng.randrange(1, n + 1))
        lines = ["H " + " ".join(map(str, prepared)), f"I {n - 1}"]
        for _ in range(36):
            gate = rng.choice(("CX", "SWAP", "CZ", "T", "T_DAG", "S", "S_DAG", "Z"))
            args = rng.sample(range(n), 2 if gate in ("CX", "SWAP", "CZ") else 1)
            lines.append(gate + " " + " ".join(map(str, args)))
        source = "\n".join(lines)
        result = reduce_phase_polynomial(source, allow_t_expansion=True)
        assert result.applied, result.reason
        actual = clifft.get_statevector(clifft.compile(result.prepared_state))
        assert_statevectors_equiv(actual, unitary_reference(_oracle_source(source)))


def test_stabilizer_quotient_preserves_joint_records(sampling_mode: SamplingMode) -> None:
    source = "\n".join(
        [
            _all_parities(4),
            "CX 0 3",
            "CX 1 3",
            "CX 2 3",
            "T 3",
            "CX 2 3",
            "CX 1 3",
            "CX 0 3",
            "MPP X0*X1 X1*X2 X2*X3",
            "DETECTOR rec[-3]",
            "DETECTOR rec[-2]",
            "DETECTOR rec[-1]",
            "MX 0 1 2 3",
            "OBSERVABLE_INCLUDE(0) rec[-1]",
        ]
    )
    result = reduce_phase_polynomial(source)
    assert result.applied
    assert result.initial_width == 4
    assert result.logical_width == 1
    assert result.output_t_count == 1
    assert result.deterministic_records == 3
    reference = clifft.compile(source, normalize_syndromes=False)
    reduced = clifft.compile(result.circuit, normalize_syndromes=False)
    assert reduced.peak_active_width == 1
    assert reduced.num_qubits == reference.num_qubits == 4
    assert reduced.num_measurements == reference.num_measurements == 7
    assert reduced.num_detectors == reference.num_detectors == 3
    records = ["".join(bits) for bits in itertools.product("01", repeat=7)]
    exact_reference = clifft.compile(_without_annotations(source))
    exact_reduced = clifft.compile(_without_annotations(result.circuit))
    expected = clifft.record_probabilities(exact_reference, records)
    np.testing.assert_allclose(
        clifft.record_probabilities(exact_reduced, records), expected, atol=1e-12
    )
    # Histogram bins use the first record as the least-significant bit.
    expected_le = clifft.record_probabilities(
        exact_reference, [f"{i:07b}"[::-1] for i in range(128)]
    )
    mode_program = sampling_mode.compile(result.circuit, normalize_syndromes=False)
    sampled = sampling_mode.sample(mode_program, shots=8192, seed=61)
    assert_joint_distribution(sampled.measurements, expected_le)
    np.testing.assert_array_equal(sampled.detectors, 0)
    np.testing.assert_array_equal(sampled.observables[:, 0], sampled.measurements[:, -1])
    survivor_program = sampling_mode.compile(
        result.circuit, normalize_syndromes=False, postselection_mask=[1, 1, 1]
    )
    survivors = sampling_mode.sample_survivors(
        survivor_program, shots=8192, seed=71, keep_records=True
    )
    assert survivors.total_shots == survivors.passed_shots == 8192
    assert_joint_distribution(survivors.measurements, expected_le)


def test_prefix_check_does_not_imply_final_symmetry() -> None:
    source = "H 0 1\nMPP X0*X1\nT 0\nMX 0 1"
    result = reduce_phase_polynomial(source)
    assert result.applied
    assert result.logical_width == 2
    assert result.deterministic_records == 1
    records = [f"{i:03b}" for i in range(8)]
    np.testing.assert_allclose(
        clifft.record_probabilities(clifft.compile(result.circuit), records),
        clifft.record_probabilities(clifft.compile(source), records),
        atol=1e-12,
    )
    assert_statevectors_equiv(
        clifft.get_statevector(clifft.compile(result.prepared_state)),
        unitary_reference("H 0 1\nT 0"),
    )


def test_negative_check_preserves_its_record() -> None:
    result = reduce_phase_polynomial("H 0\nZ 0\nMX 0\nDETECTOR rec[-1]")
    assert result.applied
    assert "MPAD 1" in result.circuit
    np.testing.assert_allclose(
        clifft.record_probabilities(
            clifft.compile(_without_annotations(result.circuit)), ["0", "1"]
        ),
        [0, 1],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "source",
    [
        "H 0\nT 0\nX_ERROR(0.1) 0\nMX 0",
        "H 0\nZ_ERROR(0.1) 0\nT 0\nMX 0",
        "H 0\nT 0\nMX(0.1) 0",
        "H 0\nT 0\nH 0",
        "H 0\nT 0\nMX 0\nT 0",
        "H 0\nM 0\nCX rec[-1] 1",
        "H 0\nR 0",
        "REPEAT 2 {\nH 0\nT 0\n}",
        "H 0\nR_Z(0.123) 0",
        "H 0\nMPP Y0",
    ],
)
def test_unsupported_circuits_are_unchanged(source: str) -> None:
    result = reduce_phase_polynomial(source)
    assert not result.applied
    assert result.circuit == source
    assert result.reason


def test_variable_limit_leaves_input_unchanged() -> None:
    source = "H 0 1 2\nT 0"
    assert not reduce_phase_polynomial(source, max_variables=2).applied
    with pytest.raises(ValueError, match="max_variables"):
        reduce_phase_polynomial(source, max_variables=-1)


def test_cost_guard_keeps_a_compact_parity_rotation() -> None:
    source = "H 0 1 2 3\nCX 0 3\nCX 1 3\nCX 2 3\nT 3\nCX 2 3\nCX 1 3\nCX 0 3"
    guarded = reduce_phase_polynomial(source)
    assert not guarded.applied
    assert guarded.circuit == source
    assert "increase T count" in guarded.reason
    unguarded = reduce_phase_polynomial(source, allow_t_expansion=True)
    assert unguarded.applied
    assert unguarded.output_t_count > unguarded.input_t_count
    assert_statevectors_equiv(
        clifft.get_statevector(clifft.compile(unguarded.prepared_state)), unitary_reference(source)
    )


def test_pauli_noise_fallback_matches_stim(sampling_mode: SamplingMode) -> None:
    import stim

    source = "H 0\nCX 0 1\nZ_ERROR(0.2) 0\nMX 0 1"
    result = reduce_phase_polynomial(source)
    assert not result.applied
    expected = [0.4, 0.1, 0.1, 0.4]
    reference = stim.Circuit(source).compile_sampler(seed=19).sample(shots=8192)
    assert_joint_distribution(reference.astype(np.uint8), expected)
    program = sampling_mode.compile(result.circuit)
    actual = sampling_mode.sample(program, shots=8192, seed=29)
    assert_joint_distribution(actual.measurements, expected)
