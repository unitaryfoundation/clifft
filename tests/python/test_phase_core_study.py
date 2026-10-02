"""Independent checks of the Clifford-core and selected Pauli-fault study."""

import itertools
import random
from functools import lru_cache

import numpy as np
import pytest
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from utils_conformance import unitary_reference
from utils_qiskit import stim_to_qiskit_noiseless

import clifft
from tools.prototypes.phase_core_study import (
    Check,
    compress_phase,
    defer_fixed_faults,
    deferred_pauli,
    is_diagonal_clifford,
    single_t_parity,
    stabilizer_kernel,
    trace,
)
from tools.prototypes.phase_polynomial import _add


@lru_cache(maxsize=512)
def _aer_unitary(source: str, n: int) -> np.ndarray:
    lines = [f"Z {n - 1}", f"Z {n - 1}"]
    for line in source.splitlines():
        gate, *args = line.split()
        if gate == "I":
            continue
        if gate == "SWAP":
            a, b = args
            lines.extend([f"CX {a} {b}", f"CX {b} {a}", f"CX {a} {b}"])
        else:
            lines.append(line)
    circuit = stim_to_qiskit_noiseless("\n".join(lines))
    circuit.save_unitary()
    result = AerSimulator(method="unitary").run(circuit, shots=1).result()
    assert result.success
    return np.asarray(result.data()["unitary"], dtype=np.complex128)


def _aer_branches(source: str, n: int) -> dict[tuple[int, ...], np.ndarray]:
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1
    branches: dict[tuple[int, ...], np.ndarray] = {(): state}
    unitary = []
    for line in source.splitlines() + ["END"]:
        gate, *args = line.split()
        if gate.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
            continue
        if gate not in ("MPP", "MX", "END"):
            unitary.append(line)
            continue
        matrix = _aer_unitary("\n".join(unitary), n)
        branches = {record: matrix @ vector for record, vector in branches.items()}
        unitary = []
        products = args if gate == "MPP" else [f"X{q}" for q in args] if gate == "MX" else []
        for product in products:
            negative = product.startswith("!")
            axes = ["I"] * n
            for factor in product.lstrip("!").split("*"):
                axes[int(factor[1:])] = factor[0]
            pauli = Pauli("".join(axes[::-1])).to_matrix() * (-1 if negative else 1)
            updated = {}
            for record, vector in branches.items():
                action = pauli @ vector
                for bit in (0, 1):
                    projected = (vector + (-1) ** bit * action) / 2
                    if np.vdot(projected, projected).real > 1e-25:
                        updated[record + (bit,)] = projected
            branches = updated
    return branches


def test_stabilizer_kernel_matches_exhaustive_aer_paulis() -> None:
    rng = random.Random(8801)
    for _ in range(16):
        n = rng.randrange(1, 5)
        lines = ["H " + " ".join(map(str, range(n)))]
        for _ in range(20):
            gate = rng.choice(("T", "T_DAG", "S", "S_DAG", "Z", "CX", "CZ"))
            if n == 1 and gate in ("CX", "CZ"):
                continue
            targets = rng.sample(range(n), 2 if gate in ("CX", "CZ") else 1)
            lines.append(gate + " " + " ".join(map(str, targets)))
        source = "\n".join(lines)
        result = trace(source)
        kernel = stabilizer_kernel(result.phase, n)
        state = unitary_reference(source)
        compressed = defer_fixed_faults(source)
        actual = _aer_unitary(compressed, n)[:, 0]
        np.testing.assert_allclose(np.abs(np.vdot(state, actual)), 1, atol=1e-12)
        assert clifft.compile(compressed).peak_active_width <= n - len(kernel)
        stabilizer_count = 0
        for axes in itertools.product("IXYZ", repeat=n):
            action = Pauli("".join(axes)).to_matrix() @ state
            expectation = np.vdot(state, action)
            if abs(abs(expectation) - 1) < 1e-12:
                stabilizer_count += 1
        assert stabilizer_count == 1 << len(kernel)


def _single_core_source() -> str:
    lines = ["H 0 1 2 3", "MPP X0*X1", "DETECTOR rec[-1]"]
    # The fifteen distinct nonzero parities cancel collectively modulo eight.
    for mask in range(1, 16):
        qubits = [q for q in range(4) if mask & (1 << q)]
        target = qubits[-1]
        compute = [f"CX {q} {target}" for q in qubits[:-1]]
        lines.extend(compute + [f"T {target}"] + compute[::-1])
    lines.extend(
        [
            "CX 0 3",
            "CX 1 3",
            "CX 2 3",
            "T 3",
            "CX 2 3",
            "CX 1 3",
            "CX 0 3",
            "MPP X1*X2",
            "DETECTOR rec[-1]",
            "S 0",
            "CZ 0 1",
            "CX 3 4",
            "MPP X2*X3*X4",
            "DETECTOR rec[-1]",
            "MX 0 1 2 3 4",
            "OBSERVABLE_INCLUDE(0) rec[-1]",
        ]
    )
    return "\n".join(lines)


def _fault_source(seed: int) -> str:
    rng = random.Random(seed)
    lines = []
    for line in _single_core_source().splitlines():
        lines.append(line)
        if line.startswith(("H", "CX", "T", "S", "CZ", "MPP")):
            for _ in range(int(rng.random() < 0.3)):
                lines.append(f"{rng.choice('XYZ')} {rng.randrange(5)}")
    return "\n".join(lines)


def test_pauli_faults_change_only_diagonal_clifford_correction() -> None:
    ideal = trace(_single_core_source())
    ideal_kernel = stabilizer_kernel(ideal.phase, ideal.width)
    assert ideal.width - len(ideal_kernel) == 1
    assert single_t_parity(ideal.phase) == 15
    ideal_magic, _, ideal_encoding = compress_phase(ideal.phase, ideal.width)
    for seed in range(64):
        faulty = trace(_fault_source(seed))
        correction = dict(faulty.phase)
        for mask, coefficient in ideal.phase.items():
            _add(correction, mask, -coefficient)
        assert is_diagonal_clifford(correction)
        assert stabilizer_kernel(faulty.phase, faulty.width) == ideal_kernel
        assert single_t_parity(faulty.phase) == 15
        magic, _, encoding = compress_phase(faulty.phase, faulty.width)
        assert magic == ideal_magic
        assert encoding == ideal_encoding
        for record in faulty.records:
            if isinstance(record, Check):
                assert deferred_pauli(faulty.phase, record) is not None


def test_deferred_fault_paths_preserve_aer_joint_records_and_branch_states() -> None:
    saw_nondeterministic_check = False
    for seed in range(12):
        source = _fault_source(seed)
        rewritten = defer_fixed_faults(source)
        expected = _aer_branches(source, 5)
        actual = _aer_branches(rewritten, 5)
        assert expected.keys() == actual.keys()
        for record, state in expected.items():
            np.testing.assert_allclose(
                np.outer(actual[record], actual[record].conj()),
                np.outer(state, state.conj()),
                atol=1e-12,
            )
        check_marginal = np.zeros(2)
        for record, state in expected.items():
            check_marginal[record[1]] += np.vdot(state, state).real
        saw_nondeterministic_check |= bool(np.all(check_marginal > 0.1))
        program = clifft.compile(
            "\n".join(
                line
                for line in rewritten.splitlines()
                if not line.startswith(("DETECTOR", "OBSERVABLE_INCLUDE"))
            )
        )
        assert program.peak_active_width <= 1
        records = ["".join(map(str, record)) for record in itertools.product((0, 1), repeat=8)]
        oracle = [
            np.vdot(expected[tuple(map(int, record))], expected[tuple(map(int, record))]).real
            if tuple(map(int, record)) in expected
            else 0
            for record in records
        ]
        np.testing.assert_allclose(
            clifft.record_probabilities(program, records), oracle, atol=1e-12
        )
    assert saw_nondeterministic_check


def test_non_pauli_measurement_suffix_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a Pauli"):
        defer_fixed_faults("H 0\nMPP X0\nT 0\nMX 0")


def test_coupled_magic_can_require_more_than_one_core_qubit() -> None:
    result = trace("H 0 1 2\nT 0 1")
    assert result.width - len(stabilizer_kernel(result.phase, result.width)) == 2
    assert single_t_parity(result.phase) is None
