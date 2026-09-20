"""Check code growth, flagged cats and complete physical f5 cultivation."""

from collections import Counter
from pathlib import Path

import pytest
import stim
from folded_msc import FOLD, FOLD_PAIRS, POINTS, X_CHECKS, Z_CHECKS
from folded_msc_family import F5Builder, Surface, make_circuit
from validate_folded_msc import bind_faults, noise_labels

import clifft


def pauli(axis, support, n=47):
    result = stim.PauliString(n)
    for q in support:
        result[q] = axis
    return result


def logicals(surface, n=47):
    end = 2 * surface.distance - 1
    x = pauli("X", [surface.index[0, y] for y in range(0, end, 2)], n)
    z = pauli("Z", [surface.index[x, 0] for x in range(0, end, 2)], n)
    return {"X": x, "Y": 1j * x * z, "Z": z}


def test_geometry_preserves_small_reference():
    surface = Surface(3)
    assert surface.points == POINTS
    assert surface.checks("X") == X_CHECKS
    assert surface.checks("Z") == Z_CHECKS
    assert surface.fold == FOLD
    assert surface.pairs == FOLD_PAIRS
    reference = Path("tools/profile/fixtures/folded_msc/f3_fed_p1e-3.stim")
    assert make_circuit(0.001, distance=3).text() == reference.read_text()


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_growth_preserves_the_encoded_qubit(axis):
    small, large = Surface(3), Surface(5)
    mapping = {q: large.index[2 * x, 2 * y] for q, (x, y) in enumerate(POINTS)}
    checks = [pauli(a, [mapping[q] for q in row]) for a in ("X", "Z") for row in small.checks(a)]
    logical = stim.PauliString(47)
    for q, value in enumerate(logicals(small)[axis]):
        if value:
            logical[mapping[q]] = value
    logical.sign = logicals(small)[axis].sign
    checks.append(logical)
    checks.extend(pauli("Z", [q]) for q in range(47) if q not in mapping.values())
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(stim.Tableau.from_stabilizers(checks).inverse())
    builder = F5Builder(0)
    builder.growth_to_rotated()
    builder.growth_to_regular()
    for op in bind_faults(builder.circuit, {}).operations:
        sim.do(stim.Circuit(op.text()))
    for a in ("X", "Z"):
        for row in large.checks(a):
            assert sim.peek_observable_expectation(pauli(a, row)) == 1
    assert sim.peek_observable_expectation(logicals(large)[axis]) == 1


def test_folded_clifford_has_the_expected_logical_action():
    surface = Surface(5)
    checks = [pauli(a, row, 41) for a in ("X", "Z") for row in surface.checks(a)]
    logical = logicals(surface, 41)
    for before, after, sign in (("X", "Y", 1), ("Y", "X", 1), ("Z", "Z", -1)):
        sim = stim.TableauSimulator()
        sim.set_inverse_tableau(stim.Tableau.from_stabilizers(checks + [logical[before]]).inverse())
        for j, q in enumerate(surface.fold):
            sim.x(q)
            sim.do(stim.Circuit(f"{'S' if j % 2 == 0 else 'S_DAG'} {q}"))
        for pairs in surface.pairs:
            for a, b in pairs:
                sim.cz(a, b)
        assert all(sim.peek_observable_expectation(check) == 1 for check in checks)
        assert sim.peek_observable_expectation(logical[after]) == sign


def test_flagged_cat_detects_single_fault_spread():
    builder = F5Builder(0.001)
    builder.cat_preparation(noisy=True)
    source, a = builder.circuit, builder.ancilla
    choices = [
        (i, label)
        for i, op in enumerate(source.operations)
        if op.probability is not None
        for label in noise_labels(op)
    ]
    accepted = 0
    for choice in [None] + choices:
        sim = stim.TableauSimulator()
        physical = bind_faults(source, {} if choice is None else {choice[0]: choice[1]})
        for op in physical.operations:
            sim.do(stim.Circuit(op.text()))
        if sim.current_measurement_record()[0]:
            continue
        accepted += 1
        assert abs(sim.peek_observable_expectation(pauli("X", range(a, a + 5)))) == 1
        signs = [sim.peek_observable_expectation(pauli("Z", (a, q))) for q in range(a + 1, a + 5)]
        assert all(sign in (-1, 1) for sign in signs)
        flipped = sum(sign == -1 for sign in signs)
        assert min(flipped, 5 - flipped) <= (0 if choice is None else 1)
    assert 0 < accepted < len(choices)


def test_complete_noiseless_f5_and_published_resource_counts():
    source = make_circuit(0)
    counts = Counter(op.name for op in make_circuit(0, evaluation="none").operations)
    assert sum(counts[gate] for gate in ("T", "T_DAG", "CCZ")) == 97
    assert source.manifest()["num_qubits"] == 47
    result = clifft.sample(clifft.compile(source.text()), 4, seed=177, threads=1)
    assert result.measurements.shape == (4, 151)
    assert result.detectors.shape == (4, 150)
    assert not result.detectors.any()
    assert not result.observables.any()


def test_growth_does_not_reset_existing_data():
    builder = F5Builder(0)
    builder.growth_to_rotated()
    builder.growth_to_regular()
    existing = {builder.surface.index[2 * x, 2 * y] for x, y in POINTS}
    reset = [op.qubits[0] for op in builder.circuit.operations if op.name == "R"]
    assert len(reset) == len(set(reset)) == 28
    assert existing.isdisjoint(reset)
    assert existing | set(reset) == set(range(41))


def test_noisy_controlled_factors_against_dense_aer():
    pytest.importorskip("cirq")
    import itertools

    import numpy as np
    from clifford_gadget import Atom, CoherentState
    from folded_msc import Operation
    from folded_msc_oracle import conjugated_cx
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuits, expected = [], []
    for first, bit, before, control_fault, after in itertools.product(
        ("T", "T_DAG"), (0, 1), "IXYZ", "IXYZ", "IXYZ"
    ):
        # Entangling the data with a reference checks the whole local map,
        # including coherent phases, instead of just one input state.
        qc, state = QuantumCircuit(3), CoherentState(3)
        if bit:
            qc.x(0)
            state.gate(Atom("X", (0,)))
        qc.h(1)
        qc.cx(1, 2)
        state.gate(Atom("H", (1,)))
        state.gate(Atom("CX", (1, 2)))
        last = "T_DAG" if first == "T" else "T"
        operations = [Operation(first, (1,), "test", "dense reference")]
        operations += [Operation(before, (1,), "test", "dense reference")] if before != "I" else []
        operations += [Operation("CX", (0, 1), "test", "dense reference")]
        if control_fault != "I":
            operations += [Operation(control_fault, (0,), "test", "dense reference")]
        operations += [Operation(after, (1,), "test", "dense reference")] if after != "I" else []
        operations += [Operation(last, (1,), "test", "dense reference")]
        for op in operations:
            getattr(qc, {"T_DAG": "tdg"}.get(op.name, op.name.lower()))(*op.qubits)
        qc.save_statevector()
        circuits.append(qc)
        conjugated_cx(state, operations, 3)
        term = state.terms[0]
        expected.append(
            np.array([term.coefficient * term.ch.inner_product_of_state_and_x(i) for i in range(8)])
        )
    results = AerSimulator(method="statevector", max_parallel_threads=1).run(circuits).result()
    reverse_bits = [int(f"{i:03b}"[::-1], 2) for i in range(8)]
    for i, vector in enumerate(expected):
        np.testing.assert_allclose(
            np.asarray(results.get_statevector(i))[reverse_bits], vector, atol=1e-12
        )
