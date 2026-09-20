"""Independently verify the f7 growth isometry and verified-cat instrument."""

import pytest
import stim
from folded_msc_f7 import F7Builder
from folded_msc_family import Surface
from test_folded_msc_family import logicals, pauli
from validate_folded_msc import bind_faults, noise_labels


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_regular_growth_preserves_the_encoded_qubit(axis):
    small, large, width = Surface(5), Surface(7), 99
    mapping = {q: large.index[x + 2, y + 2] for q, (x, y) in enumerate(small.points)}
    checks = [pauli(a, [mapping[q] for q in row], width) for a in "XZ" for row in small.checks(a)]
    logical = stim.PauliString(width)
    original = logicals(small, 41)[axis]
    for q, value in enumerate(original):
        if value:
            logical[mapping[q]] = value
    logical.sign = original.sign
    checks.append(logical)
    checks.extend(pauli("Z", [q], width) for q in range(width) if q not in mapping.values())
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(stim.Tableau.from_stabilizers(checks).inverse())
    builder = F7Builder(0)
    builder.growth_to_seven()
    for op in bind_faults(builder.circuit, {}).operations:
        sim.do(stim.Circuit(op.text()))
    assert all(
        sim.peek_observable_expectation(pauli(a, row, width)) == 1
        for a in "XZ"
        for row in large.checks(a)
    )
    assert sim.peek_observable_expectation(logicals(large, width)[axis]) == 1


def test_verified_cat_accepts_both_uniform_outcomes_and_limits_single_fault_spread():
    builder = F7Builder(0.001)
    builder.cat_preparation(noisy=True)
    choices = [
        (i, label)
        for i, op in enumerate(builder.circuit.operations)
        if op.probability is not None
        for label in noise_labels(op)
    ]
    seen = set()
    for seed, choice in enumerate([*[None] * 32, *choices]):
        physical = bind_faults(builder.circuit, {} if choice is None else {choice[0]: choice[1]})
        sim = stim.TableauSimulator(seed=seed)
        first_readout = True
        for op in physical.operations:
            if first_readout and op.name == "M":
                assert sim.peek_z(op.qubits[0]) == 0
                first_readout = False
            sim.do(stim.Circuit(op.text()))
        records = tuple(sim.current_measurement_record())
        if choice is None:
            seen.add(records)
            assert len(set(records)) == 1
        if len(set(records)) != 1:
            continue
        a = builder.ancilla
        assert abs(sim.peek_observable_expectation(pauli("X", range(a, a + 8), 99))) == 1
        signs = [
            sim.peek_observable_expectation(pauli("Z", [a, q], 99)) for q in range(a + 1, a + 8)
        ]
        assert all(abs(s) == 1 for s in signs)
        weight = sum(s == -1 for s in signs)
        assert min(weight, 8 - weight) <= (0 if choice is None else 1)
    assert seen == {(False,) * 6, (True,) * 6}
