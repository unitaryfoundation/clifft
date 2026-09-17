import importlib.util
import unittest

import numpy as np
import stim
from fold_check import PhaseMonomial
from fold_contraction import Plan, compose, fixtures
from fold_cultivation import logical, pauli, stabilizers
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

HAS_CIRQ = importlib.util.find_spec("cirq") is not None
if HAS_CIRQ:
    from cirq import StabilizerStateChForm  # type: ignore[import-not-found]
    from clifford_branches import act


class ChReference:
    """Independent code preparation and complex overlap via Stim synthesis/CH."""

    def __init__(self, distance):
        self.width = distance**2 + (distance - 1) ** 2
        generators = [pauli(distance, axis, support) for axis, support in stabilizers(distance)]
        self.states = []
        self.inverses = []
        self.scales = []
        for bit in (0, 1):
            prep = stim.Tableau.from_stabilizers(
                generators + [(-1 if bit else 1) * logical(distance, "Z")]
            ).to_circuit()
            state = StabilizerStateChForm(self.width)
            self.circuit(state, prep)
            anchor = 0
            if bit:
                anchor = sum(
                    int(logical(distance, "X")[q] != 0) << (self.width - 1 - q)
                    for q in range(self.width)
                )
            amplitude = state.inner_product_of_state_and_x(anchor)
            self.scales.append(abs(amplitude) / amplitude)
            self.states.append(state)
            self.inverses.append(prep.inverse())

    @staticmethod
    def circuit(state, circuit):
        for instruction in circuit:
            targets = [t.value for t in instruction.targets_copy()]
            arity = 2 if instruction.name in {"CX", "CZ", "SWAP"} else 1
            for j in range(0, len(targets), arity):
                act(state, instruction.name, tuple(targets[j : j + arity]))

    def matrix(self, op):
        result = np.zeros((2, 2), dtype=complex)
        for source in (0, 1):
            state = self.states[source].copy()
            for q, power in enumerate(op.linear):
                for _ in range((power % 8) // 2):
                    act(state, "S", (q,))
            for q, r in sorted(op.edges):
                act(state, "CZ", (q, r))
            for q in range(op.width):
                if (op.flips >> q) & 1:
                    act(state, "X", (q,))
            for target in (0, 1):
                projected = state.copy()
                self.circuit(projected, self.inverses[target])
                result[target, source] = (
                    self.scales[source]
                    * self.scales[target].conjugate()
                    * np.exp(1j * np.pi * op.global_phase / 4)
                    * projected.inner_product_of_state_and_x(0)
                )
        return result


class ContractionTest(unittest.TestCase):
    def test_composition_keeps_internal_fault_phases(self):
        plan = Plan(3)
        cases = fixtures(plan, 8)
        for (_, before), (_, after) in zip(cases[1:9], cases[-8:], strict=True):
            combined = compose(after, before)
            for bits in range(1 << plan.width):
                middle, a = before.column(bits)
                output, b = after.column(middle)
                self.assertEqual(combined.column(bits), (output, (a + b) % 8))

    def test_small_code_against_dense_qiskit(self):
        plan = Plan(3)
        basis = np.zeros((1 << plan.width, 2), dtype=complex)
        for bits in range(1 << len(plan.x_masks)):
            physical = 0
            for g, mask in enumerate(plan.x_masks):
                if (bits >> g) & 1:
                    physical ^= mask
            for logical_bit in (0, 1):
                basis[physical ^ (plan.logical_x * logical_bit), logical_bit] = 1 / 8
        for label, op in fixtures(plan, 4):
            circuit = QuantumCircuit(plan.width)
            for q, power in enumerate(op.linear):
                for _ in range(power // 2):
                    circuit.s(q)
            for edge in sorted(op.edges):
                circuit.cz(*edge)
            for q in range(plan.width):
                if (op.flips >> q) & 1:
                    circuit.x(q)
            circuit.global_phase = op.global_phase * np.pi / 4
            expected = np.column_stack(
                [
                    basis.conj().T @ Statevector(basis[:, b].copy()).evolve(circuit).data
                    for b in (0, 1)
                ]
            )
            np.testing.assert_allclose(plan.matrix(op), expected, rtol=0, atol=2e-12, err_msg=label)

    @unittest.skipUnless(HAS_CIRQ, "large-code reference requires cirq-core==1.6.1")
    def test_all_sizes_and_complex_amplitudes_against_independent_ch(self):
        for distance in (3, 5, 7):
            plan = Plan(distance)
            reference = ChReference(distance)
            for label, op in fixtures(plan):
                np.testing.assert_allclose(
                    plan.matrix(op),
                    reference.matrix(op),
                    rtol=0,
                    atol=2e-12,
                    err_msg=f"d{distance} {label}",
                )
                # Check contraction even when X support would reject the branch.
                diagonal = PhaseMonomial(op.width, 0, op.global_phase, op.linear, op.edges)
                np.testing.assert_allclose(
                    plan.phase_sums(op),
                    np.diag(reference.matrix(diagonal)),
                    rtol=0,
                    atol=2e-12,
                    err_msg=f"d{distance} {label} phases",
                )

    def test_projection_must_not_be_inserted_between_consecutive_checks(self):
        plan = Plan(3)
        error = PhaseMonomial(plan.width, flips=1)
        np.testing.assert_array_equal(plan.matrix(error), np.zeros((2, 2)))
        np.testing.assert_allclose(plan.matrix(compose(error, error)), np.eye(2), atol=1e-12)

    def test_declines_unsupported_operator_family(self):
        plan = Plan(3)
        for op in [
            PhaseMonomial(12),
            PhaseMonomial(13, flips=1 << 13),
            PhaseMonomial(13, linear=[1] * 13),
            PhaseMonomial(13, edges={(0, 1)}),
        ]:
            with self.assertRaises(ValueError):
                plan.matrix(op)
        with self.assertRaises(ValueError):
            Plan(9)


if __name__ == "__main__":
    unittest.main()
