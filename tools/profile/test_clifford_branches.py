import importlib.util
import unittest

import numpy as np
from fold_cultivation import Operation, Reconstruction, logical
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

HAS_CIRQ = importlib.util.find_spec("cirq") is not None
if HAS_CIRQ:
    from clifford_branches import (
        BranchState,
        logical_tail_history,
        materialize,
        paired_hook_histories,
        run_history,
        value,
    )


def dense_reference(width, stages):
    state = Statevector.from_int(0, 1 << width)
    for stage in stages:
        qc = QuantumCircuit(width)
        for op in stage:
            if op.name in {"M", "R"}:
                state = state.evolve(qc)
                qc = QuantumCircuit(width)
                # The fixtures use only fresh or previously measured-zero resets.
                projected = state.evolve(Operator([[1, 0], [0, 0]]), [op.targets[0]])
                if op.name == "R":
                    np.testing.assert_allclose(projected.data, state.data, atol=1e-12)
                state = projected
            else:
                name = "tdg" if op.name == "T_DAG" else op.name.lower()
                getattr(qc, name)(*op.targets)
        state = state.evolve(qc)
    return state.data


def dense_branches(state):
    result = np.zeros(1 << state.width, dtype=complex)
    for term in state.terms:
        # Cirq uses big-endian axes; Clifft/Qiskit qubit zero is least significant.
        vector = term.ch.to_state_vector().reshape([2] * state.width).transpose().reshape(-1)
        result += value(term.coefficient) * vector
    return result


@unittest.skipUnless(HAS_CIRQ, "offline composition reference requires cirq-core==1.6.1")
class CompositionTest(unittest.TestCase):
    def test_phase_sensitive_merging_and_overlaps_against_dense_reference(self):
        rng = np.random.default_rng(923)
        for basis in ("computational", "pauli"):
            for _ in range(12):
                state = BranchState(3, basis)
                operations = [Operation("H", (0,)), Operation("T", (0,))]
                for _ in range(16):
                    name = str(rng.choice(["H", "S", "X", "Y", "Z", "CX", "T"]))
                    targets = tuple(
                        int(q) for q in rng.choice(3, 2 if name == "CX" else 1, replace=False)
                    )
                    operations.append(Operation(name, targets))
                operations.append(Operation("M", (int(rng.integers(3)),)))
                for op in operations:
                    state.operation(op)
                state.merge()
                expected = dense_reference(3, [operations])
                np.testing.assert_allclose(dense_branches(state), expected, atol=1e-12)
                self.assertAlmostEqual(state.expectation(), float(np.vdot(expected, expected).real))

    def test_exact_destructive_interference_removes_all_terms(self):
        state = BranchState(1)
        for name in ["H", "T", "T", "T", "T", "H", "M"]:
            state.operation(Operation(name, (0,)))
        self.assertEqual(state.terms, [])
        self.assertEqual(state.expectation(), 0)

    def test_complete_f3_fixed_fault_histories_match_unnormalized_output(self):
        r = Reconstruction(3).build()
        # Keep the full output state and zero branches, not just accepted ratios.
        for probability, seed in [(0, 1), (0.006, 42), (0.006, 84), (0.006, 126)]:
            stages, _ = materialize(r, probability, seed)
            expected = dense_reference(16, stages)
            for basis in ("computational", "pauli"):
                state, _ = run_history(r, stages, basis)
                np.testing.assert_allclose(dense_branches(state), expected, atol=2e-12)
                self.assertAlmostEqual(state.expectation(), float(np.vdot(expected, expected).real))

    def test_paired_cat_faults_preserve_fractional_acceptance(self):
        r = Reconstruction(3).build()
        label, stages = paired_hook_histories(r)[0]
        expected = dense_reference(16, stages)
        probability = float(np.vdot(expected, expected).real)
        self.assertGreater(probability, 0)
        self.assertLess(probability, 1)
        state, _ = run_history(r, stages)
        np.testing.assert_allclose(dense_branches(state), expected, atol=2e-12, err_msg=label)
        self.assertAlmostEqual(state.expectation(), probability)

    def test_accepted_logical_error_is_not_replaced_by_the_target_state(self):
        r = Reconstruction(3).build()
        stages = logical_tail_history(r)
        expected = dense_reference(16, stages)
        self.assertAlmostEqual(float(np.vdot(expected, expected).real), 1)
        for basis in ("computational", "pauli"):
            state, _ = run_history(r, stages, basis)
            np.testing.assert_allclose(dense_branches(state), expected, atol=2e-12)
            self.assertAlmostEqual(state.expectation(logical(3, "X")), -1 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
