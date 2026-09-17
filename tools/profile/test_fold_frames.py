"""Independent small-state validation of large-fold tensor bounds and checkpoints."""

import random
import unittest
from types import SimpleNamespace

import numpy as np
import stim
from clifford_branches import BranchState, paired_hook_histories, run_history, value
from fold_blocks import Protocol
from fold_cultivation import Operation, Reconstruction, logical
from fold_frame_certificate import common_stabilizers, intersection, lower_bounds
from fold_frame_screen import (
    Snapshot,
    ccz_decomposition,
    copy_state,
    framed_generators,
    generators,
    multiple_hook_histories,
    partial_fold,
    phase_gate,
    profile,
    snapshots,
    support_profile,
)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def dense(state):
    # Cirq indexes q0 first; Aer indexes it in the least significant bit.
    big = sum(value(t.coefficient) * t.ch.to_state_vector() for t in state.terms)
    return big.reshape([2] * state.width).transpose(list(reversed(range(state.width)))).reshape(-1)


def matrix_rank(state, width, left):
    axes = [width - 1 - q for q in left + [q for q in range(width) if q not in left]]
    matrix = state.reshape([2] * width).transpose(axes).reshape(1 << len(left), -1)
    return int(np.count_nonzero(np.linalg.svd(matrix, compute_uv=False) > 1e-10))


def aer(state, operations):
    circuit = QuantumCircuit(state.width)
    circuit.set_statevector(dense(state))
    for op in operations:
        name = {"T_DAG": "tdg", "S_DAG": "sdg"}.get(op.name, op.name.lower())
        getattr(circuit, name)(*op.targets)
    circuit.save_statevector()
    return np.asarray(
        AerSimulator(method="statevector", max_parallel_threads=1)
        .run(circuit)
        .result()
        .get_statevector()
    )


def clifford_state(width, seed):
    rng = random.Random(seed)
    state = BranchState(width)
    for _ in range(30):
        a, b = rng.sample(range(width), 2)
        state.operation(Operation(rng.choice(["H", "S", "X"]), (a,)))
        state.operation(Operation("CX", (a, b)))
    return state


class FoldFramesTest(unittest.TestCase):
    def test_signed_stabilizer_intersection_retains_only_shared_eigenvalues(self):
        left = list(map(stim.PauliString, ["XX", "ZZ"]))
        right = list(map(stim.PauliString, ["-XX", "-ZZ"]))
        common = intersection(left, right)
        self.assertEqual(len(common), 1)
        self.assertEqual(str(common[0]), "-YY")
        for group in (left, right):
            simulator = stim.TableauSimulator()
            simulator.set_inverse_tableau(stim.Tableau.from_stabilizers(group).inverse())
            self.assertEqual(simulator.peek_observable_expectation(common[0]), 1)

    def test_common_constraints_bound_entanglement_of_coherent_states(self):
        for seed in range(6):
            state = clifford_state(7, seed)
            identity = stim.Tableau(7)
            single = lower_bounds(common_stabilizers(state, identity), 7, list(range(7)))
            self.assertEqual(
                single, [matrix_rank(dense(state), 7, list(range(c))) for c in range(1, 7)]
            )
            phase_gate(state, Operation("T", (1,)))
            phase_gate(state, Operation("T_DAG", (4,)))
            lower = lower_bounds(common_stabilizers(state, identity), 7, list(range(7)))
            for cut in range(1, 7):
                self.assertLessEqual(lower[cut - 1], matrix_rank(dense(state), 7, list(range(cut))))

    def test_stabilizer_cut_ranks_match_dense_svd(self):
        for seed in range(6):
            state = clifford_state(7, seed)
            vector = dense(state)
            order = list(range(7))
            random.Random(seed + 9).shuffle(order)
            result = profile([generators(state.terms[0].tableau)], 7, order)
            exact = [matrix_rank(vector, 7, order[:cut]) for cut in range(1, 7)]
            self.assertEqual(result["bond_upper"], exact)
            self.assertEqual(result["bond_lower"], exact)

    def test_coherent_sum_bounds_enclose_dense_ranks(self):
        state = BranchState(6)
        state.terms = [clifford_state(6, seed).terms[0] for seed in (3, 5, 19, 29)]
        state.terms[1].gate("S", (2,))
        vector = dense(state)
        result = profile([generators(t.tableau) for t in state.terms], 6, list(range(6)))
        for cut in range(1, 6):
            exact = matrix_rank(vector, 6, list(range(cut)))
            self.assertLessEqual(result["bond_lower"][cut - 1], exact)
            self.assertGreaterEqual(result["bond_upper"][cut - 1], exact)

    def test_upper_bound_does_not_confuse_coherent_sum_with_mixture(self):
        state = BranchState(2)
        other = state.terms[0].copy()
        other.gate("X", (1,))
        state.terms.append(other)
        result = profile([generators(t.tableau) for t in state.terms], 2, [0, 1])
        self.assertEqual(result["max_bond_upper"], 2)
        self.assertEqual(matrix_rank(dense(state), 2, [0]), 1)

    def test_signed_phase_expansion_and_ccz_against_aer(self):
        initial = clifford_state(5, 101)
        operations = [
            Operation("T", (0,)),
            Operation("T_DAG", (3,)),
            Operation("CCZ", (0, 2, 4)),
            Operation("Y", (2,)),
        ]
        actual = copy_state(initial)
        for op in operations:
            for gate in ccz_decomposition(op.targets) if op.name == "CCZ" else [op]:
                phase_gate(actual, gate)
        np.testing.assert_allclose(dense(actual), aer(initial, operations), atol=2e-12)

    def test_partial_fold_including_unpaired_phase_against_aer(self):
        initial = BranchState(6)
        for q in range(4):
            initial.operation(Operation("H", (q,)))
        phase_gate(initial, Operation("T", (1,)))
        for q in (4, 5):
            initial.operation(Operation("CX", (3, q)))
        reconstruction = SimpleNamespace(ancilla=3, check_mapping=lambda _: [0, 1, 2])
        operations = [
            Operation("T_DAG", (0,)),
            Operation("Y", (0,)),
            Operation("CX", (3, 0)),
            Operation("T", (0,)),
            Operation("CCZ", (3, 1, 2)),
            Operation("X", (3,)),
        ]
        for count in range(len(operations) + 1):
            actual = partial_fold(initial, reconstruction, 3, operations[:count])
            np.testing.assert_allclose(dense(actual), aer(initial, operations[:count]), atol=2e-12)

    def test_fixed_frame_bounds_and_affine_support_against_dense(self):
        state = clifford_state(6, 15)
        frame = state.terms[0].tableau.current_inverse_tableau()
        phase_gate(state, Operation("T", (0,)))
        phase_gate(state, Operation("T_DAG", (4,)))
        before = dense(state)
        operations = []
        for instruction in frame.to_circuit():
            targets = instruction.targets_copy()
            arity = 2 if instruction.name == "CX" else 1
            for offset in range(0, len(targets), arity):
                operations.append(
                    Operation(
                        instruction.name, tuple(t.value for t in targets[offset : offset + arity])
                    )
                )
        expected = aer(state, operations)
        point = Snapshot(0, "test", state)
        bounds = profile(framed_generators(point, frame), 6, list(range(6)))
        support = support_profile(point, frame)
        self.assertLessEqual(
            np.count_nonzero(abs(expected) > 1e-12), support["sparse_coefficients_upper"]
        )
        self.assertLessEqual(
            support["sparse_coefficients_upper"], support["affine_dense_coefficients"]
        )
        for cut in range(1, 6):
            self.assertLessEqual(
                matrix_rank(expected, 6, list(range(cut))), bounds["bond_upper"][cut - 1]
            )
        np.testing.assert_array_equal(dense(state), before)

    def test_full_snapshot_walk_preserves_reference_output(self):
        for distance in (3, 7):
            reconstruction = Reconstruction(distance).build()
            cases = [
                [s.operations for s in reconstruction.stages],
                paired_hook_histories(reconstruction)[0][1],
            ]
            for history in cases:
                points, actual = snapshots(reconstruction, history)
                expected, _ = run_history(reconstruction, history)
                self.assertEqual(len(points), 26)
                for observable in [None] + [logical(distance, axis) for axis in "XYZ"]:
                    self.assertAlmostEqual(
                        actual.expectation(observable),
                        expected.expectation(observable),
                        delta=2e-12,
                    )

    def test_bad_order_and_incomplete_generators_decline(self):
        rows = [generators(BranchState(3).terms[0].tableau)]
        with self.assertRaises(ValueError):
            profile(rows, 3, [0, 0, 2])
        with self.assertRaises(ValueError):
            profile([rows[0][:-1]], 3, [0, 1, 2])

    def test_multiple_hook_histories_are_legal_and_match_block_reference(self):
        protocol = Protocol(7)
        for _, history in multiple_hook_histories(protocol.reconstruction):
            expected = protocol.evaluate(protocol.encode(history))
            points, state = snapshots(protocol.reconstruction, history)
            self.assertTrue(points)
            probability = state.expectation()
            self.assertAlmostEqual(probability, expected.acceptance, delta=2e-12)
            if probability > 1e-12:
                assert expected.logical_xyz is not None
                np.testing.assert_allclose(
                    [state.expectation(logical(7, axis)) / probability for axis in "XYZ"],
                    expected.logical_xyz,
                    atol=2e-12,
                )


if __name__ == "__main__":
    unittest.main()
