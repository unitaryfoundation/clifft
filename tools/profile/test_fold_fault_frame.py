"""Independent unitary and boundary tests for compiled Clifford fault frames."""

import random
import unittest

import numpy as np
from fold_blocks import NoiseLayout
from fold_cultivation import Operation
from fold_fault_frame import FramePlan
from fold_frame_screen import ccz_decomposition, copy_state, phase_gate
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def unitary(width, operations):
    circuit = QuantumCircuit(width)
    for op in operations:
        getattr(circuit, {"T_DAG": "tdg"}.get(op.name, op.name.lower()))(*op.targets)
    circuit.save_unitary()
    return np.asarray(
        AerSimulator(method="unitary", max_parallel_threads=1).run(circuit).result().get_unitary()
    )


def matrix(plan, frame):
    width = plan.layout.width
    result = np.zeros((1 << width, 1 << width), dtype=complex)
    for b in range(1 << width):
        phase = frame.phase + sum(c * ((b >> q) & 1) for q, c in enumerate(frame.linear))
        phase += 4 * sum(
            ((b >> a) & (b >> c) & (frame.edges >> e) & 1) for e, (a, c) in enumerate(plan.edges)
        )
        result[b ^ frame.flips, b] = np.exp(1j * np.pi * (phase % 8) / 4)
    return result


class FaultFrameTest(unittest.TestCase):
    def test_inverse_frame_preserves_coherent_phases(self):
        from test_fold_frames import clifford_state, dense

        state = clifford_state(5, 191)
        phase_gate(state, Operation("T", (2,)))
        ideal, noisy = copy_state(state), copy_state(state)
        operations = [
            Operation("T_DAG", (1,)),
            Operation("CX", (1, 3)),
            Operation("CCZ", (0, 2, 3)),
            Operation("CX", (4, 2)),
        ]
        layout = NoiseLayout(operations, [True] * len(operations), 5)
        plan = FramePlan(layout)
        faults = random.Random(4).getrandbits(len(layout.slots))
        for i, op in enumerate(operations):
            for gate in ccz_decomposition(op.targets) if op.name == "CCZ" else [op]:
                phase_gate(ideal, gate)
                phase_gate(noisy, gate)
            for q, x, z in layout.boundaries[i + 1]:
                if faults & z:
                    noisy.operation(Operation("Z", (q,)))
                if faults & x:
                    noisy.operation(Operation("X", (q,)))
        corrected = plan.inverse_state(noisy, plan.evaluate(faults))
        np.testing.assert_allclose(dense(corrected), dense(ideal), atol=3e-12)

    def test_arbitrary_monomial_regions_and_faults_against_aer(self):
        rng = random.Random(8092)
        for trial in range(8):
            width = 5
            operations = []
            for _ in range(24):
                name = rng.choice(["T", "T_DAG", "CX", "CCZ"])
                arity = {"T": 1, "T_DAG": 1, "CX": 2, "CCZ": 3}[name]
                operations.append(Operation(name, tuple(rng.sample(range(width), arity))))
            layout = NoiseLayout(operations, [True] * len(operations), width)
            initial_edges = [(a, b) for a in range(width) for b in range(a + 1, width)]
            plan = FramePlan(layout, initial_edges)
            faults = rng.getrandbits(len(layout.slots))
            initial = plan.empty()
            initial.flips = rng.getrandbits(width)
            initial.linear = [2 * rng.randrange(4) for _ in range(width)]
            initial.edges = rng.getrandbits(len(initial_edges))
            initial.phase = rng.randrange(8)
            actual = []
            for i, op in enumerate(operations):
                actual.append(op)
                for q, x, z in layout.boundaries[i + 1]:
                    if faults & z:
                        actual.append(Operation("Z", (q,)))
                    if faults & x:
                        actual.append(Operation("X", (q,)))
                if i in (0, 5, 13, 23):
                    correction = plan.evaluate(faults, initial, i + 1)
                    np.testing.assert_allclose(
                        unitary(width, actual) @ matrix(plan, initial),
                        matrix(plan, correction) @ unitary(width, operations[: i + 1]),
                        atol=3e-12,
                        err_msg=f"trial {trial} prefix {i + 1}",
                    )

    def test_sparse_edge_plan_matches_full_edge_plan(self):
        rng = random.Random(671)
        for _ in range(10):
            operations = [
                Operation(rng.choice(["CX", "CCZ"]), tuple(rng.sample(range(6), 3)))
                for _ in range(12)
            ]
            operations = [
                Operation(o.name, o.targets[:2] if o.name == "CX" else o.targets)
                for o in operations
            ]
            layout = NoiseLayout(operations, [True] * len(operations), 6)
            sparse = FramePlan(layout)
            full = FramePlan(layout, [(a, b) for a in range(6) for b in range(a + 1, 6)])
            faults = rng.getrandbits(len(layout.slots))
            np.testing.assert_allclose(
                matrix(sparse, sparse.evaluate(faults)), matrix(full, full.evaluate(faults))
            )

    def test_paired_cat_hooks_leave_data_cz_and_pass_boundary(self):
        operations = [Operation("CCZ", (0, 1, 2))] * 2
        layout = NoiseLayout(operations, [True, True], 3)
        plan = FramePlan(layout, [(1, 2)])
        history = [operations[0], Operation("X", (0,)), operations[1], Operation("X", (0,))]
        frame = plan.evaluate(layout.encode(history))
        self.assertTrue(plan.data_only(frame, {1, 2}))
        np.testing.assert_allclose(matrix(plan, frame), unitary(3, [Operation("CZ", (1, 2))]))

    def test_data_fault_creates_cat_edge_and_declines_boundary(self):
        operations = [Operation("CCZ", (0, 1, 2))] * 2
        layout = NoiseLayout(operations, [True, True], 3)
        plan = FramePlan(layout)
        frame = plan.evaluate(layout.encode([operations[0], Operation("X", (1,)), operations[1]]))
        self.assertFalse(plan.data_only(frame, {1, 2}))
        # This is a real measurement effect, not just an unnecessarily strict
        # label check: the cat X observable gains a data Z under this CZ.
        from qiskit.quantum_info import Pauli

        e = matrix(plan, frame)
        xcat = Pauli("IIX").to_matrix()
        self.assertGreater(np.linalg.norm(e.conj().T @ xcat @ e - xcat), 1)

    def test_non_monomial_gate_and_unplanned_input_decline(self):
        with self.assertRaises(ValueError):
            FramePlan(NoiseLayout([Operation("H", (0,))], [True], 3))
        plan = FramePlan(NoiseLayout([Operation("CCZ", (0, 1, 2))], [True], 3))
        initial = plan.empty()
        initial.edges = 1
        with self.assertRaises(ValueError):
            plan.evaluate(0, initial)


if __name__ == "__main__":
    unittest.main()
