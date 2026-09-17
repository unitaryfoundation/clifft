import unittest

import numpy as np
from frame_screen import apply, profile, representatives, search
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class FrameTest(unittest.TestCase):
    def test_clifford_representatives_are_unitary_and_include_identity(self):
        gates = representatives()
        self.assertEqual(len(gates), 20)
        np.testing.assert_allclose(gates[0][1], np.eye(4), atol=1e-12)
        for _, matrix in gates:
            np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(4), atol=1e-12)

    def test_nonadjacent_coordinate_application_against_aer(self):
        rng = np.random.default_rng(133)
        state = rng.normal(size=16) + 1j * rng.normal(size=16)
        state /= np.linalg.norm(state)
        for _, matrix in representatives():
            circuit = QuantumCircuit(4)
            circuit.set_statevector(state)
            circuit.unitary(matrix, [0, 3])
            circuit.save_statevector()
            expected = np.asarray(
                AerSimulator(method="statevector", max_parallel_threads=1)
                .run(circuit)
                .result()
                .get_statevector()
            )
            np.testing.assert_allclose(apply(state, 4, 0, 3, matrix), expected, atol=1e-12)

    def test_frame_recovers_product_magic_without_truncation(self):
        magic = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
        product = np.kron(magic, magic)
        state = product.copy()
        state[3] *= -1
        gates = representatives()
        self.assertEqual(profile(state, 2, [0, 1])["maximum_rank"], 2)
        transformed, frame = search(state, 2, [0, 1], gates, 2)
        self.assertTrue(frame)
        self.assertEqual(profile(transformed, 2, [0, 1])["maximum_rank"], 1)
        recovered = transformed
        for a, b, j in reversed(frame):
            recovered = apply(recovered, 2, a, b, gates[j][1].conj().T)
        np.testing.assert_allclose(recovered, state, atol=1e-12)
        held_out = state[[1, 0, 3, 2]]
        for a, b, j in frame:
            held_out = apply(held_out, 2, a, b, gates[j][1])
        self.assertEqual(profile(held_out, 2, [0, 1])["maximum_rank"], 1)


if __name__ == "__main__":
    unittest.main()
