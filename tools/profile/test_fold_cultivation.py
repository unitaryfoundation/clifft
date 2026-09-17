import unittest

import numpy as np
import stim
from fold_cultivation import Reconstruction, cat_circuits, logical, pauli, stabilizers
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator


class ConstructionTest(unittest.TestCase):
    def test_growth_preserves_code_and_logical_axes(self):
        for distance in (3, 5):
            for replacement, expected in [(None, [1, 0, 0]), ("S", [0, 1, 0]), ("H", [0, 0, 1])]:
                r = Reconstruction(distance)
                r.injection()
                r.morph(3)
                if distance == 5:
                    r.grow()
                    r.morph(5)
                simulator = stim.TableauSimulator()
                for stage in r.stages:
                    for op in stage.operations:
                        name = replacement if op.name == "T" else op.name
                        if name is not None:
                            simulator.do(stim.CircuitInstruction(name, op.targets))
                for axis, support in stabilizers(distance):
                    self.assertEqual(
                        simulator.peek_observable_expectation(pauli(distance, axis, support)), 1
                    )
                self.assertEqual(
                    [
                        simulator.peek_observable_expectation(logical(distance, axis))
                        for axis in "XYZ"
                    ],
                    expected,
                )

    def test_verified_cats_and_decoders_against_stim(self):
        for distance, size in [(3, 3), (5, 5)]:
            prepare, decode = cat_circuits(distance)
            simulator = stim.TableauSimulator()
            for op in prepare.operations:
                if op.name == "M":
                    self.assertEqual(simulator.peek_z(op.targets[0]), 1)
                else:
                    simulator.do(stim.CircuitInstruction(op.name, op.targets))
            self.assertEqual(simulator.peek_observable_expectation(stim.PauliString("X" * size)), 1)
            for q in range(1, size):
                p = stim.PauliString(size)
                p[0] = p[q] = "Z"
                self.assertEqual(simulator.peek_observable_expectation(p), 1)
            for op in decode.operations:
                if op.name == "M":
                    self.assertEqual(simulator.peek_z(op.targets[0]), 1)
                else:
                    simulator.do(stim.CircuitInstruction(op.name, op.targets))

    def test_full_ideal_f3_against_aer(self):
        r = Reconstruction(3).build()
        qc = QuantumCircuit(16, 1)
        for stage in r.stages:
            for op in stage.operations:
                if op.name == "R":
                    qc.reset(op.targets[0])
                elif op.name == "M":
                    qc.measure(op.targets[0], 0)
                else:
                    name = "tdg" if op.name == "T_DAG" else op.name.lower()
                    getattr(qc, name)(*op.targets)
        qc.save_statevector()
        result = (
            AerSimulator(method="statevector", max_parallel_threads=1)
            .run(qc, shots=1, seed_simulator=81)
            .result()
        )
        state = Statevector(result.get_statevector())
        for axis, support in stabilizers(3):
            p = pauli(3, axis, support)
            p += stim.PauliString(3)
            self.assertAlmostEqual(
                state.expectation_value(Pauli(str(p)[1:].replace("_", "I")[::-1])).real, 1
            )
        expected = [1 / np.sqrt(2), 1 / np.sqrt(2), 0]
        for axis, target in zip("XYZ", expected, strict=True):
            p = logical(3, axis) + stim.PauliString(3)
            self.assertAlmostEqual(
                state.expectation_value(Pauli(str(p)[1:].replace("_", "I")[::-1])).real, target
            )

    def test_scope_and_noise_locations_are_explicit(self):
        r = Reconstruction(5).build()
        text = r.text(0.001)
        self.assertIn("not the authors' Reg5 benchmark artifact", text)
        self.assertIn("DEPOLARIZE3(0.001)", text)
        self.assertNotIn("ERROR", text.split("# syndrome_d5_ideal")[1])
        self.assertEqual(sum(op.name == "CCZ" for s in r.stages for op in s.operations), 40)
        with self.assertRaises(ValueError):
            Reconstruction(7)


if __name__ == "__main__":
    unittest.main()
