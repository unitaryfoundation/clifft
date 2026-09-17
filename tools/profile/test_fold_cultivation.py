import unittest

import numpy as np
import stim
from fold_cultivation import (
    Reconstruction,
    cat_circuits,
    logical,
    pauli,
    regular_growth,
    stabilizers,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator


class ConstructionTest(unittest.TestCase):
    def test_growth_preserves_code_and_logical_axes(self):
        for distance in (3, 5, 7):
            for replacement, expected in [(None, [1, 0, 0]), ("S", [0, 1, 0]), ("H", [0, 0, 1])]:
                r = Reconstruction(distance)
                r.injection()
                r.morph(3)
                if distance >= 5:
                    r.grow()
                    r.morph(5)
                if distance == 7:
                    r.grow_to_seven()
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
        for distance, size in [(3, 3), (5, 5), (7, 8)]:
            prepare, decode = cat_circuits(distance)
            simulator = stim.TableauSimulator()
            for op in prepare.operations:
                if op.name == "M":
                    if distance == 7 and op.targets[0] == 8:
                        self.assertEqual(simulator.peek_z(8), 0)
                        simulator.postselect_z(8, desired_value=False)
                    else:
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
            Reconstruction(9)

    def test_regular_growth_has_four_disjoint_local_layers(self):
        fresh, plus, layers = regular_growth(7)
        self.assertEqual(len(fresh), 44)
        self.assertEqual(len(plus), 22)
        self.assertTrue(plus <= fresh)
        self.assertEqual(list(map(len, layers)), [22, 22, 18, 18])
        for layer in layers:
            targets = [q for pair in layer for q in pair]
            self.assertEqual(len(set(targets)), len(targets))
            for (ax, ay), (bx, by) in layer:
                self.assertEqual(abs(ax - bx) + abs(ay - by), 2)

    def test_f7_cat_marginalizes_both_verification_records_against_aer(self):
        prep, _ = cat_circuits(7)
        for selected, axis in [(None, "X"), (15, "Y"), (22, "X"), (29, "Z")]:
            circuit = QuantumCircuit(14)
            for j, op in enumerate(prep.operations):
                if op.name == "M":
                    continue
                if op.name != "R":
                    getattr(circuit, op.name.lower())(*op.targets)
                if j == selected:
                    getattr(circuit, axis.lower())(op.targets[-1])
            circuit.save_statevector()
            result = (
                AerSimulator(method="statevector", max_parallel_threads=1).run(circuit).result()
            )
            amplitudes = np.asarray(result.get_statevector()).reshape(64, 256)
            zero, one = amplitudes[0], amplitudes[63]
            np.testing.assert_allclose(
                np.outer(zero, zero.conj()), np.outer(one, one.conj()), atol=1e-12
            )
            if selected is None:
                self.assertAlmostEqual(float(np.vdot(zero, zero).real), 0.5)
                self.assertAlmostEqual(float(np.vdot(one, one).real), 0.5)

    def test_f7_export_keeps_equal_flag_postselection(self):
        r = Reconstruction(7).build()
        text = r.text(0.001)
        self.assertEqual(r.ancilla, 85)
        self.assertEqual(max(q for s in r.stages for op in s.operations for q in op.targets), 98)
        self.assertEqual(
            sum(op.name in {"T", "T_DAG", "CCZ"} for s in r.stages for op in s.operations), 221
        )
        self.assertEqual(text.count("DETECTOR rec[-1] rec[-2]"), 10)


if __name__ == "__main__":
    unittest.main()
