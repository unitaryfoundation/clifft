import json
import os
import subprocess
import unittest

import numpy as np
from export_pauli_proxy import encode_faults
from fold_cultivation import Operation, logical
from pauli_proxy import Compiler, Protocol, execute, logical_axis_tail
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def aer_state(operations, width, initial=None):
    circuit = QuantumCircuit(width)
    if initial is not None:
        circuit.set_statevector(initial)
    for op in operations:
        getattr(circuit, {"T_DAG": "tdg", "S_DAG": "sdg"}.get(op.name, op.name.lower()))(
            *op.targets
        )
    circuit.save_statevector()
    return np.asarray(
        AerSimulator(method="statevector", max_parallel_threads=1)
        .run(circuit)
        .result()
        .get_statevector()
    )


class ProxyTest(unittest.TestCase):
    def test_induced_clifford_corrections_match_original_gate_against_aer(self):
        rng = np.random.default_rng(813)
        for name, width in [("T", 1), ("T_DAG", 1), ("CCZ", 3)]:
            initial = rng.normal(size=1 << width) + 1j * rng.normal(size=1 << width)
            initial /= np.linalg.norm(initial)
            gate = Operation(name, tuple(range(width)))
            for flips in range(1 << width):
                faults = [Operation("X", (q,)) for q in range(width) if (flips >> q) & 1]
                compiler = Compiler(flips)
                compiler.append(gate)
                actual = aer_state([*faults, gate], width, initial)
                reconstructed = aer_state([gate, *compiler.operations, *faults], width, initial)
                phase = np.vdot(reconstructed, actual)
                np.testing.assert_allclose(actual, phase * reconstructed, atol=2e-12)

    def test_small_protocol_all_records_against_aer(self):
        ideal = [Operation("H", (0,)), Operation("T", (0,))]
        for ancilla in (1, 2):
            ideal += [
                Operation("H", (ancilla,)),
                Operation("T_DAG", (0,)),
                Operation("CX", (ancilla, 0)),
                Operation("T", (0,)),
                Operation("H", (ancilla,)),
            ]
        cases = [ideal]
        for j, gate in enumerate(ideal):
            for q in gate.targets:
                for axis in "XYZ":
                    cases.append([*ideal[: j + 1], Operation(axis, (q,)), *ideal[j + 1 :]])
        for operations in cases:
            state = aer_state(operations, 3).reshape(4, 2)
            compiler = Compiler()
            for op in operations:
                compiler.append(op)
            total = 0.0
            for record in range(4):
                measurements = [
                    Operation("POST", (q, ((record >> (q - 1)) ^ (compiler.flips >> q)) & 1))
                    for q in (1, 2)
                ]
                probability, _ = execute([*compiler.operations, *measurements])
                self.assertAlmostEqual(
                    probability, float(np.vdot(state[record], state[record]).real)
                )
                total += probability
            self.assertAlmostEqual(total, 1)

    def test_logical_probes_preserve_each_axis_tail(self):
        protocol = Protocol(3)
        s = 2**-0.5
        for axis, expected in [("X", [s, -s, 0]), ("Y", [-s, s, 0]), ("Z", [-s, -s, 0])]:
            stages = logical_axis_tail(protocol.reconstruction, axis)
            probability, xyz = protocol.evaluate(protocol.compile(stages))
            self.assertEqual(probability, 1)
            assert xyz is not None
            np.testing.assert_allclose(xyz, expected, atol=2e-12)

    def test_proxy_tableau_is_not_the_physical_state(self):
        protocol = Protocol(3)
        stages = [list(s.operations) for s in protocol.reconstruction.stages]
        compiled = protocol.compile(stages)
        probability, simulator = execute(compiled.operations)
        self.assertEqual(probability, 1)
        self.assertEqual(simulator.peek_observable_expectation(logical(3, "X")), 1)
        _, xyz = protocol.evaluate(compiled)
        assert xyz is not None
        np.testing.assert_allclose(xyz, [2**-0.5, 2**-0.5, 0], atol=2e-12)

    def test_fault_export_retains_simultaneous_pauli_components(self):
        protocol = Protocol(3)
        r = protocol.reconstruction
        stages = [list(s.operations) for s in r.stages]
        stages[0].insert(1, Operation("Y", stages[0][0].targets))
        q = stages[0][0].targets[0]
        self.assertEqual(encode_faults(protocol, stages), [(1, 1 << q, 1 << q)])
        with self.assertRaises(ValueError):
            encode_faults(protocol, [[], *stages[1:]])
        with self.assertRaises(ValueError):
            Compiler().append(Operation("RX", (0,)))

    @unittest.skipUnless(
        os.environ.get("CLIFFT_PROXY_NATIVE"), "requires optional external Stim build"
    )
    def test_native_proxy_agrees_on_exported_fault_fixtures(self):
        for distance in (3, 5, 7):
            result = subprocess.run(
                [os.environ["CLIFFT_PROXY_NATIVE"], str(distance), "2"],
                check=True,
                capture_output=True,
                text=True,
            )
            data = json.loads(result.stdout)
            self.assertGreaterEqual(data["fixtures"], 100)
            self.assertLess(data["max_error"], 2e-12)
            self.assertEqual(data["total_attempts"], 10)


if __name__ == "__main__":
    unittest.main()
