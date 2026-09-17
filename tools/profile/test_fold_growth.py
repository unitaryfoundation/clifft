"""Growth-isometry, syndrome-sector, and full noisy-circuit checks."""

import json
import os
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import ClassVar

import numpy as np
import stim
from clifford_branches import logical_tail_history, materialize, run_history
from fold_blocks import Boundary, Protocol, parity_map, syndrome_sector_histories
from fold_cultivation import Operation, coordinates, logical, pauli, stabilizers
from fold_growth import growth_fault_histories, reconstruction_with_growth


def unitary(operations):
    circuit = stim.Circuit("I 98")
    for op in operations:
        if op.name != "R":
            circuit.append(op.name, op.targets)
    return stim.Tableau.from_circuit(circuit)


def frame(width, x, z, mapping):
    result = stim.PauliString(width)
    for q, physical in enumerate(mapping):
        result[physical] = (
            "Y"
            if (x >> q) & 1 and (z >> q) & 1
            else "X"
            if (x >> q) & 1
            else "Z"
            if (z >> q) & 1
            else "I"
        )
    return result


class GrowthTest(unittest.TestCase):
    protocols: ClassVar[dict[str, Protocol]]

    @classmethod
    def setUpClass(cls):
        cls.protocols = {
            "original": Protocol(7),
            "dual": Protocol(7, reconstruction=reconstruction_with_growth(7, "dual_rotate")),
            "combined": Protocol(
                7, reconstruction=reconstruction_with_growth(7, "dual_rotate", "reverse_checks")
            ),
        }

    def test_encoder_changes_while_locality_depth_and_resources_are_preserved(self):
        original, changed = [
            next(
                s
                for s in self.protocols[key].reconstruction.stages
                if s.name == "grow_regular_d5_to_regular_d7"
            )
            for key in ("original", "dual")
        ]
        self.assertNotEqual(unitary(original.operations), unitary(changed.operations))
        self.assertEqual(
            Counter(op.name for op in changed.operations), {"R": 44, "H": 22, "CX": 80}
        )
        positions = {q: xy for xy, q in self.protocols["dual"].reconstruction.index.items()}
        pairs = [op.targets for op in changed.operations if op.name == "CX"]
        start = 0
        for size in (22, 22, 18, 18):
            layer = pairs[start : start + size]
            self.assertEqual(len({q for pair in layer for q in pair}), 2 * size)
            for a, b in layer:
                (x, y), (u, v) = positions[a], positions[b]
                self.assertEqual(abs(x - u) + abs(y - v), 2)
            start += size
        for key in ("dual", "combined"):
            protocol = self.protocols[key]
            result = protocol.evaluate(
                protocol.encode([s.operations for s in protocol.reconstruction.stages])
            )
            self.assertAlmostEqual(result.acceptance, 1)
            assert result.logical_xyz is not None
            np.testing.assert_allclose(result.logical_xyz, [2**-0.5, 2**-0.5, 0], atol=2e-12)
            self.assertEqual((result.peak_terms, result.contractions), (4, 13))
        boundaries = [
            next(
                p
                for _, p, _ in self.protocols[key].groups
                if isinstance(p, Boundary) and p.before.distance == 5 and p.after.distance == 7
            )
            for key in ("original", "dual")
        ]
        before, after = boundaries
        self.assertEqual(before.transported, after.transported)
        self.assertEqual(before.syndrome_rows, after.syndrome_rows)
        self.assertEqual(before.constraints, after.constraints)
        self.assertNotEqual(
            (before.linear.x, before.linear.z, before.linear.records),
            (after.linear.x, after.linear.z, after.linear.records),
        )

    def test_six_logical_eigenstates_forward_through_encoder(self):
        protocol = self.protocols["dual"]
        reconstruction = protocol.reconstruction
        mapping = [reconstruction.qubit(xy, 5) for xy in coordinates(5)]
        stage = next(s for s in reconstruction.stages if s.name == "grow_regular_d5_to_regular_d7")
        for axis in "XYZ":
            for sign in (1, -1):
                observable = logical(5, axis)
                observable.sign = sign
                tableau = stim.Tableau.from_stabilizers(
                    [pauli(5, a, s) for a, s in stabilizers(5)] + [observable]
                )
                simulator = stim.TableauSimulator()
                simulator.set_num_qubits(99)
                simulator.do_tableau(tableau, mapping)
                for op in stage.operations:
                    simulator.do(stim.CircuitInstruction(op.name, op.targets))
                for a, support in stabilizers(7):
                    self.assertEqual(simulator.peek_observable_expectation(pauli(7, a, support)), 1)
                self.assertEqual(simulator.peek_observable_expectation(logical(7, axis)), sign)

    def test_every_input_syndrome_generator_and_transported_frame_against_stim(self):
        protocol = self.protocols["dual"]
        reconstruction = protocol.reconstruction
        boundary = next(
            p
            for _, p, _ in protocol.groups
            if isinstance(p, Boundary) and p.before.distance == 5 and p.after.distance == 7
        )
        stage = next(s for s in reconstruction.stages if s.name == "grow_regular_d5_to_regular_d7")
        preparation = stim.Tableau.from_stabilizers(
            [pauli(5, a, s) for a, s in stabilizers(5)] + [logical(5, "Y")]
        )
        for j, ((x, z), (out_x, out_z)) in enumerate(
            zip(boundary.duals, boundary.transported, strict=True)
        ):
            simulator = stim.TableauSimulator()
            simulator.set_num_qubits(99)
            simulator.do_tableau(preparation, boundary.input_mapping)
            simulator.do_pauli_string(frame(99, x, z, boundary.input_mapping))
            for op in stage.operations:
                simulator.do(stim.CircuitInstruction(op.name, op.targets))
            record = 0
            for k, measured in enumerate(boundary.measured_stabilizers):
                expectation = simulator.peek_observable_expectation(measured)
                self.assertIn(expectation, (-1, 1))
                record |= int(expectation == -1) << k
            self.assertEqual(parity_map(boundary.constraints, record), 0)
            self.assertEqual(parity_map(boundary.syndrome_rows, record), 1 << j)
            simulator.do_pauli_string(frame(99, out_x, out_z, list(range(99))))
            for a, support in stabilizers(7):
                self.assertEqual(simulator.peek_observable_expectation(pauli(7, a, support)), 1)
            self.assertEqual(simulator.peek_observable_expectation(logical(7, "Y")), 1)

    def test_growth_faults_and_combined_schedule_against_coherent_reference(self):
        for key in ("dual", "combined"):
            protocol = self.protocols[key]
            reconstruction = protocol.reconstruction
            cases = growth_fault_histories(reconstruction)[::12]
            sectors = syndrome_sector_histories(protocol)
            self.assertEqual(len(sectors), 2)
            cases += sectors + [("logical_tail", logical_tail_history(reconstruction))]
            cases += [
                (str(seed), materialize(reconstruction, 0.001, seed)[0]) for seed in (1000, 1007)
            ]
            for _, stages in cases:
                result = protocol.evaluate(protocol.encode(stages))
                reference, _ = run_history(reconstruction, stages)
                probability = reference.expectation()
                self.assertAlmostEqual(result.acceptance, probability, delta=2e-12)
                if probability > 1e-12:
                    assert result.logical_xyz is not None
                    np.testing.assert_allclose(
                        result.logical_xyz,
                        [reference.expectation(logical(7, a)) / probability for a in "XYZ"],
                        atol=2e-12,
                        rtol=0,
                    )
                else:
                    self.assertIsNone(result.logical_xyz)

    def test_growth_rejects_use_before_reset_and_unrelated_wires(self):
        for bad in ("late_reset", "ancilla"):
            reconstruction = reconstruction_with_growth(7, "dual_rotate")
            stage = next(
                s for s in reconstruction.stages if s.name == "grow_regular_d5_to_regular_d7"
            )
            if bad == "late_reset":
                h = next(op for op in stage.operations if op.name == "H")
                stage.operations.insert(0, h)
            else:
                stage.operations.extend([Operation("H", (85,)), Operation("H", (85,))])
            with self.assertRaisesRegex(ValueError, "unrelated or uninitialized"):
                Protocol(7, reconstruction=reconstruction)

    def test_wrong_growth_or_live_reset_is_rejected(self):
        for bad in ("missing_gate", "live_reset"):
            reconstruction = reconstruction_with_growth(7, "dual_rotate")
            stage = next(
                s for s in reconstruction.stages if s.name == "grow_regular_d5_to_regular_d7"
            )
            if bad == "missing_gate":
                del stage.operations[-1]
            else:
                stage.operations.insert(0, Operation("R", (reconstruction.qubit((4, 4), 5),)))
            with self.assertRaises(ValueError):
                Protocol(7, reconstruction=reconstruction)

    @unittest.skipUnless(
        os.environ.get("CLIFFT_GROWTH_RECOGNIZER_NATIVE"), "requires growth recognizer"
    )
    def test_growth_catalog_accepts_only_its_own_full_circuit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.stim"
            for key in ("original", "dual", "combined"):
                path.write_text(self.protocols[key].reconstruction.text(0.001))
                result = json.loads(
                    subprocess.check_output(
                        [
                            os.environ["CLIFFT_GROWTH_RECOGNIZER_NATIVE"],
                            str(path),
                            "100",
                            "19331",
                            "1",
                            "1",
                            "ordinary",
                        ],
                        text=True,
                    )
                )
                self.assertEqual(bool(result["eligible"]), key == "dual")
                if result["eligible"]:
                    self.assertEqual(result["probes"], [[1, 1], [2, 1], [3, 1]])
                    self.assertEqual(result["measurement_values"], result["passed_shots"] * 350)


if __name__ == "__main__":
    unittest.main()
