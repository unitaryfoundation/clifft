"""Independent checks of measurement certificates and changed noisy schedules."""

import json
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
import stim
from clifford_branches import logical_tail_history, materialize, paired_hook_histories, run_history
from fold_blocks import Boundary, Protocol, syndrome_sector_histories
from fold_cultivation import Operation, logical
from fold_schedule import certify_syndrome, measurement_chunks, reconstruction_with_schedule
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class ScheduleTest(unittest.TestCase):
    protocols: ClassVar[dict[str, Protocol]]

    @classmethod
    def setUpClass(cls):
        cls.protocols = {
            schedule: Protocol(7, reconstruction=reconstruction_with_schedule(7, schedule))
            for schedule in ("original", "reverse_support", "reverse_checks")
        }

    def test_variants_preserve_ideal_contract_but_change_fault_maps(self):
        original = self.protocols["original"]
        for schedule in ("reverse_support", "reverse_checks"):
            candidate = self.protocols[schedule]
            self.assertNotEqual(
                candidate.reconstruction.text(0.001), original.reconstruction.text(0.001)
            )
            result = candidate.evaluate(
                candidate.encode([s.operations for s in candidate.reconstruction.stages])
            )
            self.assertAlmostEqual(result.acceptance, 1)
            assert result.logical_xyz is not None
            np.testing.assert_allclose(result.logical_xyz, [2**-0.5, 2**-0.5, 0], atol=2e-12)
            self.assertEqual(result.contractions, 13)
            self.assertEqual(result.peak_terms, 4)
            before = [p for _, p, _ in original.groups if isinstance(p, Boundary)]
            after = [p for _, p, _ in candidate.groups if isinstance(p, Boundary)]
            self.assertTrue(
                any(
                    (a.linear.x, a.linear.z, a.linear.records)
                    != (b.linear.x, b.linear.z, b.linear.records)
                    for a, b in zip(before, after, strict=True)
                )
            )
            for a, b in zip(before, after, strict=True):
                expected = (
                    a.measured_stabilizers
                    if schedule == "reverse_support"
                    else list(reversed(a.measured_stabilizers))
                )
                self.assertEqual(b.measured_stabilizers, expected)

    def test_changed_histories_against_independent_coherent_reference(self):
        for schedule in ("reverse_support", "reverse_checks"):
            protocol = self.protocols[schedule]
            reconstruction = protocol.reconstruction
            cases = [materialize(reconstruction, 0.001, seed)[0] for seed in (1000, 1005, 1020)]
            cases += [materialize(reconstruction, 0.01, 2000)[0]]
            cases += [stages for _, stages in paired_hook_histories(reconstruction)[::5]]
            cases += [logical_tail_history(reconstruction)]
            sectors = syndrome_sector_histories(protocol)
            self.assertEqual(len(sectors), 2)
            cases += [stages for _, stages in sectors]
            for stages in cases:
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
            for _, stages in sectors:
                self.assertAlmostEqual(protocol.evaluate(protocol.encode(stages)).acceptance, 0.25)

    def test_measurement_gadgets_both_kraus_outcomes_against_aer(self):
        for schedule in ("reverse_support", "reverse_checks"):
            reconstruction = self.protocols[schedule].reconstruction
            stage = next(s for s in reconstruction.stages if s.name == "syndrome_d7")
            seen = set()
            for chunk in measurement_chunks(stage.operations):
                ancilla = chunk[0].targets[0]
                data = sorted({q for op in chunk for q in op.targets} - {ancilla})
                axis = "X" if chunk[1].name == "H" else "Z"
                key = axis, len(data)
                if key in seen:
                    continue
                seen.add(key)
                mapping = {q: j for j, q in enumerate(data + [ancilla])}
                circuit = QuantumCircuit(len(mapping))
                for op in chunk[1:-1]:
                    getattr(circuit, op.name.lower())(*(mapping[q] for q in op.targets))
                circuit.save_unitary()
                result = (
                    AerSimulator(method="unitary", max_parallel_threads=1).run(circuit).result()
                )
                unitary = np.asarray(result.get_unitary())
                single = np.array([[0, 1], [1, 0]]) if axis == "X" else np.diag([1, -1])
                observable = np.array([[1]])
                for _ in data:
                    observable = np.kron(observable, single)
                size = len(observable)
                for outcome in (0, 1):
                    actual = unitary[outcome * size : (outcome + 1) * size, :size]
                    np.testing.assert_allclose(
                        actual, (np.eye(size) + (-1) ** outcome * observable) / 2, atol=2e-12
                    )
            self.assertEqual({axis for axis, _ in seen}, {"X", "Z"})

    def test_same_readout_with_extra_data_unitary_is_rejected(self):
        reconstruction = self.protocols["original"].reconstruction
        stage = next(s for s in reconstruction.stages if s.name == "syndrome_d7")
        chunks = measurement_chunks(stage.operations)
        first = chunks[0]
        used = {q for op in first for q in op.targets}
        target = next(q for q in range(85) if q not in used)
        changed = first[:-1] + [Operation("H", (target,)), first[-1]]
        readout = stim.PauliString(99)
        readout[reconstruction.ancilla] = "Z"
        pulled = []
        for chunk in (first, changed):
            unitary = stim.Circuit("I 98")
            for op in chunk[1:-1]:
                unitary.append(op.name, op.targets)
            pulled.append(readout.after(unitary.inverse()))
        self.assertEqual(pulled[0], pulled[1])
        with self.assertRaisesRegex(ValueError, "extra data or ancilla action"):
            certify_syndrome(
                changed + [op for chunk in chunks[1:] for op in chunk], list(range(85)), 7, 99
            )

    def test_unlisted_shuffled_schedule_is_compiled_from_its_gates(self):
        reconstruction = reconstruction_with_schedule(7, "original")
        rng = random.Random(197)
        for stage in reconstruction.stages:
            if not stage.name.startswith("syndrome"):
                continue
            chunks = measurement_chunks(stage.operations)
            rng.shuffle(chunks)
            for chunk in chunks:
                indices = [j for j, op in enumerate(chunk) if op.name == "CX"]
                gates = [chunk[j] for j in indices]
                rng.shuffle(gates)
                for j, gate in zip(indices, gates, strict=True):
                    chunk[j] = gate
            stage.operations = [op for chunk in chunks for op in chunk]
        protocol = Protocol(7, reconstruction=reconstruction)
        cases = [
            [list(stage.operations) for stage in reconstruction.stages],
            logical_tail_history(reconstruction),
        ]
        cases += [materialize(reconstruction, 0.001, seed)[0] for seed in (1000, 1005, 1020)]
        for stages in cases:
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

    def test_incomplete_duplicate_or_changed_stabilizers_are_rejected(self):
        reconstruction = self.protocols["original"].reconstruction
        stage = next(s for s in reconstruction.stages if s.name == "syndrome_d7")
        chunks = measurement_chunks(stage.operations)
        candidates = [chunks[:-1], [chunks[0], *chunks[:-1]]]
        changed = [list(chunk) for chunk in chunks]
        j = next(j for j, op in enumerate(changed[0]) if op.name == "CX")
        del changed[0][j]
        candidates.append(changed)
        for candidate in candidates:
            with self.assertRaises(ValueError):
                certify_syndrome(
                    [op for chunk in candidate for op in chunk], list(range(85)), 7, 99
                )

    @unittest.skipUnless(
        os.environ.get("CLIFFT_SCHEDULE_RECOGNIZER_NATIVE"), "requires variant recognizer"
    )
    def test_new_catalog_accepts_only_its_certified_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.stim"
            for schedule in ("original", "reverse_support", "reverse_checks"):
                path.write_text(self.protocols[schedule].reconstruction.text(0.001))
                completed = subprocess.run(
                    [
                        os.environ["CLIFFT_SCHEDULE_RECOGNIZER_NATIVE"],
                        str(path),
                        "100",
                        "19331",
                        "1",
                        "1",
                        "ordinary",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result = json.loads(completed.stdout)
                self.assertEqual(bool(result["eligible"]), schedule == "reverse_checks")
                if result["eligible"]:
                    self.assertEqual(result["probes"], [[1, 1], [2, 1], [3, 1]])
                    self.assertEqual(result["measurement_values"], result["passed_shots"] * 350)


if __name__ == "__main__":
    unittest.main()
