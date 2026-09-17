"""Parsed-region planning and reusable native execution checks."""

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
from clifford_branches import logical_tail_history, materialize, run_history
from fold_adapter import Adapter
from fold_blocks import Protocol, syndrome_sector_histories
from fold_cultivation import Operation, Reconstruction, logical
from fold_growth import growth_fault_histories
from fold_plan_data import Writer
from fold_schedule import measurement_chunks
from study_fold_adapter import held_out


@unittest.skipUnless(os.environ.get("CLIFFT_PLAN_WORKER"), "requires reusable plan worker")
class AdapterTest(unittest.TestCase):
    directory: ClassVar[tempfile.TemporaryDirectory]
    root: ClassVar[Path]
    adapter: ClassVar[Adapter]
    reconstruction: ClassVar[Reconstruction]
    circuit: ClassVar[Path]
    plan: ClassVar[Path]
    protocol: ClassVar[Protocol]

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.adapter = Adapter(Path(os.environ["CLIFFT_PLAN_WORKER"]))
        cls.reconstruction = held_out()
        cls.circuit = cls.root / "held.stim"
        cls.circuit.write_text(cls.reconstruction.text(0.001))
        cls.plan = cls.root / "held.bin"
        cls.protocol, _ = cls.adapter.prepare(cls.circuit, cls.plan)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_regions_come_from_supplied_gates(self):
        self.assertEqual(self.protocol.reconstruction.stages, self.reconstruction.stages)
        original = Reconstruction(7).build()
        changed = [
            a.name
            for a, b in zip(self.reconstruction.stages, original.stages, strict=True)
            if a.operations != b.operations
        ]
        self.assertEqual(len(changed), 6)
        result = self.adapter.run(self.plan, self.circuit, shots=100)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["measurement_values"], result["passed_shots"] * 350)
        self.assertEqual(result["detector_values"], result["passed_shots"] * 348)
        self.assertEqual(result["probe_values"], result["passed_shots"] * 3)

    def test_fixed_histories_against_coherent_reference_and_native_worker(self):
        r = self.protocol.reconstruction
        cases = [
            ("ideal", [s.operations for s in r.stages]),
            ("logical_tail", logical_tail_history(r)),
        ]
        cases += syndrome_sector_histories(self.protocol) + growth_fault_histories(r)[::12]
        cases += [
            (str(seed), materialize(r, p, seed)[0]) for p in (0.001, 0.01) for seed in (1000, 1007)
        ]
        for _, stages in cases:
            actual = self.protocol.evaluate(self.protocol.encode(stages))
            reference, _ = run_history(r, stages)
            p = reference.expectation()
            self.assertAlmostEqual(actual.acceptance, p, delta=2e-12)
            if p > 1e-12:
                assert actual.logical_xyz is not None
                np.testing.assert_allclose(
                    actual.logical_xyz,
                    [reference.expectation(logical(7, a)) / p for a in "XYZ"],
                    atol=2e-12,
                )
            else:
                self.assertIsNone(actual.logical_xyz)
        plan = self.root / "fixtures.bin"
        Writer(self.protocol).write(plan, cases)
        result = json.loads(
            subprocess.check_output([str(self.adapter.worker), "--check", str(plan)])
        )
        self.assertEqual(result["fixtures"], len(cases))
        self.assertLess(result["max_error"], 2e-12)

    def test_same_worker_matches_existing_catalog_results_on_all_sizes(self):
        native = os.environ.get("CLIFFT_RECOGNIZER_NATIVE")
        if not native:
            self.skipTest("requires baseline catalog recognizer")
        for d in (3, 5, 7):
            circuit, plan = self.root / f"f{d}.stim", self.root / f"f{d}.bin"
            circuit.write_text(Reconstruction(d).build().text(0.001))
            self.adapter.prepare(circuit, plan)
            actual = self.adapter.run(plan, circuit, shots=100)
            expected = json.loads(
                subprocess.check_output(
                    [native, str(circuit), "100", "19331", "1", "1", "ordinary"]
                )
            )
            for key in (
                "total_shots",
                "passed_shots",
                "measurements",
                "detectors",
                "exp_vals",
                "probabilities",
                "probes",
            ):
                self.assertEqual(actual[key], expected[key])

    def test_optional_noise_aliases_prefix_and_clifford_outputs(self):
        text = self.reconstruction.text(0.001)
        lines, index = [], 0
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if "ERROR(" in line or "DEPOLARIZE" in line:
                index += 1
                if index % 7 == 0:
                    continue
                line = line.replace("0.001", "0.0003141592653589793" if index % 2 else "0.002")
            lines.append(line.replace("CX ", "CNOT "))
        text = "H 0\nTICK\n" + "\n".join(lines) + "\nH 0\nEXP_VAL Z0\n"
        circuit, plan = self.root / "composed.stim", self.root / "composed.bin"
        circuit.write_text(text)
        self.adapter.prepare(circuit, plan)
        result = self.adapter.run(plan, circuit, shots=100)
        self.assertTrue(result["eligible"])
        self.assertEqual((result["prefix_gates"], result["suffix_gates"]), (1, 1))
        self.assertEqual(result["probes"][-1], [0, 0])
        self.assertEqual(set(result["probabilities"]), {0, 0.0003141592653589793, 0.002})

    def test_invalid_syndrome_and_growth_are_declined(self):
        for variant in ("extra_data", "duplicate", "growth_gate", "live_reset"):
            r = held_out()
            if variant in {"extra_data", "duplicate"}:
                stage = next(s for s in r.stages if s.name == "syndrome_d7")
                chunks = measurement_chunks(stage.operations)
                if variant == "extra_data":
                    used = {q for op in chunks[0] for q in op.targets}
                    q = next(q for q in range(85) if q not in used)
                    chunks[0].insert(-1, Operation("H", (q,)))
                else:
                    chunks[1] = chunks[0]
                stage.operations = [op for chunk in chunks for op in chunk]
            else:
                stage = next(s for s in r.stages if s.name == "grow_regular_d5_to_regular_d7")
                if variant == "growth_gate":
                    del stage.operations[-1]
                else:
                    stage.operations.insert(0, Operation("R", (r.qubit((4, 4), 5),)))
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.adapter.derive(r.text(0.001))

    def test_changed_noise_detectors_and_nonclifford_suffix_are_declined(self):
        text = self.reconstruction.text(0.001)
        variants = [
            text.replace("DETECTOR rec[-1]", "DETECTOR rec[-1] rec[-1]", 1),
            text.replace("X_ERROR(0.001)", "Z_ERROR(0.001)", 1),
            text.replace("R ", "R[tag] ", 1),
            text + "T 0\n",
        ]
        for value in variants:
            with self.subTest(value=value[:40]), self.assertRaises(ValueError):
                self.adapter.derive(value)

    def test_unsupported_requests_decline_before_sampling(self):
        for mode in ("fixed", "records", "partial", "nonzero"):
            result = self.adapter.run(self.plan, self.circuit, request=mode)
            self.assertFalse(result["eligible"])
            self.assertNotIn("total_shots", result)

    def test_explicit_ccz_decomposition_preserves_the_noise_boundary(self):
        text = self.reconstruction.text(0.001)
        line = next(line for line in text.splitlines() if line.startswith("CCZ "))
        nodes = self.adapter.parse(line)["nodes"]
        gates = [node["gate"] + " " + " ".join(map(str, node["targets"])) for node in nodes]
        expanded = text.replace(line, "\n".join(gates), 1)
        protocol = self.adapter.derive(expanded)
        self.assertEqual(protocol.reconstruction.stages, self.reconstruction.stages)
        circuit = self.root / "lowered.stim"
        circuit.write_text(expanded)
        actual = self.adapter.run(self.plan, circuit, shots=100)
        expected = self.adapter.run(self.plan, self.circuit, shots=100)
        for key in ("measurements", "detectors", "exp_vals", "passed_shots"):
            self.assertEqual(actual[key], expected[key])
        gates.insert(1, "X_ERROR(0.001) " + line.split()[1])
        with self.assertRaisesRegex(ValueError, "noise location"):
            self.adapter.derive(text.replace(line, "\n".join(gates), 1))

    def test_relabeling_outside_canonical_layout_is_declined(self):
        nodes = self.adapter.parse(self.reconstruction.text(0))["nodes"]
        # Swap two data labels while retaining the physical width, so the fixed
        # injection anchor, not just the width filter, must reject this input.
        node = next(node for node in nodes if node["gate"] == "T")
        q = node["targets"][0]
        other = next(k for k in range(85) if k != q)
        mapping = {q: other, other: q}
        lines = []
        for line in self.reconstruction.text(0).splitlines():
            tokens = line.split()
            if not tokens or tokens[0].startswith("#"):
                continue
            for k in range(1, len(tokens)):
                token = tokens[k]
                prefix = token[0] if token[0] in "XYZ" else ""
                value = token[len(prefix) :]
                if value.isdigit():
                    tokens[k] = prefix + str(mapping.get(int(value), int(value)))
            lines.append(" ".join(tokens))
        text = "\n".join(lines) + "\n"
        with self.assertRaisesRegex(ValueError, "fixed region"):
            self.adapter.derive(text)

    def test_truncated_oversized_and_trailing_plan_data_decline(self):
        data = self.plan.read_bytes()
        variants = [
            data[:7],
            data[:-1],
            data + b"extra",
            data[:16] + struct.pack("<Q", 1 << 63) + data[24:],
        ]
        for k, value in enumerate(variants):
            path = self.root / f"bad{k}.bin"
            path.write_bytes(value)
            result = subprocess.run(
                [str(self.adapter.worker), "--check", str(path)], capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("AddressSanitizer", result.stderr)


if __name__ == "__main__":
    unittest.main()
