"""Behavioral checks for the optional native parsed-circuit recognizer."""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
import stim
from fold_cultivation import Reconstruction, coordinates, logical, pauli, stabilizers


@unittest.skipUnless(os.environ.get("CLIFFT_RECOGNIZER_NATIVE"), "requires native recognizer")
class RecognitionTest(unittest.TestCase):
    circuits: ClassVar[dict[int, Reconstruction]]

    @classmethod
    def setUpClass(cls):
        cls.circuits = {d: Reconstruction(d).build() for d in (3, 5, 7)}

    def run_circuit(self, text, shots=0, keep=0, mode="ordinary", seed=19331):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.stim"
            path.write_text(text)
            completed = subprocess.run(
                [
                    os.environ["CLIFFT_RECOGNIZER_NATIVE"],
                    str(path),
                    str(shots),
                    str(seed),
                    str(keep),
                    "1",
                    mode,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        return json.loads(completed.stdout)

    def test_all_families_and_noise_probabilities(self):
        for distance, sites in [(3, 253), (5, 933), (7, 2205)]:
            for probability in (0, 0.001, 1):
                with self.subTest(distance=distance, probability=probability):
                    result = self.run_circuit(self.circuits[distance].text(probability))
                    self.assertTrue(result["eligible"])
                    self.assertEqual(result["distance"], distance)
                    self.assertEqual(result["probabilities"], [probability] * sites)
                    self.assertEqual(result["probes"], [[1, 1], [2, 1], [3, 1]])

    def test_parser_equivalence_and_sparse_qubit_relabeling(self):
        original = self.circuits[7].text(0.001)
        lines = []
        mapping = {q: 300 - 3 * q for q in range(99)}
        for line in original.splitlines():
            if line.startswith("#"):
                continue
            gate, targets = line.split(" ", 1)
            if gate != "DETECTOR":
                targets = re.sub(r"\d+", lambda m: str(mapping[int(m[0])]), targets)
            gate = {"CX": "CNOT", "H": "H_XZ", "R": "RZ", "M": "MZ"}.get(gate, gate)
            lines.extend([f"{gate} {targets}", "TICK"])
        text = "QUBIT_COORDS(1,2) 300\nREPEAT 1 {\n" + "\n".join(lines) + "\n}\n"
        result = self.run_circuit(text)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["qubits"], list(mapping.values()))

    def test_omitted_and_nonuniform_noise_sites(self):
        lines: list[str] = []
        probabilities: list[float] = []
        for line in self.circuits[5].text(0.125).splitlines():
            if "(0.125)" in line:
                index = len(probabilities)
                probability = (index % 11) * 0.0001
                probabilities.append(probability)
                if index % 11 == 0:
                    continue
                line = line.replace("(0.125)", f"({probability})")
            lines.append(line)
        result = self.run_circuit("\n".join(lines))
        self.assertTrue(result["eligible"])
        self.assertEqual(result["probabilities"], probabilities)

    def test_changed_body_and_unsupported_suffixes_decline(self):
        text = self.circuits[7].text(0.001)
        mutations = [
            text.replace("\nT ", "\nT_DAG ", 1),
            text.replace(
                "DEPOLARIZE2(0.001)", "PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001)", 1
            ),
            text.replace("DETECTOR rec[-1]", "DETECTOR rec[-1] rec[-1]", 1),
            text.replace("\nM ", "\nM(0.001) ", 1),
            text + "T 0\n",
            text + "M 0\n",
            text + "EXP_VAL X85\n",
            text + "H 85\n",
            text + "DEPOLARIZE1(0.001) 0\n",
            text + "OBSERVABLE_INCLUDE(0) rec[-1]\n",
            "H 1000\n" + text,
        ]
        for index, changed in enumerate(mutations):
            with self.subTest(index=index):
                self.assertFalse(self.run_circuit(changed)["eligible"])

    def test_request_contract_declines_unsupported_modes(self):
        for mode in ("records", "fixed", "partial", "nonzero"):
            with self.subTest(mode=mode):
                self.assertFalse(self.run_circuit(self.circuits[3].text(0), mode=mode)["eligible"])

    def test_explicit_ccz_decomposition_and_internal_noise_boundary(self):
        text = self.circuits[5].text(0.001)
        original = next(line for line in text.splitlines() if line.startswith("CCZ "))
        a, b, c = map(int, original.split()[1:])
        lowered = (
            f"T {a} {b} {c}\nCX {a} {b}\nT_DAG {b}\nCX {a} {b}\n"
            f"CX {a} {c}\nT_DAG {c}\nCX {b} {c}\nT {c}\nCX {a} {c}\n"
            f"T_DAG {c}\nCX {b} {c}"
        )
        self.assertTrue(self.run_circuit(text.replace(original, lowered, 1))["eligible"])
        changed = lowered.replace(f"CX {a} {b}", f"DEPOLARIZE1(0.001) {a}\nCX {a} {b}", 1)
        self.assertFalse(self.run_circuit(text.replace(original, changed, 1))["eligible"])

    def test_declined_circuit_still_compiles_through_ordinary_planner(self):
        text = "R 0 1\nH 0\nT 0\nCX 0 1\nM 1\nDETECTOR rec[-1]\nEXP_VAL X0\n"
        result = self.run_circuit(text, mode="fallback")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["fallback_peak_width"], 0)
        changed = self.circuits[3].text(0.001) + "T 0\n"
        result = self.run_circuit(changed, mode="fallback")
        self.assertFalse(result["eligible"])
        self.assertGreater(result["fallback_compile_ms"], 0)

    def test_reset_erases_clifford_prefix(self):
        text = self.circuits[5].text(0.001)
        original = self.run_circuit(text, shots=100, keep=1)
        prefixed = self.run_circuit("H 0\nCX 0 40\nS 22\n" + text, shots=100, keep=1)
        self.assertEqual(prefixed["prefix_gates"], 3)
        for key in ("passed_shots", "measurements", "detectors", "exp_vals"):
            self.assertEqual(prefixed[key], original[key])

    def test_clifford_suffix_probes_against_stim(self):
        rng = np.random.default_rng(713)
        for distance in (3, 5, 7):
            width = len(coordinates(distance))
            checks = [pauli(distance, a, s) for a, s in stabilizers(distance)]
            axes = [stim.PauliString(width)] + [logical(distance, a) for a in "XYZ"]
            body = (
                "\n".join(
                    line
                    for line in self.circuits[distance].text(0).splitlines()
                    if not line.startswith("EXP_VAL")
                )
                + "\n"
            )
            suffix = stim.Circuit()
            expected = []
            for index in range(80):
                gate = ["H", "S", "CX", "SQRT_YY", "ISWAP", "H_YZ"][index % 6]
                targets = rng.choice(width, size=2 if index % 6 in (2, 3, 4) else 1, replace=False)
                suffix.append(gate, targets.tolist())
                body += gate + " " + " ".join(map(str, targets)) + "\n"
                component = index % 4
                observable = axes[component].copy()
                for _ in range(4):
                    observable *= checks[int(rng.integers(len(checks)))]
                if index % 7 == 0:
                    observable = stim.PauliString(width)
                    observable[0] = "X"
                    self.assertTrue(any(not observable.commutes(check) for check in checks))
                    sign = 0
                    component = 0
                else:
                    sign = 1
                transformed = observable.after(suffix)
                sign *= transformed.sign.real
                product = "*".join(
                    f"{'_XYZ'[transformed[q]]}{q}" for q in range(width) if transformed[q]
                )
                if not product:
                    continue
                body += f"EXP_VAL {product}\n"
                expected.append([component, sign])
            result = self.run_circuit(body)
            self.assertTrue(result["eligible"])
            self.assertEqual(result["probes"], expected)

    def test_full_survivor_outputs_and_independent_flag_groups(self):
        reconstruction = self.circuits[7]
        text = reconstruction.text(0)
        result = self.run_circuit(text, shots=1000, keep=1)
        self.assertEqual(result["passed_shots"], 1000)
        rows = np.array(result["measurements"]).reshape(1000, -1)
        offset, groups = 0, []
        for stage in reconstruction.stages:
            count = sum(op.name == "M" for op in stage.operations)
            if stage.equal_records:
                groups.append(list(range(offset, offset + count)))
            offset += count
        self.assertEqual(len(groups), 2)
        for group in groups:
            np.testing.assert_array_equal(rows[:, group], np.repeat(rows[:, group[:1]], 6, axis=1))
        counts = np.bincount(rows[:, groups[0][0]] + 2 * rows[:, groups[1][0]], minlength=4)
        self.assertTrue(np.all((counts > 190) & (counts < 310)), counts)
        rows[:, np.ravel(groups)] = 0
        self.assertFalse(rows.any())
        self.assertFalse(any(result["detectors"]))
        np.testing.assert_allclose(
            np.array(result["exp_vals"]).reshape(-1, 3),
            np.tile([2**-0.5, 2**-0.5, 0], (1000, 1)),
            atol=2e-12,
        )
        counts_only = self.run_circuit(text, shots=1000)
        self.assertEqual(counts_only["passed_shots"], result["passed_shots"])
        for name in ("measurements", "detectors", "exp_vals"):
            self.assertEqual(counts_only[name], [])

    def test_noisy_seed_reproducibility_and_record_mode_counts(self):
        text = self.circuits[7].text(0.001)
        first = self.run_circuit(text, shots=400, keep=1)
        second = self.run_circuit(text, shots=400, keep=1)
        for name in ("passed_shots", "measurements", "detectors", "exp_vals"):
            self.assertEqual(first[name], second[name])
        self.assertEqual(first["passed_shots"], self.run_circuit(text, shots=400)["passed_shots"])
        self.assertEqual(first["measurement_values"], first["passed_shots"] * 350)
        self.assertEqual(first["detector_values"], first["passed_shots"] * 348)
        self.assertEqual(first["probe_values"], first["passed_shots"] * 3)

    def test_selected_physical_flag_readout_faults(self):
        lines = self.circuits[7].text(0.125).splitlines()
        flag_sites = []
        for index, line in enumerate(lines):
            if line.startswith("# cat_prepare_d7"):
                group = []
                for j in range(index + 1, len(lines)):
                    if lines[j].startswith("#"):
                        break
                    if lines[j].startswith("M "):
                        self.assertTrue(lines[j - 1].startswith("X_ERROR"))
                        group.append(j - 1)
                flag_sites.append(group)
        self.assertEqual(list(map(len, flag_sites)), [6, 6])
        for selected, expected in [
            (flag_sites[0][:1], 0),
            (flag_sites[0], 100),
            (flag_sites[0] + flag_sites[1], 100),
        ]:
            text = "\n".join(
                line.replace("(0.125)", "(1)" if j in selected else "(0)")
                for j, line in enumerate(lines)
            )
            result = self.run_circuit(text, shots=100, keep=1)
            self.assertTrue(result["eligible"])
            self.assertEqual(result["passed_shots"], expected)


if __name__ == "__main__":
    unittest.main()
