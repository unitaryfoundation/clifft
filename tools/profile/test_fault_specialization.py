"""Conformance tests for the opt-in native fault-specialization experiment."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


class FaultSpecializationTest(unittest.TestCase):
    binary: Path

    @classmethod
    def setUpClass(cls):
        cls.binary = Path(
            os.environ.get("CLIFFT_FAULT_BINARY", "build-research/profile_fault_specialization")
        ).resolve()
        if not cls.binary.exists():
            raise unittest.SkipTest("build profile_fault_specialization first")

    def run_study(self, circuit, patterns=1, shots=1, k=-1):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.stim"
            source.write_text(circuit)
            output = Path(directory) / "patterns"
            result = subprocess.run(
                [
                    str(self.binary),
                    str(source),
                    str(patterns),
                    str(shots),
                    str(k),
                    str(output),
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            data = json.loads(result.stdout)
            texts = [(output / f"{i}.stim").read_text() for i in range(patterns)]
            return data, texts

    def test_two_and_three_qubit_channel_argument_order(self):
        # Each base-four digit labels a physical target in source order.
        for arity in [2, 3]:
            for code in range(1, 4**arity):
                probabilities = ["1" if j == code else "0" for j in range(1, 4**arity)]
                targets = " ".join(map(str, range(arity)))
                text = f"PAULI_CHANNEL_{arity}({','.join(probabilities)}) {targets}\nM {targets}\n"
                data, _ = self.run_study(text)
                expected = sum(
                    (1 << q) for q in range(arity) if (code >> (2 * (arity - 1 - q))) & 3 in [1, 2]
                )
                actual = data["patterns"][0]["record_probabilities"]
                self.assertEqual(int(np.argmax(actual)), expected)
                self.assertAlmostEqual(actual[expected], 1)

    def test_nonclifford_fixed_fault_matches_aer(self):
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        for gate, method in [("X_ERROR", "x"), ("Y_ERROR", "y"), ("Z_ERROR", "z")]:
            data, _ = self.run_study(f"H 0\nT 0\n{gate}(1) 0\nT_DAG 0\nH 0\nM 0\n")
            oracle = QuantumCircuit(1)
            oracle.h(0)
            oracle.t(0)
            getattr(oracle, method)(0)
            oracle.tdg(0)
            oracle.h(0)
            oracle.save_probabilities()
            probabilities = (
                AerSimulator(method="statevector").run(oracle).result().data()["probabilities"]
            )
            np.testing.assert_allclose(
                data["patterns"][0]["record_probabilities"], probabilities, atol=1e-12
            )

    def test_correlated_else_chain_and_stochastic_cliffords_match_stim(self):
        import stim

        text = "H 0\nCX 0 1\nE(0.3) X0\nELSE_CORRELATED_ERROR(0.7) X1 Z0\nM 0 1\n"
        data, texts = self.run_study(text, patterns=512)
        actual = np.mean([p["record_probabilities"] for p in data["patterns"]], axis=0)
        records = stim.Circuit(text).compile_sampler(seed=42).sample(100000)
        reference = np.bincount(records @ np.array([1, 2]), minlength=4) / len(records)
        np.testing.assert_allclose(actual, reference, atol=0.04)
        self.assertTrue(all(p["quantum_faults"] <= 1 for p in data["patterns"]))
        for converted in texts:
            self.assertNotIn("CORRELATED_ERROR", converted)

    def test_readout_and_feedback_remain_dynamic(self):
        data, texts = self.run_study(
            "X_ERROR(1) 0\nM 0\nREADOUT_NOISE(0.1,0.3) rec[-1]\n"
            "CX rec[-1] 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
            shots=20000,
        )
        self.assertIn("READOUT_NOISE", texts[0])
        self.assertIn("CX rec[-1] 1", texts[0])
        self.assertNotIn("record_probabilities", data["patterns"][0])
        fraction = data["patterns"][0]["metrics"]["survivor_observable_ones"][0] / 20000
        self.assertAlmostEqual(fraction, 0.7, delta=0.02)

    def test_reset_hidden_records_and_feedback(self):
        data, texts = self.run_study(
            "H 0\nT 0\nX_ERROR(0.3) 0\nR 0\nH 1\nM 1\nCX rec[-1] 0\nM 0\n",
            patterns=16,
        )
        for pattern in data["patterns"]:
            np.testing.assert_allclose(
                pattern["record_probabilities"], [0.5, 0, 0, 0.5], atol=1e-12
            )
        self.assertTrue(all("R 0" in text for text in texts))

    def test_quantum_fault_count_excludes_readout(self):
        data, _ = self.run_study(
            "X_ERROR(0.1) 0\nY_ERROR(0.4) 1\nDEPOLARIZE1(0.2) 2\nM(0.3) 0 1 2\n",
            patterns=32,
            k=2,
        )
        self.assertTrue(all(p["quantum_faults"] == 2 for p in data["patterns"]))

    def test_certain_depolarizing_faults_never_become_identity(self):
        data, _ = self.run_study("DEPOLARIZE2(1) 0 1\nM 0 1\n", patterns=64, k=1)
        self.assertTrue(all(p["quantum_faults"] == 1 for p in data["patterns"]))

    def test_fixed_count_uses_conditional_site_weights(self):
        data, _ = self.run_study("X_ERROR(0.1) 0\nX_ERROR(0.4) 1\nM 0 1\n", patterns=1024, k=1)
        actual = np.mean([p["record_probabilities"] for p in data["patterns"]], axis=0)
        np.testing.assert_allclose(actual, [0, 1 / 7, 6 / 7, 0], atol=0.03)

    def test_export_round_trip_preserves_measurement_parity_and_feedback(self):
        data, texts = self.run_study(
            "H 0\nT 0\nX_ERROR(0.5) 0\nMPP X0*Z1\n"
            "CZ rec[-1] 0\nMX 0\nDETECTOR rec[-1] rec[-2]\n",
            patterns=8,
        )
        for pattern, text in zip(data["patterns"], texts):
            replay, _ = self.run_study(text)
            np.testing.assert_allclose(
                pattern["record_probabilities"],
                replay["patterns"][0]["record_probabilities"],
                atol=1e-12,
            )

    def test_fresh_latency_paths_collect_the_same_requested_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.stim"
            source.write_text("X_ERROR(1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
            for strategy in ["shared", "specialized"]:
                output = subprocess.check_output(
                    [str(self.binary), "--latency", strategy, str(source), "8", "--postselect-all"],
                    text=True,
                )
                data = json.loads(output)
                self.assertEqual(data["discarded"], 0)
                self.assertEqual(data["survivor_observable_zero_ones"], 8)

    def test_unsupported_instruments_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.stim"
            source.write_text("LOSS(0.1) 0\nM 0\n")
            result = subprocess.run([str(self.binary), str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported specialization instruction", result.stderr)


if __name__ == "__main__":
    unittest.main()
