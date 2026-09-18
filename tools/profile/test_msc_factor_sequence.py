"""Independent physical-gate checks for repeated noisy logical-block handoffs."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fold_contraction import ROOTS
from msc_factor_sequence import Sequence, local_table
from msc_protocol import Program
from msc_reference import reference
from qiskit_aer import AerSimulator
from study_msc_factor_sequence import validate_sample
from study_msc_sequence_mps import original_circuit


class FactorSequenceTest(unittest.TestCase):
    def test_every_local_fault_binding_against_elementary_matrices(self):
        matrices = [
            np.eye(2),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.diag([1, -1]),
        ]
        t = np.diag([1, ROOTS[1]])
        for faults, branches in enumerate(local_table()):
            p = [matrices[(faults >> (2 * k)) & 3] for k in range(4)]
            for branch, (flip, phase, linear) in enumerate(branches):
                expected = (
                    p[3]
                    @ t.conj().T
                    @ p[2]
                    @ (matrices[2] if branch else matrices[0])
                    @ p[1]
                    @ t
                    @ p[0]
                )
                actual = np.zeros((2, 2), complex)
                for bit in (0, 1):
                    actual[bit ^ flip, bit] = ROOTS[(phase + bit * linear) % 8]
                np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)

    def test_five_rounds_against_original_gate_aer(self):
        for axis in "XYZ":
            sequence = Sequence(3, 0.02, axis)
            history = sequence.history(145)
            aer = reference(Program(sequence.circuit(history)), {}, 731)
            result = sequence.run(history=history, outcomes=aer["outcomes"])
            self.assertLess(abs(result["probability"] / np.prod(aer["probabilities"]) - 1), 1e-10)

    def test_mps_comparator_translation_against_logical_reference(self):
        sequence = Sequence(3, 0, "Y")
        circuit, noise = original_circuit(sequence, expectations=True)
        simulator = AerSimulator(
            method="matrix_product_state",
            max_parallel_threads=1,
            matrix_product_state_truncation_threshold=0.0,
            noise_model=noise,
        )
        result = simulator.run(circuit, shots=1, memory=True, seed_simulator=843).result()
        self.assertTrue(result.success)
        for word in result.get_memory():
            outcomes = list(map(int, reversed(word)))
            sample = sequence.run(history=sequence.history(0), outcomes=outcomes)
            chance = np.prod(
                [
                    (1 + (-1) ** bit * float(result.data()[f"p{k}"])) / 2
                    for k, bit in enumerate(outcomes)
                ]
            )
            self.assertLess(abs(sample["probability"] / chance - 1), 1e-10)

    def test_runtime_uses_precompiled_code_and_binding(self):
        sequence = Sequence(3, 0.02)
        with (
            patch(
                "msc_factor_sampling.elimination_order",
                side_effect=AssertionError("runtime planning"),
            ),
            patch(
                "msc_factor_sampling.FactorPlan.__init__",
                side_effect=AssertionError("runtime compilation"),
            ),
            patch(
                "msc_factor_sequence.local_table",
                side_effect=AssertionError("runtime binding compilation"),
            ),
        ):
            result = sequence.run(741)
        self.assertEqual(len(result["rounds"]), 5)
        self.assertEqual(len(result["records"]), 36)
        self.assertEqual(len(result["detectors"]), 24)

    @unittest.skipUnless(os.environ.get("CLIFFT_MSC_SEQUENCE_NATIVE"), "requires sequence worker")
    def test_native_full_outputs_and_handoffs(self):
        with tempfile.TemporaryDirectory(prefix="msc-sequence-test-") as folder:
            directory = Path(folder)
            plan = directory / "plan.txt"
            for distance in (3, 7, 9):
                sequence = Sequence(distance, 0.02, "Y")
                sequence.export(plan)
                output = subprocess.check_output(
                    [os.environ["CLIFFT_MSC_SEQUENCE_NATIVE"], str(plan), "sample", "2", "1721"],
                    text=True,
                )
                for k, line in enumerate(output.splitlines()):
                    validate_sample(
                        sequence,
                        json.loads(line),
                        directory,
                        os.environ.get("CLIFFT_MSC_RECORD_PROBE") if distance <= 7 else None,
                        coherent=k == 0,
                    )


if __name__ == "__main__":
    unittest.main()
