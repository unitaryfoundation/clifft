"""Full-protocol records and fixed-noise composition checks."""

import itertools
import math
import os
import unittest

import numpy as np
import stim
from msc_protocol import Program, evaluate
from msc_reference import materialize, reference
from study_msc_protocol import clifft_reference, compare, load, targeted_histories


class MscProtocolTest(unittest.TestCase):
    def test_full_d3_conditional_probabilities_against_aer(self):
        program = load(3)
        for seed, scale in [(381, 0), (7481, 1), (9672, 10)]:
            history = program.history(seed, scale)
            oracle = reference(program, history, seed)
            actual = compare(program, history, oracle)
            self.assertEqual(len(actual["records"]), 21)
            self.assertEqual(len(actual["detectors"]), 20)
            self.assertLessEqual(actual["peak_terms"], 8)

    def test_readout_flip_changes_records_without_changing_quantum_outcomes(self):
        program = load(3)
        history = dict(targeted_histories(program))["root_readout_flip"]
        ideal = reference(program, {}, 1203)
        noisy = reference(program, history, 1203)
        self.assertEqual(ideal["outcomes"], noisy["outcomes"])
        self.assertEqual(
            sum(a != b for a, b in zip(ideal["records"], noisy["records"], strict=True)), 1
        )
        self.assertNotEqual(ideal["detectors"], noisy["detectors"])
        np.testing.assert_allclose(ideal["probabilities"], noisy["probabilities"], atol=1e-13)
        compare(program, history, noisy)

    def test_root_reset_erases_physical_fault_but_keeps_hidden_outcome(self):
        program = load(3)
        history = dict(targeted_histories(program))["reset_erased_Y"]
        ideal = reference(program, {}, 4410)
        noisy = reference(program, history, 4410)
        self.assertEqual(
            sum(a != b for a, b in zip(ideal["outcomes"], noisy["outcomes"], strict=True)), 1
        )
        for field in ("records", "detectors", "observables"):
            self.assertEqual(ideal[field], noisy[field])
        compare(program, history, noisy)

    def test_accepted_logical_error_is_retained(self):
        program = load(3)
        history = dict(targeted_histories(program))["logical_z_tail"]
        actual = compare(program, history, reference(program, history, 6107))
        self.assertNotIn("1", actual["detectors"])
        self.assertEqual(actual["observables"], {"0": 1})

    @unittest.skipUnless(
        os.environ.get("CLIFFT_MSC_RECORD_PROBE"), "requires elementary-gate probe"
    )
    def test_full_d5_growth_and_feedforward_against_elementary_clifft(self):
        program = load(5)
        targeted = dict(targeted_histories(program))
        histories = [targeted["logical_z_tail"], program.history(9604, 10)]
        histories.append(
            next(h for n, h in targeted.items() if n.startswith("feedforward_readout_"))
        )
        for k, history in enumerate(histories):
            oracle = clifft_reference(
                program, history, 8720 + k, os.environ["CLIFFT_MSC_RECORD_PROBE"], True
            )
            actual = compare(program, history, oracle)
            self.assertEqual(len(actual["records"]), 112)
            self.assertEqual(len(actual["detectors"]), 107)
            self.assertLessEqual(actual["peak_terms"], 16)
            self.assertEqual(len(oracle["probabilities"]), 230)

    def test_hidden_reset_trajectory_probabilities_sum_to_one(self):
        program = Program(
            "RX 0\nT 0\nMX 0\nR 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
        )
        total = 0.0
        measured_one = 0.0
        for outcomes in itertools.product((0, 1), repeat=4):
            try:
                result = evaluate(program, {}, outcomes)
            except ValueError as error:
                self.assertIn("zero probability", str(error))
                continue
            total += result["trajectory_probability"]
            if result["records"][0]:
                measured_one += result["trajectory_probability"]
        self.assertAlmostEqual(total, 1.0, delta=2e-13)
        self.assertAlmostEqual(measured_one, (1 - 1 / math.sqrt(2)) / 2, delta=2e-13)

    def test_classical_dependencies_and_fixed_readout_against_stim(self):
        text = (
            "R 0 1\nX 0\nM(1) 0\nCX rec[-1] 1\nM 1\n"
            "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        )
        program = Program(text)
        history = {0: "flip"}
        oracle = reference(program, history, 41)
        actual = evaluate(program, history, oracle["outcomes"])
        circuit = stim.Circuit(text)
        samples = circuit.compile_sampler(seed=41).sample(8)
        self.assertTrue(np.all(samples == np.array(actual["records"])))
        # This contract returns declared parities. Stim's ordinary detector
        # sampler instead reports flips relative to a noiseless reference.
        detectors, observables = circuit.compile_m2d_converter(skip_reference_sample=True).convert(
            measurements=samples, separate_observables=True
        )
        self.assertTrue(np.all(detectors == np.array(actual["detectors"])))
        self.assertTrue(np.all(observables == np.array([actual["observables"]["0"]])))
        self.assertEqual(
            len(stim.Circuit(materialize(program, history)).compile_sampler().sample(1)[0]),
            len(program.measurements),
        )

    def test_invalid_dependencies_and_unsupported_region_controls_decline(self):
        with self.assertRaises(ValueError):
            Program("M 0\nDETECTOR rec[-2]\n")
        with self.assertRaises(ValueError):
            Program("T(0.25) 0\n")
        with self.assertRaises(ValueError):
            Program("M 0\nT 0\nMPP Y0\nOBSERVABLE_INCLUDE(0) rec[-1]\nT_DAG 0\n")
        with self.assertRaises(ValueError):
            load(3).validate_history({999999: "X"})


if __name__ == "__main__":
    unittest.main()
