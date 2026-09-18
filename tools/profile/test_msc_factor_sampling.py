"""Coherent CSS factor marginals against sparse, dense, and stabilizer references."""

import json
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from msc_factor_sampling import FactorCode, generated_color_checks
from msc_protocol import evaluate
from msc_sampling import Sampler, SparseCode
from study_msc_factor_sampling import clifford_reference, export_native, synthetic
from study_msc_protocol import load
from test_msc_terminal_contraction import DenseReference


class FactorSamplingTest(unittest.TestCase):
    def test_all_small_prefix_marginals_and_coherent_outputs(self):
        for distance in (3, 5):
            checks, width = generated_color_checks(distance)
            code = FactorCode(checks, width)
            sparse = SparseCode(checks, width)
            for seed in range(3):
                logical, frame, terms = synthetic(code, 20 + seed)
                prepared = code.prepare(logical, frame, terms)
                labels, weights = sparse.syndrome_weights(logical, frame, terms)
                for count in range(code.rank + 1):
                    selected = code.sampling_order[:count]
                    mask = sum(1 << k for k in selected)
                    for pattern in range(min(8, 1 << count)):
                        bits = sum(((pattern >> j) & 1) << k for j, k in enumerate(selected))
                        for label in set(labels):
                            expected = sum(
                                weights[t, s]
                                for t, z in enumerate(labels)
                                if z == label
                                for s in range(1 << code.rank)
                                if s & mask == bits
                            )
                            self.assertAlmostEqual(
                                prepared.mass(label, count, bits), expected, places=11
                            )
                outcomes, probability = prepared.sample(random.Random(seed))
                output = code.contract(logical, frame, terms, outcomes)
                self.assertAlmostEqual(float(np.vdot(output, output).real), probability, places=11)

    def test_original_histories_preserve_weights_and_amplitudes(self):
        circuits = json.loads(
            (Path(__file__).parent / "research/msc_protocol_data.json").read_text()
        )["circuits"]
        for circuit in circuits:
            plan = Sampler(load(circuit["distance"]))
            code = FactorCode(plan.terminal.checks, plan.terminal.width)
            logical = np.array([1, 0.3 + 0.7j])
            logical /= np.linalg.norm(logical)
            for case in circuit["cases"]:
                history, outcomes = dict(case["faults"]), list(map(int, case["outcomes"]))
                frame, payload = plan.terminal_payload.bind(history, outcomes)
                expected = plan.terminal_code.norm(logical, frame, payload)
                self.assertAlmostEqual(code.norm(logical, frame, payload), expected, places=11)
                syndrome = [outcomes[k] for k in plan.terminal.events]
                np.testing.assert_allclose(
                    code.contract(logical, frame, payload, syndrome),
                    plan.terminal.contract(logical, frame, payload, syndrome),
                    rtol=0,
                    atol=1e-11,
                )

    def test_full_original_sampling_composes_with_factor_replacement(self):
        for distance in (3, 5):
            sampler = Sampler(load(distance))
            code = FactorCode(sampler.terminal.checks, sampler.terminal.width)
            with (
                patch.object(sampler, "terminal_code", code),
                patch.object(sampler.terminal, "contract", code.contract),
            ):
                for seed in range(4):
                    sample = sampler.sample(seed, sampler.program.history(923 + seed, scale=20))
                    reference = evaluate(sampler.program, sample["faults"], sample["outcomes"])
                    self.assertLess(
                        abs(sample["probability"] / reference["trajectory_probability"] - 1), 1e-10
                    )
                    for key in ("records", "detectors", "observables"):
                        self.assertEqual(sample[key], reference[key])

    def test_large_blocks_against_independent_stabilizer_projection(self):
        for distance in (7, 9):
            checks, width = generated_color_checks(distance)
            with patch(
                "msc_sampling.SparseCode.__init__",
                side_effect=AssertionError("sparse basis construction"),
            ):
                code = FactorCode(checks, width)
                logical, frame, terms = synthetic(code, 763)
                with (
                    patch(
                        "msc_factor_sampling.elimination_order",
                        side_effect=AssertionError("runtime topology"),
                    ),
                    patch(
                        "msc_factor_sampling.FactorPlan.__init__",
                        side_effect=AssertionError("runtime compilation"),
                    ),
                ):
                    prepared = code.prepare(logical, frame, terms)
                    outcomes, _ = prepared.sample(random.Random(79))
                    xs = sum(
                        bit << k for bit, (x, k) in zip(outcomes, code.order, strict=True) if x
                    )
                    zs = sum(
                        bit << k for bit, (x, k) in zip(outcomes, code.order, strict=True) if not x
                    )
                    for count in (0, code.rank // 2, code.rank):
                        expected = clifford_reference(
                            checks, logical, frame, terms, (xs, zs), count, code.sampling_order
                        )
                        actual = prepared.mass(zs, count, xs)
                        self.assertLess(abs(actual / expected - 1), 1e-9)
                self.assertLessEqual(max(p.max_joint for p in code.prefixes), code.single.max_joint)

    def test_non_clifford_diagonal_phases_against_aer(self):
        checks, width = generated_color_checks(3)
        code = FactorCode(checks, width)
        reference = DenseReference(SimpleNamespace(width=width, checks=checks, ancillas=[]))
        for seed in (415, 423):
            logical, frame, terms = synthetic(code, seed)
            for k, term in enumerate(terms):
                term.data.linear[k] = (term.data.linear[k] + 1) % 8
            syndrome, _ = code.sample_syndrome(logical, frame, terms, random.Random(seed))
            actual = code.contract(logical, frame, terms, syndrome)
            expected = reference.density(logical, frame, terms, syndrome, aer=True)
            np.testing.assert_allclose(
                np.outer(actual, actual.conj()), expected, rtol=0, atol=1e-11
            )

    def test_compilation_declines_a_factor_budget_overflow(self):
        checks, width = generated_color_checks(9)
        with self.assertRaisesRegex(ValueError, "budget"):
            FactorCode(checks, width, entry_limit=1)

    @unittest.skipUnless(os.environ.get("CLIFFT_MSC_FACTOR_NATIVE"), "requires factor worker")
    def test_native_sampling_matches_compiled_probabilities(self):
        with tempfile.TemporaryDirectory(prefix="msc-factor-test-") as directory:
            path = Path(directory) / "plan.txt"
            for distance in (3, 7, 9):
                checks, width = generated_color_checks(distance)
                code = FactorCode(checks, width)
                cases = [synthetic(code, 762 + k) for k in range(4)]
                export_native(code, cases, path)
                samples = subprocess.check_output(
                    [os.environ["CLIFFT_MSC_FACTOR_NATIVE"], str(path), "sample", "8", "713"],
                    text=True,
                )
                for line in samples.splitlines():
                    sample = json.loads(line)
                    logical, frame, terms = cases[sample["case"]]
                    expected = code.prepare(logical, frame, terms).mass(
                        sample["zs"], code.rank, sample["xs"]
                    )
                    self.assertLess(abs(expected / sample["probability"] - 1), 1e-10)


if __name__ == "__main__":
    unittest.main()
