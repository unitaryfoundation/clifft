"""Static fault phases and complete sampled MSC histories against independent paths."""

import json
import os
import random
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
from export_msc_sampler import export
from msc_gadgets import terms
from msc_protocol import evaluate
from msc_reference import classical_reference
from msc_sampling import Sampler
from msc_static_gadgets import GadgetPlan
from study_msc_growth_contraction import gadget_payload, logical_bloch
from study_msc_protocol import load
from study_msc_sampling import validate
from study_msc_terminal_contraction import terminal_payload


class SamplingTests(unittest.TestCase):
    plans: ClassVar[dict[int, Sampler]]

    @classmethod
    def setUpClass(cls):
        cls.plans = {d: Sampler(load(d)) for d in (3, 5)}

    def test_static_payloads_preserve_all_stored_relative_phases(self):
        circuits = json.loads(
            (Path(__file__).parent / "research/msc_protocol_data.json").read_text()
        )["circuits"]
        for circuit in circuits:
            plan = self.plans[circuit["distance"]]
            pairs: list[tuple] = [
                (plan.terminal_payload, terminal_payload, plan.terminal, plan.boundaries[-1])
            ]
            if plan.growth:
                pairs.append((plan.growth_payload, gadget_payload, plan.growth, plan.boundaries[0]))
            for case in circuit["cases"]:
                history, outcomes = dict(case["faults"]), list(map(int, case["outcomes"]))
                for compiled, offline, contraction, boundary in pairs:
                    frame, payload = compiled.bind(history, outcomes)
                    expected_frame, expected = offline(
                        plan.program, boundary, contraction, history, outcomes
                    )
                    self.assertEqual(frame, expected_frame)
                    for a, b in zip(payload, expected, strict=True):
                        self.assertEqual(
                            (a.data.flips, a.data.global_phase, a.data.linear),
                            (b.data.flips, b.data.global_phase, b.data.linear),
                        )
                        np.testing.assert_allclose(a.ancillas, b.ancillas, atol=1e-14, rtol=0)
                        self.assertAlmostEqual(a.weight, b.weight)

    def test_static_gadgets_cover_all_single_paulis_and_mixed_histories(self):
        for plan in self.plans.values():
            program = plan.program
            bits = plan.terminal_payload.bits
            for region in program.regions.values():
                compiled = GadgetPlan(program, region, bits)
                eligible = [
                    (k, s)
                    for k, s in enumerate(program.sites)
                    if not s.readout and region.start <= s.line <= region.end
                ]
                histories: list[dict[int, str]] = [{}]
                for site, item in eligible:
                    for q in range(len(item.wires)):
                        for axis in "XYZ":
                            choice = "I" * q + axis + "I" * (len(item.wires) - q - 1)
                            if choice in item.choices:
                                histories.append({site: choice})
                for seed in range(12):
                    rng = random.Random(seed)
                    histories.append(
                        {k: rng.choice(s.choices) for k, s in eligible if rng.random() < 0.2}
                    )
                for history in histories:
                    local = [
                        (sum(g.line < line for g in region.gates), axis, q)
                        for line, actions in program.faults(history).items()
                        for axis, q in actions
                        if q in region.wires
                    ]
                    for outcome in (0, 1):
                        for a, b in zip(
                            compiled.bind(bits.encode(program, history), outcome),
                            terms(region, outcome, local),
                            strict=True,
                        ):
                            self.assertEqual(
                                (a.flips, a.global_phase, a.linear),
                                (b.flips, b.global_phase, b.linear),
                            )

    def test_complete_sample_weights_states_and_outputs(self):
        for distance, plan in self.plans.items():
            for seed in range(16):
                history = plan.program.history(100 + seed, scale=20) if seed >= 8 else None
                sampled = plan.sample(seed, history)
                probability, state = {}, {}

                def observe(op, reference, norm):
                    if op.line == plan.terminal.measurement.line + 1:
                        state["bloch"] = logical_bloch(reference, plan.terminal.data, norm)

                actual = evaluate(
                    plan.program, sampled["faults"], sampled["outcomes"], before_instruction=observe
                )
                probability["error"] = abs(
                    sampled["probability"] / actual["trajectory_probability"] - 1
                )
                self.assertLess(probability["error"], 1e-10, (distance, seed))
                x, y, z = state["bloch"]
                density = np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2
                np.testing.assert_allclose(
                    np.outer(sampled["logical"], sampled["logical"].conj()),
                    density,
                    atol=1e-10,
                    rtol=0,
                )
                outputs = classical_reference(plan.program, sampled["faults"], sampled["outcomes"])
                self.assertEqual(outputs, {k: sampled[k] for k in outputs})

    def test_terminal_fourier_probabilities_match_every_small_sector(self):
        plan = self.plans[3]
        rng = np.random.default_rng(9)
        for seed in range(5):
            sampled = plan.sample(seed, plan.program.history(seed, scale=10))
            frame, payload = plan.terminal_payload.bind(sampled["faults"], sampled["outcomes"])
            logical = rng.normal(size=2) + 1j * rng.normal(size=2)
            logical /= np.linalg.norm(logical)
            labels, weights = plan.terminal_code.syndrome_weights(logical, frame, payload)
            for sector in range(64):
                syndrome = [(sector >> k) & 1 for k in range(6)]
                x = sum(
                    syndrome[j] << k for j, (axis, k) in enumerate(plan.terminal_code.order) if axis
                )
                z = sum(
                    syndrome[j] << k
                    for j, (axis, k) in enumerate(plan.terminal_code.order)
                    if not axis
                )
                expected = plan.terminal.contract(logical, frame, payload, syndrome)
                actual = sum(weights[k, x] for k, label in enumerate(labels) if label == z)
                self.assertAlmostEqual(actual, float(np.vdot(expected, expected).real), places=12)
            self.assertAlmostEqual(
                weights.sum(), plan.terminal_code.norm(logical, frame, payload), places=12
            )

    @unittest.skipUnless(os.environ.get("CLIFFT_MSC_SAMPLER"), "requires native research sampler")
    def test_native_complete_records_weights_and_states(self):
        with tempfile.TemporaryDirectory(prefix="msc-native-test-") as directory:
            path = Path(directory) / "plan.txt"
            for plan in self.plans.values():
                original = plan.program.sites
                try:
                    for scale in (1, 20):
                        plan.program.sites = [
                            replace(s, probability=scale * s.probability) for s in original
                        ]
                        export(plan, path)
                        samples = subprocess.check_output(
                            [os.environ["CLIFFT_MSC_SAMPLER"], str(path), "783", "4", "sample"],
                            text=True,
                        )
                        for k, line in enumerate(samples.splitlines()):
                            probe = os.environ.get("CLIFFT_MSC_RECORD_PROBE") if k == 0 else None
                            validate(plan, json.loads(line), probe)
                finally:
                    plan.program.sites = original

    def test_sampling_does_not_reenter_topology_compilation(self):
        with (
            patch("msc_gadgets.terms", side_effect=AssertionError("offline gadget binder")),
            patch("msc_sampling.FixedHistoryPlan.__init__", side_effect=AssertionError("compiler")),
            patch(
                "msc_static_gadgets.GadgetPlan.__init__",
                side_effect=AssertionError("gadget compiler"),
            ),
            patch(
                "msc_sampling.AffineRecords.__init__",
                side_effect=AssertionError("parity elimination"),
            ),
            patch("msc_protocol.evaluate", side_effect=AssertionError("CH reference")),
        ):
            for plan in self.plans.values():
                plan.sample(44)


if __name__ == "__main__":
    unittest.main()
