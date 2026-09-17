"""Coherent cat-measurement validation for the compiled fault-frame bridge."""

import json
import os
import random
import subprocess
import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np
from fold_blocks import Fold, LinearPlan, NoiseLayout, Protocol
from fold_cultivation import Operation
from fold_frame_bridge import Bridge
from study_frame_bridge import boundary_cases
from test_fold_fault_frame import unitary


def signature(terms):
    return [
        (
            abs(weight),
            (op.global_phase + (4 if weight < 0 else 0)) % 8,
            op.flips,
            tuple(op.linear),
            sorted(op.edges),
        )
        for weight, op in terms
    ]


def noisy_unitaries(layout, faults):
    result = []
    for boundary, sites in enumerate(layout.boundaries):
        for q, x, z in sites:
            if faults & z:
                result.append(Operation("Z", (q,)))
            if faults & x:
                result.append(Operation("X", (q,)))
        if boundary < len(layout.operations) and layout.operations[boundary].name != "M":
            result.append(layout.operations[boundary])
    return result


class FrameBridgeTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CLIFFT_FRAME_BRIDGE"), "requires native bridge probe")
    def test_native_full_sampler_and_incoming_frame_contract(self):
        result = json.loads(
            subprocess.check_output([os.environ["CLIFFT_FRAME_BRIDGE"], "100", "1"])
        )
        self.assertEqual(result["fixtures"], 96)
        self.assertGreater(result["branch_histories"], 2000)
        self.assertEqual(result["incoming_pauli_histories"], 768)
        self.assertLess(result["max_fixture_error"], 2e-12)
        self.assertLess(result["max_output_error"], 2e-12)
        self.assertEqual(result["measurement_values"], result["passed"][0] * 350)
        self.assertEqual(result["probe_values"], result["passed"][0] * 5)

    @unittest.skipUnless(
        os.environ.get("CLIFFT_FRAME_BRIDGE_HELD_OUT"), "requires held-out bridge probe"
    )
    def test_native_changed_growth_and_syndrome_schedule(self):
        result = json.loads(
            subprocess.check_output([os.environ["CLIFFT_FRAME_BRIDGE_HELD_OUT"], "100", "1"])
        )
        self.assertEqual(result["fixtures"], 96)
        self.assertLess(result["max_fixture_error"], 2e-12)
        self.assertLess(result["max_output_error"], 2e-12)
        self.assertEqual(result["detector_values"], result["passed"][0] * 348)

    def test_legal_cross_check_faults_create_cat_data_corrections_and_survive(self):
        protocol = Protocol(7)
        folds = [
            (f, g[1][0])
            for _, f, g in protocol.groups
            if isinstance(f, Fold) and f.plan.distance == 7
        ]
        first, second = (Bridge(f) for f, _ in folds)
        for name, history in boundary_cases(protocol):
            if not name.startswith("carried_data"):
                continue
            incoming = first.frame.evaluate(first.fold.core.encode(history[folds[0][1]]))
            self.assertTrue(first.frame.data_only(incoming, set(first.fold.mapping)))
            outgoing = second.frame.evaluate(
                second.fold.core.encode(history[folds[1][1]]), incoming
            )
            self.assertFalse(second.frame.data_only(outgoing, set(second.fold.mapping)))
            self.assertEqual(outgoing.flips, 0)
            self.assertEqual(outgoing.edges.bit_count(), 1)
            self.assertTrue(
                any((outgoing.edges >> edge) & 1 and kind == 1 for edge, kind, *_ in second.routes)
            )
            result = protocol.evaluate(protocol.encode(history))
            self.assertAlmostEqual(result.acceptance, 0.25, delta=2e-12)

    def test_noisy_cat_instrument_against_aer_on_arbitrary_data_inputs(self):
        def layout(ops):
            return NoiseLayout(ops, [True] * len(ops), 3)

        preparation = layout([Operation("H", (2,))])
        core = layout(
            [
                Operation("CCZ", (2, 0, 1)),
                Operation("T", (0,)),
                Operation("CX", (2, 0)),
                Operation("T_DAG", (0,)),
            ]
        )
        decode = layout([Operation("H", (2,)), Operation("M", (2,))])
        fold = SimpleNamespace(
            core=core,
            mapping=[0, 1],
            cats=[2],
            data_index={0: 0, 1: 1},
            cat_index={2: 0},
            plan=SimpleNamespace(width=2, edges=[(0, 1)]),
            preparation=LinearPlan(preparation),
            decode=LinearPlan(decode),
            equal_flag_mask=0,
            decode_table=np.array([[1, 1], [1, -1]]),
        )
        bridge = Bridge(cast(Fold, fold))
        rng = random.Random(188)
        for _ in range(64):
            faults = [rng.getrandbits(len(p.slots)) for p in (preparation, core, decode)]
            operations = [
                op
                for p, f in zip((preparation, core, decode), faults, strict=True)
                for op in noisy_unitaries(p, f)
            ]
            expected = unitary(3, operations)[:4, :4]
            actual = np.zeros((4, 4), dtype=complex)
            for weight, op in bridge.branches(*faults):
                for column in range(4):
                    row, phase = op.column(column)
                    actual[row, column] += weight * np.exp(1j * np.pi * phase / 4)
            # Preparation/decoder Pauli maps discard a whole-history phase,
            # which is shared by every data input and both coherent branches.
            overlap = np.vdot(actual, expected)
            if abs(overlap) > 1e-12:
                actual *= overlap / abs(overlap)
            np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_all_single_faults_and_dense_histories_match_existing_fold_operators(self):
        protocol = Protocol(7)
        rng = random.Random(61103)
        seen = set()
        for _, fold, _ in protocol.groups:
            if not isinstance(fold, Fold) or fold.plan.distance in seen:
                continue
            seen.add(fold.plan.distance)
            bridge = Bridge(fold)
            cases: list[tuple[int, ...]] = [(0, 0, 0)]
            layouts = [fold.preparation.layout, fold.core, fold.decode.layout]
            for index, layout in enumerate(layouts):
                for sites in layout.boundaries:
                    for _, x, z in sites:
                        for fault in sorted({x, z, x | z} - {0}):
                            values = [0, 0, 0]
                            values[index] = fault
                            cases.append(tuple(values))
            cases += [
                (
                    0,
                    rng.getrandbits(len(fold.core.slots)),
                    rng.getrandbits(len(fold.decode.layout.slots)),
                )
                for _ in range(128)
            ]
            cases += [tuple(rng.getrandbits(len(p.slots)) for p in layouts) for _ in range(128)]
            for faults in cases:
                self.assertEqual(
                    signature(bridge.branches(*faults)), signature(fold.branches(*faults))
                )


if __name__ == "__main__":
    unittest.main()
