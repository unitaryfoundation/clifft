"""Weighted instrument checks, including impossible and coherent inputs."""

import itertools
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from msc_boundaries import compile_boundaries
from msc_instruments import InstrumentPlan
from msc_protocol import Program
from study_msc_instruments import amplitudes, observe_history
from study_msc_protocol import load


class MscInstrumentsTest(unittest.TestCase):
    def test_arbitrary_complex_inputs_against_dense_elementary_kraus_operators(self):
        # A hidden reset, random Z measurement, repeated measurement constraint,
        # readout-controlled Pauli and S phase distinguish all parts of the map.
        program = Program(
            "X_ERROR(0.1) 1\nR 1\nH 1\nCX 1 0\nZ_ERROR(0.1) 0\n"
            "M(0.1) 1\nM 1\nCZ rec[-2] 0\nS 0\n"
        )
        plan = InstrumentPlan(
            SimpleNamespace(program=program, instructions=program.instructions, data=(0,))
        )
        eye = np.eye(2)
        x = np.array([[0, 1], [1, 0]])
        y = np.array([[0, -1j], [1j, 0]])
        z = np.diag([1, -1])
        h = (x + z) / np.sqrt(2)
        s = np.diag([1, 1j])
        p = [(eye + z) / 2, (eye - z) / 2]
        cx = np.kron(p[0], eye) + np.kron(p[1], x)
        rng = np.random.default_rng(19027)
        comparisons = 0
        for physical, phase, readout in itertools.product(range(2), repeat=3):
            history = {
                k: choice
                for k, bit, choice in [(0, physical, "X"), (1, phase, "Z"), (2, readout, "flip")]
                if bit
            }
            for outcomes in itertools.product(range(2), repeat=3):
                reset, first, second = outcomes
                operator = (
                    np.kron(eye, s @ (z if first ^ readout else eye))
                    @ np.kron(p[second] @ p[first], z if phase else eye)
                    @ cx
                    @ np.kron(h @ (x if reset else eye) @ p[reset] @ (x if physical else eye), eye)
                )
                bound = plan.bind(outcomes, history)
                states = [rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(3)]
                states.extend(np.eye(4, dtype=complex))
                for state in states:
                    state /= np.linalg.norm(state)
                    actual = operator @ state
                    projected = state.copy()
                    if not bound.reachable:
                        projected *= 0
                    else:
                        for probe, sign in zip(plan.input_paulis, bound.input_signs, strict=True):
                            projected = (
                                projected
                                + (-1) ** sign
                                * probe.to_unitary_matrix(endian="little")
                                @ projected
                            ) / 2
                    expected_norm = bound.probability_scale * np.vdot(projected, projected).real
                    self.assertAlmostEqual(expected_norm, np.vdot(actual, actual).real, places=12)
                    for axis, probe, sign in zip(
                        (x, y, z), plan.logical_paulis, bound.logical_signs, strict=True
                    ):
                        expected = (
                            bound.probability_scale
                            * (-1) ** sign
                            * np.vdot(
                                projected, probe.to_unitary_matrix(endian="little") @ projected
                            )
                        )
                        self.assertAlmostEqual(
                            expected, np.vdot(actual, np.kron(eye, axis) @ actual), places=12
                        )
                    norm = np.vdot(actual, actual).real
                    if norm > 1e-20:
                        bloch = [
                            np.vdot(actual, np.kron(eye, axis) @ actual).real / norm
                            for axis in (x, y, z)
                        ]
                        ket = amplitudes(norm, bloch)
                        bipartite = actual.reshape(2, 2)
                        np.testing.assert_allclose(
                            np.outer(ket, ket.conj()), bipartite.T @ bipartite.conj(), atol=1e-12
                        )
                    comparisons += 1
        self.assertEqual(comparisons, 448)

    def test_actual_interval_ranks_and_complete_histories(self):
        source = Path(__file__).parent / "research" / "msc_protocol_data.json"
        for circuit in json.loads(source.read_text())["circuits"]:
            program = load(circuit["distance"])
            boundaries = compile_boundaries(program, load(3))
            plans = [InstrumentPlan(b) for b in boundaries]
            for plan, expected in zip(plans, [(20, 14, 3, 3), (155, 40, 97, 18)]):
                self.assertEqual(
                    (
                        len(plan.events),
                        len(plan.input_rows),
                        len(plan.constraints),
                        plan.random_power,
                    ),
                    expected,
                )
            for k in (0, 7, -1):
                self.assertEqual(
                    len(observe_history(program, boundaries, plans, circuit["cases"][k])),
                    len(plans),
                )

    def test_binding_needs_no_stim_or_topology_and_rejects_inconsistent_records(self):
        program = load(5)
        plan = InstrumentPlan(compile_boundaries(program, load(3))[-1])
        source = Path(__file__).parent / "research" / "msc_protocol_data.json"
        case = json.loads(source.read_text())["circuits"][1]["cases"][0]
        outcomes = list(map(int, case["outcomes"]))
        expected = plan.bind(outcomes, {})
        self.assertTrue(expected.reachable)
        with (
            patch("msc_instruments.stim", None),
            patch("msc_instruments.eliminate", side_effect=AssertionError("runtime elimination")),
        ):
            self.assertEqual(plan.bind(outcomes, {}), expected)
            for row in plan.constraints:
                changed = outcomes.copy()
                changed[(row.records & -row.records).bit_length() - 1] ^= 1
                self.assertFalse(plan.bind(changed, {}).reachable)

    def test_non_clifford_and_lost_logical_axis_decline(self):
        for text, error in [("T 0\n", "not Clifford"), ("M 0\n", "no input pullback")]:
            program = Program(text)
            with self.assertRaisesRegex(ValueError, error):
                InstrumentPlan(
                    SimpleNamespace(program=program, instructions=program.instructions, data=(0,))
                )


if __name__ == "__main__":
    unittest.main()
