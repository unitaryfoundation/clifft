"""Sparse growth contractions checked against dense projectors and full histories."""

import itertools
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fold_check import PhaseMonomial
from msc_boundaries import BoundaryPlan, compile_boundaries
from msc_growth_contraction import FactoredTerm, GrowthContraction, local_pauli
from msc_instruments import BoundInstrument, InstrumentPlan
from msc_protocol import Program
from study_msc_gadgets import terminal_code
from study_msc_growth_contraction import eigenstate, gadget_payload, observe_history
from study_msc_protocol import FIXTURES, load


class MscGrowthContractionTest(unittest.TestCase):
    program: Program
    boundaries: list[BoundaryPlan]
    instrument: InstrumentPlan
    plan: GrowthContraction
    encoding: np.ndarray
    input_matrices: list[np.ndarray]
    logical_matrices: list[np.ndarray]

    @classmethod
    def setUpClass(cls):
        cls.program = load(5)
        cls.boundaries = compile_boundaries(cls.program, load(3))
        cls.instrument = InstrumentPlan(cls.boundaries[-1])
        cls.plan = GrowthContraction(cls.boundaries[0], cls.instrument)
        basis, _ = terminal_code(
            (FIXTURES / "msc_d3_inject_cultivate_p1e-3.stim").read_text(),
            list(load(3).regions.values())[-1].data,
        )
        cls.encoding = np.zeros((128, 2), dtype=complex)
        for logical, support in enumerate(basis):
            for b in support:
                cls.encoding[b, logical] = 1 / np.sqrt(len(support))
        cls.input_matrices = [
            local_pauli(cls.instrument.input_paulis[k], cls.plan.data).to_unitary_matrix(
                endian="little"
            )
            for k in cls.plan.data_rows
        ]
        cls.logical_matrices = [
            local_pauli(p, cls.plan.data).to_unitary_matrix(endian="little")
            for p in cls.instrument.logical_paulis
        ]

    def synthetic(self, sector, logical_signs=(0, 1, 0), seed=917):
        rng = np.random.default_rng(seed + sector)
        signs = rng.integers(2, size=len(self.instrument.input_paulis)).tolist()
        for j, k in enumerate(self.plan.data_rows):
            signs[k] = (sector >> j) & 1
        bound = BoundInstrument(True, tuple(signs), logical_signs, 2.0**-18)
        projected = {self.plan.ancillas[q]: (k, q) for k, q, _ in self.plan.ancilla_rows}
        terms = []
        for _ in range(2):
            ancillas = []
            for q in self.plan.ancillas:
                if q in projected:
                    k, _ = projected[q]
                    axis = "_XYZ"[self.instrument.input_paulis[k][q]]
                    v = eigenstate(axis, signs[k])
                    if q in (1, 2, 4):
                        v = rng.normal(size=2) + 1j * rng.normal(size=2)
                        v /= np.linalg.norm(v)
                else:
                    v = eigenstate("Y", 1) * np.exp(1j * rng.uniform(-np.pi, np.pi))
                ancillas.append(v)
            terms.append(
                FactoredTerm(
                    PhaseMonomial(
                        7,
                        int(rng.integers(128)),
                        int(rng.integers(8)),
                        rng.integers(8, size=7).tolist(),
                    ),
                    np.array(ancillas),
                    complex(*rng.normal(size=2)),
                )
            )
        logical = rng.normal(size=2) + 1j * rng.normal(size=2)
        logical /= np.linalg.norm(logical)
        frame = tuple(map(int, rng.integers(128, size=2)))
        return logical, frame, terms, bound

    def dense_density(self, logical, frame, terms, bound):
        source = self.encoding @ logical
        framed = np.empty(128, dtype=complex)
        x, z = frame
        for b in range(128):
            framed[b ^ x] = (-1) ** ((z & b).bit_count() % 2) * source[b]
        data = np.zeros(128, dtype=complex)
        for term in terms:
            scalar = term.weight
            for k, q, _ in self.plan.ancilla_rows:
                physical = self.plan.ancillas[q]
                p = self.instrument.input_paulis[k]
                target = eigenstate("_XYZ"[p[physical]], bound.input_signs[k])
                scalar *= np.vdot(target, term.ancillas[q])
            for q in self.plan.unprojected:
                scalar *= np.vdot(terms[0].ancillas[q], term.ancillas[q])
            for b in range(128):
                phase = term.data.global_phase + sum(
                    term.data.linear[q] * ((b >> q) & 1) for q in range(7)
                )
                data[b ^ term.data.flips] += scalar * np.exp(1j * np.pi * phase / 4) * framed[b]
        for k, matrix in zip(self.plan.data_rows, self.input_matrices, strict=True):
            data = (data + (-1) ** bound.input_signs[k] * (matrix @ data)) / 2
        norm = float(np.vdot(data, data).real)
        axes = [
            (-1) ** sign * np.vdot(data, matrix @ data).real
            for sign, matrix in zip(bound.logical_signs, self.logical_matrices, strict=True)
        ]
        x, y, z = axes
        return (
            bound.probability_scale * np.array([[norm + z, x - 1j * y], [x + 1j * y, norm - z]]) / 2
        )

    def test_every_syndrome_and_logical_frame_against_independent_dense_projectors(self):
        comparisons = 0
        for sector, sx, sz in itertools.product(range(64), range(2), range(2)):
            args = self.synthetic(sector, (sx, sx ^ sz ^ 1, sz))
            output = self.plan.contract(*args)
            expected = self.dense_density(*args)
            scale = max(float(np.trace(expected).real), 1e-20)
            error = np.max(np.abs(np.outer(output, output.conj()) - expected))
            self.assertLess(error / scale, 2e-11)
            comparisons += 1
        self.assertEqual(comparisons, 256)

    def test_destructive_interference_and_opposite_ancilla_projection(self):
        logical, frame, terms, bound = self.synthetic(0)
        first = terms[0]
        opposite = FactoredTerm(first.data, first.ancillas.copy(), -first.weight)
        np.testing.assert_allclose(
            self.plan.contract(logical, frame, [first, opposite], bound), 0, atol=1e-17
        )
        # A phase on the unmeasured spectator is relative between coherent terms.
        opposite.weight = first.weight
        opposite.ancillas[self.plan.unprojected[0]] *= -1
        np.testing.assert_allclose(
            self.plan.contract(logical, frame, [first, opposite], bound), 0, atol=1e-17
        )
        k, q, _ = self.plan.ancilla_rows[0]
        axis = "_XYZ"[self.instrument.input_paulis[k][self.plan.ancillas[q]]]
        first.ancillas[q] = eigenstate(axis, bound.input_signs[k] ^ 1)
        np.testing.assert_allclose(
            self.plan.contract(logical, frame, [first], bound), 0, atol=1e-17
        )
        np.testing.assert_array_equal(
            self.plan.contract(logical, frame, terms, replace(bound, reachable=False)), 0
        )

    def test_execution_needs_no_stim_or_dependency_discovery(self):
        args = self.synthetic(43)
        expected = self.plan.contract(*args)
        with (
            patch("msc_growth_contraction.stim", None),
            patch(
                "msc_growth_contraction.sparse_basis",
                side_effect=AssertionError("runtime compilation"),
            ),
        ):
            np.testing.assert_array_equal(self.plan.contract(*args), expected)

    def test_declines_entangled_unprojected_spectator_and_coupled_input_projector(self):
        logical, frame, terms, bound = self.synthetic(0)
        q = self.plan.unprojected[0]
        terms[1].ancillas[q] = eigenstate("Y", 0)
        with self.assertRaisesRegex(ValueError, "unprojected spectator differs"):
            self.plan.contract(logical, frame, terms, bound)
        with patch.object(
            self.instrument, "input_paulis", [p.copy() for p in self.instrument.input_paulis]
        ):
            self.instrument.input_paulis[0][self.plan.data[0]] = "X"
            with self.assertRaisesRegex(ValueError, "couples data and ancillas"):
                GrowthContraction(self.boundaries[0], self.instrument)

    def test_complete_original_histories_include_root_fault_and_readout_feedback(self):
        source = Path(__file__).parent / "research" / "msc_protocol_data.json"
        cases = json.loads(source.read_text())["circuits"][1]["cases"]
        selected = [cases[k] for k in (0, 2, 3, 4, 5, -1)]
        selected += [c for c in cases if c["name"].startswith("feedforward_readout_")][:1]
        for case in selected:
            result = observe_history(
                self.program, self.boundaries, self.instrument, self.plan, case
            )
            self.assertLess(result["normalized_density_error"], 1e-10)
        outcomes = list(map(int, cases[0]["outcomes"]))
        region = list(self.program.regions.values())[0]
        reset = next(g for g in region.gates if g.name == "RX")
        outcomes[self.program.measurement_at[reset.line, 0]] ^= 1
        with self.assertRaisesRegex(ValueError, "impossible gadget reset outcome"):
            gadget_payload(self.program, self.boundaries[0], self.plan, {}, outcomes)


if __name__ == "__main__":
    unittest.main()
