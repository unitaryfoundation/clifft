"""Sparse terminal-code contractions against independent dense quantum states."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import stim
from fold_blocks import masks
from fold_check import PhaseMonomial
from fold_contraction import compose
from msc_boundaries import BoundaryPlan, compile_boundaries, pauli
from msc_growth_contraction import FactoredTerm, GrowthContraction
from msc_instruments import InstrumentPlan
from msc_protocol import Program
from msc_terminal_contraction import TerminalContraction
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from study_msc_growth_contraction import eigenstate
from study_msc_protocol import load
from study_msc_terminal_contraction import observe_history, terminal_payload


def parity(values, mask):
    result = np.zeros_like(values)
    while mask:
        bit = mask & -mask
        result ^= (values & bit) != 0
        mask ^= bit
    return result


class DenseReference:
    def __init__(self, plan):
        self.plan = plan
        n = plan.width
        self.indices = np.arange(1 << n)
        self.all_x = (1 << n) - 1
        self.z_sign = 1 - 2 * parity(self.indices, self.all_x)
        # Independent stabilizer preparation checks the source CSS support.
        tableau = stim.Tableau.from_stabilizers([*plan.checks, pauli(n, "Z", range(n))])
        zero = np.asarray(tableau.to_state_vector(endian="little"), dtype=complex)
        zero /= np.linalg.norm(zero)
        anchor = zero[np.flatnonzero(zero)[0]]
        zero *= abs(anchor) / anchor
        self.encoding = np.column_stack([zero, zero[self.indices ^ self.all_x]])
        self.actions = []
        for p in plan.checks:
            x, z = masks(p)
            self.actions.append((x, 1 - 2 * parity(self.indices, z)))

    def density(self, logical, frame, terms, syndrome, aer=False):
        source = self.encoding @ logical
        x, z = frame
        framed = ((1 - 2 * parity(self.indices, z)) * source)[self.indices ^ x]
        state = np.zeros_like(source)
        for term in terms:
            scalar = term.weight
            for q in range(len(self.plan.ancillas)):
                scalar *= np.vdot(terms[0].ancillas[q], term.ancillas[q])
            op = term.data
            if aer:
                circuit = QuantumCircuit(self.plan.width)
                circuit.set_statevector(framed)
                # set_statevector overwrites a circuit-level global phase in Aer;
                # an explicit scalar gate preserves the relative phase of terms.
                circuit.unitary(np.eye(2) * np.exp(1j * np.pi * op.global_phase / 4), [0])
                for q, coefficient in enumerate(op.linear):
                    circuit.p(np.pi * coefficient / 4, q)
                    if op.flips >> q & 1:
                        circuit.x(q)
                circuit.save_statevector()
                result = (
                    AerSimulator(method="statevector", max_parallel_threads=1).run(circuit).result()
                )
                if not result.success:
                    raise AssertionError(result.status)
                transformed = np.asarray(result.get_statevector())
            else:
                phases = np.full(len(source), op.global_phase, dtype=np.int16)
                for q, coefficient in enumerate(op.linear):
                    phases += coefficient * ((self.indices >> q) & 1)
                transformed = (np.exp(1j * np.pi * phases / 4) * framed)[self.indices ^ op.flips]
            state += scalar * transformed
        for sign, (x, zsign) in zip(syndrome, self.actions, strict=True):
            state = (state + (-1) ** sign * (zsign * state)[self.indices ^ x]) / 2
        norm = float(np.vdot(state, state).real)
        x = np.vdot(state, state[self.indices ^ self.all_x]).real
        z = np.vdot(state, self.z_sign * state).real
        y = np.vdot(state, 1j * (self.z_sign * state)[self.indices ^ self.all_x]).real
        return np.array([[norm + z, x - 1j * y], [x + 1j * y, norm - z]]) / 2


class MscTerminalContractionTest(unittest.TestCase):
    programs: dict[int, Program]
    boundaries: dict[int, BoundaryPlan]
    plans: dict[int, TerminalContraction]

    @classmethod
    def setUpClass(cls):
        cls.programs = {d: load(d) for d in (3, 5)}
        cls.boundaries = {d: compile_boundaries(p, load(3))[-1] for d, p in cls.programs.items()}
        cls.plans = {d: TerminalContraction(p, cls.boundaries[d]) for d, p in cls.programs.items()}

    def synthetic(self, plan, syndrome, seed):
        rng = np.random.default_rng(seed)
        logical = rng.normal(size=2) + 1j * rng.normal(size=2)
        logical /= np.linalg.norm(logical)
        x = z = 0
        for bit, (dx, dz) in zip(syndrome, plan.duals, strict=True):
            if bit:
                x ^= dx
                z ^= dz
        terms = []
        for k in range(4):
            ancillas = np.array([eigenstate("Y", 0) for _ in plan.ancillas])
            ancillas[0] *= np.exp(1j * (k + 1) / 3)
            terms.append(
                FactoredTerm(
                    PhaseMonomial(
                        plan.width,
                        ((1 << plan.width) - 1) if k & 1 else 0,
                        int(rng.integers(8)),
                        rng.integers(8, size=plan.width).tolist(),
                    ),
                    ancillas,
                    complex(*rng.normal(size=2)) / 4,
                )
            )
        return logical, (x, z), terms, syndrome

    def compare_dense(self, plan, reference, args, aer=False):
        output = plan.contract(*args)
        density = reference.density(*args, aer=aer)
        scale = max(float(np.trace(density).real), 1e-20)
        error = float(np.max(np.abs(np.outer(output, output.conj()) - density))) / scale
        self.assertLess(error, 2e-10)
        return float(np.trace(density).real)

    def test_all_small_sectors_and_large_basis_directions_against_dense_projectors(self):
        for distance, plan in self.plans.items():
            reference = DenseReference(plan)
            if distance == 3:
                sectors = list(range(64))
            else:
                rng = np.random.default_rng(15155)
                sectors = [0] + [1 << k for k in range(18)]
                sectors += rng.integers(1 << 18, size=32).tolist()
            positive = 0
            for index, sector in enumerate(sectors):
                syndrome = [(sector >> k) & 1 for k in range(len(plan.duals))]
                args = self.synthetic(plan, syndrome, 9575 + index)
                positive += self.compare_dense(plan, reference, args) > 1e-15
            self.assertEqual(positive, len(sectors))

    def test_large_monomial_sum_against_aer_statevectors(self):
        plan = self.plans[5]
        reference = DenseReference(plan)
        for sector in (0, 0x2A395):
            syndrome = [(sector >> k) & 1 for k in range(18)]
            args = self.synthetic(plan, syndrome, 381 + sector)
            self.compare_dense(plan, reference, args, aer=True)

    def test_syndrome_duals_preserve_axes_and_one_basis_covers_all_sectors(self):
        for distance, plan in self.plans.items():
            self.assertEqual(list(map(len, plan.supports)), [8, 8] if distance == 3 else [512, 512])
            self.assertEqual(len(plan.decoder), 16 if distance == 3 else 1024)
            for k, (x, z) in enumerate(plan.duals):
                self.assertEqual(x.bit_count() & 1, 0)
                self.assertEqual(z.bit_count() & 1, 0)
                for j, p in enumerate(plan.checks):
                    px, pz = masks(p)
                    self.assertEqual(((x & pz).bit_count() + (z & px).bit_count()) & 1, int(k == j))

    def test_intermediate_code_projection_changes_the_instrument(self):
        plan = self.plans[5]
        logical = np.array([1, 1j]) / np.sqrt(2)
        ancillas = np.array([eigenstate("Z", 0) for _ in plan.ancillas])
        forward = PhaseMonomial(plan.width)
        forward.append("S", (0,))
        backward = PhaseMonomial(plan.width)
        backward.append("S_DAG", (0,))
        syndrome = [0] * 18
        direct = plan.contract(
            logical, (0, 0), [FactoredTerm(compose(backward, forward), ancillas, 1)], syndrome
        )
        halfway = plan.contract(logical, (0, 0), [FactoredTerm(forward, ancillas, 1)], syndrome)
        wrong = plan.contract(halfway, (0, 0), [FactoredTerm(backward, ancillas, 1)], syndrome)
        self.assertAlmostEqual(float(np.vdot(direct, direct).real), 1, places=12)
        self.assertAlmostEqual(float(np.vdot(wrong, wrong).real), 0.25, places=12)

    def test_cancellation_spectator_phase_and_execution_without_stim(self):
        plan = self.plans[5]
        logical, frame, payload, syndrome = self.synthetic(plan, [0] * 18, 317)
        first = payload[0]
        second = FactoredTerm(first.data, first.ancillas.copy(), first.weight)
        second.ancillas[0] *= -1
        with (
            patch("msc_terminal_contraction.stim", None),
            patch(
                "msc_terminal_contraction.masks",
                side_effect=AssertionError("runtime Pauli analysis"),
            ),
        ):
            np.testing.assert_allclose(
                plan.contract(logical, frame, [first, second], syndrome), 0, atol=1e-15
            )
            self.assertGreater(
                np.linalg.norm(plan.contract(logical, frame, payload, syndrome)), 1e-8
            )
        second.ancillas[0] = eigenstate("Y", 1)
        with self.assertRaisesRegex(ValueError, "spectator differs"):
            plan.contract(logical, frame, [first, second], syndrome)

    def test_actual_histories_and_impossible_root_reset(self):
        source = Path(__file__).parent / "research" / "msc_protocol_data.json"
        for circuit in json.loads(source.read_text())["circuits"]:
            d = circuit["distance"]
            program, boundary, plan = self.programs[d], self.boundaries[d], self.plans[d]
            growth = None
            if d == 5:
                source_boundary = compile_boundaries(program, load(3))[0]
                instrument = InstrumentPlan(boundary)
                growth = (
                    source_boundary,
                    instrument,
                    GrowthContraction(source_boundary, instrument),
                )
            for k in (0, 1, 7, -1):
                case = circuit["cases"][k]
                self.assertLess(
                    observe_history(program, boundary, plan, case, growth=growth)[
                        "normalized_density_error"
                    ],
                    1e-10,
                )
            outcomes = list(map(int, circuit["cases"][0]["outcomes"]))
            region = list(program.regions.values())[-2]
            reset = next(g for g in region.gates if g.name == "RX")
            outcomes[program.measurement_at[reset.line, 0]] ^= 1
            with self.assertRaisesRegex(ValueError, "impossible terminal gadget reset"):
                terminal_payload(program, boundary, plan, {}, outcomes)


if __name__ == "__main__":
    unittest.main()
