"""Static signed-code membership, fault transport, and decline checks."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from msc_boundaries import BoundaryPlan, compile_boundaries
from msc_protocol import Program
from study_msc_boundaries import observe_history, validate_fault_rows
from study_msc_protocol import FIXTURES, load


class MscBoundariesTest(unittest.TestCase):
    def test_actual_intervals_fix_one_logical_qubit_and_preserve_growth_axes(self):
        for distance in (3, 5):
            program = load(distance)
            plans = compile_boundaries(program, load(3))
            self.assertEqual([p.code_count for p in plans], [6] if distance == 3 else [6, 18])
            for plan in plans:
                self.assertEqual(len(plan.checks), program.width - 1)
                for k, (x, z) in enumerate(plan.duals):
                    self.assertEqual(x.bit_count() % 2, 0)
                    self.assertEqual(z.bit_count() % 2, 0)
                    for j, check in enumerate(plan.checks[: plan.code_count]):
                        anti = (
                            sum(
                                ((z >> n) & 1) * int(check[q] in (1, 2))
                                + ((x >> n) & 1) * int(check[q] in (2, 3))
                                for n, q in enumerate(plan.data)
                            )
                            % 2
                        )
                        self.assertEqual(anti, int(k == j))
            if distance == 5:
                self.assertEqual(set(plans[-1].logical), {"X", "Z"})
                self.assertEqual(len(plans[-1].input_rows), 6)
                self.assertEqual(plans[-1].isolated, [5])

    def test_full_noisy_histories_have_the_compiled_signed_boundaries(self):
        path = Path(__file__).parent / "research" / "msc_protocol_data.json"
        for circuit in json.loads(path.read_text())["circuits"]:
            program = load(circuit["distance"])
            plans = compile_boundaries(program, load(3))
            cases = [circuit["cases"][k] for k in (0, 7, -1)]
            cases += [r for r in circuit["cases"] if r["name"].startswith("feedforward_readout_")][
                :1
            ]
            for case in cases:
                self.assertEqual(len(observe_history(program, plans, case)), len(plans))

    def test_sign_evaluation_does_not_use_stim_or_discover_dependencies(self):
        program = load(5)
        plan = compile_boundaries(program, load(3))[-1]
        records = [k % 2 for k in range(len(program.records))]
        history = program.history(9501, 10)
        expected = plan.evaluate(records, history)
        with (
            patch("msc_boundaries.stim", None),
            patch.object(plan, "flow_row", side_effect=AssertionError("runtime flow solving")),
            patch("msc_boundaries.masks", side_effect=AssertionError("runtime Pauli analysis")),
        ):
            self.assertEqual(plan.evaluate(records, history), expected)

    def test_signed_preparation_and_fault_parity_against_stim(self):
        program = Program("R 1\nX 1\nX_ERROR(0.25) 1\n")
        plan = BoundaryPlan(program, 1, 3, (0,), [])
        self.assertEqual(plan.evaluate([], {})[0], [1])
        self.assertEqual(plan.evaluate([], {0: "X"})[0], [0])
        self.assertEqual(validate_fault_rows(plan)["fault_basis_histories"], 2)

    def test_inherited_spectator_readout_is_not_mistaken_for_a_quantum_flip(self):
        program = load(5)
        plans = compile_boundaries(program, load(3))
        growth = plans[-1]
        row = next(row for check, row in zip(growth.checks, growth.rows, strict=True) if check[5])
        self.assertEqual(row.records.bit_count(), 1)
        record = row.records.bit_length() - 1
        self.assertNotIn(record, program.controls.values())
        event = program.measurements[program.records[record]]
        self.assertLess(event.line, growth.start)
        site = program.site_at[event.line, event.offset]
        data = json.loads(
            (Path(__file__).parent / "research" / "msc_protocol_data.json").read_text()
        )
        original = data["circuits"][1]["cases"][0]
        outcomes = list(map(int, original["outcomes"]))
        history = {site: "flip"}
        records = program.outputs(outcomes, history)["records"]
        case = {**original, "faults": [[site, "flip"]], "records": "".join(map(str, records))}
        entries = observe_history(program, plans, case)
        self.assertEqual(len(entries), 2)
        baseline = growth.evaluate(list(map(int, original["records"])), {})
        self.assertEqual(growth.evaluate(records, history), baseline)

    def test_missing_code_projection_and_entangled_spectator_decline(self):
        text = (FIXTURES / "msc_d3_inject_cultivate_p1e-3.stim").read_text()
        changed = text.replace("CX 3 2 7 6 9 8 12 11", "TICK")
        self.assertNotEqual(text, changed)
        with self.assertRaisesRegex(ValueError, "missing boundary stabilizer flow"):
            compile_boundaries(Program(changed), load(3))
        entangled = Program("CX 0 1\n")
        with self.assertRaisesRegex(ValueError, "spectator interacts"):
            BoundaryPlan(entangled, 1, 1, (0,), [])

    def test_unseen_commuting_gate_order_is_certified_from_gates(self):
        text = (FIXTURES / "msc_d5_inject_cultivate_p1e-3.stim").read_text()
        lines = []
        changed = 0
        for line in text.splitlines():
            words = line.split()
            if (
                words[0] == "CX"
                and not any(w.startswith("rec[") for w in words[1:])
                and len(set(words[1:])) == len(words) - 1
            ):
                pairs = [words[k : k + 2] for k in range(1, len(words), 2)]
                if len(pairs) > 1:
                    line = "CX " + " ".join(q for pair in reversed(pairs) for q in pair)
                    changed += 1
            lines.append(line)
        self.assertGreater(changed, 10)
        plans = compile_boundaries(Program("\n".join(lines)), load(3))
        self.assertEqual([p.code_count for p in plans], [6, 18])
        self.assertEqual(len(plans[-1].input_rows), 6)


if __name__ == "__main__":
    unittest.main()
