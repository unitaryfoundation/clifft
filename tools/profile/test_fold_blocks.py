import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from export_fold_blocks import Exporter
from fold_blocks import Protocol, f7_diagnostic_histories, syndrome_sector_histories
from fold_cultivation import Operation, logical

HAS_CIRQ = importlib.util.find_spec("cirq") is not None
if HAS_CIRQ:
    from clifford_branches import (
        logical_tail_history,
        materialize,
        paired_hook_histories,
        run_history,
    )


@unittest.skipUnless(HAS_CIRQ, "independent block reference requires cirq-core==1.6.1")
class BlockTest(unittest.TestCase):
    def compare(self, protocol, stages):
        result = protocol.evaluate(protocol.encode(stages))
        reference, _ = run_history(protocol.reconstruction, stages)
        acceptance = reference.expectation()
        self.assertAlmostEqual(result.acceptance, acceptance, delta=2e-12)
        if acceptance > 1e-12:
            expected = [
                reference.expectation(logical(protocol.reconstruction.distance, axis)) / acceptance
                for axis in "XYZ"
            ]
            np.testing.assert_allclose(result.logical_xyz, expected, rtol=0, atol=2e-12)
        else:
            self.assertIsNone(result.logical_xyz)
        return result

    def test_complete_protocol_fixed_fault_histories(self):
        for distance in (3, 5, 7):
            protocol = Protocol(distance)
            self.assertAlmostEqual(
                self.compare(
                    protocol, [list(s.operations) for s in protocol.reconstruction.stages]
                ).acceptance,
                1,
            )
            for seed in (1000, 1005, 1016, 1020, 1043, 1063):
                stages, _ = materialize(protocol.reconstruction, 0.001, seed)
                self.compare(protocol, stages)
            for _, stages in paired_hook_histories(protocol.reconstruction)[::4]:
                result = self.compare(protocol, stages)
                self.assertAlmostEqual(result.acceptance, 0.25)
            result = self.compare(protocol, logical_tail_history(protocol.reconstruction))
            np.testing.assert_allclose(result.logical_xyz, [-(2**-0.5), -(2**-0.5), 0], atol=2e-12)

    def test_nonzero_sectors_through_growth_retain_leaked_components(self):
        for distance in (5, 7):
            protocol = Protocol(distance)
            histories = syndrome_sector_histories(protocol)
            self.assertEqual(len(histories), 2)
            for _, stages in histories:
                result = self.compare(protocol, stages)
                self.assertAlmostEqual(result.acceptance, 0.25)

    def test_faults_on_both_sides_of_injection(self):
        protocol = Protocol(3)
        base = [list(s.operations) for s in protocol.reconstruction.stages]
        index = next(j for j, op in enumerate(base[0]) if op.name == "T")
        target = base[0][index].targets
        preceding = max(j for j in range(index) if base[0][j].targets == target)
        for boundary in (preceding + 1, index + 1):
            for axis in "XYZ":
                stages = [list(s) for s in base]
                stages[0].insert(boundary, Operation(axis, target))
                self.compare(protocol, stages)

    def test_execution_does_not_call_planners_or_tableaux(self):
        protocol = Protocol(5)
        stages = logical_tail_history(protocol.reconstruction)
        history = protocol.encode(stages)
        with (
            patch("fold_blocks.span_solver", side_effect=AssertionError("runtime elimination")),
            patch("fold_blocks.Plan", side_effect=AssertionError("runtime contraction planning")),
            patch("fold_blocks.stim.Tableau", side_effect=AssertionError("runtime tableau")),
            patch(
                "fold_blocks.stim.TableauSimulator", side_effect=AssertionError("runtime tableau")
            ),
        ):
            self.assertAlmostEqual(protocol.evaluate(history).acceptance, 1)

    def test_input_adapter_declines_changed_circuit(self):
        protocol = Protocol(3)
        stages = [list(s.operations) for s in protocol.reconstruction.stages]
        stages[0].append(Operation("H", (0,)))
        with self.assertRaises(ValueError):
            protocol.encode(stages)

    @unittest.skipUnless(shutil.which("c++"), "native diagnostic requires a C++20 compiler")
    def test_generated_native_plan_preserves_full_protocol_outputs(self):
        protocol = Protocol(7)
        r = protocol.reconstruction
        cases = [
            ("ideal", [list(s.operations) for s in r.stages]),
            ("logical_tail", logical_tail_history(r)),
        ]
        cases += paired_hook_histories(r)[::4]
        cases += syndrome_sector_histories(protocol) + f7_diagnostic_histories(protocol)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "program.cpp"
            executable = Path(directory) / "program"
            metadata = Exporter(protocol).write(cases, source)
            self.assertEqual(metadata["noise_locations"], 2205)
            subprocess.run(
                ["c++", "-std=c++20", "-O2", str(source), "-o", str(executable)],
                check=True,
                capture_output=True,
                text=True,
            )
            run = subprocess.run(
                [str(executable), "1", "10"], check=True, capture_output=True, text=True
            )
            result = json.loads(run.stdout)
            self.assertEqual(result["fixtures"], len(cases))
            self.assertLess(result["max_error"], 2e-12)

    def test_uniform_flag_flip_accepts_and_nonuniform_flip_rejects(self):
        protocol = Protocol(7)
        r = protocol.reconstruction
        stage_index = next(i for i, s in enumerate(r.stages) if s.name == "cat_prepare_d7")
        original = r.stages[stage_index].operations
        root_h = next(
            i for i, op in enumerate(original) if op.name == "H" and op.targets == (r.ancilla + 8,)
        )
        stages = [list(s.operations) for s in r.stages]
        stages[stage_index].insert(root_h + 1, Operation("X", (r.ancilla + 8,)))
        self.assertAlmostEqual(self.compare(protocol, stages).acceptance, 1)
        stages = [list(s.operations) for s in r.stages]
        first_m = next(i for i, op in enumerate(original) if op.name == "M")
        stages[stage_index].insert(first_m, Operation("X", original[first_m].targets))
        self.assertEqual(self.compare(protocol, stages).acceptance, 0)


if __name__ == "__main__":
    unittest.main()
