"""Static injection maps against original-gate Aer prefixes and complete histories."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from msc_boundaries import compile_boundaries
from msc_injection import InjectionPlan
from msc_protocol import Program
from msc_reference import materialize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from study_msc_injection import FixedHistoryPlan, compare
from study_msc_protocol import FIXTURES, load


def prefix_reference(program, boundary, history, seed):
    """Exact active-wire reduction followed by original elementary-gate Aer.

    Only wires untouched by every quantum gate may be dropped. Fixed Pauli
    faults on those product spectators change neither data nor event weights.
    """
    prefix_ops = [op for op in program.instructions if op.line <= boundary.end]
    quantum = {"H", "S", "S_DAG", "X", "Y", "Z", "T", "T_DAG", "CX", "CZ", "R", "RX", "M", "MX"}
    active = sorted({int(t) for op in prefix_ops if op.name in quantum for t in op.targets})
    mapping = {q: k for k, q in enumerate(active)}
    prefix_text = "\n".join(
        op.name
        + ("(" + ",".join(map(str, op.args)) + ")" if op.args else "")
        + " "
        + " ".join(op.targets)
        for op in prefix_ops
    )
    prefix = Program(prefix_text)
    lines = {op.line: k + 1 for k, op in enumerate(prefix_ops)}
    local_history = {
        prefix.site_at[lines[program.sites[k].line], program.sites[k].offset]: value
        for k, value in history.items()
        if program.sites[k].line <= boundary.end
    }
    compact = []
    for line in materialize(prefix, local_history).splitlines():
        words = line.split()
        if words[0] in {"X", "Y", "Z"} and int(words[1]) not in mapping:
            continue
        compact.append(
            words[0]
            + " "
            + " ".join(t if t.startswith("rec[") else str(mapping[int(t)]) for t in words[1:])
        )
    original = Program("\n".join(compact))
    circuit = QuantumCircuit(len(active), len(original.measurements))
    event = 0
    for op in original.instructions:
        if op.name in {"M", "MX"}:
            for word in op.targets:
                q = int(word)
                circuit.save_expectation_value(
                    Pauli("X" if op.name == "MX" else "Z"), [q], label=f"p{event}"
                )
                if op.name == "MX":
                    circuit.h(q)
                circuit.measure(q, event)
                if op.name == "MX":
                    circuit.h(q)
                event += 1
        elif op.name in {"CX", "CZ"}:
            for offset in range(0, len(op.targets), 2):
                a, b = op.targets[offset : offset + 2]
                if a.startswith("rec["):
                    source = original.controls[op.line, offset]
                    with circuit.if_test((circuit.clbits[source], 1)):
                        getattr(circuit, "x" if op.name == "CX" else "z")(int(b))
                else:
                    getattr(circuit, op.name.lower())(int(a), int(b))
        else:
            for word in op.targets:
                getattr(circuit, {"T_DAG": "tdg", "S_DAG": "sdg"}.get(op.name, op.name.lower()))(
                    int(word)
                )
    circuit.save_statevector()
    result = (
        AerSimulator(method="statevector", max_parallel_threads=1)
        .run(circuit, shots=1, memory=True, seed_simulator=seed)
        .result()
    )
    if not result.success:
        raise AssertionError(result.status)
    outcomes = [int(b) for b in reversed(result.get_memory()[0])]
    data = result.data()
    probability = float(
        np.prod([(1 + (-1) ** b * float(data[f"p{k}"])) / 2 for k, b in enumerate(outcomes)])
    )
    state = np.asarray(result.get_statevector())
    bits = np.arange(len(state))
    mask = sum(1 << mapping[q] for q in boundary.data)
    parity = np.zeros(len(state), dtype=int)
    for q in boundary.data:
        parity ^= (bits >> mapping[q]) & 1
    zsign = 1 - 2 * parity
    x = np.vdot(state, state[bits ^ mask]).real
    z = np.vdot(state, zsign * state).real
    y = np.vdot(state, 1j * (zsign * state)[bits ^ mask]).real
    density = probability * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2
    return outcomes, density, len(active)


class MscInjectionTest(unittest.TestCase):
    def test_original_prefixes_against_aer_for_both_t_orientations_and_faults(self):
        count = 0
        for distance in (3, 5):
            text = (FIXTURES / f"msc_d{distance}_inject_cultivate_p1e-3.stim").read_text()
            for inverse in (True, False):
                program = Program(text if inverse else text.replace("T_DAG 3\n", "T 3\n", 1))
                boundary = compile_boundaries(program, load(3))[0]
                plan = InjectionPlan(program, boundary)
                rotation = next(op for op in program.instructions if op.name in {"T", "T_DAG"})
                site = max(
                    k
                    for k, s in enumerate(program.sites)
                    if s.line < rotation.line and s.wires == (3,) and len(s.choices) == 3
                )
                histories = [{}, {site: "X"}, {site: "Y"}, {site: "Z"}, program.history(7905, 10)]
                for k, history in enumerate(histories):
                    outcomes, density, active = prefix_reference(
                        program, boundary, history, 897 + k
                    )
                    self.assertEqual(active, 15)
                    output = plan.evaluate(outcomes, history)
                    scale = float(np.trace(density).real)
                    error = np.max(np.abs(np.outer(output, output.conj()) - density)) / scale
                    self.assertLess(error, 2e-11)
                    count += 1
        self.assertEqual(count, 20)

    def test_normalization_and_all_four_logical_maps(self):
        for distance in (3, 5):
            program = load(distance)
            plan = InjectionPlan(program, compile_boundaries(program, load(3))[0])
            self.assertEqual(
                (len(plan.event_map), len(plan.constraints), plan.free_outcomes), (33, 16, 17)
            )
            self.assertEqual(plan.random_power, 18)
            self.assertEqual([str(p) for p in plan.joint_paulis], ["+XX", "+YY"])
            for matrix in plan.matrices:
                np.testing.assert_allclose(
                    matrix.conj().T @ matrix, np.eye(2) * 2.0**-17, atol=1e-20
                )
            self.assertEqual(2.0**plan.free_outcomes * 2.0 ** (1 - plan.random_power), 1)

    def test_inconsistent_records_and_binding_without_stim(self):
        program = load(5)
        whole = FixedHistoryPlan(program)
        plan = whole.injection
        source = Path(__file__).parent / "research" / "msc_protocol_data.json"
        case = json.loads(source.read_text())["circuits"][1]["cases"][0]
        outcomes = list(map(int, case["outcomes"]))
        expected = plan.evaluate(outcomes, {})
        inverse_events = {target: source for source, target in plan.event_map}
        with (
            patch("msc_injection.stim", None),
            patch("msc_injection.eliminate", side_effect=AssertionError("runtime elimination")),
        ):
            np.testing.assert_array_equal(plan.evaluate(outcomes, {}), expected)
            for row in plan.constraints:
                mask = row.records & ~(1 << plan.postselection_event)
                changed = outcomes.copy()
                changed[inverse_events[(mask & -mask).bit_length() - 1]] ^= 1
                np.testing.assert_array_equal(plan.evaluate(changed, {}), np.zeros(2))
        with (
            patch(
                "study_msc_injection.gadget_payload",
                side_effect=AssertionError("unreachable suffix binding"),
            ),
            patch(
                "study_msc_injection.terminal_payload",
                side_effect=AssertionError("unreachable suffix binding"),
            ),
        ):
            initial, final = whole.coefficients({}, changed)
            np.testing.assert_array_equal(initial, np.zeros(2))
            np.testing.assert_array_equal(final, np.zeros(2))

    def test_complete_coefficient_path_does_not_call_the_coherent_reference(self):
        source = Path(__file__).parent / "research" / "msc_protocol_data.json"
        for circuit in json.loads(source.read_text())["circuits"]:
            plan = FixedHistoryPlan(load(circuit["distance"]))
            for case in [circuit["cases"][k] for k in (0, 7, -1)]:
                history = dict(case["faults"])
                outcomes = list(map(int, case["outcomes"]))
                with (
                    patch(
                        "msc_protocol.BranchState",
                        side_effect=AssertionError("CH coefficient path"),
                    ),
                    patch(
                        "study_msc_injection.evaluate",
                        side_effect=AssertionError("reference used by coefficient path"),
                    ),
                ):
                    initial, final = plan.coefficients(history, outcomes)
                self.assertGreater(np.vdot(initial, initial).real, 0)
                self.assertGreater(np.vdot(final, final).real, 0)
                self.assertIn("complete", compare(plan, case))

    def test_multiple_magic_gates_decline(self):
        text = (FIXTURES / "msc_d3_inject_cultivate_p1e-3.stim").read_text()
        program = Program(text.replace("T_DAG 3\n", "T_DAG 3\nT 3\n", 1))
        boundary = compile_boundaries(program, load(3))[0]
        with self.assertRaisesRegex(ValueError, "exactly one T"):
            InjectionPlan(program, boundary)


if __name__ == "__main__":
    unittest.main()
