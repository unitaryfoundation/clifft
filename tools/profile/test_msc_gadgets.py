"""Independent fixed-instrument checks on the original color-code MSC corpus."""

import random
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from msc_gadgets import contract_ancillas, regions, terms
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from study_msc_gadgets import ideal_boundary, logical_matrix, terminal_code

FIXTURES = Path(__file__).parent / "fixtures" / "msc"


def corpus(distance):
    return (FIXTURES / f"msc_d{distance}_inject_cultivate_p1e-3.stim").read_text()


def sparse_reference(region, outcome, faults, column):
    """Follow elementary gates numerically, explicitly projecting then resetting."""
    index = {q: k for k, q in enumerate(region.wires)}
    state = {column: 1.0 + 0j}
    slots = defaultdict(list)
    for boundary, axis, q in faults:
        slots[boundary].append((axis, q))
    root = next((g.targets[0] for g in region.gates if g.name == "MX"), None)
    reset_bit = outcome
    measured = False
    for boundary in range(len(region.gates) + 1):
        operations = list(slots[boundary])
        for axis, wire in operations:
            q = index[wire]
            updated = {}
            for b, a in state.items():
                phase = (-1) ** ((b >> q) & 1) if axis in {"Y", "Z"} else 1
                if axis == "Y":
                    phase *= 1j
                updated[b ^ ((1 << q) if axis in {"X", "Y"} else 0)] = a * phase
            state = updated
            if measured and wire == root and axis in {"Y", "Z"}:
                reset_bit ^= 1
        if boundary == len(region.gates):
            break
        g = region.gates[boundary]
        qs = [index[q] for q in g.targets]
        q = qs[0]
        updated = defaultdict(complex)
        for b, a in state.items():
            if g.name in {"T", "T_DAG"}:
                angle = (1 if g.name == "T" else -1) * np.pi / 4
                updated[b] += a * np.exp(1j * angle * ((b >> q) & 1))
            elif g.name == "CX":
                updated[b ^ (((b >> q) & 1) << qs[1])] += a
            elif g.name == "MX":
                updated[b] += a / 2
                updated[b ^ (1 << q)] += a * (-1) ** outcome / 2
            elif g.name == "RX":
                updated[b] += a * (-1) ** (reset_bit * ((b >> q) & 1))
            elif g.name == "MPP_Y":
                flipped = b ^ sum(1 << r for r in qs)
                phase = 1j ** len(qs) * (-1) ** sum((b >> r) & 1 for r in qs)
                updated[b] += a / 2
                updated[flipped] += a * (-1) ** outcome * phase / 2
            else:
                raise AssertionError(g)
        state = dict(updated)
        if g.name in {"MX", "RX"}:
            measured = g.name == "MX"
    return state


def compiled_column(operators, column):
    result: defaultdict[int, complex] = defaultdict(complex)
    for op in operators:
        row, phase = op.column(column)
        result[row] += np.exp(1j * np.pi * phase / 4) / 2
    return result


def compare_columns(region, outcome, faults, columns):
    operators = terms(region, outcome, faults)
    error = 0.0
    common_phase = None
    for column in columns:
        expected = sparse_reference(region, outcome, faults, column)
        actual = compiled_column(operators, column)
        if common_phase is None:
            for row, amplitude in actual.items():
                if abs(amplitude) > 1e-10:
                    common_phase = expected.get(row, 0) / amplitude
                    break
        phase = 1 if common_phase is None else common_phase
        error = max(error, abs(abs(phase) - 1))
        error = max(
            error,
            *(
                abs(actual.get(r, 0) * phase - expected.get(r, 0))
                for r in actual.keys() | expected.keys()
            ),
        )
    return error


def aer_reference(region, outcome, faults, state):
    index = {q: k for k, q in enumerate(region.wires)}
    slots = defaultdict(list)
    for boundary, axis, wire in faults:
        slots[boundary].append((axis, wire))
    root = next(g.targets[0] for g in region.gates if g.name == "MX")
    root_index = index[root]
    reset_bit = outcome
    measured = False
    circuit = QuantumCircuit(len(index))
    circuit.set_statevector(state)
    weight = 1.0

    def run(c):
        c.save_statevector()
        return np.asarray(
            AerSimulator(method="statevector", max_parallel_threads=1)
            .run(c)
            .result()
            .get_statevector()
        )

    for boundary in range(len(region.gates) + 1):
        for axis, wire in slots[boundary]:
            getattr(circuit, axis.lower())(index[wire])
            if measured and wire == root and axis in {"Y", "Z"}:
                reset_bit ^= 1
        if boundary == len(region.gates):
            break
        g = region.gates[boundary]
        targets = [index[q] for q in g.targets]
        if g.name == "MX":
            circuit.h(root_index)
            state = run(circuit)
            state[((np.arange(len(state)) >> root_index) & 1) != outcome] = 0
            weight = float(np.linalg.norm(state))
            circuit = QuantumCircuit(len(index))
            circuit.set_statevector(state / weight)
            circuit.h(root_index)
            measured = True
        elif g.name == "RX":
            if reset_bit:
                circuit.z(root_index)
            measured = False
        else:
            getattr(circuit, {"T_DAG": "tdg"}.get(g.name, g.name.lower()))(*targets)
    return run(circuit) * weight


class MscGadgetsTest(unittest.TestCase):
    def test_ideal_cultivation_boundary_keeps_nonzero_record_sector(self):
        for distance in (3, 5):
            text = corpus(distance)
            found = regions(text)
            basis, _ = terminal_code(text, found[-1].data)
            check = logical_matrix(terms(found[-1], 0)[1], basis)
            result = ideal_boundary(found[-2], basis, check)
            for entry in result:
                self.assertLess(entry["code_preservation_error"], 3e-13)
                self.assertLess(entry["logical_projector_error"], 3e-13)
            self.assertEqual(result[0]["nonzero_ancilla_records"], [])
            self.assertEqual(len(result[1]["nonzero_ancilla_records"]), 1)
            self.assertEqual(len(result[1]["data_z_correction"]), 1)

    def test_terminal_check_preserves_actual_single_logical_code(self):
        expected = np.array([[0, np.exp(1j * np.pi / 4)], [np.exp(-1j * np.pi / 4), 0]])
        for distance in (3, 5):
            text = corpus(distance)
            terminal = regions(text)[-1]
            basis, rank = terminal_code(text, terminal.data)
            self.assertEqual(rank, len(terminal.data) - 1)
            check = logical_matrix(terms(terminal, 0)[1], basis)
            np.testing.assert_allclose(check, expected, atol=3e-13)
            np.testing.assert_allclose(check.conj().T @ check, np.eye(2), atol=3e-13)

    def test_wire_relabeling_is_derived_from_gates(self):
        rng = random.Random(82014)
        for original in regions(corpus(5)):
            shuffled = rng.sample(range(100, 100 + len(original.wires)), len(original.wires))
            mapping = dict(zip(original.wires, shuffled, strict=True))
            text = []
            for g in original.gates:
                if g.name == "MPP_Y":
                    text.append("MPP " + "*".join(f"Y{mapping[q]}" for q in g.targets))
                else:
                    text.append(g.name + " " + " ".join(str(mapping[q]) for q in g.targets))
            relabeled = regions("\n".join(text))
            self.assertEqual(len(relabeled), 1)
            changed = relabeled[0]
            destinations = [changed.wires.index(mapping[q]) for q in original.wires]

            def permute(bits):
                return sum(((bits >> q) & 1) << r for q, r in enumerate(destinations))

            for outcome in (0, 1):
                before, after = terms(original, outcome), terms(changed, outcome)
                for _ in range(32):
                    column = rng.getrandbits(len(original.wires))
                    for a, b in zip(before, after, strict=True):
                        row, phase = a.column(column)
                        self.assertEqual(b.column(permute(column)), (permute(row), phase))

    def test_corpus_recognition_and_noise_retention(self):
        for distance, sizes, count in [(3, [7, 7], 28), (5, [7, 19, 19], 90)]:
            found = regions(corpus(distance))
            self.assertEqual([len(r.data) for r in found], sizes)
            self.assertEqual(sum(r.t_count for r in found), count)
            self.assertTrue(any("DEPOLARIZE2" in s for s in found[0].source_lines))
            self.assertTrue(any("MX(0.001)" in s for s in found[0].source_lines))

    def test_changed_structure_declines_without_whole_circuit_claim(self):
        text = corpus(3)
        self.assertEqual(len(regions(text.replace("RX 7\n", "RX 6\n"))), 1)
        self.assertEqual(
            len(regions(text.replace("T 0 3 7 9 10 12 13\n", "T_DAG 0 3 7 9 10 12 13\n"))), 0
        )
        altered = text.replace("MX(0.001) 7", "H 7\nMX(0.001) 7")
        self.assertEqual(len(regions(altered)), 1)
        start = text.index("RX 7\n")
        altered = text[:start] + text[start:].replace("CX 7 11", "CX 11 7", 1)
        self.assertEqual(len(regions(altered)), 1)

    def test_fault_instruments_on_all_five_actual_regions(self):
        rng = random.Random(22071)
        for distance in (3, 5):
            for region in regions(corpus(distance)):
                histories: list[list[tuple[int, str, int]]] = [[]]
                for boundary in range(len(region.gates) + 1):
                    histories.extend(
                        [[(boundary, axis, q)] for axis in "XYZ" for q in region.wires]
                    )
                histories.extend(
                    [
                        [
                            (
                                rng.randrange(len(region.gates) + 1),
                                rng.choice("XYZ"),
                                rng.choice(region.wires),
                            )
                            for _ in range(12)
                        ]
                        for _ in range(32)
                    ]
                )
                columns = [0, (1 << len(region.wires)) - 1] + [
                    rng.getrandbits(len(region.wires)) for _ in range(2)
                ]
                for history in histories:
                    for outcome in (0, 1):
                        self.assertLess(compare_columns(region, outcome, history, columns), 4e-12)

    def test_d3_entangled_inputs_and_ancilla_records_against_aer(self):
        region = regions(corpus(3))[0]
        rng = np.random.default_rng(96132)
        index = {q: k for k, q in enumerate(region.wires)}
        for trial in range(8):
            # Both arbitrary entanglement and the actual plus-prepared contract.
            if trial < 4:
                initial = rng.normal(size=1 << len(index)) + 1j * rng.normal(size=1 << len(index))
            else:
                data = rng.normal(size=1 << len(region.data)) + 1j * rng.normal(
                    size=1 << len(region.data)
                )
                initial = np.array(
                    [
                        data[sum(((b >> index[q]) & 1) << j for j, q in enumerate(region.data))]
                        for b in range(1 << len(index))
                    ]
                )
            initial /= np.linalg.norm(initial)
            faults = [
                (
                    int(rng.integers(len(region.gates) + 1)),
                    str(rng.choice(list("XYZ"))),
                    int(rng.choice(region.wires)),
                )
                for _ in range(8 if trial % 2 else 0)
            ]
            for outcome in (0, 1):
                operators = terms(region, outcome, faults)
                actual = sum(op.apply(initial) for op in operators) / 2
                expected = aer_reference(region, outcome, faults, initial)
                phase = np.vdot(actual, expected)
                if abs(phase) > 1e-12:
                    actual *= phase / abs(phase)
                np.testing.assert_allclose(actual, expected, atol=4e-12)
                if trial >= 4:
                    # Include the nonzero outcome patterns predicted by either
                    # branch as well as an impossible record when available.
                    records = [
                        {q: op.linear[index[q]] // 4 for q in region.ancillas} for op in operators
                    ]
                    records.append({q: 1 for q in region.ancillas})
                    data /= np.linalg.norm(data)
                    for record in records:
                        projected = np.zeros(1 << len(region.data), dtype=complex)
                        for b, amplitude in enumerate(expected):
                            row = sum(((b >> index[q]) & 1) << j for j, q in enumerate(region.data))
                            sign = (-1) ** sum(
                                record[q] * ((b >> index[q]) & 1) for q in region.ancillas
                            )
                            projected[row] += amplitude * sign / np.sqrt(1 << len(region.ancillas))
                        reduced = (
                            sum(
                                (
                                    op.apply(data)
                                    for op in contract_ancillas(region, operators, record)
                                ),
                                np.zeros_like(data),
                            )
                            / 2
                        )
                        if abs(phase) > 1e-12:
                            reduced *= phase / abs(phase)
                        np.testing.assert_allclose(reduced, projected, atol=4e-12)


if __name__ == "__main__":
    unittest.main()
