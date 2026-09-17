import unittest

import numpy as np
from fold_check import Check, Event, elementary_check, export_control, fold_check, kraus_terms
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def native_circuit(check, reference=False, state=None):
    n, a = check.data, check.ancillas
    qc = QuantumCircuit(n + a + (n if reference else 0))
    if reference:
        for q in range(n):
            qc.h(q)
            qc.cx(q, n + a + q)
    elif state is not None:
        initial = np.zeros(1 << (n + a), dtype=complex)
        initial[: 1 << n] = state
        qc.set_statevector(initial)
    qc.h(n)
    for q in range(1, a):
        qc.cx(n, n + q)
    for event in check.events:
        if event.kind in {"X", "Y", "Z"}:
            getattr(qc, event.kind.lower())(*event.targets)
        elif event.kind == "CZ":
            qc.ccz(*event.targets)
        elif event.kind in {"T", "T_DAG", "CX"}:
            getattr(qc, {"T": "t", "T_DAG": "tdg", "CX": "cx"}[event.kind])(*event.targets)
        else:
            control, target = event.targets
            if event.kind == "HXY":
                qc.tdg(target)
                qc.cx(control, target)
                qc.t(target)
            else:
                qc.t(target)
                qc.cx(control, target)
                qc.tdg(target)
    for q in range(1, a):
        qc.cx(n, n + q)
    qc.h(n)
    qc.save_statevector()
    return qc


def with_faults(check, faults):
    events = []
    for boundary in range(len(check.events) + 1):
        events.extend(faults.get(boundary, ()))
        if boundary < len(check.events):
            events.append(check.events[boundary])
    return Check(check.data, check.ancillas, tuple(events))


class FoldCheckTest(unittest.TestCase):
    def test_all_kraus_matrices_against_aer_with_internal_faults(self):
        # A Choi input checks the complete channel on arbitrary, entangled inputs.
        core = Check(
            3, 2, (Event("HXY", (3, 0)), Event("CZ", (4, 1, 2)), Event("HXY_MINUS", (4, 1)))
        )
        core = elementary_check(core)
        cases = [core]
        for boundary in range(len(core.events) + 1):
            for q in range(5):
                for pauli in "XYZ":
                    cases.append(with_faults(core, {boundary: [Event(pauli, (q,))]}))
        rng = np.random.default_rng(495)
        for _ in range(32):
            faults: dict[int, list[Event]] = {}
            for _ in range(5):
                boundary, q = int(rng.integers(len(core.events) + 1)), int(rng.integers(5))
                faults.setdefault(boundary, []).append(Event(str(rng.choice(list("XYZ"))), (q,)))
            cases.append(with_faults(core, faults))
        simulator = AerSimulator(method="statevector", max_parallel_threads=1)
        result = simulator.run([native_circuit(c, reference=True) for c in cases]).result()
        for i, check in enumerate(cases):
            observed = np.asarray(result.get_statevector(i)).reshape(8, 4, 8).transpose(
                1, 2, 0
            ) * np.sqrt(8)
            total = np.zeros((8, 8), dtype=complex)
            for outcome in range(4):
                terms = kraus_terms(check, outcome)
                self.assertIn(len(terms), (0, 2))
                expected = np.zeros((8, 8), dtype=complex)
                for term in terms:
                    for column in range(8):
                        row, phase = term.column(column)
                        expected[row, column] += np.exp(1j * np.pi * phase / 4) / 2
                np.testing.assert_allclose(observed[outcome], expected, atol=2e-14)
                total += expected.conj().T @ expected
            np.testing.assert_allclose(total, np.eye(8), atol=2e-14)

    def test_full_distance_three_core_on_dense_input_against_aer(self):
        core = elementary_check(fold_check(3))
        rng = np.random.default_rng(47)
        state = rng.normal(size=1 << core.data) + 1j * rng.normal(size=1 << core.data)
        state /= np.linalg.norm(state)
        cases = [
            core,
            with_faults(
                core, {0: [Event("Y", (13,))], 3: [Event("X", (7,))], 8: [Event("Z", (15,))]}
            ),
        ]
        simulator = AerSimulator(method="statevector", max_parallel_threads=1)
        result = simulator.run([native_circuit(c, state=state) for c in cases]).result()
        for i, check in enumerate(cases):
            observed = np.asarray(result.get_statevector(i)).reshape(1 << check.ancillas, -1)
            for outcome in range(1 << check.ancillas):
                expected = np.zeros_like(state)
                for term in kraus_terms(check, outcome):
                    expected += term.apply(state) / 2
                np.testing.assert_allclose(observed[outcome], expected, atol=2e-14)

    def test_published_core_sizes_and_hermitian_involution(self):
        rng = np.random.default_rng(63)
        for d, data, ancillas, factors, t_count in [
            (3, 13, 3, 9, 38),
            (5, 41, 5, 25, 130),
            (7, 85, 8, 49, 278),
        ]:
            check = fold_check(d)
            self.assertEqual(
                (check.data, check.ancillas, len(check.events)), (data, ancillas, factors)
            )
            self.assertEqual(sum(7 if e.kind == "CZ" else 2 for e in check.events), t_count)
            identity, unitary = kraus_terms(check, 0)
            expanded_identity, expanded_unitary = kraus_terms(elementary_check(check), 0)
            for _ in range(100):
                bits = sum(int(rng.integers(2)) << q for q in range(data))
                self.assertEqual(identity.column(bits), (bits, 0))
                self.assertEqual(expanded_identity.column(bits), identity.column(bits))
                self.assertEqual(expanded_unitary.column(bits), unitary.column(bits))
                other, phase = unitary.column(bits)
                back, phase_back = unitary.column(other)
                self.assertEqual(back, bits)
                self.assertEqual((phase + phase_back) % 8, 0)
            self.assertIn("not a full cultivation workload", export_control(check))

    def test_reject_unsupported_core_instead_of_dropping_operations(self):
        for event in [
            Event("H", (0,)),
            Event("CZ", (3, 0, 0)),
            Event("X", (5,)),
            Event("HXY", (0, 1)),
            Event("T", (0,)),
        ]:
            with self.assertRaises(ValueError):
                kraus_terms(Check(3, 2, (event,)), 0)
        with self.assertRaises(ValueError):
            kraus_terms(fold_check(3), 8)


if __name__ == "__main__":
    unittest.main()
