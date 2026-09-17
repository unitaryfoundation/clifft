"""Slow offline composition reference using Cirq CH states and Stim tableaux.

Requires cirq-core==1.6.1 in addition to the usual research dependencies.
This deliberately performs per-history tableau work and is not an executor.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import cirq  # type: ignore[import-not-found]
import numpy as np
import stim
from fold_check import Check, Event, branch_operator
from fold_cultivation import Operation, Reconstruction, logical

Weight = tuple[Fraction, ...]
ZERO: Weight = (Fraction(0),) * 4
ONE: Weight = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
INV_SQRT2: Weight = (Fraction(0), Fraction(1, 2), Fraction(0), Fraction(-1, 2))


def root(exponent: int) -> Weight:
    values = list(ZERO)
    values[exponent % 4] = Fraction(1 if exponent % 8 < 4 else -1)
    return tuple(values)


def add(a: Weight, b: Weight) -> Weight:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def multiply(a: Weight, b: Weight) -> Weight:
    values = list(ZERO)
    for i in range(4):
        for j in range(4):
            values[(i + j) % 4] += a[i] * b[j] * (1 if i + j < 4 else -1)
    return tuple(values)


def value(a: Weight) -> complex:
    return complex(sum(float(c) * np.exp(1j * np.pi * k / 4) for k, c in enumerate(a)))


def act(ch: Any, name: str, targets: tuple[int, ...]) -> None:
    if name in {"S", "S_DAG"}:
        ch.apply_z(targets[0], exponent=0.5 if name == "S" else -0.5)
    else:
        getattr(ch, "apply_" + name.lower())(*targets)


@dataclass
class Term:
    coefficient: Weight
    ch: Any
    tableau: Any

    def copy(self) -> Term:
        return Term(self.coefficient, self.ch.copy(), self.tableau.copy())

    def gate(self, name: str, targets: tuple[int, ...]) -> None:
        act(self.ch, name, targets)
        self.tableau.do(stim.CircuitInstruction(name, targets))

    def project(self, q: int, outcome: int) -> bool:
        expectation = self.tableau.peek_z(q)
        if expectation == -(1 - 2 * outcome):
            return False
        if expectation == 0:
            self.coefficient = multiply(self.coefficient, INV_SQRT2)
        self.ch.project_Z(q, outcome)
        self.tableau.postselect_z(q, desired_value=bool(outcome))
        return True

    def canonicalize(self) -> tuple[str, ...]:
        generators = self.tableau.canonical_stabilizers()
        key = tuple(map(str, generators))
        # An amplitude anchor resolves the global phase that tableaux discard.
        # Choose the first nonzero computational amplitude without enumeration.
        anchor = self.tableau.copy()
        bits = 0
        width = len(generators)
        for q in range(width):
            bit = int(anchor.peek_z(q) == -1)
            anchor.postselect_z(q, desired_value=bool(bit))
            bits |= bit << (width - 1 - q)
        amplitude = self.ch.inner_product_of_state_and_x(bits)
        if abs(amplitude) == 0:
            raise ValueError("CH state and tableau disagree about amplitude support")
        phase = amplitude / abs(amplitude)
        exponent = int(round(np.angle(phase) * 4 / np.pi)) % 8
        expected = np.exp(1j * np.pi * exponent / 4)
        if abs(phase - expected) > 1e-8:
            raise ValueError("stabilizer phase is not an eighth root of unity")
        self.coefficient = multiply(self.coefficient, root(exponent))
        self.ch.apply_global_phase(expected.conjugate())
        return key


class BranchState:
    def __init__(self, width: int, injection_basis: str = "computational"):
        if injection_basis not in {"computational", "pauli"}:
            raise ValueError("unknown injection decomposition")
        self.injection_basis = injection_basis
        tab = stim.TableauSimulator()
        tab.set_num_qubits(width)
        self.width = width
        self.terms = [Term(ONE, cirq.StabilizerStateChForm(width), tab)]
        self.peak_terms = 1
        self.merges = 0
        self.exact_cancellations = 0
        self.overlap_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], complex] = {}

    def merge(self) -> None:
        self.peak_terms = max(self.peak_terms, len(self.terms))
        grouped: dict[tuple[str, ...], Term] = {}
        for term in self.terms:
            key = term.canonicalize()
            if key in grouped:
                grouped[key].coefficient = add(grouped[key].coefficient, term.coefficient)
                self.merges += 1
            else:
                grouped[key] = term
        self.exact_cancellations += sum(t.coefficient == ZERO for t in grouped.values())
        self.terms = [t for t in grouped.values() if t.coefficient != ZERO]

    def operation(self, op: Operation) -> None:
        if op.name == "M":
            self.terms = [t for t in self.terms if t.project(op.targets[0], 0)]
            self.merge()
        elif op.name == "R":
            if any(t.tableau.peek_z(op.targets[0]) != 1 for t in self.terms):
                raise ValueError("reference reset requires a fresh or postselected-zero qubit")
        elif op.name == "T":
            if self.injection_basis == "computational":
                expanded = []
                for source in self.terms:
                    for bit in (0, 1):
                        term = source.copy()
                        if term.project(op.targets[0], bit):
                            term.coefficient = multiply(term.coefficient, root(bit))
                            expanded.append(term)
                self.terms = expanded
                self.merge()
                return
            half = (Fraction(1, 2), Fraction(0), Fraction(0), Fraction(0))
            a = multiply(add(ONE, root(1)), half)
            b = multiply(add(ONE, root(5)), half)
            expanded = []
            for term in self.terms:
                other = term.copy()
                other.gate("Z", op.targets)
                other.coefficient = multiply(other.coefficient, b)
                term.coefficient = multiply(term.coefficient, a)
                expanded += [term, other]
            self.terms = expanded
            self.merge()
        elif op.name in {"H", "X", "Y", "Z", "S", "S_DAG", "CX", "CZ"}:
            for term in self.terms:
                term.gate(op.name, op.targets)
        else:
            raise ValueError(f"unsupported operation outside a recognized fold core: {op.name}")

    def fold(self, check: Check, mapping: list[int]) -> None:
        check.validate()
        if len(mapping) != check.data + check.ancillas or len(set(mapping)) != len(mapping):
            raise ValueError("invalid fold-core qubit mapping")
        expanded = []
        for source in self.terms:
            for bit in (0, 1):
                term = source.copy()
                if not term.project(mapping[check.data], bit):
                    continue
                cat = []
                for q in mapping[check.data :]:
                    expectation = term.tableau.peek_z(q)
                    if expectation == 0:
                        raise ValueError("cat register is not two computational strings")
                    cat.append(int(expectation == -1))
                operator, final_cat = branch_operator(check, tuple(cat))
                term.coefficient = multiply(term.coefficient, root(operator.global_phase))
                for q, phase in enumerate(operator.linear):
                    for _ in range(phase // 2):
                        term.gate("S", (mapping[q],))
                for q, r in sorted(operator.edges):
                    term.gate("CZ", (mapping[q], mapping[r]))
                for q in range(check.data):
                    if (operator.flips >> q) & 1:
                        term.gate("X", (mapping[q],))
                for q, before, after in zip(mapping[check.data :], cat, final_cat, strict=True):
                    if before != after:
                        term.gate("X", (q,))
                expanded.append(term)
        self.terms = expanded
        self.merge()

    def overlap(self, a: Term, b: Term) -> complex:
        ka = tuple(map(str, a.tableau.canonical_stabilizers()))
        kb = tuple(map(str, b.tableau.canonical_stabilizers()))
        if ka == kb:
            return 1.0
        if (ka, kb) not in self.overlap_cache:
            inverse = (
                stim.Tableau.from_stabilizers([stim.PauliString(p) for p in ka])
                .to_circuit()
                .inverse()
            )
            left, right = a.ch.copy(), b.ch.copy()
            for op in inverse:
                targets = op.targets_copy()
                arity = 2 if op.name in {"CX", "CZ"} else 1
                for start in range(0, len(targets), arity):
                    group = tuple(t.value for t in targets[start : start + arity])
                    act(left, op.name, group)
                    act(right, op.name, group)
            phase = left.inner_product_of_state_and_x(0)
            if abs(abs(phase) - 1) > 1e-8:
                raise ValueError("canonical stabilizer preparation failed to invert")
            result = phase.conjugate() * right.inner_product_of_state_and_x(0)
            self.overlap_cache[ka, kb] = result
            self.overlap_cache[kb, ka] = result.conjugate()
        return self.overlap_cache[ka, kb]

    def expectation(self, observable: stim.PauliString | None = None) -> float:
        self.merge()
        transformed = []
        for source in self.terms:
            term = source.copy()
            if observable is not None:
                if observable.sign != 1:
                    raise ValueError("only positive Pauli probes are supported")
                for q in range(len(observable)):
                    if observable[q]:
                        term.gate("_XYZ"[observable[q]], (q,))
                term.canonicalize()
            transformed.append(term)
        result = sum(
            value(a.coefficient).conjugate() * value(b.coefficient) * self.overlap(a, b)
            for a in self.terms
            for b in transformed
        )
        if abs(result.imag) > 1e-8:
            raise ValueError("Hermitian expectation has an imaginary component")
        return float(result.real)


def materialize(
    reconstruction: Reconstruction, probability: float, seed: int
) -> tuple[list[list[Operation]], int]:
    if not 0 <= probability <= 1:
        raise ValueError("fault probability must lie in [0,1]")
    rng = random.Random(seed)
    stages = []
    faults = 0
    for stage in reconstruction.stages:
        operations = []
        for op in stage.operations:
            errors = []
            if stage.noisy and rng.random() < probability:
                faults += 1
                if op.name in {"R", "M"}:
                    errors = [Operation("X", op.targets)]
                else:
                    paulis = rng.randrange(1, 4 ** len(op.targets))
                    for q in op.targets:
                        if paulis % 4:
                            errors.append(Operation("_XYZ"[paulis % 4], (q,)))
                        paulis //= 4
            operations += errors + [op] if op.name == "M" else [op] + errors
        stages.append(operations)
    return stages, faults


def paired_hook_histories(
    reconstruction: Reconstruction,
) -> list[tuple[str, list[list[Operation]]]]:
    """Two legal gate-location faults that expose coherent Clifford hooks.

    Flip a cat control after its preceding gate, then after a selected CCZ.
    These are diagnostic strata, not samples from the physical noise model.
    """
    base, _ = materialize(reconstruction, 0, 0)
    histories = []
    for stage_index, stage in enumerate(reconstruction.stages):
        if stage.name != f"fold_core_d{reconstruction.distance}":
            continue
        operations = base[stage_index]
        cczs = [i for i, op in enumerate(operations) if op.name == "CCZ"]
        for selected in sorted(
            {0, len(cczs) // 4, len(cczs) // 2, 3 * len(cczs) // 4, len(cczs) - 1}
        ):
            last = cczs[selected]
            control = operations[last].targets[0]
            first = max(i for i in range(last) if control in operations[i].targets)
            changed = []
            for i, op in enumerate(operations):
                changed.append(op)
                if i in {first, last}:
                    changed.append(Operation("X", (control,)))
            stages = [list(ops) for ops in base]
            stages[stage_index] = changed
            histories.append((f"stage{stage_index}_ccz{selected}_cat{control}", stages))
    return histories


def logical_tail_history(reconstruction: Reconstruction) -> list[list[Operation]]:
    """A legal high-weight logical Z history after the last noisy X checks."""
    stages, _ = materialize(reconstruction, 0, 0)
    index = max(
        i
        for i, s in enumerate(reconstruction.stages)
        if s.name == f"syndrome_d{reconstruction.distance}"
    )
    operations = stages[index]
    observable = logical(reconstruction.distance, "Z")
    insertions = {}
    for q in range(len(observable)):
        if observable[q]:
            site = max(i for i, op in enumerate(operations) if q in op.targets)
            if operations[site].name != "CX":
                raise ValueError("logical tail fault is not at a data CNOT")
            insertions[site] = Operation("Z", (q,))
    changed = []
    for i, op in enumerate(operations):
        changed.append(op)
        if i in insertions:
            changed.append(insertions[i])
    stages[index] = changed
    return stages


def run_history(
    reconstruction: Reconstruction,
    stages: list[list[Operation]],
    injection_basis: str = "computational",
) -> tuple[BranchState, list[dict]]:
    width = max(q for stage in stages for op in stage for q in op.targets) + 1
    state = BranchState(width, injection_basis)
    trace = []
    for stage, operations in zip(reconstruction.stages, stages, strict=True):
        if stage.name.startswith("fold_core_"):
            distance = int(stage.name.rsplit("d", 1)[1])
            mapping = reconstruction.check_mapping(distance)
            data = len(mapping)
            ancillas = 3 if distance == 3 else 5
            mapping += [reconstruction.ancilla + q for q in range(ancillas)]
            inverse = {q: i for i, q in enumerate(mapping)}
            check = Check(
                data,
                ancillas,
                tuple(
                    Event(
                        "CZ" if op.name == "CCZ" else op.name, tuple(inverse[q] for q in op.targets)
                    )
                    for op in operations
                ),
            )
            state.fold(check, mapping)
        else:
            for op in operations:
                state.operation(op)
        state.merge()
        trace.append({"stage": stage.name, "terms": len(state.terms)})
        if not state.terms:
            break
    return state, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5), required=True)
    parser.add_argument("--histories", type=int, default=32)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument(
        "--injection-basis", choices=("computational", "pauli"), default="computational"
    )
    parser.add_argument("--paired-hooks", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reconstruction = Reconstruction(args.distance).build()
    if args.histories < 1:
        parser.error("histories must be positive")
    if args.paired_hooks:
        cases = [(label, stages, 2) for label, stages in paired_hook_histories(reconstruction)]
    else:
        cases = []
        for i in range(args.histories):
            stages, faults = materialize(reconstruction, args.probability, 1000 + i)
            cases.append((str(1000 + i), stages, faults))
    results = []
    for label, stages, faults in cases:
        start = time.perf_counter()
        state, trace = run_history(reconstruction, stages, args.injection_basis)
        acceptance = state.expectation()
        probes = (
            [state.expectation(logical(args.distance, axis)) / acceptance for axis in "XYZ"]
            if acceptance > 1e-12
            else None
        )
        row = {
            "case": label,
            "faults": faults,
            "acceptance": acceptance,
            "logical_xyz": probes,
            "peak_terms": state.peak_terms,
            "merges": state.merges,
            "exact_cancellations": state.exact_cancellations,
            "overlap_cache_entries": len(state.overlap_cache),
            "reference_seconds": time.perf_counter() - start,
            "trace": trace,
        }
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "distance": args.distance,
                "probability": None if args.paired_hooks else args.probability,
                "selection": "paired hooks"
                if args.paired_hooks
                else "natural faults with seeds starting at 1000",
                "injection_basis": args.injection_basis,
                "histories": results,
            },
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
