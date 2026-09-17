"""Offline compile-once block prototype for the reconstructed f3/f5/f7 protocol.

Tableau and dependency analysis occur during construction only. Python execution
still allocates; this is a correctness prototype, not a production executor.
Every result conditions on a fixed physical fault history and accepted records.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim
from fold_check import PhaseMonomial
from fold_contraction import Plan, compose
from fold_cultivation import Operation, Reconstruction, coordinates, logical, pauli, stabilizers


def parity_map(rows: list[int], bits: int) -> int:
    return sum(((row & bits).bit_count() % 2) << j for j, row in enumerate(rows))


def masks(p: stim.PauliString) -> tuple[int, int]:
    return (
        sum(int(p[q] in (1, 2)) << q for q in range(len(p))),
        sum(int(p[q] in (2, 3)) << q for q in range(len(p))),
    )


def pauli_operator(width: int, x: int, z: int, *, inverse: bool = False) -> PhaseMonomial:
    # X^x Z^z has no Y convention phase; its inverse can differ by a sign.
    phase = 4 * (x & z).bit_count() if inverse else 0
    return PhaseMonomial(width, x, phase % 8, [4 * ((z >> q) & 1) for q in range(width)])


def project_mask(bits: int, mapping: list[int]) -> int:
    return sum(((bits >> physical) & 1) << q for q, physical in enumerate(mapping))


def span_solver(columns: list[int]) -> Callable[[int], tuple[int, int]]:
    """Compile a GF2 span basis with coordinates, never called during execution."""
    pivots: dict[int, tuple[int, int]] = {}
    for j, column in enumerate(columns):
        tag = 1 << j
        while column:
            pivot = column.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = column, tag
                break
            row, mask = pivots[pivot]
            column ^= row
            tag ^= mask
        else:
            raise ValueError("dependent compile-time generators")

    def solve(bits: int) -> tuple[int, int]:
        tag = 0
        for pivot in sorted(pivots, reverse=True):
            row, mask = pivots[pivot]
            if (bits >> pivot) & 1:
                bits ^= row
                tag ^= mask
        return tag, bits

    return solve


class NoiseLayout:
    def __init__(self, operations: list[Operation], noisy: list[bool], width: int):
        self.operations = operations
        self.noisy = noisy
        self.width = width
        self.slots: dict[tuple[int, int, str], int] = {}
        for j, (op, enabled) in enumerate(zip(operations, noisy, strict=True)):
            if op.name in {"X", "Y", "Z"}:
                raise ValueError("prototype requires ideal circuits without explicit Paulis")
            if enabled:
                boundary = j if op.name == "M" else j + 1
                for q in op.targets:
                    for axis in "X" if op.name in {"R", "M"} else "XZ":
                        key = boundary, q, axis
                        if key not in self.slots:
                            self.slots[key] = len(self.slots)
        self.boundaries: list[list[tuple[int, int, int]]] = []
        for boundary in range(len(operations) + 1):
            self.boundaries.append(
                [
                    (
                        q,
                        (1 << self.slots[boundary, q, "X"])
                        if (boundary, q, "X") in self.slots
                        else 0,
                        (1 << self.slots[boundary, q, "Z"])
                        if (boundary, q, "Z") in self.slots
                        else 0,
                    )
                    for q in range(width)
                    if any((boundary, q, axis) in self.slots for axis in "XZ")
                ]
            )

    def encode(self, actual: list[Operation]) -> int:
        """Input adapter, outside execution: validate and encode inserted Paulis."""
        boundary = 0
        bits = 0
        for op in actual:
            if boundary < len(self.operations) and op == self.operations[boundary]:
                boundary += 1
            elif op.name in {"X", "Y", "Z"} and len(op.targets) == 1:
                for axis in "XZ" if op.name == "Y" else op.name:
                    key = boundary, op.targets[0], axis
                    if key not in self.slots:
                        raise ValueError("fault is outside a supported noise location")
                    bits ^= 1 << self.slots[key]
            else:
                raise ValueError("history does not match the compiled circuit")
        if boundary != len(self.operations):
            raise ValueError("incomplete history")
        return bits


class LinearPlan:
    def __init__(self, layout: NoiseLayout, *, injection: bool = False):
        self.layout = layout
        width = layout.width
        self.x = [0] * width
        self.z = [0] * width
        self.records: list[int] = []
        ideal = stim.TableauSimulator() if injection else None
        seen_t = 0
        for boundary, faults in enumerate(layout.boundaries):
            for q, x, z in faults:
                self.x[q] ^= x
                self.z[q] ^= z
            if boundary == len(layout.operations):
                break
            op = layout.operations[boundary]
            q = op.targets[0]
            if op.name == "R":
                self.x[q] = self.z[q] = 0
            elif op.name == "H":
                self.x[q], self.z[q] = self.z[q], self.x[q]
            elif op.name == "CX":
                r = op.targets[1]
                self.x[r] ^= self.x[q]
                self.z[q] ^= self.z[r]
            elif op.name == "M":
                self.records.append(self.x[q])
            elif op.name == "T" and ideal is not None and ideal.peek_x(q) == 1:
                # The ideal input is a separate |+> factor. Its X error can be
                # removed before T, up to a history-wide irrelevant phase.
                self.x[q] = 0
                seen_t += 1
            else:
                raise ValueError("unsupported linear-propagation operation")
            if ideal is not None and op.name != "T":
                if op.name == "M":
                    if ideal.peek_z(q) != 1:
                        raise ValueError("prefix has a nondeterministic ideal record")
                    ideal.postselect_z(q, desired_value=False)
                else:
                    ideal.do(stim.CircuitInstruction(op.name, op.targets))
        if injection and seen_t != 1:
            raise ValueError("expected one injection on an independent X eigenstate")

    def evaluate(self, faults: int) -> tuple[int, int, int]:
        return (
            parity_map(self.x, faults),
            parity_map(self.z, faults),
            parity_map(self.records, faults),
        )


class Boundary:
    def __init__(
        self,
        reconstruction: Reconstruction,
        before: int,
        after: int,
        growth: list[Operation],
        layout: NoiseLayout,
    ):
        self.linear = LinearPlan(layout)
        self.before = Plan(before)
        self.after = Plan(after)
        self.input_mapping = [reconstruction.qubit(p, before) for p in coordinates(before)]
        self.output_mapping = [reconstruction.qubit(p, after) for p in coordinates(after)]
        width = layout.width
        unitary = stim.Circuit()
        fresh = set(self.output_mapping) - set(self.input_mapping)
        reset = set()
        for op in growth:
            if op.name == "R":
                if op.targets[0] not in fresh or op.targets[0] in reset:
                    raise ValueError("growth resets a live qubit")
                reset.add(op.targets[0])
            elif op.name in {"H", "CX"}:
                unitary.append(op.name, op.targets)
            else:
                raise ValueError("growth is not a supported Clifford isometry")
        if reset != fresh:
            raise ValueError("growth does not initialize every new qubit")

        def embed(p: stim.PauliString, mapping: list[int]) -> stim.PauliString:
            out = stim.PauliString(width)
            out.sign = p.sign
            for q, target in enumerate(mapping):
                out[target] = p[q]
            return out

        input_generators = [pauli(before, a, s) for a, s in stabilizers(before)]
        basis = [embed(p, self.input_mapping) for p in input_generators]
        for q in sorted(fresh):
            p = stim.PauliString(width)
            p[q] = "Z"
            basis.append(p)

        def vector(p):
            x, z = masks(p)
            return x | (z << width)

        def coordinates_of(p, generators):
            solve = span_solver([vector(g) for g in generators])
            tag, residue = solve(vector(p))
            product = stim.PauliString(width)
            for j, g in enumerate(generators):
                if (tag >> j) & 1:
                    product *= g
            if residue or product != p:
                raise ValueError("growth does not preserve the declared signed code basis")
            return tag

        rows = []
        for axis, support in stabilizers(after):
            pulled = embed(pauli(after, axis, support), self.output_mapping).after(
                unitary.inverse()
            )
            rows.append(coordinates_of(pulled, basis) & ((1 << len(input_generators)) - 1))
        for axis in "XZ":
            pulled = embed(logical(after, axis), self.output_mapping).after(unitary.inverse())
            tag = coordinates_of(pulled, basis + [embed(logical(before, axis), self.input_mapping)])
            if not (tag >> len(basis)) & 1:
                raise ValueError("growth loses a logical axis")
        if len(rows) != len(self.linear.records):
            raise ValueError("boundary does not contain exactly one complete syndrome round")
        columns = [
            sum(((row >> j) & 1) << k for k, row in enumerate(rows))
            for j in range(len(input_generators))
        ]
        solve = span_solver(columns)
        unit_solutions = [solve(1 << j) for j in range(len(rows))]
        self.syndrome_rows = [
            sum(((tag >> j) & 1) << k for k, (tag, _) in enumerate(unit_solutions))
            for j in range(len(input_generators))
        ]
        self.constraints = [
            sum(((residue >> j) & 1) << k for k, (_, residue) in enumerate(unit_solutions))
            for j in range(len(rows))
        ]
        duals = stim.Tableau.from_stabilizers(input_generators + [logical(before, "Z")])
        self.duals = [masks(duals.x_output(j)) for j in range(len(input_generators))]
        self.transported = [
            masks(embed(duals.x_output(j), self.input_mapping).after(unitary))
            for j in range(len(input_generators))
        ]

    def reduce(
        self, terms: list[tuple[complex, PhaseMonomial]], state: np.ndarray, faults: int
    ) -> tuple[np.ndarray, PhaseMonomial]:
        x, z, records = self.linear.evaluate(faults)
        if parity_map(self.constraints, records):
            return np.zeros(2, dtype=complex), PhaseMonomial(self.after.width)
        syndrome = parity_map(self.syndrome_rows, records)
        dx = dz = 0
        for j, ((a, b), (c, d)) in enumerate(zip(self.duals, self.transported, strict=True)):
            if (syndrome >> j) & 1:
                dx ^= a
                dz ^= b
                x ^= c
                z ^= d
        inverse = pauli_operator(self.before.width, dx, dz, inverse=True)
        matrix = np.zeros((2, 2), dtype=complex)
        for weight, op in terms:
            matrix += weight * self.before.matrix(compose(inverse, op))
        frame = pauli_operator(
            self.after.width,
            project_mask(x, self.output_mapping),
            project_mask(z, self.output_mapping),
        )
        return matrix @ state, frame


class Fold:
    def __init__(
        self,
        reconstruction: Reconstruction,
        distance: int,
        preparation: NoiseLayout,
        core: NoiseLayout,
        decode: NoiseLayout,
    ):
        self.preparation = LinearPlan(preparation)
        self.decode = LinearPlan(decode)
        self.core = core
        self.plan = Plan(distance)
        self.mapping = [reconstruction.qubit(p, distance) for p in self.plan.coords]
        self.cats = list(
            range(reconstruction.ancilla, reconstruction.ancilla + {3: 3, 5: 5, 7: 8}[distance])
        )
        self.equal_flag_mask = 63 if distance == 7 else 0
        self.data_index = {q: j for j, q in enumerate(self.mapping)}
        self.cat_index = {q: j for j, q in enumerate(self.cats)}
        self.actions: list[tuple[str, tuple[int, ...]]] = []
        for op in core.operations:
            if op.name in {"T", "T_DAG"}:
                self.actions.append((op.name, (self.data_index[op.targets[0]],)))
            elif op.name == "CX":
                self.actions.append(
                    ("CX", (self.cat_index[op.targets[0]], self.data_index[op.targets[1]]))
                )
            elif op.name == "CCZ":
                q, r = sorted(self.data_index[p] for p in op.targets[1:])
                self.actions.append(
                    ("CZ", (self.cat_index[op.targets[0]], q, r, self.plan.edges.index((q, r))))
                )
            else:
                raise ValueError("unsupported core")
        self.fault_actions = [
            [
                (
                    q in self.cat_index,
                    self.cat_index[q] if q in self.cat_index else self.data_index[q],
                    x,
                    z,
                )
                for q, x, z in faults
            ]
            for faults in core.boundaries
        ]
        size = 1 << len(self.cats)
        self.decode_table = np.zeros((size, size), dtype=int)
        for bits in range(size):
            transformed = bits
            root = None
            measured = []
            for op in decode.operations:
                if op.name == "CX" and root is None:
                    q, r = (self.cat_index[p] for p in op.targets)
                    transformed ^= ((transformed >> q) & 1) << r
                elif op.name == "H" and root is None:
                    root = self.cat_index[op.targets[0]]
                elif op.name == "M" and root is not None:
                    measured.append(self.cat_index[op.targets[0]])
                else:
                    raise ValueError("unsupported cat decoder")
            if root is None or measured != list(range(len(self.cats))):
                raise ValueError("decoder records do not cover the cat in order")
            for record in range(size):
                if ((record ^ transformed) & ~(1 << root)) == 0:
                    self.decode_table[record, bits] = (-1) ** (
                        ((record >> root) & 1) * ((transformed >> root) & 1)
                    )
        ideal = stim.TableauSimulator()
        for op in preparation.operations:
            if op.name == "M":
                if distance != 7 and ideal.peek_z(op.targets[0]) != 1:
                    raise ValueError("ideal cat flag is not deterministic")
            else:
                ideal.do(stim.CircuitInstruction(op.name, op.targets))
        for axes in ["X" * len(self.cats)] + [
            "Z" + "_" * (j - 1) + "Z" + "_" * (len(self.cats) - j - 1)
            for j in range(1, len(self.cats))
        ]:
            p = stim.PauliString(core.width)
            for q, axis in zip(self.cats, axes, strict=True):
                p[q] = axis
            if ideal.peek_observable_expectation(p) != 1:
                raise ValueError("preparation is not the declared GHZ state")
        if distance == 7:
            flags = [op.targets[0] for op in preparation.operations if op.name == "M"]
            if len(flags) != 6:
                raise ValueError("expected six verification qubits")
            for supports, axis in [(flags, "X")] + [([flags[0], q], "Z") for q in flags[1:]]:
                p = stim.PauliString(core.width)
                for q in supports:
                    p[q] = axis
                if ideal.peek_observable_expectation(p) != 1:
                    raise ValueError("verification register is not an independent GHZ state")

    def branches(
        self, preparation_faults: int, core_faults: int, decode_faults: int
    ) -> list[tuple[complex, PhaseMonomial]]:
        x, z, flags = self.preparation.evaluate(preparation_faults)
        if flags not in (0, self.equal_flag_mask):
            return []
        _, _, records = self.decode.evaluate(decode_faults)
        x = project_mask(x, self.cats)
        z = project_mask(z, self.cats)
        result = []
        for initial in (0, (1 << len(self.cats)) - 1):
            cat = initial ^ x
            flips = phase = edges = 0
            linear = [0] * self.plan.width
            for boundary, faults in enumerate(self.fault_actions):
                for on_cat, q, fx, fz in faults:
                    a, b = int(bool(core_faults & fx)), int(bool(core_faults & fz))
                    if on_cat:
                        phase += 4 * b * ((cat >> q) & 1)
                        cat ^= a << q
                    else:
                        phase += 4 * b * ((flips >> q) & 1)
                        linear[q] += 4 * b
                        flips ^= a << q
                if boundary == len(self.actions):
                    break
                kind, targets = self.actions[boundary]
                if kind in {"T", "T_DAG"}:
                    q = targets[0]
                    sign = 1 if kind == "T" else -1
                    bit = (flips >> q) & 1
                    phase += sign * bit
                    linear[q] += sign * (1 - 2 * bit)
                elif (cat >> targets[0]) & 1:
                    q = targets[1]
                    if kind == "CX":
                        flips ^= 1 << q
                    else:
                        r, edge = targets[2:]
                        a, b = (flips >> q) & 1, (flips >> r) & 1
                        phase += 4 * a * b
                        linear[q] += 4 * b
                        linear[r] += 4 * a
                        edges ^= 1 << edge
            weight = (
                0.5 * (-1) ** ((initial & z).bit_count() % 2) * int(self.decode_table[records, cat])
            )
            if weight:
                op = PhaseMonomial(
                    self.plan.width,
                    flips,
                    phase % 8,
                    [c % 8 for c in linear],
                    {pair for j, pair in enumerate(self.plan.edges) if (edges >> j) & 1},
                )
                assert all(c % 2 == 0 for c in linear)
                result.append((weight, op))
        return result


@dataclass
class Result:
    acceptance: float
    logical_xyz: list[float] | None
    contractions: int
    peak_terms: int


class Protocol:
    def __init__(self, distance: int):
        self.reconstruction = Reconstruction(distance).build()
        r = self.reconstruction
        self.width = r.ancilla + {3: 3, 5: 6, 7: 14}[distance]

        def layout(indices):
            operations = [op for i in indices for op in r.stages[i].operations]
            noisy = [r.stages[i].noisy for i in indices for _ in r.stages[i].operations]
            return NoiseLayout(operations, noisy, self.width)

        self.prefix = LinearPlan(layout(range(3)), injection=True)
        self.groups: list[tuple[str, Fold | Boundary, list[list[int]]]] = []
        self.input_mapping = [r.qubit(p, 3) for p in coordinates(3)]
        current = 3
        i = 3
        while i < len(r.stages):
            name = r.stages[i].name
            if name.startswith("cat_prepare"):
                groups = [[i], [i + 1], [i + 2]]
                self.groups.append(("fold", Fold(r, current, *(layout(g) for g in groups)), groups))
                i += 3
            else:
                start = i
                growth = []
                while not r.stages[i].name.startswith("syndrome"):
                    growth.extend(r.stages[i].operations)
                    i += 1
                after = int(r.stages[i].name.removeprefix("syndrome_d").split("_")[0])
                indices = list(range(start, i + 1))
                self.groups.append(
                    ("boundary", Boundary(r, current, after, growth, layout(indices)), [indices])
                )
                current = after
                i += 1

    def encode(self, stages: list[list[Operation]]) -> list[list[int]]:
        if len(stages) != len(self.reconstruction.stages):
            raise ValueError("wrong number of stages")
        encoded = [[self.prefix.layout.encode([op for s in stages[:3] for op in s])]]
        for kind, plan, groups in self.groups:
            layouts = (
                [plan.preparation.layout, plan.core, plan.decode.layout]
                if isinstance(plan, Fold)
                else [plan.linear.layout]
            )
            encoded.append(
                [
                    layout.encode([op for i in group for op in stages[i]])
                    for layout, group in zip(layouts, groups, strict=True)
                ]
            )
        return encoded

    def evaluate(self, history: list[list[int]]) -> Result:
        x, z, records = self.prefix.evaluate(history[0][0])
        if records:
            return Result(0, None, 0, 0)
        state = np.array([1, np.exp(1j * np.pi / 4)], dtype=complex) / np.sqrt(2)
        frame = pauli_operator(
            13, project_mask(x, self.input_mapping), project_mask(z, self.input_mapping)
        )
        terms = [(1 + 0j, frame)]
        contractions = 0
        peak = 1
        for (_, plan, _), faults in zip(self.groups, history[1:], strict=True):
            if isinstance(plan, Fold):
                branches = plan.branches(*faults)
                terms = [
                    (a * b, compose(after, before)) for a, before in terms for b, after in branches
                ]
                peak = max(peak, len(terms))
            else:
                contractions += len(terms)
                state, frame = plan.reduce(terms, state, faults[0])
                terms = [(1 + 0j, frame)]
            if not terms or not np.any(state):
                return Result(0, None, contractions, peak)
        acceptance = float(np.vdot(state, state).real)
        cross = state[0].conjugate() * state[1]
        xyz = (
            [
                float(2 * cross.real / acceptance),
                float(2 * cross.imag / acceptance),
                float((abs(state[0]) ** 2 - abs(state[1]) ** 2) / acceptance),
            ]
            if acceptance > 1e-12
            else None
        )
        return Result(acceptance, xyz, contractions, peak)


def syndrome_sector_histories(protocol: Protocol) -> list[tuple[str, list[list[Operation]]]]:
    """Adversarial strata: select leaked sectors and physically return them to code.

    These high-weight histories exercise nonzero measured sectors and growth;
    they have no natural-noise statistical weight in this diagnostic.
    """
    if protocol.reconstruction.distance not in (5, 7):
        raise ValueError("sector stress fixtures require a regular-code growth boundary")
    r = protocol.reconstruction
    after = r.distance
    before = after - 2
    boundary = next(
        p
        for _, p, _ in protocol.groups
        if isinstance(p, Boundary) and p.before.distance == before and p.after.distance == after
    )
    core_index = next(j for j, stage in enumerate(r.stages) if stage.name == f"fold_core_d{before}")
    syndrome_index = next(
        j for j, stage in enumerate(r.stages) if stage.name == f"syndrome_d{after}"
    )
    core = r.stages[core_index].operations
    last = next(j for j, op in enumerate(core) if op.name == "CCZ")
    control = core[last].targets[0]
    first = max(j for j in range(last) if control in core[j].targets)
    hooked = [
        piece
        for j, op in enumerate(core)
        for piece in ([op, Operation("X", (control,))] if j in {first, last} else [op])
    ]
    generators = [pauli(after, a, s) for a, s in stabilizers(after)]
    histories = []
    sectors = [1 << j for j in range(len(boundary.duals))]
    sectors += [
        (1 << j) | (1 << k)
        for j in range(len(boundary.duals))
        for k in range(j + 1, len(boundary.duals))
    ]
    for sector in sectors:
        x = z = 0
        for j, (a, b) in enumerate(boundary.transported):
            if (sector >> j) & 1:
                x ^= a
                z ^= b
        record = 0
        for j, p in enumerate(generators):
            a, b = masks(p)
            record |= (((x & b).bit_count() + (z & a).bit_count()) % 2) << j
        ops = r.stages[syndrome_index].operations
        last_touch = {
            q: max(j for j, op in enumerate(ops) if q in op.targets)
            for q in range(r.ancilla)
            if ((x | z) >> q) & 1
        }
        changed = []
        m = 0
        for j, op in enumerate(ops):
            if op.name == "M":
                if (record >> m) & 1:
                    changed.append(Operation("X", op.targets))
                m += 1
            changed.append(op)
            for q, touch in last_touch.items():
                if j == touch:
                    assert op.name == "CX"
                    a, b = (x >> q) & 1, (z >> q) & 1
                    changed.append(Operation("Y" if a and b else "X" if a else "Z", (q,)))
        stages = [list(s.operations) for s in r.stages]
        stages[core_index] = hooked
        stages[syndrome_index] = changed
        if protocol.evaluate(protocol.encode(stages)).acceptance > 1e-12:
            histories.append((f"growth_sector_{sector}", stages))
    return histories


def f7_diagnostic_histories(protocol: Protocol) -> list[tuple[str, list[list[Operation]]]]:
    """Exercise equality flags and data coordinates beyond a single machine word."""
    r = protocol.reconstruction
    if r.distance != 7:
        raise ValueError("these diagnostics require f7")
    base = [list(s.operations) for s in r.stages]
    cases = [("ideal", base)]
    index = next(j for j, s in enumerate(r.stages) if s.name == "cat_prepare_d7")
    root = next(
        j for j, op in enumerate(base[index]) if op.name == "H" and op.targets == (r.ancilla + 8,)
    )
    stages = [list(s) for s in base]
    stages[index].insert(root + 1, Operation("X", (r.ancilla + 8,)))
    cases.append(("uniform_flag_flip", stages))
    measured = next(j for j, op in enumerate(base[index]) if op.name == "M")
    stages = [list(s) for s in base]
    stages[index].insert(measured, Operation("X", base[index][measured].targets))
    cases.append(("nonuniform_flag_flip", stages))
    index = max(j for j, s in enumerate(r.stages) if s.name == "syndrome_d7")
    last = max(j for j, op in enumerate(base[index]) if 84 in op.targets)
    for axis in "XYZ":
        stages = [list(s) for s in base]
        stages[index].insert(last + 1, Operation(axis, (84,)))
        cases.append((f"high_data_tail_{axis}", stages))
    return cases


def main():
    from clifford_branches import (
        logical_tail_history,
        materialize,
        paired_hook_histories,
        run_history,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--histories", type=int, default=64)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--stress-histories", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    protocol = Protocol(args.distance)
    construction_seconds = time.perf_counter() - start
    cases: list[tuple[str, list[list[Operation]], int | None]] = []
    for seed in range(1000, 1000 + args.histories):
        stages, count = materialize(protocol.reconstruction, args.probability, seed)
        cases.append((str(seed), stages, count))
    cases += [
        (label, stages, 2) for label, stages in paired_hook_histories(protocol.reconstruction)
    ]
    cases.append(("logical_tail", logical_tail_history(protocol.reconstruction), args.distance))
    if args.distance in (5, 7):
        cases += [(label, stages, None) for label, stages in syndrome_sector_histories(protocol)]
    if args.distance == 7:
        cases += [(label, stages, None) for label, stages in f7_diagnostic_histories(protocol)]
    for seed in range(2000, 2000 + args.stress_histories):
        stages, count = materialize(protocol.reconstruction, 0.01, seed)
        cases.append((f"stress_{seed}", stages, count))
    results = []
    for label, stages, fault_count in cases:
        history = protocol.encode(stages)
        start = time.perf_counter()
        result = protocol.evaluate(history)
        results.append(
            {
                "case": label,
                "faults": fault_count,
                **vars(result),
                "python_evaluate_seconds": time.perf_counter() - start,
            }
        )
        if args.validate:
            reference, _ = run_history(protocol.reconstruction, stages)
            probability = reference.expectation()
            xyz = (
                [
                    reference.expectation(logical(args.distance, axis)) / probability
                    for axis in "XYZ"
                ]
                if probability > 1e-12
                else None
            )
            error = abs(result.acceptance - probability)
            probe_error = (
                max(abs(a - b) for a, b in zip(result.logical_xyz, xyz, strict=True))
                if result.logical_xyz is not None and xyz is not None
                else 0
            )
            if (
                error > 2e-12
                or probe_error > 2e-12
                or (xyz is None) != (result.logical_xyz is None)
            ):
                raise ValueError(f"block and coherent reference disagree for {label}")
            results[-1].update(
                reference_acceptance=probability,
                reference_logical_xyz=xyz,
                probability_error=error,
                probe_error=probe_error,
            )
        print(json.dumps(results[-1]), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "distance": args.distance,
                "construction_seconds": construction_seconds,
                "histories": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
