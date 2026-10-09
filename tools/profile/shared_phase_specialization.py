"""Shared affine-support and mod-eight phase analysis for a sampling circuit.

This research compiler accepts a Clifford preparation followed by CNOT and
diagonal phase operations with terminal Pauli readouts. It derives support
coordinates from stabilizers, without identifying a code or logical operation.
Only control evaluation, Clifford correction synthesis, and ordinary lowering
remain per shot. All work is outside Clifft's existing executor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from operator import xor
from typing import Any

import stim
from automatic_specialization import ANNOTATIONS, FaultModel, History

import clifft


def bits(mask: int) -> list[int]:
    result = []
    while mask:
        low = mask & -mask
        result.append(low.bit_length() - 1)
        mask ^= low
    return result


def parity(mask: int, value: int) -> int:
    return (mask & value).bit_count() & 1


def pauli_masks(p: stim.PauliString) -> tuple[int, int]:
    x = sum(1 << q for q in range(len(p)) if p[q] in (1, 2))
    z = sum(1 << q for q in range(len(p)) if p[q] in (2, 3))
    return x, z


def add(poly: dict[int, int], term: int, coefficient: int) -> None:
    value = (poly.get(term, 0) + coefficient) % 8
    if value:
        poly[term] = value
    else:
        poly.pop(term, None)


def add_parity(poly: dict[int, int], mask: int, coefficient: int) -> None:
    support = bits(mask)
    for degree in (1, 2, 3):
        value = coefficient * (-2) ** (degree - 1) % 8
        if value:
            for indices in combinations(support, degree):
                add(poly, sum(1 << q for q in indices), value)


def add_cz(poly: dict[int, int], a: int, b: int) -> None:
    for i in bits(a):
        for j in bits(b):
            add(poly, (1 << i) | (1 << j), 4)


def support_basis(
    generators: list[stim.PauliString],
) -> tuple[list[stim.PauliString], list[int], list[int]]:
    rows = [p.copy() for p in generators]
    n, rank = len(rows), 0
    free = []
    for q in range(n):
        found = next((i for i in range(rank, n) if rows[i][q] in (1, 2)), None)
        if found is None:
            continue
        rows[rank], rows[found] = rows[found], rows[rank]
        for i in range(n):
            if i != rank and rows[i][q] in (1, 2):
                rows[i] *= rows[rank]
        free.append(q)
        rank += 1
    pivots = []
    position = rank
    for q in range(n):
        if q in free:
            continue
        found = next((i for i in range(position, n) if rows[i][q] == 3), None)
        if found is None:
            raise ValueError("Stabilizers do not specify a pure affine-support state")
        rows[position], rows[found] = rows[found], rows[position]
        for i in range(rank, n):
            if i != position and rows[i][q] == 3:
                rows[i] *= rows[position]
        pivots.append(q)
        position += 1
    for row in rows:
        row.sign = 1
    return rows, free, pivots


def inverse_rows(rows: list[int]) -> list[int]:
    width = len(rows)
    augmented = [row | (1 << (width + i)) for i, row in enumerate(rows)]
    for q in range(width):
        found = next(i for i in range(q, width) if augmented[i] >> q & 1)
        augmented[q], augmented[found] = augmented[found], augmented[q]
        for i in range(width):
            if i != q and augmented[i] >> q & 1:
                augmented[i] ^= augmented[q]
    return [row >> width for row in augmented]


def pullback(mask: int, rows: list[int]) -> int:
    return reduce(xor, (rows[i] for i in bits(mask)), 0)


def terminal_bases(model: FaultModel) -> list[str]:
    """Absorb local terminal Cliffords into readout axes, respecting noise sites."""
    lines = model.lines.copy()
    sites = {s.line: i for i, s in enumerate(model.sites)}
    pending: dict[int, tuple[int, stim.PauliString]] = {}
    for index in reversed(range(len(lines))):
        line = lines[index]
        if index in sites and model.sites[sites[index]].gate != "READOUT_NOISE":
            for replacement in model.sites[sites[index]].replacements:
                for instruction in replacement.splitlines():
                    pending.pop(int(instruction.split()[1]), None)
            continue
        if not line:
            continue
        gate, *targets = line.split()
        if gate.split("(")[0] in ANNOTATIONS:
            continue
        if gate in {"M", "MX", "MY"}:
            q = int(targets[0].lstrip("!"))
            axis = {"M": "Z", "MX": "X", "MY": "Y"}[gate]
            pending[q] = (
                index,
                stim.PauliString(("-" if targets[0].startswith("!") else "+") + axis),
            )
            continue
        try:
            data = stim.gate_data(gate)
        except IndexError:
            data = None
        if data is not None and data.is_unitary and data.is_single_qubit_gate and len(targets) == 1:
            q = int(targets[0])
            if q in pending:
                measured, p = pending[q]
                p = p.before(stim.Circuit(f"{gate} 0"))
                pending[q] = measured, p
                name = {1: "MX", 2: "MY", 3: "M"}[p[0]]
                lines[measured] = name + " " + ("!" if p.sign == -1 else "") + str(q)
                lines[index] = ""
                continue
        for target in targets:
            if target.lstrip("!").isdigit():
                pending.pop(int(target.lstrip("!")), None)
            else:
                for q in list(pending):
                    pending.pop(q)
    return lines


@dataclass(frozen=True)
class Readout:
    gate: str
    qubit: int
    flip: int


class SharedPhase:
    def __init__(
        self,
        source: str,
        *,
        max_variables: int = 128,
        max_terms: int = 200000,
        preserve_state: bool = False,
    ):
        self.model = model = FaultModel(source)
        self.preserve_state = preserve_state
        # Absorbing a Clifford into a terminal readout preserves its record law,
        # but can change the post-measurement state needed by a continuation.
        self.lines = lines = model.lines.copy() if preserve_state else terminal_bases(model)
        n = model.num_qubits
        first = None
        for index, line in enumerate(lines):
            if line:
                try:
                    stim.gate_data(line.split()[0].split("(")[0])
                except IndexError:
                    first = index
                    break
        if first is None:
            raise ValueError("No non-Clifford region")
        if lines[first].split()[0] not in {"T", "T_DAG"}:
            raise ValueError("First non-Clifford instruction is outside the phase representation")
        self.first = first
        ideal_prefix = stim.Circuit("\n".join(lines[:first]))
        self.prefix_records = m = ideal_prefix.num_measurements
        simulator = stim.TableauSimulator(seed=71)
        simulator.set_num_qubits(n)
        simulator.do(ideal_prefix)
        # Stim counts an MPAD literal 1 as an additional wire in a one-wire
        # circuit. That unused |0> wire is not part of the physical interface.
        simulator.set_num_qubits(n)
        rows, free, pivots = support_basis(simulator.canonical_stabilizers())
        self.variables = r = len(free)
        if r > max_variables:
            raise ValueError("Affine support exceeds the variable budget")
        self.probe_rows = rows
        probes = stim.Circuit()
        for row in rows:
            product_ = "*".join(f"{'IXYZ'[row[q]]}{q}" for q in range(n) if row[q])
            probes += stim.Circuit("MPP " + product_)
        self.probes = probes
        self.prefix = ideal_prefix + probes
        self.prefix_sampler = self.prefix.compile_sampler(seed=19319)
        self.fault_base = 1 + m + n
        self.site_generators: list[list[tuple[int, int, int]]] = []
        self.outcome_controls: list[list[int]] = []
        generators = 0
        for site in model.sites:
            masks = []
            if site.gate == "READOUT_NOISE":
                entries = [(-1, 0, 1 << generators)]
                generators += 1
                masks = [0, entries[0][2]]
            else:
                decoded = []
                xs = zs = 0
                for replacement in site.replacements:
                    x = z = 0
                    for operation in replacement.splitlines():
                        axis, target = operation.split()
                        if axis in "XY":
                            x ^= 1 << int(target)
                        if axis in "YZ":
                            z ^= 1 << int(target)
                    decoded.append((x, z))
                    xs |= x
                    zs |= z
                entries = []
                for q in bits(xs | zs):
                    xb = (1 << generators) if xs >> q & 1 else 0
                    generators += bool(xb)
                    zb = (1 << generators) if zs >> q & 1 else 0
                    generators += bool(zb)
                    entries.append((q, xb, zb))
                for x, z in decoded:
                    masks.append(
                        reduce(
                            xor,
                            (xb * ((x >> q) & 1) ^ zb * ((z >> q) & 1) for q, xb, zb in entries),
                            0,
                        )
                    )
            self.site_generators.append(entries)
            self.outcome_controls.append(masks)
        self.generator_count = generators
        by_line = {site.line: i for i, site in enumerate(model.sites)}
        self.prefix_responses = self._prefix_responses(first, rows, by_line)

        # Choose support coordinates at the output, without recognizing a decoder.
        xrows = [pauli_masks(row)[0] for row in rows[:r]]
        coordinates = [sum(1 << i for i, row in enumerate(xrows) if row >> q & 1) for q in range(n)]
        output_coordinates = coordinates.copy()
        measured: set[int] = set()
        records = m
        for index in range(first, len(lines)):
            site_id = by_line.get(index)
            if site_id is not None and model.sites[site_id].gate != "READOUT_NOISE":
                if any(q in measured for q, _, _ in self.site_generators[site_id]):
                    raise ValueError("Noise after a deferred readout")
            line = lines[index]
            if not line:
                continue
            gate, *targets = line.split()
            if gate.split("(")[0] in ANNOTATIONS:
                continue
            if preserve_state and gate in {"M", "MX", "MY", "MPAD"}:
                raise ValueError("State-preserving phase regions must end before measurement")
            if gate not in {
                "CX",
                "CZ",
                "T",
                "T_DAG",
                "S",
                "S_DAG",
                "X",
                "Y",
                "Z",
                "I",
                "M",
                "MX",
                "MY",
                "MPAD",
            }:
                raise ValueError(f"Unsupported operation inside phase region: {gate}")
            if gate == "MPAD":
                records += 1
                continue
            qs = [int(t.lstrip("!")) for t in targets if not t.startswith("rec[")]
            if measured.intersection(qs):
                raise ValueError("A measured qubit is used again inside the phase region")
            if targets[0].startswith("rec["):
                rec = records + int(targets[0][4:-1])
                if gate not in {"CX", "CZ"} or not 0 <= rec < m:
                    raise ValueError("Feedback requires a sampled prefix record")
            elif gate == "CX":
                output_coordinates[qs[1]] ^= output_coordinates[qs[0]]
            if gate in {"M", "MX", "MY"}:
                measured.add(qs[0])
                records += 1
        selected, basis = [], {}
        for q, coordinate in enumerate(output_coordinates):
            reduced = coordinate
            while reduced:
                pivot = reduced.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = reduced
                    selected.append(q)
                    break
                reduced ^= basis[pivot]
        if len(selected) != r:
            raise AssertionError("Reversible coordinates lost affine rank")
        inverse = inverse_rows([output_coordinates[q] for q in selected])
        coordinates = [pullback(mask, inverse) for mask in coordinates]
        offsets = [0] * n
        for i, q in enumerate(pivots):
            offsets[q] = 1 << (1 + m + r + i)
        self.pairs = list(combinations(range(r), 2))
        self.pair_indices = {pair: i for i, pair in enumerate(self.pairs)}
        nominal: dict[int, int] = {}
        events: dict[int, tuple[int, int, int]] = {}

        def controlled(mask: int, coefficient: int, dependency: int) -> None:
            if dependency & 1:
                add_parity(nominal, mask, coefficient)
                coefficient = -coefficient
            dependency &= ~1
            if dependency:
                polynomial: dict[int, int] = {}
                add_parity(polynomial, mask, coefficient)
                correction = self.pack_clifford(polynomial)
                events[dependency] = self.plus(events.get(dependency, (0, 0, 0)), correction)

        def rotation(mask: int, coefficient: int, offset: int) -> None:
            signed = -coefficient if offset & 1 else coefficient
            add_parity(nominal, mask, signed)
            controlled(mask, -2 * signed, offset & ~1)

        for i, row in enumerate(rows[:r]):
            x, z = pauli_masks(row)
            add_parity(nominal, inverse[i], 2 * (x & z).bit_count())
            dependency = (1 << (1 + m + i)) ^ pullback(z, offsets)
            controlled(inverse[i], 4, dependency)
            for j in range(i):
                if parity(z, xrows[j]):
                    add_cz(nominal, inverse[i], inverse[j])

        self.output: list[str | Readout | int] = []
        records = 0
        for index, line in enumerate(lines):
            site_id = by_line.get(index)
            readout_control = 0
            if site_id is not None:
                site = model.sites[site_id]
                if site.gate == "READOUT_NOISE":
                    readout_control = self.outcome_controls[site_id][1] << self.fault_base
                elif index >= first:
                    for q, xb, zb in self.site_generators[site_id]:
                        controlled(coordinates[q], 4, zb << self.fault_base)
                        offsets[q] ^= xb << self.fault_base
            if not line:
                continue
            gate, *targets = line.split()
            bare = gate.split("(")[0]
            if bare in ANNOTATIONS:
                if bare in {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}:
                    self.output.append(line)
                continue
            if index < first:
                instruction = stim.Circuit(line)
                for _ in range(instruction.num_measurements):
                    self.output.append(records)
                    records += 1
                continue
            if gate == "MPAD":
                self.output.append(Readout("MPAD", int(targets[0]), readout_control))
                records += 1
                continue
            q = int(targets[-1].lstrip("!"))
            if gate in {"M", "MX", "MY"}:
                flip = readout_control ^ int(targets[0].startswith("!"))
                if gate in {"M", "MY"}:
                    flip ^= offsets[q]
                self.output.append(Readout(gate, q, flip))
                records += 1
            elif gate in {"CX", "CZ"}:
                if targets[0].startswith("rec["):
                    dependency = 1 << (1 + records + int(targets[0][4:-1]))
                    if gate == "CX":
                        offsets[q] ^= dependency
                    else:
                        controlled(coordinates[q], 4, dependency)
                else:
                    control = int(targets[0])
                    if gate == "CX":
                        coordinates[q] ^= coordinates[control]
                        offsets[q] ^= offsets[control]
                    else:
                        add_cz(nominal, coordinates[control], coordinates[q])
                        controlled(coordinates[q], 4, offsets[control])
                        controlled(coordinates[control], 4, offsets[q])
            elif gate in {"T", "T_DAG", "S", "S_DAG", "Z", "Y"}:
                coefficient = {"T": 1, "T_DAG": -1, "S": 2, "S_DAG": -2, "Z": 4, "Y": 4}[gate]
                rotation(coordinates[q], coefficient, offsets[q])
            if gate in {"X", "Y"}:
                offsets[q] ^= 1
            if len(nominal) > max_terms or len(events) > max_terms:
                raise ValueError("Phase representation exceeds the term budget")
        if records != model.num_records:
            raise AssertionError("The output template lost records")
        self.events = [
            (dependency, correction) for dependency, correction in events.items() if any(correction)
        ]
        self.event_controls = [0] * (self.fault_base + self.generator_count)
        for index, (dependency, _) in enumerate(self.events):
            for bit in bits(dependency):
                self.event_controls[bit] ^= 1 << index
        self.nominal = nominal
        self.base, self.nonclifford = self.synthesize(nominal)
        self.quantum_map = {q: i for i, q in enumerate(selected)}
        self.encoder = []
        for q, coordinate in enumerate(coordinates):
            if q in self.quantum_map or not coordinate:
                continue
            output_qubit = len(self.quantum_map)
            self.quantum_map[q] = output_qubit
            self.encoder += [f"CX {i} {output_qubit}" for i in bits(coordinate)]
        self.preparation = ["H " + " ".join(map(str, range(r)))] if r else []
        self.exit_offsets = offsets
        self.symbolic_payload_bytes = sum((dep.bit_length() + 7) // 8 for dep, _ in self.events)

    def _prefix_responses(
        self, first: int, rows: list[stim.PauliString], by_line: dict[int, int]
    ) -> list[list[int]]:
        n = self.model.num_qubits
        x, z = [0] * n, [0] * n
        records: list[int] = []
        for index in range(first):
            site_id = by_line.get(index)
            if site_id is not None and self.model.sites[site_id].gate != "READOUT_NOISE":
                for q, xb, zb in self.site_generators[site_id]:
                    x[q] ^= xb
                    z[q] ^= zb
            line = self.lines[index]
            if not line:
                continue
            instruction = next(iter(stim.Circuit(line)))
            gate = instruction.name
            data = stim.gate_data(gate)
            targets = instruction.targets_copy()
            if gate in ANNOTATIONS:
                continue
            if gate == "MPAD":
                records.append(0)
            elif data.produces_measurements:
                if gate == "MPP":
                    flip = 0
                    for t in targets:
                        if t.is_x_target or t.is_y_target:
                            flip ^= z[t.value]
                        if t.is_z_target or t.is_y_target:
                            flip ^= x[t.value]
                    records.append(flip)
                elif gate in {"M", "MX", "MY"}:
                    records.extend(
                        (
                            x[t.value]
                            if gate == "M"
                            else z[t.value]
                            if gate == "MX"
                            else x[t.value] ^ z[t.value]
                        )
                        for t in targets
                    )
                else:
                    raise ValueError(f"Unsupported preparation measurement: {gate}")
            elif data.is_reset:
                for t in targets:
                    x[t.value] = z[t.value] = 0
            elif data.is_unitary:
                if any(t.is_measurement_record_target for t in targets):
                    if (
                        gate not in {"CX", "CY", "CZ"}
                        or len(targets) != 2
                        or not targets[0].is_measurement_record_target
                    ):
                        raise ValueError("Unsupported preparation feedback")
                    dep, q = records[len(records) + targets[0].value], targets[1].value
                    if gate in {"CX", "CY"}:
                        x[q] ^= dep
                    if gate in {"CZ", "CY"}:
                        z[q] ^= dep
                else:
                    tableau = stim.Tableau.from_named_gate(gate)
                    arity = len(tableau)
                    for start in range(0, len(targets), arity):
                        qs = [t.value for t in targets[start : start + arity]]
                        old_x, old_z = [x[q] for q in qs], [z[q] for q in qs]
                        for j, q in enumerate(qs):
                            x[q] = z[q] = 0
                            for i in range(arity):
                                for p, dep in (
                                    (tableau.x_output(i), old_x[i]),
                                    (tableau.z_output(i), old_z[i]),
                                ):
                                    if p[j] in (1, 2):
                                        x[q] ^= dep
                                    if p[j] in (2, 3):
                                        z[q] ^= dep
            else:
                raise ValueError(f"Unsupported preparation operation: {gate}")
            if site_id is not None and self.model.sites[site_id].gate == "READOUT_NOISE":
                records[-1] ^= self.outcome_controls[site_id][1]
        dependencies = records.copy()
        for row in rows:
            a, b = pauli_masks(row)
            dependencies.append(pullback(a, z) ^ pullback(b, x))
        if len(dependencies) != self.prefix_records + n:
            raise AssertionError("Incomplete prefix record response")
        transposed = [0] * self.generator_count
        for output, dep in enumerate(dependencies):
            for generator in bits(dep):
                transposed[generator] ^= 1 << output
        return [
            [pullback(mask, transposed) for mask in outcomes] for outcomes in self.outcome_controls
        ]

    @staticmethod
    def plus(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
        return a[0] ^ b[0], a[1] ^ b[1] ^ (a[0] & b[0]), a[2] ^ b[2]

    def pack_clifford(self, polynomial: dict[int, int]) -> tuple[int, int, int]:
        s = z = cz = 0
        for mask, coefficient in polynomial.items():
            support = bits(mask)
            if len(support) == 1 and coefficient % 2 == 0:
                s ^= mask if coefficient & 2 else 0
                z ^= mask if coefficient & 4 else 0
            elif len(support) == 2 and coefficient == 4:
                cz ^= 1 << self.pair_indices[support[0], support[1]]
            else:
                raise ValueError("A conditional correction is not Clifford")
        return s, z, cz

    def synthesize(self, polynomial: dict[int, int]) -> tuple[tuple[int, int, int], list[str]]:
        clifford: dict[int, int] = {}
        rotations: dict[int, int] = {}
        for mask, coefficient in polynomial.items():
            support = bits(mask)
            degree = len(support)
            if degree == 1:
                add(clifford, mask, coefficient & 6)
                if coefficient & 1:
                    add(rotations, mask, 1)
            elif degree == 2:
                add(clifford, mask, coefficient & 4)
                if coefficient & 2:
                    for q in support:
                        add(rotations, 1 << q, 1)
                    add(rotations, mask, -1)
            elif degree == 3 and coefficient == 4:
                for count in (1, 2, 3):
                    for subset in combinations(support, count):
                        add(rotations, sum(1 << q for q in subset), 1 if count % 2 else -1)
            else:
                raise ValueError("Unsupported weighted phase monomial")
        gates = []
        for mask, coefficient in sorted(rotations.items()):
            support = bits(mask)
            target = support[-1]
            if coefficient % 2 == 0:
                add_parity(clifford, mask, coefficient)
                continue
            signed = -1 if coefficient == 7 else 1
            add_parity(clifford, mask, coefficient - signed)
            compute = [f"CX {q} {target}" for q in support[:-1]]
            gates += compute + [f"{'T' if signed == 1 else 'T_DAG'} {target}"] + compute[::-1]
        return self.pack_clifford(clifford), gates

    def controls(self, history: History, prefix_bits: int | None = None) -> int:
        if prefix_bits is None:
            sample = self.prefix_sampler.sample(1, bit_packed=True)[0]
            prefix_bits = int.from_bytes(sample.tobytes(), "little")
        faults = flips = 0
        seen = set()
        for site, outcome in history:
            if (
                site in seen
                or not 0 <= site < len(self.model.sites)
                or not 0 < outcome < len(self.outcome_controls[site])
            ):
                raise ValueError("Invalid categorical fault history")
            seen.add(site)
            faults ^= self.outcome_controls[site][outcome]
            flips ^= self.prefix_responses[site][outcome]
        return int(1 | ((prefix_bits ^ flips) << 1) | (faults << self.fault_base))

    def quantum_lines(self, controls: int) -> list[str]:
        correction = self.base
        active = pullback(controls, self.event_controls)
        for event in bits(active):
            correction = self.plus(correction, self.events[event][1])
        s, z, cz = correction
        lines = self.preparation + self.nonclifford
        for q in bits(s | z):
            coefficient = 2 * ((s >> q) & 1) + 4 * ((z >> q) & 1)
            lines.append(f"{ {2: 'S', 4: 'Z', 6: 'S_DAG'}[coefficient] } {q}")
        lines += [f"CZ {self.pairs[i][0]} {self.pairs[i][1]}" for i in bits(cz)]
        lines += self.encoder
        return lines

    def render_state(self, controls: int) -> str:
        """Prepare the conditional state on the original physical wire labels."""
        if not self.preserve_state:
            raise ValueError("A sampling-only reduction cannot expose a quantum-state interface")
        physical = {compact: q for q, compact in self.quantum_map.items()}
        lines = [f"I {self.model.num_qubits - 1}"]
        for line in self.quantum_lines(controls):
            gate, *targets = line.split()
            lines.append(gate + " " + " ".join(str(physical[int(q)]) for q in targets))
        for q, offset in enumerate(self.exit_offsets):
            if parity(offset, controls):
                lines.append(f"X {q}")
        for output in self.output:
            if isinstance(output, str):
                lines.append(output)
            elif isinstance(output, int):
                lines.append(f"MPAD {(controls >> (1 + output)) & 1}")
            else:
                raise AssertionError("A quantum-state exit cannot contain deferred readouts")
        return "\n".join(lines) + "\n"

    def render(self, controls: int, rng: random.Random | None) -> str:
        if self.preserve_state:
            raise ValueError("Use the quantum-state renderer for a state-preserving region")
        lines = self.quantum_lines(controls)
        auxiliary = len(self.quantum_map)
        for output in self.output:
            if isinstance(output, str):
                lines.append(output)
            elif isinstance(output, int):
                lines.append(f"MPAD {(controls >> (1 + output)) & 1}")
            else:
                flip = parity(output.flip, controls)
                if output.gate == "MPAD":
                    lines.append(f"MPAD {output.qubit ^ flip}")
                elif output.qubit not in self.quantum_map:
                    if output.gate != "M" and rng is None:
                        lines.append(f"MX {'!' if flip else ''}{auxiliary}")
                        auxiliary += 1
                    else:
                        value = flip
                        if output.gate != "M":
                            assert rng is not None
                            value ^= rng.getrandbits(1)
                        lines.append(f"MPAD {value}")
                else:
                    lines.append(
                        f"{output.gate} {'!' if flip else ''}{self.quantum_map[output.qubit]}"
                    )
        return "\n".join(lines) + "\n"

    def program(self, history: History, rng: random.Random) -> tuple[Any, str]:
        text = self.render(self.controls(history), rng)
        return clifft.lower(clifft.trace(clifft.parse(text))), text

    def metadata(self) -> dict[str, Any]:
        return {
            "physical_qubits": self.model.num_qubits,
            "support_variables": self.variables,
            "quantum_core_qubits": len(self.quantum_map),
            "prefix_records": self.prefix_records,
            "primitive_fault_controls": self.generator_count,
            "phase_monomials": len(self.nominal),
            "conditional_clifford_responses": len(self.events),
            "residual_t": sum(line.startswith(("T ", "T_DAG ")) for line in self.nonclifford),
            "encoder_cnot": len(self.encoder),
            "dependency_mask_payload_bytes": self.symbolic_payload_bytes,
        }
