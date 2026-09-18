"""Compile noisy MSC monomials into parity polynomials and fixed payload steps.

Only construction propagates CNOT coordinates. Binding evaluates fixed parity
products modulo eight; no circuit traversal or stabilizer analysis is needed.
This allocating Python implementation is a research reference for native lowering.
"""

import numpy as np
from fold_check import PhaseMonomial
from fold_contraction import ROOTS, compose
from msc_boundaries import FaultBits
from msc_growth_contraction import FactoredTerm

PAULIS = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.diag([1, -1]).astype(complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
)


def eigenstate(axis, sign):
    if axis == 3:
        return np.array([1 - sign, sign], dtype=complex)
    return np.array([1, (-1) ** sign * (1j if axis == 2 else 1)]) / np.sqrt(2)


class Polynomial:
    def __init__(self):
        self.terms = {}

    def add(self, coefficient, a=1, b=1):
        if not a or not b:
            return
        if a == b:
            b = 1
        key = tuple(sorted((a, b)))
        self.terms[key] = (self.terms.get(key, 0) + coefficient) % 8
        if not self.terms[key]:
            del self.terms[key]

    def evaluate(self, bits):
        return (
            sum(
                c * ((a & bits).bit_count() & 1) * ((b & bits).bit_count() & 1)
                for (a, b), c in self.terms.items()
            )
            % 8
        )


class GadgetPlan:
    def __init__(self, program, region, bits):
        self.region = region
        self.outcome_bit = 1 << (len(bits.slots) + 1)
        mapping = {q: k for k, q in enumerate(region.wires)}
        self.outside = []
        slots: dict[int, list[tuple[int, int, int]]] = {}
        measured = next(g for g in region.gates if g.name in {"MX", "MPP_Y"})
        self.event = program.measurement_at[measured.line, 0]
        reset = next((g for g in region.gates if g.name == "RX"), None)
        self.hidden = None if reset is None else program.measurement_at[reset.line, 0]
        self.reset_flip = 0
        for site, item in enumerate(program.sites):
            if item.readout or not region.start <= item.line <= region.end:
                continue
            offset = sum(g.line < item.line for g in region.gates)
            for q in item.wires:
                x, z = (bits.bit(site, q, axis) for axis in "XZ")
                if reset and measured.line < item.line < reset.line and q == measured.targets[0]:
                    self.reset_flip ^= z
                if q in mapping:
                    slots.setdefault(offset, []).append((mapping[q], x << 1, z << 1))
                else:
                    self.outside.append((q, x, z))
        self.branches = []
        for branch in (0, 1):
            width = len(mapping)
            rows = [1 << q for q in range(width)]
            flips = [0] * width
            linear = [Polynomial() for _ in range(width)]
            phase = Polynomial()

            def pauli(q, x, z):
                phase.add(4, z, flips[q])
                phase.add(2, x, z)
                for r in range(width):
                    if rows[q] >> r & 1:
                        linear[r].add(4, z)
                flips[q] ^= x

            for offset in range(len(region.gates) + 1):
                gate = region.gates[offset] if offset < len(region.gates) else None
                erased = mapping[gate.targets[0]] if gate and gate.name == "RX" else -1
                for q, x, z in slots.get(offset, []):
                    if q != erased:
                        pauli(q, x, z)
                if gate is None:
                    break
                targets = [mapping[q] for q in gate.targets]
                q = targets[0]
                if gate.name in {"T", "T_DAG"}:
                    if rows[q].bit_count() != 1:
                        raise ValueError("T acts on a parity")
                    sign = 1 if gate.name == "T" else -1
                    phase.add(sign, flips[q])
                    linear[rows[q].bit_length() - 1].add(sign)
                    linear[rows[q].bit_length() - 1].add(-2 * sign, flips[q])
                elif gate.name == "CX":
                    r = targets[1]
                    rows[r] ^= rows[q]
                    flips[r] ^= flips[q]
                elif gate.name == "MX":
                    pauli(q, 0, self.outcome_bit)
                    pauli(q, branch, 0)
                elif gate.name == "MPP_Y":
                    if branch:
                        phase.add(4, self.outcome_bit)
                        for r in targets:
                            pauli(r, 1, 1)
                elif gate.name != "RX":
                    raise ValueError("unsupported gadget gate")
            if rows != [1 << q for q in range(width)]:
                raise ValueError("gadget has an uncancelled coordinate permutation")
            self.branches.append((tuple(flips), phase, tuple(linear)))

    def bind(self, faults, outcome):
        packed = (faults << 1) | 1 | (self.outcome_bit * outcome)
        return tuple(
            PhaseMonomial(
                len(flips),
                sum(((row & packed).bit_count() & 1) << q for q, row in enumerate(flips)),
                phase.evaluate(packed),
                [row.evaluate(packed) for row in linear],
            )
            for flips, phase, linear in self.branches
        )


class PayloadPlan:
    def __init__(self, program, boundary, contraction, *, terminal):
        self.program, self.boundary = program, boundary
        self.bits = FaultBits(program)
        self.data, self.ancillas = contraction.data, contraction.ancillas
        self.width = len(self.data)
        self.initial = []
        for k, p in enumerate(boundary.checks[boundary.code_count :], boundary.code_count):
            support = [q for q in self.ancillas if p[q]]
            if len(support) != 1:
                raise ValueError("input spectator is not a single-wire state")
            q = support[0]
            self.initial.append((self.ancillas.index(q), p[q], k))
        regions = (
            list(program.regions.values())[-2:]
            if terminal
            else [next(iter(program.regions.values()))]
        )
        by_line = {r.start: r for r in regions}
        end = contraction.measurement.line if terminal else regions[0].end + 1
        skip = boundary.end
        self.steps: list[tuple[str, tuple]] = []
        for op in program.instructions:
            if op.line <= skip or op.line >= end:
                continue
            if op.line in by_line:
                region = by_line[op.line]
                gadget = GadgetPlan(program, region, self.bits)
                mapping = {q: k for k, q in enumerate(region.wires)}
                dm = tuple(mapping[q] for q in self.data)
                am = tuple((j, mapping[q]) for j, q in enumerate(self.ancillas) if q in mapping)
                for q, x, z in gadget.outside:
                    self.steps.append(("fault", (False, self.ancillas.index(q), x, z)))
                self.steps.append(("gadget", (gadget, dm, am)))
                skip = region.end
            elif op.name == "MX":
                for offset, word in enumerate(op.targets):
                    self.steps.append(
                        (
                            "measure",
                            (
                                self.ancillas.index(int(word)),
                                program.measurement_at[op.line, offset],
                            ),
                        )
                    )
            elif op.name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
                for site, item in enumerate(program.sites):
                    if item.line == op.line and not item.readout:
                        for q in item.wires:
                            data = q in self.data
                            j = (self.data if data else self.ancillas).index(q)
                            self.steps.append(
                                (
                                    "fault",
                                    (
                                        data,
                                        j,
                                        self.bits.bit(site, q, "X"),
                                        self.bits.bit(site, q, "Z"),
                                    ),
                                )
                            )
            elif op.name not in {
                "TICK",
                "DETECTOR",
                "OBSERVABLE_INCLUDE",
                "SHIFT_COORDS",
                "QUBIT_COORDS",
            }:
                raise ValueError("unsupported operation between gadgets")

    def start(self, history, outcomes):
        records = self.program.outputs(outcomes, history)["records"]
        signs, frame = self.boundary.evaluate(records, history)
        ancillas = np.zeros((len(self.ancillas), 2), dtype=complex)
        for j, axis, k in self.initial:
            ancillas[j] = eigenstate(axis, signs[k])
        return frame, [FactoredTerm(PhaseMonomial(self.width), ancillas, 1)]

    def expand(self, payload, step, faults, outcome):
        gadget, dm, am = step
        result = []
        for branch in gadget.bind(faults, outcome):
            data = PhaseMonomial(
                self.width,
                sum(((branch.flips >> k) & 1) << j for j, k in enumerate(dm)),
                branch.global_phase,
                [branch.linear[k] for k in dm],
            )
            for term in payload:
                spectators = term.ancillas.copy()
                for j, k in am:
                    spectators[j, 1] *= ROOTS[branch.linear[k]]
                    if branch.flips >> k & 1:
                        spectators[j] = spectators[j, ::-1].copy()
                result.append(FactoredTerm(compose(data, term.data), spectators, term.weight / 2))
        return result

    @staticmethod
    def fault(payload, step, faults):
        data, j, x, z = step
        choice = bool(faults & x) + 2 * bool(faults & z)
        if choice:
            for term in payload:
                if data:
                    term.data.append("_XZY"[choice], (j,))
                else:
                    term.ancillas[j] = PAULIS[choice] @ term.ancillas[j]

    @staticmethod
    def measure(payload, j, outcome):
        target = eigenstate(1, outcome)
        for term in payload:
            term.weight *= np.vdot(target, term.ancillas[j])
            term.ancillas[j] = target.copy()

    def bind(self, history, outcomes):
        faults = self.bits.encode(self.program, history)
        frame, payload = self.start(history, outcomes)
        for kind, step in self.steps:
            if kind == "gadget":
                gadget = step[0]
                if gadget.hidden is not None and outcomes[gadget.hidden] != outcomes[
                    gadget.event
                ] ^ ((gadget.reset_flip & faults).bit_count() & 1):
                    raise ValueError("impossible gadget reset outcome")
                payload = self.expand(payload, step, faults, outcomes[gadget.event])
            elif kind == "measure":
                j, event = step
                self.measure(payload, j, outcomes[event])
            else:
                self.fault(payload, step, faults)
        return frame, payload
