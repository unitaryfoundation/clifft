"""Offline certification of T-conjugated measurement regions in MSC inputs.

This is a fixed-fault instrument diagnostic, not a noisy circuit sampler.
Noise and record annotations are retained by source line but are not sampled.
Unsupported region structure declines instead of approximating its action.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fold_check import PhaseMonomial


@dataclass(frozen=True)
class Gate:
    name: str
    targets: tuple[int, ...]
    line: int


@dataclass(frozen=True)
class Region:
    wires: tuple[int, ...]
    gates: tuple[Gate, ...]
    source_lines: tuple[str, ...]
    start: int
    end: int

    @property
    def data(self):
        return tuple(sorted({g.targets[0] for g in self.gates if g.name in {"T", "T_DAG"}}))

    @property
    def ancillas(self):
        return tuple(q for q in self.wires if q not in self.data)

    @property
    def t_count(self):
        return sum(g.name in {"T", "T_DAG"} for g in self.gates)


def read_gates(text: str) -> list[Gate]:
    """Small diagnostic reader; every other quantum operation remains a barrier."""
    result = []
    metadata = {"QUBIT_COORDS", "TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}
    noise = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}
    for line, raw in enumerate(text.splitlines(), 1):
        raw = raw.split("#", 1)[0].strip()
        if not raw:
            continue
        match = re.fullmatch(r"([A-Z_0-9]+)(?:\([^)]*\))?(?:\s+(.*))?", raw)
        if match is None:
            result.append(Gate("BARRIER", (), line))
            continue
        name, arguments = match.groups()
        if name in metadata | noise:
            continue
        fields = (arguments or "").split()
        if name == "MPP" and len(fields) == 1:
            paulis = fields[0].split("*")
            if all(re.fullmatch(r"Y[0-9]+", p) for p in paulis):
                result.append(Gate("MPP_Y", tuple(int(p[1:]) for p in paulis), line))
                continue
        if name not in {"T", "T_DAG", "CX", "MX", "RX"} or not all(s.isdecimal() for s in fields):
            result.append(Gate("BARRIER", (), line))
            continue
        arity = 2 if name == "CX" else 1
        if not fields or len(fields) % arity:
            result.append(Gate("BARRIER", (), line))
            continue
        for k in range(0, len(fields), arity):
            result.append(Gate(name, tuple(map(int, fields[k : k + arity])), line))
    return result


def regions(text: str) -> list[Region]:
    """Find and certify paired T layers enclosing CX/MX/RX or one Y product."""
    gates = read_gates(text)
    lines = text.splitlines()
    found = []
    i = 0
    while i < len(gates):
        start = i
        if gates[i].name not in {"T", "T_DAG"}:
            i += 1
            continue
        while i < len(gates) and gates[i].name in {"T", "T_DAG"}:
            i += 1
        opening_end = i
        while i < len(gates) and gates[i].name == "CX":
            i += 1
        if i < len(gates) and gates[i].name == "MX":
            root = gates[i].targets
            i += 1
            if i >= len(gates) or gates[i].name != "RX" or gates[i].targets != root:
                continue
            i += 1
            while i < len(gates) and gates[i].name == "CX":
                i += 1
        elif i == opening_end and i < len(gates) and gates[i].name == "MPP_Y":
            i += 1
        else:
            continue
        closing_start = i
        while i < len(gates) and gates[i].name in {"T", "T_DAG"}:
            i += 1
        opening, closing = gates[start:opening_end], gates[closing_start:i]
        a = {g.targets: g.name for g in opening}
        b = {g.targets: g.name for g in closing}
        if len(a) != len(opening) or len(b) != len(closing) or a.keys() != b.keys():
            continue
        if any(a[q] == b[q] for q in a):
            continue
        body = tuple(gates[start:i])
        candidate = Region(
            tuple(sorted({q for g in body for q in g.targets})),
            body,
            tuple(lines[body[0].line - 1 : body[-1].line]),
            body[0].line,
            body[-1].line,
        )
        try:
            terms(candidate, 0)
        except ValueError:
            continue
        found.append(candidate)
    return found


def terms(
    region: Region, outcome: int, faults: Sequence[tuple[int, str, int]] = ()
) -> tuple[PhaseMonomial, PhaseMonomial]:
    """Two terms of weight 1/2 for a fixed true outcome and fixed Pauli history.

    Faults are (boundary, axis, physical_wire), inserted before that gate, with
    len(gates) denoting the exit. Root faults between MX and RX are erased by
    reset, up to an irrelevant common trajectory phase. Readout flips change
    the reported record, not this true quantum outcome.

    Offline affine propagation proves cancellation of the CX permutation and
    even residual phase coefficients. No runtime topology plan is implied.
    """
    if outcome not in (0, 1):
        raise ValueError("outcome must be binary")
    index = {q: k for k, q in enumerate(region.wires)}
    slots: dict[int, list[tuple[str, int]]] = {}
    for boundary, axis, wire in faults:
        if not 0 <= boundary <= len(region.gates) or axis not in {"X", "Y", "Z"}:
            raise ValueError("invalid fixed fault")
        if wire not in index:
            raise ValueError("fault wire is outside region")
        slots.setdefault(boundary, []).append((axis, index[wire]))
    result = []
    for branch in (0, 1):
        width = len(index)
        rows = [1 << q for q in range(width)]
        flips = [0] * width
        linear = [0] * width
        constant = 0

        def pauli(axis, q):
            nonlocal constant
            if axis in {"Z", "Y"}:
                constant += 4 * flips[q] + (2 if axis == "Y" else 0)
                for r in range(width):
                    linear[r] += 4 * ((rows[q] >> r) & 1)
            if axis in {"X", "Y"}:
                flips[q] ^= 1

        for boundary in range(len(region.gates) + 1):
            erased = (
                index[region.gates[boundary].targets[0]]
                if boundary < len(region.gates) and region.gates[boundary].name == "RX"
                else -1
            )
            for axis, q in slots.get(boundary, []):
                if q != erased:
                    pauli(axis, q)
            if boundary == len(region.gates):
                break
            gate = region.gates[boundary]
            targets = [index[q] for q in gate.targets]
            q = targets[0]
            if gate.name in {"T", "T_DAG"}:
                if rows[q].bit_count() != 1:
                    raise ValueError("T on a parity is outside this certificate")
                sign = 1 if gate.name == "T" else -1
                constant += sign * flips[q]
                linear[rows[q].bit_length() - 1] += sign * (1 - 2 * flips[q])
            elif gate.name == "CX":
                r = targets[1]
                if q == r:
                    raise ValueError("repeated CX wire")
                rows[r] ^= rows[q]
                flips[r] ^= flips[q]
            elif gate.name == "MX":
                # |+><m_X| = (I + X) Z**m / 2, including the reset.
                if outcome:
                    pauli("Z", q)
                if branch:
                    pauli("X", q)
            elif gate.name == "MPP_Y":
                if branch:
                    constant += 4 * outcome
                    for r in targets:
                        pauli("Y", r)
            elif gate.name != "RX":
                raise ValueError("unsupported instrument gate")
        if rows != [1 << q for q in range(width)] or any(c % 2 for c in linear):
            raise ValueError("region does not reduce to Clifford monomials")
        result.append(
            PhaseMonomial(
                width,
                sum(x << q for q, x in enumerate(flips)),
                constant % 8,
                [c % 8 for c in linear],
            )
        )
    return result[0], result[1]


def contract_ancillas(
    region: Region, operators: Sequence[PhaseMonomial], records: Mapping[int, int]
) -> list[PhaseMonomial]:
    """Contract ideal |+> ancillas and specified X outcomes, retaining phases.

    Preparation/readout faults must already be represented as physical Paulis
    or record flips by the caller. This does not infer a code boundary.
    """
    if set(records) != set(region.ancillas) or any(v not in (0, 1) for v in records.values()):
        raise ValueError("one binary X outcome is required per ancilla")
    index = {q: k for k, q in enumerate(region.wires)}
    result = []
    for op in operators:
        phase = op.global_phase
        for q in region.ancillas:
            k = index[q]
            if op.linear[k] not in (0, 4):
                raise ValueError("ancilla phase is outside X-basis contraction")
            if records[q] != op.linear[k] // 4:
                break
            phase += 4 * records[q] * ((op.flips >> k) & 1)
        else:
            result.append(
                PhaseMonomial(
                    len(region.data),
                    sum(((op.flips >> index[q]) & 1) << j for j, q in enumerate(region.data)),
                    phase % 8,
                    [op.linear[index[q]] for q in region.data],
                )
            )
    return result
