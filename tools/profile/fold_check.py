"""Offline two-Clifford-term diagnostic for a cat-controlled fold check.

This models the check core, not injection, verified cat preparation, growth,
or a complete cultivation protocol. Fault events are fixed physical Paulis;
their probabilities and the measurement outcome must be handled by a caller.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Event:
    kind: str
    targets: tuple[int, ...]


@dataclass(frozen=True)
class Check:
    data: int
    ancillas: int
    events: tuple[Event, ...]

    def validate(self) -> None:
        if self.data < 1 or self.ancillas < 1:
            raise ValueError("a check needs data and cat qubits")
        for event in self.events:
            if event.kind in {"X", "Y", "Z"}:
                valid = (
                    len(event.targets) == 1 and 0 <= event.targets[0] < self.data + self.ancillas
                )
            elif event.kind in {"T", "T_DAG"}:
                valid = len(event.targets) == 1 and 0 <= event.targets[0] < self.data
            elif event.kind in {"HXY", "HXY_MINUS", "CX", "CZ"}:
                count = 3 if event.kind == "CZ" else 2
                valid = (
                    len(event.targets) == count
                    and self.data <= event.targets[0] < self.data + self.ancillas
                    and all(0 <= q < self.data for q in event.targets[1:])
                    and len(set(event.targets)) == count
                )
            else:
                valid = False
            if not valid:
                raise ValueError(f"unsupported check event: {event}")


@dataclass
class PhaseMonomial:
    """|b> -> omega**phase(b) |b xor flips>, retaining relative phase.

    phase(b) = global_phase + sum(linear[q] b[q])
               + 4 sum(b[a] b[b]) over quadratic edges, modulo eight.
    """

    width: int
    flips: int = 0
    global_phase: int = 0
    linear: list[int] = field(default_factory=list)
    edges: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.linear:
            self.linear = [0] * self.width

    def append(self, kind: str, targets: tuple[int, ...]) -> None:
        q = targets[0]
        x = (self.flips >> q) & 1
        if kind == "CZ":
            r = targets[1]
            y = (self.flips >> r) & 1
            self.global_phase += 4 * x * y
            self.linear[q] += 4 * y
            self.linear[r] += 4 * x
            self.edges.symmetric_difference_update({(min(q, r), max(q, r))})
        elif kind in {"S", "S_DAG", "T", "T_DAG"}:
            sign = -1 if kind.endswith("_DAG") else 1
            power = 2 if kind.startswith("S") else 1
            self.global_phase += power * sign * x
            self.linear[q] += power * sign * (1 - 2 * x)
        elif kind in {"X", "Y", "Z"}:
            if kind in {"Y", "Z"}:
                self.global_phase += 4 * x + (2 if kind == "Y" else 0)
                self.linear[q] += 4
            if kind in {"X", "Y"}:
                self.flips ^= 1 << q
        else:
            raise ValueError(f"unsupported monomial gate: {kind}")
        self.global_phase %= 8
        self.linear = [c % 8 for c in self.linear]

    def column(self, bits: int) -> tuple[int, int]:
        phase = self.global_phase
        phase += sum(c * ((bits >> q) & 1) for q, c in enumerate(self.linear))
        phase += 4 * sum(((bits >> q) & 1) * ((bits >> r) & 1) for q, r in self.edges)
        return bits ^ self.flips, phase % 8

    def apply(self, state: np.ndarray) -> np.ndarray:
        # Dense conversion is only an independent small-system validation aid.
        if self.width > 16 or state.shape != (1 << self.width,):
            raise ValueError("dense validation requires a matching state of at most 16 qubits")
        result = np.empty_like(state, dtype=complex)
        for bits, amplitude in enumerate(state):
            output, phase = self.column(bits)
            result[output] = np.exp(1j * np.pi * phase / 4) * amplitude
        return result


def fold_check(distance: int) -> Check:
    """Appendix A, Eqs. 183 and 190-194 and Table 4 of arXiv:2609.16929v1.

    Label fold-line data first, then reflected pairs by increasing k and j.
    These are abstract check-core labels, not a full physical layout.
    """
    parameters = {3: (3, (0, 0, 2)), 5: (5, (0, 3, 0, 1, 2)), 7: (8, (0, 3, 7, 2, 5, 0, 3))}
    if distance not in parameters:
        raise ValueError("only the published d=3, 5, 7 schedules are supported")
    ancillas, starts = parameters[distance]
    data = distance**2 + (distance - 1) ** 2
    events = []
    for j in range(2 * distance - 1):
        kind = "HXY" if j % 2 == 0 else "HXY_MINUS"
        events.append(Event(kind, (data + (starts[0] + j) % ancillas, j)))
    next_data = 2 * distance - 1
    for k in range(1, distance):
        for j in range(2 * (distance - k) - 1):
            events.append(
                Event("CZ", (data + (starts[k] + j) % ancillas, next_data, next_data + 1))
            )
            next_data += 2
    assert next_data == data
    result = Check(data, ancillas, tuple(events))
    result.validate()
    return result


def kraus_terms(check: Check, outcome: int) -> list[PhaseMonomial]:
    """Return K_outcome = (C_0 + C_1)/2, or zero for impossible cat syndromes.

    Phases live in C_0 and C_1. Never sample them as a classical mixture.
    Cat input is (|0...0> + |1...1>)/sqrt(2); output is CX-unencoded
    from cat qubit zero, Hadamard on zero, then all cat qubits measured in Z.
    """
    check.validate()
    if not 0 <= outcome < 1 << check.ancillas:
        raise ValueError("outcome does not fit the cat register")
    terms = []
    for branch in (0, 1):
        cat = [branch] * check.ancillas
        term = PhaseMonomial(check.data)
        for event in check.events:
            q = event.targets[0]
            if event.kind in {"X", "Y", "Z"}:
                if q < check.data:
                    term.append(event.kind, event.targets)
                else:
                    a = q - check.data
                    if event.kind in {"Y", "Z"}:
                        term.global_phase += 4 * cat[a] + (2 if event.kind == "Y" else 0)
                    if event.kind in {"X", "Y"}:
                        cat[a] ^= 1
            elif event.kind in {"T", "T_DAG"}:
                term.append(event.kind, event.targets)
            elif cat[q - check.data]:
                targets = event.targets[1:]
                if event.kind in {"CZ", "CX"}:
                    term.append("CZ" if event.kind == "CZ" else "X", targets)
                else:
                    positive = event.kind == "HXY"
                    term.append("X", targets)
                    term.append("S" if positive else "S_DAG", targets)
                    term.global_phase += -1 if positive else 1
        if any(c % 2 for c in term.linear):
            raise ValueError("check branch is not Clifford: unpaired non-Clifford phase")
        if any(((outcome >> a) & 1) != (cat[a] ^ cat[0]) for a in range(1, check.ancillas)):
            continue
        term.global_phase = (term.global_phase + 4 * cat[0] * (outcome & 1)) % 8
        terms.append(term)
    return terms


def elementary_check(check: Check) -> Check:
    """Expose the paper's noisy T, CX, T_DAG boundaries; CCZ stays native."""
    check.validate()
    events = []
    for event in check.events:
        if event.kind in {"HXY", "HXY_MINUS"}:
            control, target = event.targets
            before, after = ("T_DAG", "T") if event.kind == "HXY" else ("T", "T_DAG")
            events += [
                Event(before, (target,)),
                Event("CX", (control, target)),
                Event(after, (target,)),
            ]
        else:
            events.append(event)
    return Check(check.data, check.ancillas, tuple(events))


def export_control(check: Check) -> str:
    """A |+> product-input mechanism control, explicitly not cultivation."""
    check.validate()
    root = check.data
    lines = ["# Fold-check mechanism control; not a full cultivation workload."]
    lines += ["H " + " ".join(map(str, range(check.data + 1)))]
    lines += [f"CX {root} {root + a}" for a in range(1, check.ancillas)]
    for event in check.events:
        if event.kind in {"X", "Y", "Z", "T", "T_DAG"}:
            lines.append(f"{event.kind} {event.targets[0]}")
        elif event.kind in {"CZ", "CX"}:
            name = "CCZ" if event.kind == "CZ" else "CX"
            lines.append(name + " " + " ".join(map(str, event.targets)))
        else:
            control, target = event.targets
            before, after = ("T_DAG", "T") if event.kind == "HXY" else ("T", "T_DAG")
            lines += [f"{before} {target}", f"CX {control} {target}", f"{after} {target}"]
    lines += [f"CX {root} {root + a}" for a in range(1, check.ancillas)]
    lines += [f"H {root}", "M " + " ".join(map(str, range(root, root + check.ancillas)))]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for distance in (3, 5, 7):
        check = fold_check(distance)
        path = args.output / f"fold_core_d{distance}.stim"
        path.write_text(export_control(check))
        rows.append(
            {
                "distance": distance,
                "data_qubits": check.data,
                "cat_qubits": check.ancillas,
                "controlled_factors": len(check.events),
                "decomposed_t_count": sum(7 if e.kind == "CZ" else 2 for e in check.events),
                "coherent_clifford_terms_per_nonzero_outcome": len(kraus_terms(check, 0)),
                "control_path": str(path),
            }
        )
    print(
        json.dumps(
            {"scope": "isolated check mechanism controls, not cultivation", "checks": rows},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
