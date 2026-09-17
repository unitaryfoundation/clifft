"""Offline full-history MSC composition reference, never a hot executor.

Physical noise is materialized at its original source location. Coherent CH
states handle Clifford regions and the injection; certified gadget monomials
replace the paired T layers. Tableau evolution here is validation work only.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from fractions import Fraction

import stim
from clifford_branches import BranchState, multiply, root
from fold_cultivation import Operation
from msc_gadgets import regions, terms


@dataclass(frozen=True)
class Instruction:
    name: str
    targets: tuple[str, ...]
    args: tuple[float, ...]
    line: int


@dataclass(frozen=True)
class Site:
    line: int
    offset: int
    probability: float
    wires: tuple[int, ...]
    choices: tuple[str, ...]
    readout: bool = False


@dataclass(frozen=True)
class Measurement:
    line: int
    offset: int
    pauli: tuple[tuple[str, int], ...]
    reset: bool


class Program:
    def __init__(self, text):
        self.instructions: list[Instruction] = []
        self.sites: list[Site] = []
        self.measurements: list[Measurement] = []
        self.width = 0
        self.records: list[int] = []
        self.detectors: list[list[int]] = []
        self.observables: dict[int, list[int]] = {}
        self.controls: dict[tuple[int, int], int] = {}
        self.measurement_at: dict[tuple[int, int], int] = {}
        self.site_at: dict[tuple[int, int], int] = {}
        self.regions = {r.start: r for r in regions(text)}
        allowed = {
            "R",
            "RX",
            "M",
            "MX",
            "MPP",
            "H",
            "X",
            "Y",
            "Z",
            "S",
            "S_DAG",
            "T",
            "T_DAG",
            "CX",
            "CZ",
            "QUBIT_COORDS",
            "TICK",
            "SHIFT_COORDS",
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "X_ERROR",
            "Y_ERROR",
            "Z_ERROR",
        }
        for line, raw in enumerate(text.splitlines(), 1):
            raw = raw.split("#", 1)[0].strip()
            if not raw:
                continue
            match = re.fullmatch(r"([A-Z_0-9]+)(?:\(([^)]*)\))?(?:\s+(.*))?", raw)
            if match is None:
                raise ValueError("unsupported circuit syntax")
            name, arguments, targets = match.groups()
            if name not in allowed:
                raise ValueError(f"unsupported operation {name}")
            if name not in {"T", "T_DAG"}:
                stim.Circuit(raw)
            op = Instruction(
                name,
                tuple((targets or "").split()),
                tuple(float(x) for x in arguments.split(",")) if arguments else (),
                line,
            )
            if name in {"T", "T_DAG"} and (
                op.args or not op.targets or not all(t.isdecimal() for t in op.targets)
            ):
                raise ValueError("invalid T instruction")
            self.instructions.append(op)
            for word in op.targets:
                if not word.startswith("rec["):
                    for factor in word.split("*"):
                        q = int(factor[1:] if factor.startswith(("X", "Y", "Z")) else factor)
                        self.width = max(self.width, q + 1)
            if name in {"M", "MX", "MPP", "R", "RX"}:
                for offset, word in enumerate(op.targets):
                    pauli = (
                        tuple((p[0], int(p[1:])) for p in word.split("*"))
                        if name == "MPP"
                        else (("X" if name.endswith("X") else "Z", int(word)),)
                    )
                    event = Measurement(line, offset, pauli, name in {"R", "RX"})
                    self.measurement_at[line, offset] = len(self.measurements)
                    self.measurements.append(event)
                    if not event.reset:
                        self.records.append(len(self.measurements) - 1)
                        if op.args:
                            self.add_site(Site(line, offset, op.args[0], (), ("flip",), True))
            elif name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
                arity = 2 if name == "DEPOLARIZE2" else 1
                choices = (
                    tuple(a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II")
                    if arity == 2
                    else tuple("XYZ" if name == "DEPOLARIZE1" else name[0])
                )
                for offset in range(0, len(op.targets), arity):
                    self.add_site(
                        Site(
                            line,
                            offset,
                            op.args[0],
                            tuple(map(int, op.targets[offset : offset + arity])),
                            choices,
                        )
                    )
            elif name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
                deps = [self.record_reference(t) for t in op.targets]
                if name == "DETECTOR":
                    self.detectors.append(deps)
                else:
                    self.observables.setdefault(int(op.args[0]), []).extend(deps)
            elif name in {"CX", "CZ"}:
                for offset in range(0, len(op.targets), 2):
                    if op.targets[offset].startswith("rec["):
                        self.controls[line, offset] = self.record_reference(op.targets[offset])
        for region in self.regions.values():
            if any(
                op.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}
                or any(t.startswith("rec[") for t in op.targets)
                for op in self.instructions
                if region.start <= op.line <= region.end
            ):
                raise ValueError("classical dependency inside a gadget is not supported")

    def record_reference(self, target):
        match = re.fullmatch(r"rec\[(-[0-9]+)\]", target)
        if match is None or not -len(self.records) <= int(match[1]) < 0:
            raise ValueError("invalid measurement-record dependency")
        return len(self.records) + int(match[1])

    def add_site(self, site):
        if not 0 <= site.probability <= 1:
            raise ValueError("invalid noise probability")
        self.site_at[site.line, site.offset] = len(self.sites)
        self.sites.append(site)

    def history(self, seed, scale=1.0):
        rng = random.Random(seed)
        return {
            k: rng.choice(site.choices)
            for k, site in enumerate(self.sites)
            if rng.random() < min(1.0, scale * site.probability)
        }

    def validate_history(self, history):
        if any(
            not 0 <= k < len(self.sites) or choice not in self.sites[k].choices
            for k, choice in history.items()
        ):
            raise ValueError("fault outside a physical noise site")

    def faults(self, history):
        self.validate_history(history)
        result: dict[int, list[tuple[str, int]]] = {}
        for k, choice in sorted(history.items()):
            site = self.sites[k]
            if not site.readout:
                result.setdefault(site.line, []).extend(
                    (axis, q) for axis, q in zip(choice, site.wires, strict=True) if axis != "I"
                )
        return result

    def outputs(self, outcomes, history):
        records = []
        for event in self.records:
            measurement = self.measurements[event]
            site = self.site_at.get((measurement.line, measurement.offset))
            records.append(outcomes[event] ^ int(site in history))
        return {
            "records": records,
            "detectors": [sum(records[k] for k in deps) % 2 for deps in self.detectors],
            "observables": {
                str(obs): sum(records[k] for k in deps) % 2
                for obs, deps in self.observables.items()
            },
        }


def gate(state, name, targets):
    state.operation(Operation(name, tuple(targets)))


def project(state, pauli, outcome):
    preparation: list[tuple[str, tuple[int, ...]]] = []
    for axis, q in pauli:
        if axis == "Y":
            preparation.append(("S_DAG", (q,)))
        if axis in {"X", "Y"}:
            preparation.append(("H", (q,)))
    pivot = pauli[-1][1]
    preparation.extend(("CX", (q, pivot)) for _, q in pauli[:-1])
    for name, targets in preparation:
        gate(state, name, targets)
    state.terms = [t for t in state.terms if t.project(pivot, outcome)]
    for name, targets in reversed(preparation):
        gate(state, "S" if name == "S_DAG" else name, targets)
    state.merge()


def apply_instrument(state, region, outcome, faults):
    expanded = []
    half = (Fraction(1, 2), Fraction(0), Fraction(0), Fraction(0))
    for op in terms(region, outcome, faults):
        for source in state.terms:
            term = source.copy()
            term.coefficient = multiply(term.coefficient, multiply(half, root(op.global_phase)))
            for q, phase in enumerate(op.linear):
                for _ in range(phase // 2):
                    term.gate("S", (region.wires[q],))
            for q in range(op.width):
                if op.flips >> q & 1:
                    term.gate("X", (region.wires[q],))
            expanded.append(term)
    state.terms = expanded
    state.merge()


def evaluate(program, history, outcomes):
    """Evaluate all conditional Born probabilities of one specified trajectory.

    Outcomes include hidden reset measurements. They are not readout-flipped.
    Classical controls use the separately derived reported measurement record.
    """
    if len(outcomes) != len(program.measurements) or any(b not in (0, 1) for b in outcomes):
        raise ValueError("one true bit is required for every measurement and reset")
    faults = program.faults(history)
    outputs = program.outputs(outcomes, history)
    state = BranchState(program.width)
    probabilities: list[float | None] = [None] * len(outcomes)
    norm = 1.0
    skip_until = -1
    boundary_terms = []
    for op in program.instructions:
        if op.line <= skip_until:
            continue
        if op.line in program.regions:
            region = program.regions[op.line]
            measured = next(g for g in region.gates if g.name in {"MX", "MPP_Y"})
            event = program.measurement_at[measured.line, 0]
            local = []
            for line, inserted in faults.items():
                if region.start <= line <= region.end:
                    boundary = sum(g.line < line for g in region.gates)
                    for axis, q in inserted:
                        if q in region.wires:
                            local.append((boundary, axis, q))
                        else:
                            gate(state, axis, (q,))
            apply_instrument(state, region, outcomes[event], local)
            new_norm = state.expectation()
            probabilities[event] = new_norm / norm
            norm = new_norm
            reset = next((g for g in region.gates if g.name == "RX"), None)
            if reset is not None:
                hidden = program.measurement_at[reset.line, 0]
                bit = outcomes[event] ^ (
                    sum(
                        axis in {"Y", "Z"} and q == reset.targets[0]
                        for line, inserted in faults.items()
                        if measured.line < line < reset.line
                        for axis, q in inserted
                    )
                    % 2
                )
                probabilities[hidden] = float(outcomes[hidden] == bit)
                if outcomes[hidden] != bit:
                    raise ValueError("impossible reset outcome inside gadget")
            skip_until = region.end
            boundary_terms.append(len(state.terms))
        elif op.name in {"M", "MX", "MPP", "R", "RX"}:
            for offset in range(len(op.targets)):
                event = program.measurement_at[op.line, offset]
                measurement = program.measurements[event]
                project(state, measurement.pauli, outcomes[event])
                new_norm = state.expectation()
                if new_norm <= 0:
                    raise ValueError("specified trajectory has zero probability")
                probabilities[event] = new_norm / norm
                norm = new_norm
                if measurement.reset and outcomes[event]:
                    gate(state, "Z" if op.name == "RX" else "X", (measurement.pauli[0][1],))
        elif op.name in {"CX", "CZ"}:
            for offset in range(0, len(op.targets), 2):
                if (op.line, offset) in program.controls:
                    if outputs["records"][program.controls[op.line, offset]]:
                        gate(state, "X" if op.name == "CX" else "Z", (int(op.targets[offset + 1]),))
                else:
                    gate(state, op.name, tuple(map(int, op.targets[offset : offset + 2])))
        elif op.name in {"H", "S", "S_DAG", "X", "Y", "Z", "T", "T_DAG"}:
            for target in op.targets:
                if op.name == "T_DAG":
                    gate(state, "S_DAG", (int(target),))
                gate(state, "T" if op.name == "T_DAG" else op.name, (int(target),))
        else:
            for axis, q in faults.get(op.line, []):
                gate(state, axis, (q,))
        if norm <= 0:
            raise ValueError("specified trajectory has zero probability")
    if any(p is None for p in probabilities):
        raise ValueError("unaccounted measurement in composition")
    return {
        **outputs,
        "probabilities": probabilities,
        "trajectory_probability": norm,
        "peak_terms": state.peak_terms,
        "gadget_exit_terms": boundary_terms,
    }
