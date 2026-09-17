"""Offline Pauli-proxy experiment for the structured fold reconstruction.

This implements the diagonal circuit-level reduction of arXiv:2609.16929v1.
It constructs a separate Clifford program for a fixed physical fault history
and uses external Stim as a reference backend. It is not a Clifft executor or
an eligibility test for arbitrary non-Clifford circuits.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import stim
from fold_check import elementary_check, fold_check
from fold_cultivation import Operation, Reconstruction, logical


@dataclass
class Compiler:
    """Move continuing X errors forward; retain their diagonal corrections."""

    flips: int = 0
    operations: list[Operation] = field(default_factory=list)

    def emit(self, name: str, *targets: int) -> None:
        self.operations.append(Operation(name, tuple(targets)))

    def append(self, op: Operation) -> None:
        name, targets = op.name, op.targets
        q = targets[0]
        x = (self.flips >> q) & 1
        if name in {"X", "Y", "Z"}:
            if name in {"Y", "Z"}:
                self.emit("Z", q)
            if name in {"X", "Y"}:
                self.flips ^= 1 << q
        elif name == "R":
            self.flips &= ~(1 << q)
            self.emit(name, q)
        elif name == "H":
            self.emit(name, q)
            if x:
                self.emit("Z", q)
                self.flips ^= 1 << q
        elif name == "CX":
            if x:
                self.flips ^= 1 << targets[1]
            self.emit(name, *targets)
        elif name in {"T", "T_DAG"}:
            # T X = exp(i pi/4) X S_DAG T. The dropped phase is
            # common to the entire fixed history, including both cat strings.
            if x:
                self.emit("S_DAG" if name == "T" else "S", q)
        elif name in {"S", "S_DAG"}:
            self.emit(name, q)
            if x:
                self.emit("Z", q)
        elif name == "CZ":
            self.emit(name, *targets)
            if x:
                self.emit("Z", targets[1])
            if (self.flips >> targets[1]) & 1:
                self.emit("Z", q)
        elif name == "CCZ":
            # The Boolean derivative of a cubic phase is quadratic. Keeping
            # both its pair and single terms handles simultaneous X faults.
            for i in range(3):
                a, b = [targets[j] for j in range(3) if j != i]
                if (self.flips >> targets[i]) & 1:
                    self.emit("CZ", a, b)
                if ((self.flips >> a) & 1) and ((self.flips >> b) & 1):
                    self.emit("Z", targets[i])
        elif name == "M":
            self.emit("POST", q, x)
        else:
            raise ValueError(f"unsupported proxy operation: {op}")


def execute(
    operations: list[Operation], simulator: stim.TableauSimulator | None = None
) -> tuple[float, stim.TableauSimulator]:
    """Evaluate postselection with external Stim, without sampling rare records."""
    if simulator is None:
        simulator = stim.TableauSimulator(seed=0)
    probability = 1.0
    for op in operations:
        if op.name == "POST":
            q, desired = op.targets
            expectation = simulator.peek_z(q)
            probability *= (1 + (-1 if desired else 1) * expectation) / 2
            if not probability:
                return 0.0, simulator
            if not expectation:
                simulator.postselect_z(q, desired_value=bool(desired))
        elif op.name == "EQUAL":
            a, b, desired = op.targets
            observable = stim.PauliString(max(a, b) + 1)
            observable[a] = observable[b] = "Z"
            expectation = simulator.peek_observable_expectation(observable)
            probability *= (1 + (-1 if desired else 1) * expectation) / 2
            if not probability:
                return 0.0, simulator
            if not expectation:
                simulator.postselect_observable(observable, desired_value=bool(desired))
        else:
            simulator.do(stim.CircuitInstruction(op.name, op.targets))
    return probability, simulator


def probe_operations(reconstruction: Reconstruction, *, minus: bool) -> list[Operation]:
    """Fictitious ideal measurements of Hplus and Hminus, after the ideal syndrome."""
    ancilla = reconstruction.ancilla
    result = [Operation("R", (ancilla,)), Operation("H", (ancilla,))]
    mapping = reconstruction.check_mapping(reconstruction.distance)
    check = elementary_check(fold_check(reconstruction.distance))
    for event in check.events:
        targets = tuple(mapping[q] if q < check.data else ancilla for q in event.targets)
        result.append(Operation("CCZ" if event.kind == "CZ" else event.kind, targets))
    if minus:
        # Hminus = i Z_L Hplus on the code. The controlled i is S on
        # the ancilla, and must not be discarded as a global data phase.
        z = logical(reconstruction.distance, "Z")
        result += [Operation("CZ", (ancilla, q)) for q in range(len(z)) if z[q]]
        result.append(Operation("S", (ancilla,)))
    result.append(Operation("H", (ancilla,)))
    return result


class Protocol:
    def __init__(self, distance: int):
        self.reconstruction = Reconstruction(distance).build()
        self.probes = [probe_operations(self.reconstruction, minus=m) for m in (False, True)]
        self.z = logical(distance, "Z")

    def compile(self, stages: list[list[Operation]]) -> Compiler:
        compiler = Compiler()
        for original, operations in zip(self.reconstruction.stages, stages, strict=True):
            last = None
            for op in operations:
                if original.equal_records and op.name == "M":
                    q = op.targets[0]
                    if last is not None:
                        desired = ((compiler.flips >> last) ^ (compiler.flips >> q)) & 1
                        compiler.emit("EQUAL", last, q, desired)
                    last = q
                else:
                    compiler.append(op)
        return compiler

    def evaluate(self, compiled: Compiler) -> tuple[float, list[float] | None]:
        probability, simulator = execute(compiled.operations)
        if not probability:
            return probability, None
        z_sign = sum((compiled.flips >> q) & 1 for q in range(len(self.z)) if self.z[q]) % 2
        z = simulator.peek_observable_expectation(self.z) * (-1 if z_sign else 1)
        probes = []
        for operations in self.probes:
            compiler = Compiler(compiled.flips)
            for op in operations:
                compiler.append(op)
            _, state = execute(compiler.operations, simulator.copy())
            ancilla = self.reconstruction.ancilla
            probes.append(state.peek_z(ancilla) * (-1 if (compiler.flips >> ancilla) & 1 else 1))
        plus, minus = probes
        return probability, [(plus + minus) / math.sqrt(2), (plus - minus) / math.sqrt(2), z]


def logical_axis_tail(reconstruction: Reconstruction, axis: str) -> list[list[Operation]]:
    stages = [list(s.operations) for s in reconstruction.stages]
    index = max(
        j
        for j, s in enumerate(reconstruction.stages)
        if s.name == f"syndrome_d{reconstruction.distance}"
    )
    operations = stages[index]
    observable = logical(reconstruction.distance, axis)
    insertions: dict[int, list[Operation]] = {}
    for q in range(len(observable)):
        if observable[q]:
            site = max(i for i, op in enumerate(operations) if q in op.targets)
            if operations[site].name != "CX":
                raise ValueError("logical tail is not at a physical CNOT noise site")
            insertions.setdefault(site, []).append(Operation("_XYZ"[observable[q]], (q,)))
    stages[index] = [
        part for i, op in enumerate(operations) for part in [op, *insertions.get(i, [])]
    ]
    return stages


def validation_cases(distance: int) -> list[tuple[str, list[list[Operation]]]]:
    from clifford_branches import materialize, paired_hook_histories
    from fold_blocks import Protocol as Blocks
    from fold_blocks import f7_diagnostic_histories, syndrome_sector_histories

    r = Reconstruction(distance).build()
    cases = [("ideal", [list(s.operations) for s in r.stages])]
    cases += [(f"natural_{s}", materialize(r, 0.001, s)[0]) for s in range(1000, 1064)]
    cases += [(f"stress_{s}", materialize(r, 0.01, s)[0]) for s in range(2000, 2032)]
    cases += paired_hook_histories(r)
    cases += [(f"logical_tail_{a}", logical_axis_tail(r, a)) for a in "XYZ"]
    if distance > 3:
        cases += syndrome_sector_histories(Blocks(distance))
    if distance == 7:
        cases += [(k, v) for k, v in f7_diagnostic_histories(Blocks(distance)) if k != "ideal"]
    return cases


def main() -> None:
    from fold_blocks import Protocol as Blocks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proxy = Protocol(args.distance)
    blocks = Blocks(args.distance)
    rows = []
    for label, stages in validation_cases(args.distance):
        start = time.perf_counter()
        compiled = proxy.compile(stages)
        compiled_at = time.perf_counter()
        probability, xyz = proxy.evaluate(compiled)
        finished = time.perf_counter()
        expected = blocks.evaluate(blocks.encode(stages))
        error = abs(probability - expected.acceptance)
        if probability and expected.logical_xyz is not None:
            assert xyz is not None
            error = max(
                error, *(abs(a - b) for a, b in zip(xyz, expected.logical_xyz, strict=True))
            )
        if error > 2e-12:
            raise ValueError(f"proxy mismatch at {label}: {error}")
        rows.append(
            {
                "case": label,
                "acceptance": probability,
                "logical_xyz": xyz,
                "maximum_error": error,
                "clifford_operations": len(compiled.operations),
                "python_compile_us": 1e6 * (compiled_at - start),
                "python_evaluate_us": 1e6 * (finished - compiled_at),
            }
        )
    args.output.write_text(
        json.dumps({"distance": args.distance, "histories": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
