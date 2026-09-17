"""Paper-guided f3/f5 reconstruction, not the authors' Reg5 benchmark artifact.

Sources: arXiv:2609.16929v1, Appendix A and Figs. 7, 18, 19;
Sahay's grow_3u5r at MSC_foldedH revision 9378fd228ba83c592875d155136fcbcd16a45680.
Syndrome CNOT ordering and terminal probes are explicit reconstruction choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import stim
from fold_check import elementary_check, fold_check


@dataclass(frozen=True)
class Operation:
    name: str
    targets: tuple[int, ...]


@dataclass
class Stage:
    name: str
    operations: list[Operation] = field(default_factory=list)
    noisy: bool = True

    def add(self, name: str, *targets: int) -> None:
        self.operations.append(Operation(name, tuple(targets)))


def coordinates(distance: int) -> list[tuple[int, int]]:
    return [
        (x, y) for y in range(2 * distance - 1) for x in range(2 * distance - 1) if (x + y) % 2 == 0
    ]


def stabilizers(distance: int) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    data = set(coordinates(distance))
    result = []
    for axis in "XZ":
        for y in range(2 * distance - 1):
            for x in range(2 * distance - 1):
                if (x + y) % 2 != 1 or ("X" if x % 2 else "Z") != axis:
                    continue
                # The paper specifies sequential X then Z extraction. The
                # within-stabilizer neighbor order is a reconstruction choice.
                support = tuple(
                    p for p in [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)] if p in data
                )
                result.append((axis, support))
    return result


def pauli(distance: int, axis: str, support: tuple[tuple[int, int], ...]) -> stim.PauliString:
    index = {p: i for i, p in enumerate(coordinates(distance))}
    result = stim.PauliString(len(index))
    for p in support:
        result[index[p]] = axis
    return result


def logical(distance: int, axis: str) -> stim.PauliString:
    x = pauli(distance, "X", tuple((0, y) for y in range(0, 2 * distance - 1, 2)))
    z = pauli(distance, "Z", tuple((x, 0) for x in range(0, 2 * distance - 1, 2)))
    return {"X": x, "Y": 1j * x * z, "Z": z}[axis]


def cat_circuits(distance: int) -> tuple[Stage, Stage]:
    """Local ancilla labels, transcribed from Fig. 19 for f3 and f5."""
    if distance == 3:
        size, root = 3, 1
        preparation = [(1, 2), (1, 0)]
        decode = [(1, 0), (1, 2)]
    elif distance == 5:
        size, root = 6, 2
        preparation = [(2, 3), (2, 1), (3, 4), (1, 0), (4, 5), (0, 5)]
        decode = [(1, 0), (3, 4), (2, 1), (2, 3)]
    else:
        raise ValueError("only f3 and f5 complete cat schedules are transcribed")
    prep = Stage(f"cat_prepare_d{distance}")
    for q in range(size):
        prep.add("R", q)
    prep.add("H", root)
    for a, b in preparation:
        prep.add("CX", a, b)
    if distance == 5:
        prep.add("M", 5)
    finish = Stage(f"cat_decode_d{distance}")
    for a, b in decode:
        finish.add("CX", a, b)
    finish.add("H", root)
    for q in range(3 if distance == 3 else 5):
        finish.add("M", q)
    return prep, finish


class Reconstruction:
    def __init__(self, distance: int):
        if distance not in (3, 5):
            raise ValueError("full reconstruction currently supports f3 and f5 only")
        self.distance = distance
        self.index = {p: i for i, p in enumerate(coordinates(distance))}
        self.ancilla = len(self.index)
        self.stages: list[Stage] = []

    def qubit(self, xy: tuple[int, int], distance: int) -> int:
        scale = 2 if self.distance == 5 and distance == 3 else 1
        return self.index[scale * xy[0], scale * xy[1]]

    def injection(self) -> None:
        stage = Stage("inject_rotated_d3")
        main = [self.qubit((2 * x, 2 * y), 3) for y in range(3) for x in range(3)]
        for q in main:
            stage.add("R", q)
        for q in (0, 2, 4, 5, 6):
            stage.add("H", main[q])
        stage.add("T", main[4])
        for layer in [
            [(0, 3), (5, 8), (6, 7)],
            [(4, 1)],
            [(1, 7), (5, 4)],
            [(2, 1), (0, 4), (5, 7)],
            [(0, 1)],
        ]:
            for a, b in layer:
                stage.add("CX", main[a], main[b])
        self.stages.append(stage)

    def morph(self, distance: int) -> None:
        stage = Stage(f"morph_rotated_to_regular_d{distance}")
        centers = [
            (x, y) for y in range(1, 2 * distance - 1, 2) for x in range(1, 2 * distance - 1, 2)
        ]
        for x, y in centers:
            stage.add("R", self.qubit((x, y), distance))
        for x, y in centers:
            if ((x // 2) + (y // 2)) % 2 == 0:
                stage.add("H", self.qubit((x, y), distance))
        for layer in range(2):
            for x, y in centers:
                plus = ((x // 2) + (y // 2)) % 2 == 0
                neighbor = (x - 1 + 2 * layer, y - 1) if plus else (x - 1, y - 1 + 2 * layer)
                a, b = self.qubit((x, y), distance), self.qubit(neighbor, distance)
                stage.add("CX", *((a, b) if plus else (b, a)))
        self.stages.append(stage)

    def grow(self) -> None:
        if self.distance != 5:
            raise ValueError("only the regular d3 to rotated d5 growth is implemented")
        stage = Stage("grow_regular_d3_to_rotated_d5")

        def q(x: int, y: int) -> int:
            return self.index[2 * x, 2 * y]

        for y in range(5):
            for x in range(5):
                if (x + y) % 2:
                    stage.add("R", q(x, y))
                    if x % 2:
                        stage.add("H", q(x, y))
        for y in (0, 2):
            for x in (0, 2, 4):
                stage.add("CX", q(x, y), q(x, y + 1))
        for y in (1, 3):
            for x in (1, 3):
                stage.add("CX", q(x, y + 1), q(x, y))
        for y in (0, 2, 4):
            for x in (2, 4):
                stage.add("CX", q(x - 1, y), q(x, y))
        for y in (1, 3):
            for x in (1, 3):
                stage.add("CX", q(x, y), q(x + 1, y))
        self.stages.append(stage)

    def syndrome(self, distance: int, *, noisy: bool = True) -> None:
        stage = Stage(f"syndrome_d{distance}" + ("" if noisy else "_ideal"), noisy=noisy)
        a = self.ancilla
        for axis, support in stabilizers(distance):
            stage.add("R", a)
            if axis == "X":
                stage.add("H", a)
            for xy in support:
                q = self.qubit(xy, distance)
                stage.add("CX", *((a, q) if axis == "X" else (q, a)))
            if axis == "X":
                stage.add("H", a)
            stage.add("M", a)
        self.stages.append(stage)

    def check(self, distance: int) -> None:
        prep, finish = cat_circuits(distance)
        for stage in (prep, finish):
            stage.operations = [
                Operation(op.name, tuple(self.ancilla + q for q in op.targets))
                for op in stage.operations
            ]
        self.stages.append(prep)
        mapping = self.check_mapping(distance)
        core = elementary_check(fold_check(distance))
        mapping += [self.ancilla + q for q in range(core.ancillas)]
        stage = Stage(f"fold_core_d{distance}")
        for event in core.events:
            stage.add(
                "CCZ" if event.kind == "CZ" else event.kind, *(mapping[q] for q in event.targets)
            )
        self.stages.append(stage)
        self.stages.append(finish)

    def check_mapping(self, distance: int) -> list[int]:
        local = [(j, j) for j in range(2 * distance - 1)]
        for k in range(1, distance):
            for j in range(2 * (distance - k) - 1):
                local += [(j, j + 2 * k), (j + 2 * k, j)]
        # Parallel-to-fold diagonals have offset 2k; their j increments are
        # one lattice step, so data remain on the even-parity checkerboard.
        return [self.qubit(xy, distance) for xy in local]

    def build(self) -> Reconstruction:
        self.injection()
        self.morph(3)
        self.syndrome(3)
        self.check(3)
        self.check(3)
        if self.distance == 5:
            self.grow()
            self.morph(5)
            self.syndrome(5)
            self.check(5)
            self.check(5)
        self.syndrome(self.distance)
        self.syndrome(self.distance, noisy=False)
        return self

    def text(self, probability: float) -> str:
        if not 0 <= probability <= 1:
            raise ValueError("noise probability must lie in [0,1]")
        lines = ["# Paper-guided reconstruction; not the authors' Reg5 benchmark artifact."]
        for stage in self.stages:
            lines.append("# " + stage.name)
            p = probability if stage.noisy else 0
            for op in stage.operations:
                targets = " ".join(map(str, op.targets))
                if op.name == "M":
                    if p:
                        lines.append(f"X_ERROR({p}) {targets}")
                    lines += [f"M {targets}", "DETECTOR rec[-1]"]
                else:
                    lines.append(f"{op.name} {targets}")
                    if p:
                        noise = "X_ERROR" if op.name == "R" else f"DEPOLARIZE{len(op.targets)}"
                        lines.append(f"{noise}({p}) {targets}")
        for axis in "XYZ":
            observable = logical(self.distance, axis)
            if observable.sign != 1:
                raise ValueError("terminal probe requires positive Pauli convention")
            targets = "*".join(
                f"{'_XYZ'[observable[q]]}{q}" for q in range(len(observable)) if observable[q]
            )
            lines.append("EXP_VAL " + targets)
        return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability", type=float, default=0.001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for distance in (3, 5):
        reconstruction = Reconstruction(distance).build()
        for p in (0.0, args.probability):
            text = reconstruction.text(p)
            name = f"reconstructed_f{distance}_p{p:g}.stim"
            (args.output / name).write_text(text)
            rows.append(
                {
                    "distance": distance,
                    "probability": p,
                    "file": name,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "stages": [s.name for s in reconstruction.stages],
                }
            )
    manifest = {"scope": __doc__, "circuits": rows}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
