"""Physical f3/f5/f7 fold-transversal MSC, including published unitary growth.

The f3 generator is retained byte-for-byte as the small-circuit reference.
Growth follows Sahay's grow_3u5r at revision
9378fd228ba83c592875d155136fcbcd16a45680 and Takada Appendix A.1.
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from folded_msc import POINTS, Builder, Circuit
from folded_msc import make_circuit as make_f3


@dataclass(frozen=True)
class Surface:
    distance: int

    def __post_init__(self):
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("surface distance must be odd and at least three")

    @property
    def points(self):
        return tuple(
            (x, y)
            for y in range(2 * self.distance - 1)
            for x in range(2 * self.distance - 1)
            if (x + y) % 2 == 0
        )

    @property
    def index(self):
        return {point: i for i, point in enumerate(self.points)}

    def checks(self, axis):
        end = 2 * self.distance - 1
        if axis == "X":
            centers = [(x, y) for y in range(0, end, 2) for x in range(1, end, 2)]
        elif axis == "Z":
            centers = [(x, y) for x in range(0, end, 2) for y in range(1, end, 2)]
        else:
            raise ValueError("CSS axis must be X or Z")
        index = self.index
        return tuple(
            tuple(
                sorted(
                    index[p] for p in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)) if p in index
                )
            )
            for x, y in centers
        )

    @property
    def fold(self):
        return tuple(self.index[j, j] for j in range(2 * self.distance - 1))

    @property
    def pairs(self):
        index = self.index
        return tuple(
            tuple(
                (index[x, x + 2 * k], index[x + 2 * k, x])
                for x in range(2 * (self.distance - k) - 1)
            )
            for k in range(1, self.distance)
        )


class F5Builder(Builder):
    def __init__(self, probability):
        super().__init__(probability)
        self.surface = Surface(5)
        self.circuit = Circuit(coordinates=self.surface.points, protocol="f5")
        self.ancilla = len(self.surface.points)

    def prefix(self):
        prefix = Builder(self.probability)
        prefix.injection()
        prefix.growth()
        prefix.stabilizers("pre_checks", noisy=True)
        prefix.logical_check("logical_check_1")
        prefix.logical_check("logical_check_2")
        # The old patch occupies the checkerboard subset of the new rotated
        # patch; its measured ancillas can be reused at the new boundary.
        mapping = {q: self.surface.index[2 * x, 2 * y] for q, (x, y) in enumerate(POINTS)}
        mapping.update({q: self.ancilla + q - 13 for q in (13, 14, 15)})
        self.circuit.operations.extend(
            replace(op, qubits=tuple(mapping[q] for q in op.qubits))
            for op in prefix.circuit.operations
        )
        self.circuit.measurements.extend(prefix.circuit.measurements)

    def growth_to_rotated(self):
        self.stage = "grow_regular3_to_rotated5"
        self.source = (
            "Sahay grow_3u5r at 9378fd228ba83c592875d155136fcbcd16a45680; Takada Appendix A.1"
        )
        index = self.surface.index
        horizontal = [(x, y) for y in (0, 4, 8) for x in (2, 6)]
        vertical = [(x, y) for y in (2, 6) for x in (0, 4, 8)]
        for row in range(3):
            for x in (2, 6):
                self.reset(index[x, 4 * row])
            if row != 2:
                for x in (0, 4, 8):
                    self.reset(index[x, 4 * row + 2])
        for p in horizontal:
            self.gate("H", index[p])
        assert set(horizontal + vertical).isdisjoint({(2 * x, 2 * y) for x, y in POINTS})
        main = [(x, y) for y in (0, 4, 8) for x in (0, 4, 8)]
        alternate = [(x, y) for y in (2, 6) for x in (2, 6)]
        for x, y in main[:6]:
            self.gate("CX", index[x, y], index[x, y + 2])
        for x, y in alternate:
            self.gate("CX", index[x, y + 2], index[x, y])
        for x, y in (main[i] for i in (1, 2, 4, 5, 7, 8)):
            self.gate("CX", index[x - 2, y], index[x, y])
        for x, y in alternate:
            self.gate("CX", index[x, y], index[x + 2, y])

    def growth_to_regular(self):
        self.stage = "grow_rotated5_to_regular5"
        self.source = "Takada Appendix A.1 half-cycle; Fig. 7 construction at d5"
        index = self.surface.index
        centers = [(x, y) for y in range(1, 8, 2) for x in range(1, 8, 2)]
        for point in centers:
            self.reset(index[point])
        for x, y in centers:
            if (x + y) % 4 == 2:
                self.gate("H", index[x, y])
        for step in range(2):
            for x, y in centers:
                if (x + y) % 4 == 2:
                    self.gate("CX", index[x, y], index[x - 1 + 2 * step, y - 1])
                else:
                    self.gate("CX", index[x - 1, y - 1 + 2 * step], index[x, y])

    def stabilizers(self, stage, *, noisy):
        self.stage = stage
        self.source = "Takada surface-code geometry; Appendix A.2 sequential extraction"
        for axis in ("X", "Z"):
            for i, support in enumerate(self.surface.checks(axis)):
                self.reset(self.ancilla, noisy)
                if axis == "X":
                    self.gate("H", self.ancilla, noisy=noisy)
                for q in support:
                    pair = (self.ancilla, q) if axis == "X" else (q, self.ancilla)
                    self.gate("CX", *pair, noisy=noisy)
                if axis == "X":
                    self.gate("H", self.ancilla, noisy=noisy)
                self.measure(self.ancilla, f"{axis}{i + 1}", noisy=noisy)

    def controlled_factors(self, cat, *, noisy):
        for j, q in enumerate(self.surface.fold):
            first, last = ("T_DAG", "T") if j % 2 == 0 else ("T", "T_DAG")
            self.gate(first, q, noisy=noisy)
            self.gate("CX", cat[j % len(cat)], q, noisy=noisy)
            self.gate(last, q, noisy=noisy)
        for offset, pairs in zip((3, 0, 1, 2), self.surface.pairs, strict=True):
            for j, (a, b) in enumerate(pairs):
                self.gate("CCZ", cat[(offset + j) % len(cat)], a, b, noisy=noisy)

    def cat_preparation(self, *, noisy):
        a = self.ancilla
        for q in range(a, a + 6):
            self.reset(q, noisy)
        self.gate("H", a + 2, noisy=noisy)
        for control, target in ((2, 3), (2, 1), (3, 4), (1, 0), (4, 5), (0, 5)):
            self.gate("CX", a + control, a + target, noisy=noisy)
        self.measure(a + 5, "cat_flag", noisy=noisy)

    def logical_check(self, stage, *, noisy=True, terminal=False):
        self.stage, self.source = stage, "Takada Appendix A.1; Table 4; Fig. 19b"
        cat: tuple[int, ...]
        if terminal:
            cat = (self.ancilla,)
            self.reset(cat[0], noisy)
            self.gate("H", cat[0], noisy=noisy)
        else:
            cat = tuple(range(self.ancilla, self.ancilla + 5))
            self.cat_preparation(noisy=noisy)
        self.controlled_factors(cat, noisy=noisy)
        if not terminal:
            for control, target in ((1, 0), (2, 1), (3, 4), (2, 3)):
                self.gate("CX", cat[control], cat[target], noisy=noisy)
        root = cat[len(cat) // 2]
        self.gate("H", root, noisy=noisy)
        for q in cat:
            self.measure(
                q,
                "logical" if q == root else f"cat_{q}",
                noisy=noisy,
                postselect=not terminal,
                logical=terminal,
            )


def make_circuit(probability=0.001, *, distance=5, evaluation="fed"):
    if distance == 7:
        from folded_msc_f7 import make_f7

        return make_f7(probability, evaluation=evaluation)
    if distance == 3:
        return make_f3(probability, evaluation=evaluation)
    if distance != 5 or evaluation not in {"fed", "none"}:
        raise ValueError("implemented distances are 3, 5 and 7, with fed or none evaluation")
    builder = F5Builder(probability)
    builder.prefix()
    builder.growth_to_rotated()
    builder.growth_to_regular()
    builder.stabilizers("d5_pre_checks", noisy=True)
    builder.logical_check("d5_logical_check_1")
    builder.logical_check("d5_logical_check_2")
    builder.stabilizers("d5_post_checks", noisy=True)
    if evaluation == "fed":
        builder.stabilizers("final_syndrome", noisy=False)
        builder.logical_check("final_logical", noisy=False, terminal=True)
    return builder.circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--distance", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--evaluation", choices=("fed", "none"), default="fed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    circuit = make_circuit(args.probability, distance=args.distance, evaluation=args.evaluation)
    args.output.write_text(circuit.text())
    args.output.with_suffix(".json").write_text(json.dumps(circuit.manifest(), indent=1) + "\n")


if __name__ == "__main__":
    main()
