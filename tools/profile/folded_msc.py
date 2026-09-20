"""Physical f3 fold-transversal cultivation reconstructed from arXiv:2609.16929v1.

This research generator uses the published injection, growth and cat check,
with an explicit sequential stabilizer schedule. It is not an author export.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

POINTS = tuple((x, y) for y in range(5) for x in range(5) if (x + y) % 2 == 0)
X_CHECKS = ((0, 1, 3), (1, 2, 4), (3, 5, 6, 8), (4, 6, 7, 9), (8, 10, 11), (9, 11, 12))
Z_CHECKS = ((0, 3, 5), (5, 8, 10), (1, 3, 4, 6), (6, 8, 9, 11), (2, 4, 7), (7, 9, 12))
FOLD = (0, 3, 6, 9, 12)
FOLD_PAIRS = (((5, 1), (8, 4), (11, 7)), ((10, 2),))
ROTATED = (0, 1, 2, 5, 6, 7, 10, 11, 12)
INJECTION_CX = ((0, 3), (5, 8), (6, 7), (4, 1), (5, 4), (1, 7), (2, 1), (5, 7), (0, 4), (0, 1))
GROWTH_CX = ((3, 0), (1, 4), (5, 8), (9, 6), (3, 1), (6, 4), (10, 8), (9, 7))


@dataclass(frozen=True)
class Operation:
    name: str
    qubits: tuple[int, ...]
    stage: str
    source: str
    probability: float | None = None

    def text(self):
        argument = "" if self.probability is None else f"({self.probability:.17g})"
        return f"{self.name}{argument} " + " ".join(map(str, self.qubits))


@dataclass
class Circuit:
    operations: list[Operation] = field(default_factory=list)
    measurements: list[dict] = field(default_factory=list)
    coordinates: tuple[tuple[int, int], ...] = POINTS
    protocol: str = "f3"

    def text(self):
        lines = [
            f"# Reconstructed {self.protocol} fold-transversal MSC; "
            "see folded_msc.py and its report."
        ]
        lines += [f"QUBIT_COORDS({x},{y}) {q}" for q, (x, y) in enumerate(self.coordinates)]
        record = 0
        for op in self.operations:
            lines.append(op.text())
            if op.name == "M":
                info = self.measurements[record]
                if info["postselect"]:
                    partner = info.get("parity_with")
                    extra = "" if partner is None else f" rec[{partner - record - 1}]"
                    lines.append("DETECTOR rec[-1]" + extra)
                if info["logical"]:
                    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
                record += 1
        return "\n".join(lines) + "\n"

    def manifest(self):
        return {
            "source": "https://arxiv.org/html/2609.16929v1#A1",
            "status": "independent reconstruction; sequential check ordering is explicit",
            "evaluation": "FED"
            if any(m["stage"] == "final_syndrome" for m in self.measurements)
            else "none",
            "num_qubits": 1 + max(q for op in self.operations for q in op.qubits),
            "num_measurements": len(self.measurements),
            "num_detectors": sum(m["postselect"] for m in self.measurements),
            "sha256": hashlib.sha256(self.text().encode()).hexdigest(),
            "gate_counts": dict(Counter(op.name for op in self.operations)),
            "operations": [dict(index=i, **asdict(op)) for i, op in enumerate(self.operations)],
            "measurements": self.measurements,
        }


class Builder:
    def __init__(self, probability):
        if not 0 <= probability <= 1:
            raise ValueError("noise probability must be between zero and one")
        self.probability = probability
        self.circuit = Circuit()
        self.stage = ""
        self.source = ""

    def append(self, name, *qubits, probability=None):
        self.circuit.operations.append(
            Operation(name, qubits, self.stage, self.source, probability)
        )

    def gate(self, name, *qubits, noisy=True):
        self.append(name, *qubits)
        if noisy:
            self.append(f"DEPOLARIZE{len(qubits)}", *qubits, probability=self.probability)

    def reset(self, q, noisy=True):
        self.append("R", q)
        if noisy:
            self.append("X_ERROR", q, probability=self.probability)

    def measure(self, q, label, *, noisy=True, postselect=True, logical=False):
        if noisy:
            self.append("X_ERROR", q, probability=self.probability)
        self.append("M", q)
        self.circuit.measurements.append(
            dict(
                index=len(self.circuit.measurements),
                stage=self.stage,
                label=label,
                postselect=postselect,
                logical=logical,
            )
        )

    def injection(self):
        self.stage, self.source = "injection", "Fig. 18"
        for q in ROTATED:
            self.reset(q)
        for q in (0, 2, 4, 5, 6):
            self.gate("H", ROTATED[q])
        # Disjoint first-column CNOTs commute with the injection T.
        self.gate("T", ROTATED[4])
        for a, b in INJECTION_CX:
            self.gate("CX", ROTATED[a], ROTATED[b])

    def growth(self):
        self.stage, self.source = "growth", "Fig. 7 half-cycle; Appendix A.1"
        for q in (3, 4, 8, 9):
            self.reset(q)
        for q in (3, 9):
            self.gate("H", q)
        for a, b in GROWTH_CX:
            self.gate("CX", a, b)

    def stabilizers(self, stage, *, noisy):
        self.stage = stage
        self.source = "Sec. III.A checks; Appendix A.2 sequential extraction"
        # Fix all ordering choices so hook-error locations are reproducible.
        # X then Z, in published generator order, ascending data index per check.
        for axis, checks in (("X", X_CHECKS), ("Z", Z_CHECKS)):
            for i, support in enumerate(checks):
                self.reset(13, noisy)
                if axis == "X":
                    self.gate("H", 13, noisy=noisy)
                for q in support:
                    self.gate("CX", *((13, q) if axis == "X" else (q, 13)), noisy=noisy)
                if axis == "X":
                    self.gate("H", 13, noisy=noisy)
                self.measure(13, f"{axis}{i + 1}", noisy=noisy)

    def controlled_factors(self, cat, *, noisy):
        for j, q in enumerate(FOLD):
            first, last = ("T_DAG", "T") if j % 2 == 0 else ("T", "T_DAG")
            self.gate(first, q, noisy=noisy)
            self.gate("CX", cat[j % len(cat)], q, noisy=noisy)
            self.gate(last, q, noisy=noisy)
        for offset, pairs in zip((0, 2), FOLD_PAIRS, strict=True):
            for j, (a, b) in enumerate(pairs):
                self.gate("CCZ", cat[(offset + j) % len(cat)], a, b, noisy=noisy)

    def logical_check(self, stage, *, noisy=True, terminal=False):
        self.stage, self.source = stage, "Appendix A.1 Hermitian factors; Table 4; Fig. 19a"
        cat = (13,) if terminal else (13, 14, 15)
        for q in cat:
            self.reset(q, noisy)
        root = cat[len(cat) // 2]
        self.gate("H", root, noisy=noisy)
        if not terminal:
            self.gate("CX", 14, 15, noisy=noisy)
            self.gate("CX", 14, 13, noisy=noisy)
        self.controlled_factors(cat, noisy=noisy)
        if not terminal:
            self.gate("CX", 14, 13, noisy=noisy)
            self.gate("CX", 14, 15, noisy=noisy)
        self.gate("H", root, noisy=noisy)
        for q in cat:
            self.measure(
                q,
                "logical" if q == root else f"cat_{q}",
                noisy=noisy,
                postselect=not terminal,
                logical=terminal,
            )


def make_circuit(probability=0.001, *, evaluation="fed"):
    if evaluation not in {"fed", "none"}:
        raise ValueError("supported evaluations are fed and none")
    builder = Builder(probability)
    builder.injection()
    builder.growth()
    builder.stabilizers("pre_checks", noisy=True)
    builder.logical_check("logical_check_1")
    builder.logical_check("logical_check_2")
    builder.stabilizers("post_checks", noisy=True)
    if evaluation == "fed":
        builder.stabilizers("final_syndrome", noisy=False)
        builder.logical_check("final_logical", noisy=False, terminal=True)
    return builder.circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--evaluation", choices=("fed", "none"), default="fed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    circuit = make_circuit(args.probability, evaluation=args.evaluation)
    args.output.write_text(circuit.text())
    args.output.with_suffix(".json").write_text(json.dumps(circuit.manifest(), indent=2) + "\n")


if __name__ == "__main__":
    main()
