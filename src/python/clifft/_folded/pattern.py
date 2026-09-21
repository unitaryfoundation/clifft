"""Concrete folded MSC gate patterns; these require no source annotations."""

from dataclasses import dataclass

from .contraction import Surface


@dataclass(frozen=True)
class Operation:
    name: str
    qubits: tuple[int, ...]
    stage: str
    probability: float | None = None

    def text(self):
        arg = "" if self.probability is None else f"({self.probability:.17g})"
        return f"{self.name}{arg} " + " ".join(map(str, self.qubits))


def terminal_pattern(distance):
    surface = Surface(distance)
    n = len(surface.points)
    operations = []
    stage = ""

    def add(name, *qubits, probability=None):
        operations.append(Operation(name, qubits, stage, probability))

    def gate(name, *qubits, noisy):
        add(name, *qubits)
        if noisy:
            add(f"DEPOLARIZE{len(qubits)}", *qubits, probability=0.0)

    def reset(q, noisy):
        add("R", q)
        if noisy:
            add("X_ERROR", q, probability=0.0)

    def measure(q, noisy):
        if noisy:
            add("X_ERROR", q, probability=0.0)
        add("M", q)

    def check(terminal=False):
        cat: tuple[int, ...]
        noisy = not terminal
        if terminal:
            cat, root = (n,), n
            reset(n, False)
            gate("H", n, noisy=False)
        elif distance == 3:
            cat, root = tuple(range(n, n + 3)), n + 1
            for q in cat:
                reset(q, True)
            gate("H", root, noisy=True)
            for q in (n + 2, n):
                gate("CX", root, q, noisy=True)
        elif distance == 5:
            cat, root = tuple(range(n, n + 5)), n + 2
            for q in range(n, n + 6):
                reset(q, True)
            gate("H", root, noisy=True)
            for a, b in ((2, 3), (2, 1), (3, 4), (1, 0), (4, 5), (0, 5)):
                gate("CX", n + a, n + b, noisy=True)
            measure(n + 5, True)
        else:
            cat, root = tuple(range(n, n + 8)), n + 4
            for q in range(n, n + 14):
                reset(q, True)
            for q in (n, n + 8):
                gate("H", q, noisy=True)
            for a, b in (
                (0, 4),
                (0, 2),
                (4, 6),
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 12),
                (8, 10),
                (12, 13),
                (8, 9),
                (10, 11),
                (1, 10),
                (2, 13),
                (3, 8),
                (5, 12),
                (6, 11),
                (7, 9),
            ):
                gate("CX", n + a, n + b, noisy=True)
            for q in range(n + 8, n + 14):
                measure(q, True)
        for j, q in enumerate(surface.fold):
            first, last = ("T_DAG", "T") if j % 2 == 0 else ("T", "T_DAG")
            gate(first, q, noisy=noisy)
            gate("CX", cat[j % len(cat)], q, noisy=noisy)
            gate(last, q, noisy=noisy)
        offsets = {3: (0, 2), 5: (3, 0, 1, 2), 7: (3, 7, 2, 5, 0, 3)}[distance]
        for offset, pairs in zip(offsets, surface.pairs, strict=True):
            for j, (a, b) in enumerate(pairs):
                gate("CCZ", cat[(offset + j) % len(cat)], a, b, noisy=noisy)
        if not terminal:
            uncompute = {
                3: ((1, 0), (1, 2)),
                5: ((1, 0), (2, 1), (3, 4), (2, 3)),
                7: ((1, 0), (2, 1), (3, 2), (6, 7), (4, 3), (5, 6), (4, 5)),
            }[distance]
            for a, b in uncompute:
                gate("CX", n + a, n + b, noisy=True)
        gate("H", root, noisy=noisy)
        for q in cat:
            measure(q, noisy)

    for stage in ("check_1", "check_2"):
        check()
    for stage in ("post", "final_syndrome"):
        noisy = stage == "post"
        for axis in "XZ":
            for support in surface.checks(axis):
                reset(n, noisy)
                if axis == "X":
                    gate("H", n, noisy=noisy)
                for q in support:
                    gate("CX", *((n, q) if axis == "X" else (q, n)), noisy=noisy)
                if axis == "X":
                    gate("H", n, noisy=noisy)
                measure(n, noisy)
    stage = "final"
    check(terminal=True)
    return operations
