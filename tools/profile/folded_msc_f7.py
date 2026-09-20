"""Reconstruct the f7 cultivation sequence in Takada Appendix A, through FED.

Growth is the Hadamard-dual form of Higgott et al. Figure 2, matching this
generator's X/Z convention. Cat preparation follows Takada Figure 19c.
"""

from dataclasses import replace

from folded_msc import Builder, Circuit
from folded_msc_family import F5Builder, Surface

CAT_DATA_CX = ((0, 4), (0, 2), (4, 6), (0, 1), (2, 3), (4, 5), (6, 7))
CAT_VERIFY_CX = ((8, 12), (8, 10), (12, 13), (8, 9), (10, 11))
CAT_TRANSVERSAL_CX = ((1, 10), (2, 13), (3, 8), (5, 12), (6, 11), (7, 9))
CAT_UNCOMPUTE_CX = ((1, 0), (2, 1), (3, 2), (6, 7), (4, 3), (5, 6), (4, 5))


def growth_layers(distance):
    """Return new-qubit H sites and four local CNOT layers for d to d+2."""
    surface = Surface(distance)
    e = 2 * distance - 2
    interior = {(x + 2, y + 2) for x, y in Surface(distance - 2).points}
    paper_h = {(x, y) for x in range(2, e, 2) for y in (0, e)}
    paper_h |= {(x, y) for x in (1, e - 1) for y in range(1, e, 2)}
    hadamards = [p for p in surface.points if p not in interior and p not in paper_h]
    blue = [((x0, y), (x1, y)) for x0, x1 in ((2, 0), (e - 2, e)) for y in range(2, e, 2)]
    blue += [
        ((x0, y0), (x1, y1)) for x0, x1 in ((1, 0), (e - 1, e)) for y0, y1 in ((1, 0), (e - 1, e))
    ]
    green = [((x, y0), (x, y1)) for y0, y1 in ((0, 2), (e, e - 2)) for x in range(2, e, 2)]
    green += [
        ((x0, y0), (x1, y1))
        for x0, x1 in ((1, 0), (e - 1, e))
        for y0, y1 in ((1, 2), (e - 1, e - 2))
    ]
    layers = [blue, green]
    for step in (0, 1):
        layer = [((x, 0), (x + 1 - 2 * step, 1)) for x in range(2, e, 2)]
        layer += [((x, e), (x - 1 + 2 * step, e - 1)) for x in range(2, e, 2)]
        layer += [((1, y), (0, y - 1 + 2 * step)) for y in range(3, e - 1, 2)]
        layer += [((e - 1, y), (e, y + 1 - 2 * step)) for y in range(3, e - 1, 2)]
        layers.append(layer)
    return hadamards, [[(target, control) for control, target in layer] for layer in layers]


class F7Builder(F5Builder):
    def __init__(self, probability):
        Builder.__init__(self, probability)
        self.surface = Surface(7)
        self.circuit = Circuit(coordinates=self.surface.points, protocol="f7")
        self.ancilla = len(self.surface.points)

    def prefix(self):
        small = F5Builder(self.probability)
        small.prefix()
        small.growth_to_rotated()
        small.growth_to_regular()
        small.stabilizers("d5_pre_checks", noisy=True)
        small.logical_check("d5_logical_check_1")
        small.logical_check("d5_logical_check_2")
        mapping = {
            q: self.surface.index[x + 2, y + 2] for q, (x, y) in enumerate(small.surface.points)
        }
        mapping.update(
            {q: self.ancilla + q - small.ancilla for q in range(small.ancilla, small.ancilla + 6)}
        )
        self.circuit.operations.extend(
            replace(op, qubits=tuple(mapping[q] for q in op.qubits))
            for op in small.circuit.operations
        )
        self.circuit.measurements.extend(small.circuit.measurements)

    def growth_to_seven(self):
        self.stage = "grow_regular5_to_regular7"
        self.source = "Higgott et al. Quantum 5, 517, Fig. 2, Hadamard dual; Takada Appendix A.1"
        inside = {(x + 2, y + 2) for x, y in Surface(5).points}
        for point in self.surface.points:
            if point not in inside:
                self.reset(self.surface.index[point])
        hadamards, layers = growth_layers(7)
        for point in hadamards:
            self.gate("H", self.surface.index[point])
        for layer in layers:
            for control, target in layer:
                self.gate("CX", self.surface.index[control], self.surface.index[target])

    def controlled_factors(self, cat, *, noisy):
        for j, q in enumerate(self.surface.fold):
            first, last = ("T_DAG", "T") if j % 2 == 0 else ("T", "T_DAG")
            self.gate(first, q, noisy=noisy)
            self.gate("CX", cat[j % len(cat)], q, noisy=noisy)
            self.gate(last, q, noisy=noisy)
        for offset, pairs in zip((3, 7, 2, 5, 0, 3), self.surface.pairs, strict=True):
            for j, (a, b) in enumerate(pairs):
                self.gate("CCZ", cat[(offset + j) % len(cat)], a, b, noisy=noisy)

    def cat_preparation(self, *, noisy):
        a = self.ancilla
        for q in range(a, a + 14):
            self.reset(q, noisy)
        self.gate("H", a, noisy=noisy)
        self.gate("H", a + 8, noisy=noisy)
        for control, target in CAT_DATA_CX + CAT_VERIFY_CX + CAT_TRANSVERSAL_CX:
            self.gate("CX", a + control, a + target, noisy=noisy)
        first = len(self.circuit.measurements)
        for q in range(a + 8, a + 14):
            self.measure(q, f"cat_verify_{q-a-8}", noisy=noisy, postselect=q != a + 8)
            if q != a + 8:
                self.circuit.measurements[-1]["parity_with"] = first

    def logical_check(self, stage, *, noisy=True, terminal=False):
        if terminal:
            return super().logical_check(stage, noisy=noisy, terminal=True)
        self.stage, self.source = stage, "Takada Table 4 and Fig. 19c"
        self.cat_preparation(noisy=noisy)
        cat = tuple(range(self.ancilla, self.ancilla + 8))
        self.controlled_factors(cat, noisy=noisy)
        for control, target in CAT_UNCOMPUTE_CX:
            self.gate("CX", cat[control], cat[target], noisy=noisy)
        self.gate("H", cat[4], noisy=noisy)
        for q in cat:
            self.measure(q, "logical" if q == cat[4] else f"cat_{q}", noisy=noisy)


def make_f7(probability=0.001, *, evaluation="fed"):
    if evaluation not in {"fed", "none"}:
        raise ValueError("supported final evaluations are fed and none")
    builder = F7Builder(probability)
    builder.prefix()
    builder.growth_to_seven()
    builder.stabilizers("d7_pre_checks", noisy=True)
    builder.logical_check("d7_logical_check_1")
    builder.logical_check("d7_logical_check_2")
    builder.stabilizers("d7_post_checks", noisy=True)
    if evaluation == "fed":
        builder.stabilizers("final_syndrome", noisy=False)
        builder.logical_check("final_logical", noisy=False, terminal=True)
    return builder.circuit
