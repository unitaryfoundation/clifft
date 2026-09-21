"""Compile fixed folded-code contractions before native execution."""

from dataclasses import dataclass

import numpy as np


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


def binary_duals(rows, width):
    """Plan a fixed right inverse of independent binary check rows."""
    if any(row < 0 or row >> width for row in rows):
        raise ValueError("check exceeds physical width")
    work = [(row, 1 << i) for i, row in enumerate(rows)]
    pivots = []
    for i in range(len(rows)):
        candidate = next((j for j in range(i, len(rows)) if work[j][0]), None)
        if candidate is None:
            raise ValueError("dependent code checks")
        work[i], work[candidate] = work[candidate], work[i]
        row, labels = work[i]
        pivot = (row & -row).bit_length() - 1
        pivots.append(pivot)
        for j, (other, label) in enumerate(work):
            if j != i and other >> pivot & 1:
                work[j] = (other ^ row, label ^ labels)
    return [
        sum(((labels >> i) & 1) << p for p, (_, labels) in zip(pivots, work, strict=True))
        for i in range(len(rows))
    ]


class FactorPlan:
    """Eliminate fixed binary scopes with one- or two-parity local tables."""

    def __init__(self, masks, rank, entry_limit=2_000_000):
        if rank < 0:
            raise ValueError("negative contraction rank")
        self.rank = rank
        self.storage = self.peak_scope = self.work = self.gather_entries = 0
        self.leaves, self.steps = [], []
        active = []
        for factors in masks:
            if not 1 <= len(factors) <= 2 or any(m < 0 or m >> rank for m in factors):
                raise ValueError("invalid local parity factor")
            scope = tuple(i for i in range(rank) if any(m >> i & 1 for m in factors))
            size = 1 << len(scope)
            if self.storage + size > entry_limit:
                raise ValueError("contraction storage limit exceeded")
            labels = []
            for bits in range(size):
                expanded = sum(((bits >> j) & 1) << q for j, q in enumerate(scope))
                labels.append(
                    sum(((expanded & m).bit_count() % 2) << j for j, m in enumerate(factors))
                )
            self.leaves.append((self.storage, np.array(labels, dtype=np.int64)))
            active.append((scope, self.storage))
            self.storage += size
        for _ in range(rank):
            variables = set().union(*(set(s) for s, _ in active))
            if not variables:
                raise ValueError("sum variable missing from factor graph")
            unions = {v: set().union(*(set(s) for s, _ in active if v in s)) for v in variables}
            pivot = min(variables, key=lambda v: (len(unions[v]), v))
            union = tuple(sorted(unions[pivot] - {pivot})) + (pivot,)
            size = 1 << len(union)
            selected = [(s, offset) for s, offset in active if pivot in s]
            self.gather_entries += size * len(selected)
            if self.storage + size // 2 > entry_limit or self.gather_entries > entry_limit:
                raise ValueError("contraction storage limit exceeded")
            gathers = []
            for scope, offset in selected:
                positions = [union.index(v) for v in scope]
                gathers.append(
                    np.array(
                        [
                            offset + sum(((i >> p) & 1) << j for j, p in enumerate(positions))
                            for i in range(size)
                        ],
                        dtype=np.int64,
                    )
                )
            self.steps.append((self.storage, size // 2, gathers))
            active = [(s, offset) for s, offset in active if pivot not in s]
            active.append((union[:-1], self.storage))
            self.storage += size // 2
            self.peak_scope = max(self.peak_scope, len(union))
            self.work += size
        self.outputs = [offset for scope, offset in active]


class FoldedChecks:
    def __init__(self, distance):
        self.surface = Surface(distance)
        self.n = len(self.surface.points)
        self.x_rows, self.z_rows = [
            [sum(1 << q for q in row) for row in self.surface.checks(axis)] for axis in "XZ"
        ]
        self.rank = len(self.x_rows)
        end = 2 * distance - 1
        self.logical_x = sum(1 << self.surface.index[0, y] for y in range(0, end, 2))
        self.logical_z = sum(1 << self.surface.index[x, 0] for x in range(0, end, 2))
        self.x_duals = binary_duals(self.x_rows + [self.logical_x], self.n)
        self.z_duals = binary_duals(self.z_rows + [self.logical_z], self.n)
        self.pairs = [pair for group in self.surface.pairs for pair in group]
        self.pair_index = {pair: i for i, pair in enumerate(self.pairs)}
        masks = [
            sum(((row >> q) & 1) << i for i, row in enumerate(self.x_rows)) for q in range(self.n)
        ]
        self.masks = [(m,) for m in masks] + [(masks[a], masks[b]) for a, b in self.pairs]
        self.characters = len(self.masks)
        self.masks += [(1 << i,) for i in range(self.rank)]
        self.amplitude_plan = FactorPlan(self.masks, self.rank)
        self.marginals = []
        for measured in range(self.rank + 1):
            mask = (1 << measured) - 1
            right = [tuple((m & ~mask) | ((m & mask) << self.rank) for m in f) for f in self.masks]
            self.marginals.append(FactorPlan(self.masks + right, self.rank + measured))


def kernel_plan(plan):
    """Serialize an unbound kernel plan with compiler-selected gather storage.

    The expanded NumPy plan remains the arithmetic reference. Native loaders
    only validate and load the chosen representation; they do not optimize it.
    """
    tokens = [2, plan.rank, plan.storage, len(plan.leaves)]
    for offset, parity in plan.leaves:
        tokens.extend([offset, len(parity), *parity])
    tokens.append(len(plan.steps))
    for offset, size, gathers in plan.steps:
        tokens.extend([offset, size, len(gathers)])
        for gather in gathers:
            low, high = gather, []
            if len(gather) > 256 and len(gather) % 256 == 0:
                blocks = gather.reshape(-1, 256)
                offsets = blocks[:, 0] - gather[0]
                if np.all(offsets >= 0) and np.array_equal(blocks, gather[:256] + offsets[:, None]):
                    low, high = gather[:256], offsets
            tokens.extend([len(low), len(high), *low, *high])
    tokens.extend([len(plan.outputs), *plan.outputs])
    return "\n".join(map(str, tokens)) + "\n"
