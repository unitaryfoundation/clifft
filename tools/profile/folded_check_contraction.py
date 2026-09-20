"""Fixed contractions for two physical folded checks and CSS continuation.

Planning and physical-fault binding are offline here. The exported numeric
tables can be evaluated without tableau evolution or changing tensor scopes.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import stim
from compiled_gadget_contraction import binary_duals
from folded_msc_family import Surface


def pauli(axis, row, width):
    p = stim.PauliString(width)
    for q in range(width):
        if row >> q & 1:
            p[q] = axis
    return p


def pauli_text(p):
    if p.sign != 1:
        raise ValueError("probe requires a positive physical Pauli")
    return "*".join(f"{'IXYZ'[axis]}{q}" for q, axis in enumerate(p) if axis)


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

    def evaluate(self, local):
        scratch = np.empty(self.storage, dtype=np.complex128)
        for (offset, labels), values in zip(self.leaves, local, strict=True):
            scratch[offset : offset + len(labels)] = values[labels]
        for offset, size, gathers in self.steps:
            product = np.ones(2 * size, dtype=np.complex128)
            for gather in gathers:
                product *= scratch[gather]
            scratch[offset : offset + size] = product[:size] + product[size:]
        return np.prod(scratch[self.outputs]) / (1 << self.rank)


@dataclass
class CodeBoundary:
    offset: int
    x_signs: int
    logical: np.ndarray
    ancillas: int


@dataclass
class PathTerm:
    scalar: complex
    flips: int
    local: np.ndarray
    records: tuple[int, ...]
    ancillas: int


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

    def path(self, operations, boundary, branches, outcomes):
        """Bind physical monomial gates; no Clifford-state synthesis is used."""
        local = np.ones((self.characters, 4), dtype=complex)
        bits, flips, scalar = boundary.ancillas, 0, 1 + 0j
        records = []
        stage = None
        check = -1
        hadamards = 0

        def phase(q, exponent):
            if q < self.n:
                for x in (0, 1):
                    local[q, x] *= np.exp(1j * math.pi / 4 * exponent * (x ^ ((flips >> q) & 1)))
                return 1 + 0j
            return np.exp(1j * math.pi / 4 * exponent * ((bits >> (q - self.n)) & 1))

        for op in operations:
            if op.probability is not None:
                raise ValueError("bind physical Pauli faults before path evaluation")
            if op.stage != stage:
                stage, check, hadamards = op.stage, check + 1, 0
            q = op.qubits[0]
            a = q - self.n
            if op.name in {"X", "Y", "Z"}:
                if op.name in {"Y", "Z"}:
                    scalar *= phase(q, 4)
                if op.name == "Y":
                    scalar *= 1j
                if op.name in {"X", "Y"}:
                    if a >= 0:
                        bits ^= 1 << a
                    else:
                        flips ^= 1 << q
            elif op.name in {"T", "T_DAG"}:
                scalar *= phase(q, 1 if op.name == "T" else -1)
            elif op.name == "CX":
                if a < 0:
                    raise ValueError("expected cat-controlled monomial factor")
                target = op.qubits[1]
                if bits >> a & 1:
                    if target < self.n:
                        flips ^= 1 << target
                    else:
                        bits ^= 1 << (target - self.n)
            elif op.name == "CCZ":
                _, x, y = op.qubits
                if a < 0 or (x, y) not in self.pair_index:
                    raise ValueError("expected a folded mirror pair")
                if bits >> a & 1:
                    table = local[self.n + self.pair_index[x, y]]
                    for label in range(4):
                        if ((label & 1) ^ ((flips >> x) & 1)) and (
                            (label >> 1) ^ ((flips >> y) & 1)
                        ):
                            table[label] *= -1
            elif op.name == "R":
                if a < 0:
                    raise ValueError("folded block cannot reset data")
                records.append((bits >> a) & 1)
                bits &= ~(1 << a)
            elif op.name == "H":
                if a < 0 or hadamards > 1:
                    raise ValueError("expected cat preparation and readout Hadamards")
                bit = branches[check] if hadamards == 0 else outcomes[check]
                scalar *= (-1 if (bits >> a) & bit & 1 else 1) / math.sqrt(2)
                bits = (bits & ~(1 << a)) | (bit << a)
                hadamards += 1
            elif op.name == "M":
                if a < 0:
                    raise ValueError("folded block cannot measure data")
                records.append((bits >> a) & 1)
            else:
                raise ValueError(f"unsupported monomial gate {op.name}")
        if check != len(branches) - 1 or hadamards != 2:
            raise ValueError("incorrect folded block stage count")
        return PathTerm(scalar, flips, local, tuple(records), bits)

    def bind(self, operations, boundary, outcomes):
        paths = [
            self.path(operations, boundary, b, outcomes)
            for b in itertools.product((0, 1), repeat=len(outcomes))
        ]
        if any(p.records != paths[0].records for p in paths):
            raise ValueError("unmeasured cat branch leaked into a classical output")
        offset = boundary.offset ^ paths[0].flips
        inputs: list[list[tuple[complex, np.ndarray]]] = [[], []]
        for path in paths:
            delta = path.flips ^ paths[0].flips
            shift = sum(
                ((delta & dual).bit_count() % 2) << i for i, dual in enumerate(self.x_duals)
            )
            restored = 0
            for i, row in enumerate(self.x_rows + [self.logical_x]):
                if shift >> i & 1:
                    restored ^= row
            if restored != delta:
                raise ValueError("branches leave different computational code cosets")
            for logical in (0, 1):
                base = offset ^ path.flips ^ (self.logical_x if logical else 0)
                local = np.ones((len(self.masks), 4), dtype=complex)
                for q in range(self.n):
                    local[q, :2] = path.local[q, [(base >> q) & 1, 1 ^ ((base >> q) & 1)]]
                for i, (a, b) in enumerate(self.pairs):
                    local[self.n + i] = path.local[
                        self.n + i, np.arange(4) ^ (((base >> a) & 1) | (((base >> b) & 1) << 1))
                    ]
                for i in range(self.rank):
                    local[self.characters + i, 1] = (-1) ** ((boundary.x_signs >> i) & 1)
                coefficient = path.scalar * boundary.logical[logical ^ (shift >> self.rank)]
                coefficient *= (-1) ** ((boundary.x_signs & shift).bit_count() % 2)
                inputs[logical].append((coefficient, local))
        return dict(
            inputs=inputs, offset=offset, records=paths[0].records, ancillas=paths[0].ancillas
        )

    def marginal(self, bound, measured, syndrome):
        total = 0.0
        for terms in bound["inputs"]:
            local_terms = []
            for coefficient, local in terms:
                local = local.copy()
                for i in range(measured):
                    local[self.characters + i, 1] *= (-1) ** ((syndrome >> i) & 1)
                local_terms.append((coefficient, local))
            for i, (a, left) in enumerate(local_terms):
                for j, (b, right) in enumerate(local_terms[: i + 1]):
                    value = self.marginals[measured].evaluate(
                        np.concatenate([left, right.conjugate()])
                    )
                    total += ((1 if i == j else 2) * a * b.conjugate() * value).real
        if total < -1e-10:
            raise AssertionError("negative coherent marginal")
        return max(0.0, float(total))

    def amplitudes(self, bound, syndrome):
        result = []
        for terms in bound["inputs"]:
            amplitude = 0j
            for coefficient, local in terms:
                local = local.copy()
                for i in range(self.rank):
                    local[self.characters + i, 1] *= (-1) ** ((syndrome >> i) & 1)
                amplitude += coefficient * self.amplitude_plan.evaluate(local)
            result.append(amplitude)
        return np.array(result)
