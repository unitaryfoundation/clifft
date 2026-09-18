"""Fixed factor contractions for coherent CSS norms and syndrome marginals.

Compilation chooses one elimination order. A prefix of that order can be split
into bra/ket copies without increasing the elimination width. Runtime only
fills phase factors and follows fixed gathers; it never plans topology.
"""

from dataclasses import dataclass

import numpy as np
import stim
from fold_blocks import masks
from fold_contraction import ROOTS
from msc_sampling import SparseCode, draw, parity


@dataclass
class Leaf:
    scope: tuple[int, ...]
    parameter: int
    offset: int
    parity: np.ndarray


@dataclass
class Step:
    offset: int
    size: int
    gathers: np.ndarray


def elimination_order(scopes, variables):
    graph: dict[int, set[int]] = {k: set() for k in range(variables)}
    for scope in scopes:
        for k in scope:
            graph[k].update(set(scope) - {k})
    result = []
    while graph:

        def score(k):
            neighbors = graph[k]
            missing = sum(b not in graph[a] for a in neighbors for b in neighbors if a < b)
            return missing, len(neighbors), k

        k = min(graph, key=score)
        neighbors = graph.pop(k)
        for a in neighbors:
            graph[a].update(neighbors - {a})
            graph[a].remove(k)
        result.append(k)
    return tuple(result)


class FactorPlan:
    def __init__(self, factors, order, *, entry_limit=2000000):
        self.leaves: list[Leaf] = []
        self.steps: list[Step] = []
        self.storage = 0
        self.gather_entries = 0
        self.joint_entries = 0
        self.max_joint = 1
        for scope, parameter in factors:
            scope = tuple(sorted(scope))
            size = 1 << len(scope)
            if self.storage + size > entry_limit:
                raise ValueError("factor leaves exceed compilation budget")
            table = np.asarray([j.bit_count() & 1 for j in range(size)], dtype=np.int8)
            self.leaves.append(Leaf(scope, parameter, self.storage, table))
            self.storage += size
        active = [(leaf.scope, leaf.offset) for leaf in self.leaves]
        for variable in order:
            inputs = [(scope, offset) for scope, offset in active if variable in scope]
            active = [(scope, offset) for scope, offset in active if variable not in scope]
            neighbors = tuple(sorted(set().union(*(set(s) for s, _ in inputs)) - {variable}))
            joint = (variable, *neighbors)
            size = 1 << len(neighbors)
            entries = 2 * size * len(inputs)
            if self.storage + size > entry_limit or self.gather_entries + entries > entry_limit:
                raise ValueError("factor elimination exceeds compilation budget")
            gathers = []
            for scope, offset in inputs:
                positions = [joint.index(q) for q in scope]
                gathers.append(
                    [
                        offset + sum(((b >> p) & 1) << j for j, p in enumerate(positions))
                        for b in range(2 * size)
                    ]
                )
            self.steps.append(Step(self.storage, size, np.asarray(gathers, dtype=np.int64)))
            active.append((neighbors, self.storage))
            self.storage += size
            self.gather_entries += entries
            self.joint_entries += 2 * size
            self.max_joint = max(self.max_joint, 2 * size)
        if any(scope for scope, _ in active):
            raise ValueError("elimination order leaves unresolved variables")
        self.outputs = [offset for _, offset in active]

    def evaluate(self, parameters):
        values = np.empty(self.storage, dtype=complex)
        roots = np.asarray(ROOTS)
        for leaf in self.leaves:
            values[leaf.offset : leaf.offset + len(leaf.parity)] = roots[
                (parameters[leaf.parameter] * leaf.parity) % 8
            ]
        for step in self.steps:
            products = np.prod(values[step.gathers], axis=0)
            values[step.offset : step.offset + step.size] = products.reshape(-1, 2).mean(axis=1)
        return complex(np.prod(values[self.outputs]))

    def statistics(self):
        return {
            "coefficient_slots": self.storage,
            "max_joint_entries": self.max_joint,
            "joint_entries": self.joint_entries,
            "gather_entries": self.gather_entries,
        }


class FactorCode:
    def __init__(self, checks, width, *, entry_limit=2000000):
        if width < 1 or width % 2 != 1:
            raise ValueError("all-data logical axes require odd data width")
        self.width = width
        self.xchecks: list[int] = []
        self.zchecks: list[int] = []
        self.order: list[tuple[bool, int]] = []
        for p in checks:
            x, z = masks(p)
            if p.sign != 1 or (x and z) or not (x or z):
                raise ValueError("factor code requires positive CSS checks")
            family = self.xchecks if x else self.zchecks
            self.order.append((bool(x), len(family)))
            family.append(x or z)
        self.rank = len(self.xchecks)
        self.all_x = (1 << width) - 1
        # Certify the same one-logical-qubit convention without enumerating a
        # code basis. Tableaux and binary reduction occur only at construction.
        logical_z = stim.PauliString("Z" * width)
        if any(x.bit_count() & 1 for x in self.xchecks + self.zchecks):
            raise ValueError("all-data logical axes do not preserve the code")
        if len(checks) != width - 1:
            raise ValueError("code does not encode exactly one qubit")
        tableau = stim.Tableau.from_stabilizers([*checks, logical_z])
        logical_x = stim.PauliString("X" * width)
        self.duals = []
        for k in range(len(checks)):
            correction = tableau.x_output(k)
            if not correction.commutes(logical_x):
                correction *= logical_z
            self.duals.append(masks(correction))
        basis = {}
        for k, x in enumerate(self.xchecks):
            key, coords = x, 1 << k
            while key:
                pivot = key.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = key, coords
                    break
                row, value = basis[pivot]
                key ^= row
                coords ^= value
            if not key:
                raise ValueError("dependent X generators")
        self.coordinates = [0] * self.rank
        for q in range(width):
            key, coords = 1 << q, 0
            for pivot in sorted(basis, reverse=True):
                if key >> pivot & 1:
                    row, value = basis[pivot]
                    key ^= row
                    coords ^= value
            for k in range(self.rank):
                self.coordinates[k] |= ((coords >> k) & 1) << q
        self.supports = [
            tuple(k for k, x in enumerate(self.xchecks) if x >> q & 1) for q in range(width)
        ]
        self.sampling_order = elimination_order(self.supports, self.rank)
        self.single = FactorPlan(
            [(scope, q) for q, scope in enumerate(self.supports)]
            + [((k,), width + k) for k in range(self.rank)],
            self.sampling_order,
            entry_limit=entry_limit,
        )
        self.prefixes = []
        for count in range(self.rank + 1):
            measured = self.sampling_order[:count]
            mapping = {k: self.rank + j for j, k in enumerate(measured)}
            factors = [(scope, q) for q, scope in enumerate(self.supports)]
            factors += [
                (tuple(mapping.get(k, k) for k in scope), width + q)
                for q, scope in enumerate(self.supports)
            ]
            for k in measured:
                factors += [((k,), 2 * width + k), ((mapping[k],), 2 * width + k)]
            order = [v for k in measured for v in (k, mapping[k])] + list(
                self.sampling_order[count:]
            )
            plan = FactorPlan(factors, order, entry_limit=entry_limit)
            if plan.max_joint > self.single.max_joint:
                raise AssertionError("prefix splitting increased the certified width")
            self.prefixes.append(plan)

    def coordinate(self, bits):
        return sum(parity(bits, row) << k for k, row in enumerate(self.coordinates))

    def expand(self, bits):
        result = 0
        for k, row in enumerate(self.xchecks):
            if bits >> k & 1:
                result ^= row
        return result

    def prepare(self, logical, frame, terms):
        if len(terms) > 4:
            raise ValueError("coherent term count exceeds the bounded sampler")
        labels, amplitudes, phases = [], [], []
        x, z = frame
        if not 0 <= x < 1 << self.width or not 0 <= z < 1 << self.width:
            raise ValueError("frame exceeds data width")
        for term in terms:
            op = term.data
            if op.width != self.width or op.edges or len(op.linear) != self.width:
                raise ValueError("operator is outside the diagonal/flip certificate")
            offset = x ^ op.flips
            labels.append(sum(parity(offset, mask) << k for k, mask in enumerate(self.zchecks)))
            coefficients, parameters = [], []
            for output in (0, 1):
                source = output ^ parity(offset, self.all_x)
                delta = (
                    x
                    ^ (self.all_x * source)
                    ^ self.expand(self.coordinate(offset ^ (self.all_x * source)))
                )
                constant = (
                    op.global_phase
                    + sum(c * ((delta >> q) & 1) for q, c in enumerate(op.linear))
                    + 4 * parity(z, delta ^ x)
                )
                coefficients.append(logical[source] * ROOTS[constant % 8])
                parameters.append(
                    [
                        (c * (1 - 2 * ((delta >> q) & 1)) + 4 * ((z >> q) & 1)) % 8
                        for q, c in enumerate(op.linear)
                    ]
                )
            amplitudes.append(coefficients)
            phases.append(parameters)
        return Prepared(
            self, labels, np.asarray(amplitudes), np.asarray(phases), SparseCode.gram(terms)
        )

    def norm(self, logical, frame, terms):
        prepared = self.prepare(logical, frame, terms)
        return sum(prepared.mass(label, 0, 0) for label in dict.fromkeys(prepared.labels))

    def sample_syndrome(self, logical, frame, terms, rng):
        return self.prepare(logical, frame, terms).sample(rng)

    def contract(self, logical, frame, terms, syndrome):
        if len(syndrome) != len(self.order) or any(bit not in (0, 1) for bit in syndrome):
            raise ValueError("one binary outcome is required per CSS check")
        prepared = self.prepare(logical, frame, terms)
        xs = sum(bit << k for bit, (x, k) in zip(syndrome, self.order, strict=True) if x)
        zs = sum(bit << k for bit, (x, k) in zip(syndrome, self.order, strict=True) if not x)
        cx = cz = 0
        for bit, (x, z) in zip(syndrome, self.duals, strict=True):
            if bit:
                cx ^= x
                cz ^= z
        logical_rep = self.all_x ^ self.expand(self.coordinate(self.all_x))
        representative = (
            cx ^ self.expand(self.coordinate(cx)) ^ (logical_rep * parity(cx, self.all_x))
        )
        result = np.zeros(2, dtype=complex)
        for t, term in enumerate(terms):
            if prepared.labels[t] != zs:
                continue
            overlaps = np.sum(terms[0].ancillas.conj() * term.ancillas, axis=1)
            if not np.allclose(np.abs(overlaps), 1, rtol=0, atol=1e-10):
                raise ValueError("spectators do not leave a pure logical output")
            scalar = term.weight * np.prod(overlaps)
            for output in (0, 1):
                parameters = [
                    *prepared.phases[t, output],
                    *[4 * ((xs >> k) & 1) for k in range(self.rank)],
                ]
                delta = representative ^ (logical_rep * output) ^ cx
                target = output ^ parity(cx, self.all_x)
                result[target] += (
                    scalar
                    * prepared.amplitudes[t, output]
                    * self.single.evaluate(parameters)
                    * (-1) ** parity(delta, cz)
                )
        return result

    def statistics(self):
        return {
            "data_width": self.width,
            "x_rank": self.rank,
            "sampling_order": self.sampling_order,
            "single": self.single.statistics(),
            "prefixes": [p.statistics() for p in self.prefixes],
            "sparse_amplitudes_per_term": 2 << self.rank,
        }


class Prepared:
    def __init__(self, code, labels, amplitudes, phases, gram):
        self.code, self.labels = code, labels
        self.amplitudes, self.phases, self.gram = amplitudes, phases, gram

    def mass(self, label, count, syndrome):
        code = self.code
        result = 0j
        signs = [4 * ((syndrome >> k) & 1) for k in range(code.rank)]
        for a in range(len(self.labels)):
            if self.labels[a] != label:
                continue
            for b in range(a, len(self.labels)):
                if self.labels[b] != label:
                    continue
                for logical in (0, 1):
                    coefficient = (
                        self.gram[a, b]
                        * self.amplitudes[a, logical].conjugate()
                        * self.amplitudes[b, logical]
                    )
                    if coefficient == 0:
                        continue
                    parameters = [*(-self.phases[a, logical]), *self.phases[b, logical], *signs]
                    value = coefficient * code.prefixes[count].evaluate(parameters)
                    result += value if a == b else 2 * value.real
        if abs(result.imag) > 1e-10 or result.real < -1e-10:
            raise ArithmeticError("invalid factor probability")
        return max(0.0, float(result.real))

    def sample(self, rng):
        labels = list(dict.fromkeys(self.labels))
        weights = [self.mass(label, 0, 0) for label in labels]
        selected, probability = draw(weights, rng)
        label = labels[selected]
        mass, total = weights[selected], sum(weights)
        syndrome = 0
        for count, k in enumerate(self.code.sampling_order, 1):
            zero = self.mass(label, count, syndrome)
            if zero > mass + 1e-10 * total:
                raise ArithmeticError("factor marginal exceeds its parent")
            zero = min(zero, mass)
            bit, _ = draw([zero, mass - zero], rng)
            syndrome |= bit << k
            mass = mass - zero if bit else zero
        outcomes = [((syndrome if x else label) >> k) & 1 for x, k in self.code.order]
        return outcomes, mass / total


def generated_color_checks(distance):
    """CSS block geometry from Stim, not a full MSC cultivation circuit."""
    circuit = stim.Circuit.generated("color_code:memory_xyz", distance=distance, rounds=3)
    supports: dict[int, set[int]] = {}
    for op in circuit.flattened():
        if op.name == "CX":
            targets = [t.value for t in op.targets_copy()]
            for a, b in zip(targets[::2], targets[1::2], strict=True):
                supports.setdefault(b, set()).symmetric_difference_update({a})
        elif op.name == "MR":
            ancillas = [t.value for t in op.targets_copy()]
            if set(ancillas) != set(supports):
                raise ValueError("unexpected generated check-extraction schedule")
            break
    data = sorted(set().union(*supports.values()))
    if set(data) & set(supports):
        raise ValueError("generated checks mix data and ancillas")
    checks = []
    for support in supports.values():
        for axis in "XZ":
            p = stim.PauliString(len(data))
            for q in support:
                p[data.index(q)] = axis
            checks.append(p)
    return checks, len(data)
