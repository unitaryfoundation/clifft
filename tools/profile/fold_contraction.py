"""Offline fixed-topology overlap plans for regular surface-code fold branches.

This is a code-boundary kernel experiment, not a cultivation sampler. The
planner enumerates tensor indices; execution only follows those fixed maps.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fold_check import Check, Event, PhaseMonomial, branch_operator, elementary_check, fold_check
from fold_cultivation import coordinates, stabilizers

ROOTS = (
    1,
    (1 + 1j) / np.sqrt(2),
    1j,
    (-1 + 1j) / np.sqrt(2),
    -1,
    (-1 - 1j) / np.sqrt(2),
    -1j,
    (1 - 1j) / np.sqrt(2),
)


def compose(after: PhaseMonomial, before: PhaseMonomial) -> PhaseMonomial:
    """Compose before then after without inserting a syndrome projection."""
    if before.width != after.width:
        raise ValueError("monomial widths differ")
    flips = [(before.flips >> q) & 1 for q in range(before.width)]
    constant = before.global_phase + after.global_phase
    constant += sum(c * x for c, x in zip(after.linear, flips, strict=True))
    linear = [
        b + a * (1 - 2 * x) for a, b, x in zip(after.linear, before.linear, flips, strict=True)
    ]
    for q, r in after.edges:
        constant += 4 * flips[q] * flips[r]
        linear[q] += 4 * flips[r]
        linear[r] += 4 * flips[q]
    return PhaseMonomial(
        before.width,
        before.flips ^ after.flips,
        constant % 8,
        [c % 8 for c in linear],
        before.edges ^ after.edges,
    )


@dataclass
class Leaf:
    scope: tuple[int, ...]
    offset: int
    multipliers: tuple[list[int], list[int]]


@dataclass
class Step:
    offset: int
    size: int
    # Absolute addresses into one preallocated coefficient array.
    gathers: list[list[int]]


class Plan:
    def __init__(self, distance: int):
        if distance not in (3, 5, 7):
            raise ValueError("supported regular-code distances are 3, 5 and 7")
        self.distance = distance
        self.coords = coordinates(distance)
        index = {p: q for q, p in enumerate(self.coords)}
        self.width = len(index)
        self.x_masks = [
            sum(1 << index[p] for p in support)
            for axis, support in stabilizers(distance)
            if axis == "X"
        ]
        self.z_masks = [
            sum(1 << index[p] for p in support)
            for axis, support in stabilizers(distance)
            if axis == "Z"
        ]
        self.logical_x = sum(1 << index[0, y] for y in range(0, 2 * distance - 1, 2))
        self.logical_z = sum(1 << index[x, 0] for x in range(0, 2 * distance - 1, 2))
        self.edges = [(index[x, y], index[y, x]) for x, y in self.coords if x < y]
        self.edges = [(min(q, r), max(q, r)) for q, r in self.edges]
        supports = [
            {g for g, mask in enumerate(self.x_masks) if (mask >> q) & 1} for q in range(self.width)
        ]
        self.leaves: list[Leaf] = []
        self.storage = 0
        for physical in [(q,) for q in range(self.width)] + list(self.edges):
            scope = tuple(sorted(set().union(*(supports[q] for q in physical))))
            tables: list[list[int]] = []
            for logical in (0, 1):
                table = []
                for bits in range(1 << len(scope)):
                    product = 1
                    for q in physical:
                        parity = (
                            sum((bits >> j) & 1 for j, g in enumerate(scope) if g in supports[q])
                            % 2
                        )
                        product *= parity ^ (logical * ((self.logical_x >> q) & 1))
                    table.append(product)
                tables.append(table)
            self.leaves.append(Leaf(scope, self.storage, (tables[0], tables[1])))
            self.storage += len(tables[0])
        graph: dict[int, set[int]] = {g: set() for g in range(len(self.x_masks))}
        for leaf in self.leaves:
            for g in leaf.scope:
                graph[g].update(set(leaf.scope) - {g})
        factors = [(leaf.scope, leaf.offset) for leaf in self.leaves]
        self.steps: list[Step] = []
        self.order: list[int] = []
        while graph:

            def score(g: int) -> tuple[int, int, int]:
                neighbors = graph[g]
                missing = sum(b not in graph[a] for a in neighbors for b in neighbors if a < b)
                return missing, len(neighbors), g

            g = min(graph, key=score)
            remaining = tuple(sorted(graph[g]))
            joint = (g,) + remaining
            inputs = [(s, o) for s, o in factors if g in s]
            factors = [(s, o) for s, o in factors if g not in s]
            gathers = []
            for scope, offset in inputs:
                positions = [joint.index(v) for v in scope]
                gathers.append(
                    [
                        offset + sum(((bits >> pos) & 1) << j for j, pos in enumerate(positions))
                        for bits in range(1 << len(joint))
                    ]
                )
            size = 1 << len(remaining)
            self.steps.append(Step(self.storage, size, gathers))
            factors.append((remaining, self.storage))
            self.storage += size
            self.order.append(g)
            for a in remaining:
                graph[a].update(set(remaining) - {a})
                graph[a].remove(g)
            del graph[g]
        assert all(not scope for scope, _ in factors)
        self.outputs = [offset for _, offset in factors]

    def parameters(self, op: PhaseMonomial) -> list[int]:
        if (
            op.width != self.width
            or len(op.linear) != self.width
            or not 0 <= op.flips < 1 << self.width
            or any(c % 2 for c in op.linear)
            or not op.edges <= set(self.edges)
        ):
            raise ValueError("operator is outside the compiled fold family")
        return [c % 8 for c in op.linear] + [4 * int(edge in op.edges) for edge in self.edges]

    def phase_sums(self, op: PhaseMonomial) -> np.ndarray:
        parameters = self.parameters(op)
        result = []
        for logical in (0, 1):
            values = [0j] * self.storage
            for leaf, parameter in zip(self.leaves, parameters, strict=True):
                for j, bit in enumerate(leaf.multipliers[logical]):
                    values[leaf.offset + j] = ROOTS[parameter * bit]
            for step in self.steps:
                for j in range(step.size):
                    pair = [1 + 0j, 1 + 0j]
                    for gather in step.gathers:
                        pair[0] *= values[gather[2 * j]]
                        pair[1] *= values[gather[2 * j + 1]]
                    values[step.offset + j] = pair[0] + pair[1]
            total = complex(ROOTS[op.global_phase % 8]) / (1 << len(self.x_masks))
            for offset in self.outputs:
                total *= values[offset]
            result.append(total)
        return np.asarray(result)

    def matrix(self, op: PhaseMonomial) -> np.ndarray:
        """Return <out_L|op|in_L>, with ideal zero-syndrome boundaries."""
        self.parameters(op)
        result = np.zeros((2, 2), dtype=complex)
        if any((op.flips & mask).bit_count() % 2 for mask in self.z_masks):
            return result
        flip = (op.flips & self.logical_z).bit_count() % 2
        sums = self.phase_sums(op)
        for logical in (0, 1):
            result[logical ^ flip, logical] = sums[logical]
        return result

    def statistics(self) -> dict:
        return {
            "distance": self.distance,
            "physical_data": self.width,
            "gauge_variables": len(self.x_masks),
            "leaf_factors": len(self.leaves),
            "max_joint_entries": max(2 * s.size for s in self.steps),
            "joint_assignments": sum(2 * s.size for s in self.steps),
            "complex_multiplications_per_input": sum(
                2 * s.size * len(s.gathers) for s in self.steps
            ),
            "coefficient_slots": self.storage,
            "gather_entries": sum(len(g) for s in self.steps for g in s.gathers),
            "elimination_order": self.order,
        }


def fixtures(plan: Plan, count: int = 32) -> list[tuple[str, PhaseMonomial]]:
    """Mix actual fixed-fault cores and deliberately varied admissible phases."""
    rng = random.Random(98231 + plan.distance)
    core = elementary_check(fold_check(plan.distance))
    d = plan.distance
    order = [(j, j) for j in range(2 * d - 1)]
    for k in range(1, d):
        for j in range(2 * (d - k) - 1):
            order.extend([(j, j + 2 * k), (j + 2 * k, j)])
    mapping = [plan.coords.index(p) for p in order]
    result = [("identity", PhaseMonomial(plan.width))]
    for case in range(count):
        events = []
        for event in core.events:
            events.append(event)
            if case > 1 and rng.random() < 0.06:
                events.append(Event(rng.choice("XYZ"), (rng.choice(event.targets),)))
        op, _ = branch_operator(
            Check(core.data, core.ancillas, tuple(events)), (case % 2,) * core.ancillas
        )
        mapped = PhaseMonomial(
            plan.width,
            sum(((op.flips >> q) & 1) << mapping[q] for q in range(plan.width)),
            op.global_phase,
        )
        for q, c in enumerate(op.linear):
            mapped.linear[mapping[q]] = c
        mapped.edges = {
            (min(mapping[q], mapping[r]), max(mapping[q], mapping[r])) for q, r in op.edges
        }
        result.append((f"core_{case}", mapped))
    for case in range(count):
        result.append(
            (f"pair_{case}", compose(result[1 + case][1], result[1 + (case + 1) % count][1]))
        )
        result.append(
            (
                f"random_{case}",
                PhaseMonomial(
                    plan.width,
                    rng.getrandbits(plan.width),
                    rng.randrange(8),
                    [2 * rng.randrange(4) for _ in range(plan.width)],
                    {edge for edge in plan.edges if rng.randrange(2)},
                ),
            )
        )
    return result


def export(plan: Plan, cases: list[tuple[str, PhaseMonomial]], path: Path) -> None:
    """Simple versioned text format for the standalone native kernel diagnostic."""
    lines = [
        f"1 {len(plan.x_masks)} {plan.storage} {len(plan.leaves)} {len(plan.steps)} "
        f"{len(plan.outputs)} {len(cases)}"
    ]
    for leaf in plan.leaves:
        lines.append(f"{leaf.offset} {len(leaf.multipliers[0])}")
        lines.extend(" ".join(map(str, row)) for row in leaf.multipliers)
    for step in plan.steps:
        lines.append(f"{step.offset} {step.size} {len(step.gathers)}")
        lines.extend(" ".join(map(str, gather)) for gather in step.gathers)
    lines.append(" ".join(map(str, plan.outputs)))
    for _, op in cases:
        lines.append(" ".join(map(str, [op.global_phase % 8] + plan.parameters(op))))
        lines.append(
            " ".join(f"{part:.17g}" for v in plan.phase_sums(op) for part in (v.real, v.imag))
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stats = []
    for distance in (3, 5, 7):
        plan = Plan(distance)
        cases = fixtures(plan)
        export(plan, cases, args.output / f"d{distance}.txt")
        stats.append(plan.statistics() | {"cases": len(cases)})
    (args.output / "plans.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
