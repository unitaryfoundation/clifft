"""Offline plans for Clifford fault frames in CNOT and diagonal C3 regions.

The plan fixes every quadratic coefficient slot and CNOT update pair before
execution. The Python evaluator is a reference for the standalone native probe,
not a proposed runtime tableau or topology planner.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from clifford_branches import multiply, root
from fold_blocks import NoiseLayout
from fold_frame_screen import copy_state


@dataclass
class Frame:
    flips: int
    phase: int
    linear: list[int]
    edges: int = 0

    def copy(self):
        return Frame(self.flips, self.phase, self.linear.copy(), self.edges)


@dataclass
class Action:
    kind: str
    targets: tuple[int, ...]
    edges: tuple[int, ...] = ()
    updates: list[tuple[int, int]] = field(default_factory=list)
    x: int = -1
    z: int = -1


class FramePlan:
    def __init__(self, layout: NoiseLayout, initial_edges: Iterable[tuple[int, int]] = ()):
        self.layout = layout
        self.edges = list(sorted(initial_edges))
        if len(set(self.edges)) != len(self.edges) or any(
            not 0 <= a < b < layout.width for a, b in self.edges
        ):
            raise ValueError("invalid initial edge universe")
        self.initial_edge_count = len(self.edges)
        self.actions: list[Action] = []
        self.stops = []

        def edge(a, b):
            pair = min(a, b), max(a, b)
            if pair not in self.edges:
                self.edges.append(pair)
            return self.edges.index(pair)

        for boundary, faults in enumerate(layout.boundaries):
            for q, x, z in faults:
                self.actions.append(
                    Action("fault", (q,), x=x.bit_length() - 1, z=z.bit_length() - 1)
                )
            self.stops.append(len(self.actions))
            if boundary == len(layout.operations):
                break
            op = layout.operations[boundary]
            expected = {"T": 1, "T_DAG": 1, "CX": 2, "CCZ": 3}.get(op.name)
            if (
                expected is None
                or len(op.targets) != expected
                or len(set(op.targets)) != expected
                or any(not 0 <= q < layout.width for q in op.targets)
            ):
                raise ValueError("unsupported fault-frame region")
            action = Action(op.name, op.targets)
            if op.name == "CX":
                c, t = op.targets
                # Only edges reachable before this gate can contribute. Future
                # slots are zero here; no runtime search of adjacency is needed.
                for source, pair in enumerate(list(self.edges)):
                    if t in pair and c not in pair:
                        k = pair[0] if pair[1] == t else pair[1]
                        action.updates.append((source, edge(c, k)))
                action.edges = (edge(c, t),)
            elif op.name == "CCZ":
                a, b, c = op.targets
                action.edges = (edge(b, c), edge(a, c), edge(a, b))
            self.actions.append(action)

    def empty(self):
        return Frame(0, 0, [0] * self.layout.width)

    def evaluate(self, faults, initial=None, stop=None):
        frame = self.empty() if initial is None else initial.copy()
        if (
            len(frame.linear) != self.layout.width
            or any(c % 2 for c in frame.linear)
            or frame.edges >> self.initial_edge_count
        ):
            raise ValueError("initial frame exceeds compiled input contract")
        end = len(self.actions) if stop is None else self.stops[stop]
        for action in self.actions[:end]:
            q = action.targets[0]
            x = (frame.flips >> q) & 1
            if action.kind == "fault":
                a = (faults >> action.x) & 1 if action.x >= 0 else 0
                b = (faults >> action.z) & 1 if action.z >= 0 else 0
                # Encoded Y is XZ, differing from physical Y by a global i.
                frame.phase += 4 * b * x
                frame.linear[q] += 4 * b
                frame.flips ^= a << q
            elif action.kind in {"T", "T_DAG"}:
                sign = 1 if action.kind == "T" else -1
                frame.phase += sign * x
                frame.linear[q] -= 2 * sign * x
            elif action.kind == "CX":
                t = action.targets[1]
                e = action.edges[0]
                frame.flips ^= x << t
                frame.linear[q] += frame.linear[t] + 4 * ((frame.edges >> e) & 1)
                frame.edges ^= ((frame.linear[t] // 2) & 1) << e
                for source, target in action.updates:
                    frame.edges ^= ((frame.edges >> source) & 1) << target
            else:
                b, c = action.targets[1:]
                y, z = (frame.flips >> b) & 1, (frame.flips >> c) & 1
                frame.phase += 4 * x * y * z
                frame.linear[q] += 4 * y * z
                frame.linear[b] += 4 * x * z
                frame.linear[c] += 4 * x * y
                for bit, e in zip((x, y, z), action.edges, strict=True):
                    frame.edges ^= bit << e
        frame.phase %= 8
        frame.linear = [c % 8 for c in frame.linear]
        return frame

    def data_only(self, frame, data):
        mask = sum(1 << q for q in data)
        return (
            not frame.flips & ~mask
            and all(not c or q in data for q, c in enumerate(frame.linear))
            and all(
                not (frame.edges >> e) & 1 or (a in data and b in data)
                for e, (a, b) in enumerate(self.edges)
            )
        )

    def inverse_state(self, state, frame):
        """Offline phase-preserving application; never used in native timing."""
        result = copy_state(state)
        for term in result.terms:
            for q in range(state.width):
                if (frame.flips >> q) & 1:
                    term.gate("X", (q,))
            for q, power in enumerate(frame.linear):
                for _ in range(power // 2):
                    term.gate("S_DAG", (q,))
            for e, pair in enumerate(self.edges):
                if (frame.edges >> e) & 1:
                    term.gate("CZ", pair)
            term.coefficient = multiply(term.coefficient, root(-frame.phase))
        return result

    def remap(self, frame, other):
        """Offline descriptor conversion at an explicitly checked boundary."""
        result = frame.copy()
        result.edges = 0
        for e, pair in enumerate(self.edges):
            if (frame.edges >> e) & 1:
                if pair not in other.edges[: other.initial_edge_count]:
                    raise ValueError("frame does not fit the next compiled input contract")
                result.edges ^= 1 << other.edges.index(pair)
        return result
