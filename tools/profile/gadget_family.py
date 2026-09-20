"""Geometric color-code gadget benchmarks with ideal encoded inputs.

Geometry follows make_color_code at Strilanc/magic-state-cultivation revision
871e68ff6df2f75190b1bfd6351459d1b5a037e3. The spanning-tree extraction, noise
schedule, and ideal encoder here define a benchmark, not an author protocol.
"""

from collections import deque

import stim


def geometry(distance):
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be odd and at least three")
    faces = []
    edges = set()
    offsets = [(-1, 0), (0, -1), (1, -1), (2, 0), (1, 1), (0, 1)]

    def inside(q):
        x, y = q
        return y >= 0 and 2 * y - 1 <= 3 * x <= 6 * distance - 3 - 2 * y

    for x in range(1, 2 * distance, 2):
        for y in range((x // 2) % 2, 2 * distance, 2):
            hexagon = [(x + dx, y + dy) for dx, dy in offsets]
            face = frozenset(q for q in hexagon if inside(q))
            if len(face) not in {4, 6}:
                continue
            faces.append(face)
            for a, b in zip(hexagon, hexagon[1:] + hexagon[:1], strict=True):
                if a in face and b in face:
                    edges.add(tuple(sorted((a, b))))
    coordinates = sorted(set().union(*faces))
    index = {q: i for i, q in enumerate(coordinates)}
    rows = [sum(1 << index[q] for q in face) for face in faces]
    adjacency: list[list[int]] = [[] for _ in coordinates]
    for a, b in sorted(edges):
        adjacency[index[a]].append(index[b])
        adjacency[index[b]].append(index[a])
    queue, seen, tree = deque([0]), {0}, []
    while queue:
        parent = queue.popleft()
        for child in sorted(adjacency[parent]):
            if child not in seen:
                seen.add(child)
                queue.append(child)
                tree.append((parent, child))
    if len(seen) != len(coordinates):
        raise ValueError("disconnected data geometry")
    signs = [1 if x % 2 else -1 for x, _ in coordinates]
    return coordinates, rows, signs, tree


def make_circuit(distance, probability):
    if not 0 <= probability <= 1:
        raise ValueError("noise probability must be between zero and one")
    coordinates, rows, signs, tree = geometry(distance)
    n = len(coordinates)
    checks = [
        stim.PauliString("".join(axis if row >> q & 1 else "I" for q in range(n)))
        for axis in "XZ"
        for row in rows
    ]
    encoder = stim.Tableau.from_stabilizers(checks + [stim.PauliString("Z" * n)]).to_circuit()
    if any(op.name not in {"H", "CX"} for op in encoder):
        raise ValueError("expected a real CSS encoder")
    out = [f"QUBIT_COORDS({x},{y}) {q}" for q, (x, y) in enumerate(coordinates)]
    # A generic non-stabilizer input exercises both central outcomes. The
    # encoder maps its last input wire to the all-data logical X/Z pair.
    out += [f"H {n - 1}", f"T {n - 1}", f"H {n - 1}", f"T {n - 1}", str(encoder)]
    targets = " ".join(map(str, range(n)))
    out += [f"DEPOLARIZE1({probability}) {targets}"]

    def layer(inverse=False):
        for sign in (1, -1):
            gate = "T" if (sign == 1) != inverse else "T_DAG"
            out.append(gate + " " + " ".join(str(q) for q in range(n) if signs[q] == sign))
        out.append(f"DEPOLARIZE1({probability}) {targets}")

    layer()
    # The inverse traversal spreads X on the root to all data qubits.
    for a, b in reversed(tree):
        out += [f"CX {a} {b}", f"DEPOLARIZE2({probability}) {a} {b}"]
    out += [f"MX({probability}) 0", "RX 0", f"Z_ERROR({probability}) 0"]
    for a, b in tree:
        out += [f"CX {a} {b}", f"DEPOLARIZE2({probability}) {a} {b}"]
    layer(inverse=True)
    out.append(
        f"MPP({probability}) "
        + " ".join(
            "*".join(f"{'IXYZ'[axis]}{q}" for q, axis in enumerate(p) if axis)
            for p in [stim.PauliString("Y" * n)] + checks
        )
    )
    for offset in range(1, 2 * len(rows) + 1):
        out.append(f"DETECTOR rec[-{offset}]")
    out.append(f"OBSERVABLE_INCLUDE(0) rec[-{2 * len(rows) + 1}]")
    return "\n".join(out) + "\n"
