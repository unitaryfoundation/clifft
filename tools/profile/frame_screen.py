"""Bounded offline search for exact tensor frames on saved residual states.

This is a dense-state diagnostic. It never truncates a state, never enters
Clifft execution, and does not implement an MPS backend or a trajectory policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import stim
from residual_ranks import singular_values


def representatives() -> list[tuple[str, np.ndarray]]:
    """Twenty two-qubit Cliffords modulo local Cliffords on the output."""
    found = {}
    for tableau in stim.Tableau.iter_all(2, unsigned=True):
        inverse = tableau.inverse()
        groups = []
        for q in range(2):
            x, z = inverse.x_output(q), inverse.z_output(q)
            groups.append(tuple(sorted(tuple(p[k] for k in range(2)) for p in (x, z, x * z))))
        key = tuple(groups)
        if key not in found:
            matrix = tableau.to_unitary_matrix(endian="little").astype(complex)
            # Stim exports complex64. Clifford matrix columns have a uniform
            # power-of-two support; normalization restores double precision.
            matrix /= np.linalg.norm(matrix[:, 0])
            found[key] = (str(tableau.to_circuit()), matrix)
    if len(found) != 20:
        raise ValueError("unexpected local Clifford quotient")
    # Stabilizer synthesis may choose a different global phase. Only the
    # identity's placement matters to ensure exact ties keep the current frame.
    result = list(found.values())
    identity = next(i for i, (_, u) in enumerate(result) if abs(np.trace(u)) > 3.999)
    result.insert(0, result.pop(identity))
    return result


def apply(state: np.ndarray, width: int, a: int, b: int, matrix: np.ndarray) -> np.ndarray:
    axes = [width - 1 - b, width - 1 - a]
    axes += [q for q in range(width) if q not in axes]
    tensor = state.reshape([2] * width).transpose(axes)
    transformed = (matrix @ tensor.reshape(4, -1)).reshape([2] * width)
    return np.asarray(transformed.transpose(np.argsort(axes)).reshape(-1))


def spectrum(state: np.ndarray, width: int, left: list[int]) -> tuple[int, float]:
    values = singular_values(state, width, left)
    rank = int(np.count_nonzero(values > 1e-12 * values[0]))
    probabilities = values**2
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log2(positive)))
    return rank, entropy


def profile(state: np.ndarray, width: int, order: list[int]) -> dict:
    values = [singular_values(state, width, order[:cut]) for cut in range(1, width)]
    ranks = {
        str(tol): [int(np.count_nonzero(s > tol * s[0])) for s in values]
        for tol in (1e-8, 1e-10, 1e-12, 1e-14)
    }
    bonds = [1, *ranks["1e-12"], 1]
    return {
        "ranks": ranks,
        "maximum_rank": max(bonds),
        "mps_coefficients": sum(2 * a * b for a, b in zip(bonds[:-1], bonds[1:], strict=True)),
        "dense_coefficients": len(state),
    }


def choose_order(state: np.ndarray, width: int, candidates: int) -> list[int]:
    rng = np.random.default_rng(8394)
    best = list(range(width))
    metric = profile(state, width, best)
    cost = metric["mps_coefficients"]
    for _ in range(candidates):
        order = [int(q) for q in rng.permutation(width)]
        candidate = profile(state, width, order)["mps_coefficients"]
        if candidate < cost:
            best, cost = order, candidate
    return best


def search(
    state: np.ndarray,
    width: int,
    order: list[int],
    gates: list[tuple[str, np.ndarray]],
    sweeps: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    result = state.copy()
    frame = []
    for sweep in range(sweeps):
        cuts = list(range(1, width))
        if sweep % 2:
            cuts.reverse()
        changed = False
        for cut in cuts:
            a, b = order[cut - 1 : cut + 1]
            left = order[:cut]
            best_rank, best_entropy = spectrum(result, width, left)
            selected, best = 0, result
            for j, (_, matrix) in enumerate(gates[1:], 1):
                candidate = apply(result, width, a, b, matrix)
                rank, entropy = spectrum(candidate, width, left)
                if rank < best_rank or (rank == best_rank and entropy < best_entropy - 1e-10):
                    best_rank, best_entropy, selected, best = rank, entropy, j, candidate
            if selected:
                frame.append((a, b, selected))
                result = best
                changed = True
        if not changed:
            break
    return result, frame


def run(directory: Path, actions: list[int], sweeps: int, orders: int) -> dict:
    metadata = json.loads((directory / "snapshots.json").read_text())
    dtype = "<f8" if metadata["byte_order"] == "little" else ">f8"
    gates = representatives()
    rows = []
    for action in actions:
        items = sorted(
            [x for x in metadata["snapshots"] if x["action"] == action], key=lambda x: x["sample"]
        )
        if len(items) < 2 or items[0]["sample"] != 0:
            raise ValueError("need training sample zero and independent held-out snapshots")
        width = items[0]["width"]
        if not 2 <= width <= 16:
            raise ValueError("bounded dense diagnostic supports widths two through sixteen")
        states = []
        for item in items:
            raw = np.fromfile(directory / item["file"], dtype=dtype)
            state = raw[: 1 << width] + 1j * raw[1 << width :]
            if item["width"] != width or raw.size != 2 << width:
                raise ValueError("inconsistent snapshot widths")
            if abs(np.vdot(state, state).real - 1) > 1e-9:
                raise ValueError("unnormalized snapshot")
            states.append(state)
        start = time.perf_counter()
        order = choose_order(states[0], width, orders)
        ordered_at = time.perf_counter()
        _, frame = search(states[0], width, order, gates, sweeps)
        searched_at = time.perf_counter()
        samples = []
        for item, state in zip(items, states, strict=True):
            start_apply = time.perf_counter()
            changed = state.copy()
            for a, b, j in frame:
                changed = apply(changed, width, a, b, gates[j][1])
            apply_us = 1e6 * (time.perf_counter() - start_apply)
            recovered = changed.copy()
            for a, b, j in reversed(frame):
                recovered = apply(recovered, width, a, b, gates[j][1].conj().T)
            recovery_error = float(np.linalg.norm(recovered - state))
            if recovery_error > 1e-10:
                raise ValueError(f"frame inversion failed: {recovery_error}")
            samples.append(
                {
                    **item,
                    "sha256": hashlib.sha256((directory / item["file"]).read_bytes()).hexdigest(),
                    "baseline": profile(state, width, list(range(width))),
                    "reordered": profile(state, width, order),
                    "framed": profile(changed, width, order),
                    "frame_apply_us": apply_us,
                    "inverse_error": recovery_error,
                }
            )
        rows.append(
            {
                "action": action,
                "width": width,
                "training_sample": 0,
                "order": order,
                "frame": frame,
                "ordering_seconds": ordered_at - start,
                "frame_search_seconds": searched_at - ordered_at,
                "samples": samples,
            }
        )
    return {
        "sampling": metadata["sampling"],
        "order_candidates": orders,
        "maximum_sweeps": sweeps,
        "clifford_representatives": [circuit for circuit, _ in gates],
        "checkpoints": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--actions", type=int, nargs="+", required=True)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--orders", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sweeps < 1 or args.orders < 0:
        parser.error("sweeps must be positive and orders nonnegative")
    args.output.write_text(
        json.dumps(run(args.directory, args.actions, args.sweeps, args.orders), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
