"""Shared stabilizer constraints certify entanglement of nonzero coherent sums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import stim
from clifford_branches import paired_hook_histories
from fold_cultivation import Reconstruction
from fold_frame_screen import multiple_hook_histories, rank, snapshots


def bits(pauli):
    x, z = pauli.to_numpy()
    return sum(int(v) << q for q, v in enumerate(x)) | sum(
        int(v) << (q + len(pauli)) for q, v in enumerate(z)
    )


def intersection(left, right):
    """Intersect signed positive-eigenvalue stabilizer groups, not just supports."""
    if not left:
        return []
    pivots: dict[int, tuple[int, stim.PauliString]] = {}
    for generator in right:
        p = generator.copy()
        v = bits(p)
        while v:
            k = v.bit_length() - 1
            if k in pivots:
                row, pauli = pivots[k]
                v ^= row
                p *= pauli
            else:
                pivots[k] = v, p
                break
    residuals: dict[int, tuple[int, int]] = {}
    candidates = []
    for j, p in enumerate(left):
        v = bits(p)
        for k in sorted(pivots, reverse=True):
            if (v >> k) & 1:
                v ^= pivots[k][0]
        witness = 1 << j
        while v:
            k = v.bit_length() - 1
            if k in residuals:
                row, combination = residuals[k]
                v ^= row
                witness ^= combination
            else:
                residuals[k] = v, witness
                break
        if not v:
            product = stim.PauliString(len(p))
            for k, original in enumerate(left):
                if (witness >> k) & 1:
                    product *= original
            candidates.append(product)
    result = []
    negative = None
    for p in candidates:
        difference, v = p.copy(), bits(p)
        for k in sorted(pivots, reverse=True):
            if (v >> k) & 1:
                row, generator = pivots[k]
                v ^= row
                difference *= generator
        if v or difference.sign not in (1, -1):
            raise ValueError("invalid stabilizer intersection")
        if difference.sign == 1:
            result.append(p)
        elif negative is None:
            negative = p
        else:
            result.append(p * negative)
    return result


def common_stabilizers(state, frame):
    common = None
    for term in state.terms:
        simulator = term.tableau.copy()
        simulator.do_tableau(frame, list(range(state.width)))
        group = simulator.canonical_stabilizers()
        common = group if common is None else intersection(common, group)
    return common or []


def lower_bounds(common, width, order):
    if sorted(order) != list(range(width)):
        raise ValueError("incomplete order")
    rows = [bits(p) for p in common]
    x = [r & ((1 << width) - 1) for r in rows]
    z = [r >> width for r in rows]
    mask = 0
    bounds = []
    for q in order[:-1]:
        mask |= 1 << q
        commutations = [
            sum(
                (((a & d & mask) ^ (b & c & mask)).bit_count() % 2) << j
                for j, (c, d) in enumerate(zip(x, z, strict=True))
            )
            for a, b in zip(x, z, strict=True)
        ]
        count = rank(commutations)
        if count % 2:
            raise ValueError("restricted commutation rank must be even")
        bounds.append(1 << (count // 2))
    return bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    r = Reconstruction(study["distance"]).build()
    frame = stim.Tableau.from_circuit(stim.Circuit(study["frame_circuit"]))
    cases = [("ideal", [s.operations for s in r.stages])]
    cases += paired_hook_histories(r)[:1] + multiple_hook_histories(r)
    results = []
    for name, history in cases:
        points, _ = snapshots(r, history)
        first = points[0].stage
        source = next(c for c in study["cases"] if c["case"] == name)
        peak = max(source["snapshots"], key=lambda p: p["max_bond_upper"])
        selected = {
            (first, f"gate_{len(r.stages[first].operations)}"),
            (peak["stage"], peak["point"]),
        }
        for point in points:
            if (point.stage, point.point) not in selected:
                continue
            norm = point.state.expectation()
            if norm <= 1e-12:
                raise ValueError("certificate requires a nonzero checkpoint state")
            common = common_stabilizers(point.state, frame)
            lower = lower_bounds(common, study["width"], study["selected_order"])
            prior = next(
                p
                for p in source["snapshots"]
                if p["stage"] == point.stage and p["point"] == point.point
            )
            results.append(
                {
                    "case": name,
                    "stage": point.stage,
                    "point": point.point,
                    "norm": norm,
                    "common_stabilizers": len(common),
                    "bond_lower": lower,
                    "max_bond_lower": max(lower),
                    "max_bond_upper": prior["max_bond_upper"],
                }
            )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in row.items() if k != "bond_lower"} for row in results], indent=2
        )
    )


if __name__ == "__main__":
    main()
