"""Bounded ideal-training and held-out fault screen on large fold states."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import stim
from clifford_branches import materialize, paired_hook_histories
from fold_blocks import Protocol, syndrome_sector_histories
from fold_cultivation import Reconstruction
from fold_frame_screen import (
    basis,
    conjugated_support,
    framed_generators,
    multiple_hook_histories,
    profile,
    rank,
    snapshots,
    support_profile,
)
from fold_growth import growth_fault_histories


def compact(metrics):
    return {k: v for k, v in metrics.items() if not k.startswith("bond_")}


def study(distance, natural, stress):
    r = Reconstruction(distance).build()
    begin = time.perf_counter()
    training, _ = snapshots(r, [s.operations for s in r.stages])
    snapshot_seconds = time.perf_counter() - begin
    width = training[0].state.width
    reference = training[0].state.terms[0].tableau
    begin = time.perf_counter()
    frames = {
        "physical": stim.Tableau(width),
        "canonical": stim.Tableau.from_stabilizers(reference.canonical_stabilizers()).inverse(),
        "transported": reference.current_inverse_tableau(),
    }
    construction_seconds = time.perf_counter() - begin
    orders = {
        "row": list(range(width)),
        "fold": r.check_mapping(distance) + list(range(r.ancilla, width)),
    }
    begin = time.perf_counter()
    candidates = []
    for name, frame in frames.items():
        branches = [framed_generators(snapshot, frame) for snapshot in training]
        for order_name, order in orders.items():
            profiles = [compact(profile(b, width, order)) for b in branches]
            candidates.append(
                {
                    "frame": name,
                    "order": order_name,
                    "maximum_bond_upper": max(p["max_bond_upper"] for p in profiles),
                    "maximum_mps_coefficients_upper": max(
                        p["mps_coefficients_upper"] for p in profiles
                    ),
                    "profiles": profiles,
                }
            )
    selected = min(candidates, key=lambda p: p["maximum_mps_coefficients_upper"])
    frame, order = frames[selected["frame"]], orders[selected["order"]]
    selection_seconds = time.perf_counter() - begin
    cases = [("ideal", [s.operations for s in r.stages])]
    cases += [
        (f"natural_{seed}", materialize(r, 0.001, seed)[0]) for seed in range(1000, 1000 + natural)
    ]
    cases += [
        (f"stress_{seed}", materialize(r, 0.01, seed)[0]) for seed in range(2000, 2000 + stress)
    ]
    cases += paired_hook_histories(r)
    cases += multiple_hook_histories(r)
    cases += syndrome_sector_histories(Protocol(distance))
    if distance == 7:
        cases += growth_fault_histories(r)[::21]
    rows = []
    training_spans = {}
    union_spans = {}
    for name, history in cases:
        start = time.perf_counter()
        points, state = (training, None) if name == "ideal" else snapshots(r, history)
        sampling_seconds = time.perf_counter() - start
        profiles = []
        for point in points:
            if not point.state.terms:
                continue
            start = time.perf_counter()
            framed = framed_generators(point, frame)
            frame_apply_seconds = time.perf_counter() - start
            metrics = compact(profile(framed, width, order))
            support = support_profile(point, frame)
            span = support.pop("basis")
            key = f"{point.stage}_{point.point}"
            if name == "ideal":
                training_spans[key] = span
                union_spans[key] = span
            support["directions_outside_training_span"] = rank(training_spans[key] + span) - len(
                training_spans[key]
            )
            union_spans[key] = basis(union_spans[key] + span)
            profiles.append(
                {
                    "stage": point.stage,
                    "point": point.point,
                    "terms": len(point.state.terms),
                    "frame_tableau_apply_seconds": frame_apply_seconds,
                    **metrics,
                    **support,
                }
            )
        row = {
            "case": name,
            "training": name == "ideal",
            "snapshots": profiles,
            "reference_seconds": sampling_seconds,
            "acceptance": 1.0 if state is None else state.expectation(),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "distance": distance,
                    "case": name,
                    "snapshots": len(profiles),
                    "max_bond_upper": max((p["max_bond_upper"] for p in profiles), default=0),
                }
            ),
            flush=True,
        )
    core = next(s.operations for s in r.stages if s.name == f"fold_core_d{distance}")
    return {
        "distance": distance,
        "width": width,
        "training_snapshot_seconds": snapshot_seconds,
        "frame_construction_seconds": construction_seconds,
        "selection_seconds": selection_seconds,
        "training_points": [{"stage": s.stage, "point": s.point} for s in training],
        "candidates": candidates,
        "selected_frame": selected["frame"],
        "selected_order": order,
        "frame_circuit": str(frame.to_circuit()),
        "generator_support": conjugated_support(frame, core, order),
        "span_unions": [
            {
                "point": key,
                "training_dimension": len(training_spans[key]),
                "held_out_union_dimension": len(span),
            }
            for key, span in union_spans.items()
        ],
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5, 7), default=7)
    parser.add_argument("--natural", type=int, default=32)
    parser.add_argument("--stress", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = study(args.distance, args.natural, args.stress)
    result["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "fold_frame_screen.py",
            "study_fold_frames.py",
            "clifford_branches.py",
            "fold_check.py",
            "fold_cultivation.py",
        )
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
