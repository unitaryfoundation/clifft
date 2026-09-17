"""Fault-conditioned frame screen using the previous held-out f7 histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import stim
from clifford_branches import materialize, multiply, paired_hook_histories, root
from fold_blocks import Fold, Protocol, syndrome_sector_histories
from fold_fault_frame import FramePlan
from fold_frame_screen import (
    Snapshot,
    basis,
    copy_state,
    framed_generators,
    multiple_hook_histories,
    profile,
    snapshots,
    support_profile,
)
from fold_growth import growth_fault_histories


def cases(protocol):
    r = protocol.reconstruction
    return (
        [("ideal", [s.operations for s in r.stages])]
        + [(f"natural_{s}", materialize(r, 0.001, s)[0]) for s in range(1000, 1032)]
        + [(f"stress_{s}", materialize(r, 0.01, s)[0]) for s in range(2000, 2008)]
        + paired_hook_histories(r)
        + multiple_hook_histories(r)
        + syndrome_sector_histories(protocol)
        + growth_fault_histories(r)[::21]
    )


def exact_same(left, right):
    difference = copy_state(left)
    for source in right.terms:
        term = source.copy()
        term.coefficient = multiply(term.coefficient, root(4))
        difference.terms.append(term)
    difference.merge()
    return not difference.terms


def study(prior):
    protocol = Protocol(7)
    r = protocol.reconstruction
    frame = stim.Tableau.from_circuit(stim.Circuit(prior["frame_circuit"]))
    order = prior["selected_order"]
    data = set(range(r.ancilla))
    plans = {}
    start = time.perf_counter()
    for _, fold, groups in protocol.groups:
        if isinstance(fold, Fold) and fold.plan.distance == 7:
            initial = [
                (min(fold.mapping[a], fold.mapping[b]), max(fold.mapping[a], fold.mapping[b]))
                for a, b in fold.plan.edges
            ]
            plans[groups[1][0]] = FramePlan(fold.core, initial)
    planning_seconds = time.perf_counter() - start
    rows, ideals = [], {}
    unions: dict[str, list[int]] = {}
    for name, history in cases(protocol):
        points, _ = snapshots(r, history)
        profiles, boundaries = [], []
        incoming = None
        for stage, plan in plans.items():
            stage_points = [p for p in points if p.stage == stage and p.point.startswith("gate_")]
            if not stage_points:
                continue
            faults = plan.layout.encode(history[stage])
            for point in stage_points:
                key = f"{stage}_{point.point}"
                stop = int(point.point.removeprefix("gate_"))
                correction = plan.evaluate(faults, incoming, stop)
                corrected = plan.inverse_state(point.state, correction)
                corrected_point = Snapshot(stage, point.point, corrected)
                metrics = profile(
                    framed_generators(corrected_point, frame), point.state.width, order
                )
                support = support_profile(corrected_point, frame)
                span = support.pop("basis")
                unions[key] = basis(unions.get(key, []) + span)
                if name == "ideal":
                    ideals[key] = corrected
                equal = (
                    exact_same(corrected, ideals[key])
                    if name.startswith(("stage", "multiple_hooks_"))
                    else None
                )
                if equal is False:
                    raise AssertionError(
                        f"hook correction did not recover ideal state: {name} {key}"
                    )
                source = next(c for c in prior["cases"] if c["case"] == name)
                original = next(
                    p
                    for p in source["snapshots"]
                    if p["stage"] == stage and p["point"] == point.point
                )
                profiles.append(
                    {
                        "stage": stage,
                        "point": point.point,
                        "uncorrected_bond_upper": original["max_bond_upper"],
                        "corrected_bond_upper": metrics["max_bond_upper"],
                        "corrected_mps_coefficients_upper": metrics["mps_coefficients_upper"],
                        "correction_edges": correction.edges.bit_count(),
                        "exactly_ideal": equal,
                        **support,
                    }
                )
            outgoing = plan.evaluate(faults, incoming)
            eligible = plan.data_only(outgoing, data)
            later = [s for s in plans if s > stage]
            if later:
                next_stage = min(later)
                # Eligibility includes the actual intervening instructions,
                # not just the name of the next fold or its syndrome contract.
                eligible &= all(
                    not set(op.targets) & data
                    for ops in history[stage + 1 : next_stage]
                    for op in ops
                )
                incoming = plan.remap(outgoing, plans[next_stage]) if eligible else None
            boundaries.append(
                {
                    "stage": stage,
                    "data_only": eligible,
                    "carried_to_next_core": bool(later) and eligible,
                }
            )
        rows.append({"case": name, "snapshots": profiles, "boundaries": boundaries})
        print(
            json.dumps(
                {
                    "case": name,
                    "snapshots": len(profiles),
                    "max_corrected_bond_upper": max(
                        (p["corrected_bond_upper"] for p in profiles), default=0
                    ),
                }
            ),
            flush=True,
        )
    return {
        "distance": 7,
        "selected_frame": prior["selected_frame"],
        "planning_seconds": planning_seconds,
        "plans": [
            {
                "stage": stage,
                "edge_slots": len(p.edges),
                "actions": len(p.actions),
                "cx_update_pairs": sum(len(a.updates) for a in p.actions),
                "fault_bits": len(p.layout.slots),
            }
            for stage, p in plans.items()
        ],
        "span_unions": [{"point": k, "dimension": len(v)} for k, v in unions.items()],
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior", type=Path, default=Path(__file__).parent / "research/fold_frames_f7_data.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = study(json.loads(args.prior.read_text()))
    result["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("fold_fault_frame.py", "study_fault_frames.py", "fold_frame_screen.py")
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
