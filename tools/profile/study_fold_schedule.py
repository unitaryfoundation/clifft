"""Validate full schedule variants against an independent coherent reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from clifford_branches import logical_tail_history, materialize, paired_hook_histories, run_history
from export_fold_blocks import Exporter
from fold_blocks import Boundary, Protocol, f7_diagnostic_histories, syndrome_sector_histories
from fold_cultivation import logical
from fold_schedule import SCHEDULES, reconstruction_with_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=SCHEDULES, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--histories", default=64, type=int)
    parser.add_argument("--stress-histories", default=32, type=int)
    args = parser.parse_args()
    if min(args.histories, args.stress_histories) < 0:
        parser.error("history counts must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reconstruction = reconstruction_with_schedule(7, args.schedule)
    protocol = Protocol(7, reconstruction=reconstruction)
    construction = time.perf_counter() - start
    text = reconstruction.text(0.001)
    (args.output / "circuit.stim").write_text(text)
    cases = [
        (f"natural_{seed}", materialize(reconstruction, 0.001, seed)[0])
        for seed in range(1000, 1000 + args.histories)
    ]
    cases += [
        (f"stress_{seed}", materialize(reconstruction, 0.01, seed)[0])
        for seed in range(2000, 2000 + args.stress_histories)
    ]
    cases += paired_hook_histories(reconstruction)
    cases += [("logical_tail", logical_tail_history(reconstruction))]
    cases += syndrome_sector_histories(protocol) + f7_diagnostic_histories(protocol)
    rows = []
    for name, stages in cases:
        result = protocol.evaluate(protocol.encode(stages))
        reference, _ = run_history(reconstruction, stages)
        probability = reference.expectation()
        xyz = (
            [reference.expectation(logical(7, axis)) / probability for axis in "XYZ"]
            if probability > 1e-12
            else None
        )
        error = abs(result.acceptance - probability)
        probe_error = (
            max(abs(a - b) for a, b in zip(result.logical_xyz, xyz, strict=True))
            if result.logical_xyz is not None and xyz is not None
            else 0
        )
        if error > 2e-12 or probe_error > 2e-12 or (xyz is None) != (result.logical_xyz is None):
            raise ValueError(f"coherent reference disagreement for {name}")
        rows.append(
            {
                "case": name,
                **vars(result),
                "reference_acceptance": probability,
                "reference_logical_xyz": xyz,
                "probability_error": error,
                "probe_error": probe_error,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    metadata = Exporter(protocol).write(cases, args.output / "native.cpp")
    boundaries = []
    for _, plan, _ in protocol.groups:
        if isinstance(plan, Boundary):
            boundaries.append(
                {
                    "before": plan.before.distance,
                    "after": plan.after.distance,
                    "measured_stabilizers": list(map(str, plan.measured_stabilizers)),
                    "syndrome_rows": plan.syndrome_rows,
                    "constraints": plan.constraints,
                }
            )
    (args.output / "validation.json").write_text(
        json.dumps(
            {
                "schedule": args.schedule,
                "construction_seconds": construction,
                "circuit_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "native_export": metadata,
                "boundaries": boundaries,
                "histories": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
