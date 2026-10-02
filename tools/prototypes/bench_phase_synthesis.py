"""Compare phase synthesis before scheduling and as a final cleanup.

Requires the experimental preserve_parities option. Structural scan timings are
not an overhead benchmark. Optional paired sampling excludes compilation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import clifft

MODES = ("current", "projected", "late")


def apply(hir: Any, pass_: Any) -> None:
    manager = clifft.HirPassManager()
    manager.add(pass_)
    manager.run(hir)


def metrics(hir: Any) -> dict[str, Any]:
    trace = clifft.active_width_trace(hir)
    before = trace["initial"]
    work = 0
    for after, effect in zip(trace["widths"], trace["effects"]):
        if effect in (
            "rotation_neutral",
            "rotation_promote",
            "instrument_active",
            "instrument_activate",
        ):
            work += 2**after
        elif effect == "measure_active":
            work += 2**before
        before = after
    return dict(
        peak=trace["peak"],
        dense_work=work,
        log2_dense_work=math.log2(work) if work else None,
        t=hir.num_t_gates,
        ops=hir.num_ops,
    )


def compile_case(source: str, mask: list[int], mode: str) -> tuple[Any, dict[str, Any]]:
    hir = clifft.trace(clifft.parse(source))
    apply(hir, clifft.PeepholeFusionPass())
    phase = clifft.PhasePolynomialPass(
        use_known_stabilizers=True, preserve_parities=mode == "projected"
    )
    apply(hir, phase)
    stats: dict[str, Any] = dict(
        applied=phase.applied,
        input_t=phase.input_t_count,
        output_t=phase.output_t_count,
        phase=metrics(hir),
    )
    apply(hir, clifft.StatevectorSqueezePass())
    apply(hir, clifft.ActiveWidthSchedulePass())
    stats["scheduled"] = metrics(hir)
    if mode == "late":
        cleanup = clifft.PhasePolynomialPass(use_known_stabilizers=True, preserve_parities=True)
        apply(hir, cleanup)
        stats["cleanup_applied"] = cleanup.applied
    stats["final"] = metrics(hir)
    program = clifft.lower(hir, postselection_mask=mask)
    stats.update(peak=program.peak_active_width, actions=program.num_actions)
    assert stats["peak"] == stats["final"]["peak"]
    return program, stats


def sampling_pair(programs: list[Any], repeats: int, target: float) -> dict[str, Any]:
    shots = []
    for program in programs:
        clifft.sample_survivors(program, shots=1, threads=1, batch_size=1, seed=72)
        start = time.perf_counter()
        clifft.sample_survivors(program, shots=8, threads=1, batch_size=1, seed=72)
        elapsed = time.perf_counter() - start
        shots.append(max(1, min(1_000_000, math.ceil(target * 8 / elapsed))))
    seconds: list[list[float]] = [[], []]
    outcomes: list[list[dict[str, Any]]] = [[], []]
    for repeat in range(repeats):
        for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
            start = time.perf_counter()
            result = clifft.sample_survivors(
                programs[arm],
                shots=shots[arm],
                threads=1,
                batch_size=1,
                seed=20261003 + repeat,
            )
            seconds[arm].append(time.perf_counter() - start)
            outcomes[arm].append(dict(total=result.total_shots, passed=result.passed_shots))
    per_shot = [statistics.median(values) / n for values, n in zip(seconds, shots)]
    return dict(
        shots=shots,
        seconds=seconds,
        outcomes=outcomes,
        seconds_per_shot=per_shot,
        speedup=per_shot[0] / per_shot[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-changed", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--target-seconds", type=float, default=0.12)
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.target_seconds) or args.target_seconds <= 0:
        parser.error("repeats and target-seconds must be positive and finite")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sampling: list[dict[str, Any]] = []
    with (args.corpus / "circuits.csv").open(newline="") as handle:
        entries = list(csv.DictReader(handle))
    for entry in entries:
        source = (args.corpus / entry["circuit"]).read_text()
        mask = [1] * int(entry["postselection_detectors"])
        row: dict[str, Any] = dict(entry)
        programs = []
        for mode in MODES:
            program, row[mode] = compile_case(source, mask, mode)
            programs.append(program)
        current = row["current"]["final"]
        projected = row["projected"]["final"]
        row["candidate_choice"] = (
            "projected"
            if (projected["peak"], projected["dense_work"])
            < (current["peak"], current["dense_work"])
            else "current"
        )
        rows.append(row)
        if args.sample_changed:
            for index, mode in enumerate(MODES[1:], 1):
                if row[mode]["final"] == current:
                    continue
                sampling.append(
                    dict(
                        circuit=entry["circuit"],
                        mode=mode,
                        **sampling_pair(
                            [programs[0], programs[index]], args.repeats, args.target_seconds
                        ),
                    )
                )
                (args.output / "sampling.json").write_text(json.dumps(sampling, indent=2) + "\n")
        if len(rows) % 100 == 0:
            print(f"Scanned {len(rows)}/{len(entries)}", flush=True)
    (args.output / "scan.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary: dict[str, Any] = dict(circuits=len(rows), version=clifft.__version__)
    for mode in ("projected", "late", "candidate_choice"):
        pairs = [
            (r["current"]["final"], r[r[mode] if mode == "candidate_choice" else mode]["final"])
            for r in rows
        ]
        summary[mode] = {
            metric: dict(
                lower=sum(b[metric] < a[metric] for a, b in pairs),
                same=sum(b[metric] == a[metric] for a, b in pairs),
                higher=sum(b[metric] > a[metric] for a, b in pairs),
            )
            for metric in ("peak", "dense_work", "t")
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
