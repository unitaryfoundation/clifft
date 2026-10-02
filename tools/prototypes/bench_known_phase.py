"""Compare baseline and phase-pass modes on a circuits.csv corpus.

Scan timings are single observations, not a compilation benchmark. Optional
sampling compares the two phase modes with compilation and warmup excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import clifft


def manager(known: bool | None) -> tuple[clifft.HirPassManager, Any]:
    passes = clifft.HirPassManager()
    passes.add(clifft.PeepholeFusionPass())
    phase = None if known is None else clifft.PhasePolynomialPass(use_known_stabilizers=known)
    if phase is not None:
        passes.add(phase)
    passes.add(clifft.StatevectorSqueezePass())
    passes.add(clifft.ActiveWidthSchedulePass())
    return passes, phase


def sample(programs: list[Any], target: float, repeats: int) -> dict[str, Any]:
    shots = []
    for program in programs:
        clifft.sample_survivors(program, shots=1, threads=1, batch_size=1, seed=72)
        start = time.perf_counter()
        clifft.sample_survivors(program, shots=100, threads=1, batch_size=1, seed=72)
        elapsed = time.perf_counter() - start
        shots.append(max(1, min(1_000_000, math.ceil(target * 100 / elapsed))))
    seconds: list[list[float]] = [[], []]
    for repeat in range(repeats):
        for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
            start = time.perf_counter()
            clifft.sample_survivors(
                programs[arm], shots=shots[arm], threads=1, batch_size=1, seed=20261002 + repeat
            )
            seconds[arm].append(time.perf_counter() - start)
    per_shot = [statistics.median(values) / n for values, n in zip(seconds, shots)]
    return dict(
        shots=shots,
        seconds=seconds,
        seconds_per_shot=per_shot,
        speedup=per_shot[0] / per_shot[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-circuits", nargs="*", type=int, default=[])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--target-seconds", type=float, default=0.12)
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.target_seconds) or args.target_seconds <= 0:
        parser.error("repeats and target-seconds must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.corpus / "circuits.csv").open(newline="") as handle:
        entries = list(csv.DictReader(handle))
    selected = {f"circuits/circuit_{n:04}.stim" for n in args.sample_circuits}
    if selected - {entry["circuit"] for entry in entries}:
        parser.error("a selected circuit is absent from the manifest")
    rows: list[dict[str, Any]] = []
    sampling: list[dict[str, Any]] = []
    for entry in entries:
        source = (args.corpus / entry["circuit"]).read_text()
        row: dict[str, Any] = dict(entry)
        programs = []
        for name, known in (("baseline", None), ("current", False), ("known", True)):
            passes, phase = manager(known)
            start = time.perf_counter()
            program = clifft.compile(
                source,
                hir_passes=passes,
                postselection_mask=[1] * int(entry["postselection_detectors"]),
                normalize_syndromes=False,
            )
            elapsed = time.perf_counter() - start
            programs.append(program)
            row[name] = dict(
                peak=program.peak_active_width,
                actions=program.num_actions,
                compile_s=elapsed,
                applied=phase.applied if phase is not None else False,
                output_t=phase.output_t_count if phase is not None else None,
            )
        rows.append(row)
        if entry["circuit"] in selected:
            result = dict(
                circuit=entry["circuit"],
                peak=[p.peak_active_width for p in programs[1:]],
                **sample(programs[1:], args.target_seconds, args.repeats),
            )
            sampling.append(result)
            (args.output / "sampling.json").write_text(json.dumps(sampling, indent=2) + "\n")
        if len(rows) % 100 == 0:
            print(f"Scanned {len(rows)}/{len(entries)}", flush=True)
    (args.output / "scan.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary: dict[str, Any] = dict(
        version=clifft.__version__,
        platform=platform.platform(),
        isa=clifft.runtime_isa(),
        circuits=len(rows),
        sampling_circuits=sorted(selected),
        sampling_repeats=args.repeats,
        sampling_target_s=args.target_seconds,
        sampling_threads=1,
        sampling_batch_size=1,
    )
    for name in ("current", "known"):
        summary[name] = dict(
            applied=sum(r[name]["applied"] for r in rows),
            lower_peak=sum(r[name]["peak"] < r["baseline"]["peak"] for r in rows),
            higher_peak=sum(r[name]["peak"] > r["baseline"]["peak"] for r in rows),
            peak_one=sum(r[name]["peak"] == 1 for r in rows),
            total_t=sum(r[name]["output_t"] for r in rows),
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
