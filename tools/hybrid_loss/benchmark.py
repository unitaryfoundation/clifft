"""Single-core throughput slopes, retaining each timed call and its seed."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from experiment import python_model, run


def summarize(rows):
    counts = sorted({r["shots"] for r in rows})
    medians = {n: statistics.median(r["seconds"] for r in rows if r["shots"] == n) for n in counts}
    low, high = counts[0], counts[-1]
    slopes = []
    for repetition in sorted({r["repetition"] for r in rows}):
        times = {r["shots"]: r["seconds"] for r in rows if r["repetition"] == repetition}
        slopes.append((times[high] - times[low]) * 1e6 / (high - low))
    return {
        "us_per_shot": (medians[high] - medians[low]) * 1e6 / (high - low),
        "repeat_slopes_us": slopes,
        "median_seconds": medians,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["hybrid", "clifft-auto", "qdk"], required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    os.sched_setaffinity(0, {2})
    os.environ["RAYON_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for entry in json.loads((args.workloads / "manifest.json").read_text()):
        name = entry["name"]
        if args.case and name not in args.case:
            continue
        if args.backend == "qdk" and not (
            name.startswith("surface") and name.endswith("loss0.001")
        ):
            continue
        case = args.workloads / name
        text = (case / "circuit.stim").read_text()
        spec = json.loads((case / "model.json").read_text())
        rows = []
        if args.backend == "qdk":
            from qdk import stim as qdk_stim
            from qdk.simulation import run_qir

            qir, noise = qdk_stim.compile(text.replace("LOSS(", "LOSS_ERROR("))
            counts = [128, 512, 2048]

            def sample(shots, seed):
                return run_qir(qir, shots=shots, seed=seed, noise=noise, type="stabilizer")
        elif args.backend == "clifft-auto":
            from clifft import noncomp

            program = noncomp.compile(text, python_model(spec), final_status=False)
            # The slow trajectory cases are covered by the comparison/profile runs.
            if program.strategy != "fixed":
                continue
            counts = [2048, 8192, 32768]

            def sample(shots, seed):
                return program.sample(shots, seed=seed, threads=1, collect_stats=True)
        else:
            counts = [256, 1024, 4096]
            if name == "neutral_two_row_ldu_exact":
                counts = [32, 128, 512]
        for repetition in range(3):
            for shots in counts if repetition % 2 == 0 else reversed(counts):
                seed = 8001 + repetition * 97 + shots
                if args.backend == "hybrid":
                    info, _ = run(
                        args.binary,
                        args.output / f"{name}-{repetition}-{shots}",
                        text,
                        spec,
                        shots,
                        seed=seed,
                        capacity=20,
                        outputs=False,
                    )
                    elapsed = info["sampling_seconds"]
                    extra = {
                        "peak_width": info["peak_width"],
                        "continuations": info["continuations"],
                        "peak_rss_kib": info["peak_rss_kib"],
                    }
                else:
                    start = time.perf_counter()
                    result = sample(shots, seed)
                    elapsed = time.perf_counter() - start
                    extra = {}
                    if args.backend == "clifft-auto":
                        extra = {
                            "strategy": result.stats.strategy,
                            "peak_width": result.stats.peak_active_width,
                            "batch_size": result.stats.batch_size,
                        }
                row = {
                    "case": name,
                    "backend": args.backend,
                    "shots": shots,
                    "seed": seed,
                    "repetition": repetition,
                    "seconds": elapsed,
                    **extra,
                }
                rows.append(row)
                with (args.output / "calls.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
        summary = {
            "case": name,
            "backend": args.backend,
            **summarize(rows),
            "peak_widths": sorted({r["peak_width"] for r in rows if "peak_width" in r}),
        }
        summaries.append(summary)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
