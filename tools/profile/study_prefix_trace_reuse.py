"""Measure preparation rendering and parsed/traced fragment reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
from compiled_prefix_reuse import ReusablePhase
from prefix_trace_reuse import PreparedPhase, TraceWorker
from study_automatic_specialization import sample_parities, stress_histories
from study_factory_frontend import sources


def study(source: str, args: Any) -> dict[str, Any]:
    start = perf_counter()
    front = PreparedPhase(source, args.exporter)
    result: dict[str, Any] = {
        "frontend_setup_seconds": perf_counter() - start,
        "reuse": front.reuse_info,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    if not front.reuse_info["eligible"]:
        histories = stress_histories(front.model)
        for h in histories:
            assert front.rewrite(h, 319) == ReusablePhase.rewrite(front, h, 319)
        result["unchanged_fallback_checks"] = len(histories)
        return result
    modes = ("fresh", "parsed", "traced")
    times: dict[str, list[float]] = {m: [] for m in modes}
    stages: dict[str, dict[str, float]] = {m: defaultdict(float) for m in modes}
    widths: dict[str, Counter[int]] = {m: Counter() for m in modes}
    records: dict[str, list[str]] = {m: [] for m in modes}
    rewriters: dict[str, list[float]] = {m: [] for m in ("before", "direct")}
    draws, histories = [], []
    checked = 0
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x9811)
    setup_start = perf_counter()
    with TraceWorker(
        args.worker, front.optimized_prefix, max_width=args.max_width, phase=front.phase_required
    ) as worker:
        result["worker_setup_seconds"] = perf_counter() - setup_start
        result["native_setup"] = worker.setup
        for index in range(args.shots):
            start = perf_counter()
            history = front.model.draw(faults)
            draws.append(perf_counter() - start)
            histories.append(history)
            seed, sample_seed = quantum.getrandbits(64), quantum.getrandbits(64)
            branches = {}
            for name in ("before", "direct") if index % 2 else ("direct", "before"):
                start = perf_counter()
                branches[name] = (
                    front.rewrite(history, seed)
                    if name == "direct"
                    else ReusablePhase.rewrite(front, history, seed)
                )
                rewriters[name].append(perf_counter() - start)
            assert branches["before"] == branches["direct"]
            tail = branches["direct"].source[len(front.optimized_prefix) :]
            replies = {}
            for mode in modes[index % 3 :] + modes[: index % 3]:
                row = worker.run(mode, tail, sample_seed, check=False)
                replies[mode] = row
                checked += int(row["checked"])
                times[mode].append(row["roundtrip_seconds"] - row["validation_seconds"])
                for key, value in row["stage_seconds"].items():
                    stages[mode][key] += value
                widths[mode][row["width"]] += 1
                if row["width"] <= args.max_width:
                    records[mode].append(row["measurements"])
                    sample_parities(
                        source,
                        SimpleNamespace(
                            **{
                                key: np.asarray([list(map(int, row[key]))], dtype="u1")
                                for key in ("measurements", "detectors", "observables")
                            }
                        ),
                    )
            for mode in modes[1:]:
                verified = worker.run(mode, tail, sample_seed)
                checked += int(verified["checked"])
                for key in (
                    "width",
                    "t_count",
                    "measurements",
                    "detectors",
                    "observables",
                    "exp_vals",
                ):
                    assert replies[mode][key] == replies["fresh"][key], (mode, key)
                    assert verified[key] == replies[mode][key], (mode, key)
        stress = []
        for history in stress_histories(front.model):
            branch = front.rewrite(history, 419)
            assert branch == ReusablePhase.rewrite(front, history, 419)
            row = worker.run("traced", branch.source[len(front.optimized_prefix) :], 715)
            checked += int(row["checked"])
            stress.append({"history": history, "width": row["width"]})
        result["native_peak_kib"] = row["native_peak_kib"]
    draw_mean = statistics.mean(draws)
    before, direct = (statistics.mean(rewriters[name]) for name in ("before", "direct"))
    result.update(
        draw_seconds_per_shot=draw_mean,
        rewrite_seconds_per_shot={"before": before, "direct": direct},
        baseline_seconds_per_attempt=draw_mean + before + statistics.mean(times["fresh"]),
        modes={
            m: {
                "seconds_per_attempt": draw_mean + direct + statistics.mean(times[m]),
                "worker_seconds_per_attempt": statistics.mean(times[m]),
                "stage_seconds_per_attempt": {k: v / args.shots for k, v in stages[m].items()},
                "widths": dict(widths[m]),
                "completed_shots": len(records[m]),
                "record_sha256": hashlib.sha256("".join(records[m]).encode()).hexdigest(),
            }
            for m in modes
        },
        histories_sha256=hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
        fault_weights=dict(Counter(map(len, histories))),
        exact_trace_checks=checked,
        identical_source_checks=args.shots + len(stress),
        stress=stress,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=271021)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.shots < 32 or not 0 <= args.max_width <= 16:
        parser.error("Require at least 32 shots and width budget at most sixteen")
    panel = sources(args)
    selected = args.cases or [
        "quadcycle-noisy.stim",
        "steane-D.stim",
        "steane-E.stim",
        "bt27_direct_x",
        "bt27_scored",
        "cultivation_d3",
        "cultivation_d5",
        "unsupported_rotation",
    ]
    result: dict[str, Any] = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("prefix_trace_reuse.py"),
                Path(__file__).with_name("profile_prefix_trace_reuse.cpp"),
            ]
        },
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
        "cases": {},
    }
    for name in selected:
        print(name, "starting", flush=True)
        result["cases"][name] = study(panel[name], args)
        row = result["cases"][name]
        print(
            name, {m: r["seconds_per_attempt"] for m, r in row.get("modes", {}).items()}, flush=True
        )
        result["host_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
