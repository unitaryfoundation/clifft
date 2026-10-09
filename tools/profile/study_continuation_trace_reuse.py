"""Compare full tracing with fixed Clifford-continuation response reuse."""

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
from continuation_trace_reuse import ContinuationPhase, ContinuationWorker
from prefix_trace_reuse import PreparedPhase
from study_automatic_specialization import sample_parities, stress_histories
from study_factory_frontend import sources


def study(source: str, args: Any) -> dict[str, Any]:
    start = perf_counter()
    front = ContinuationPhase(source, args.exporter)
    result: dict[str, Any] = {
        "frontend_setup_seconds": perf_counter() - start,
        "reuse": front.reuse_info,
        "continuation": front.continuation_info,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    if not front.continuation_info["eligible"]:
        histories = stress_histories(front.model)
        for h in histories:
            assert front.rewrite(h, 319) == PreparedPhase.rewrite(front, h, 319)
        result["unchanged_fallback_checks"] = len(histories)
        return result
    modes = tuple(args.modes)
    times: dict[str, list[float]] = {m: [] for m in modes}
    host: dict[str, list[float]] = {m: [] for m in modes}
    stages: dict[str, dict[str, float]] = {m: defaultdict(float) for m in modes}
    composition: dict[str, dict[str, float]] = {m: defaultdict(float) for m in modes}
    optimizer: dict[str, dict[str, float]] = {m: defaultdict(float) for m in modes}
    guard: dict[str, float] = dict.fromkeys(modes, 0.0)
    statuses: dict[str, Counter[str]] = {m: Counter() for m in modes}
    widths: dict[str, Counter[int]] = {m: Counter() for m in modes}
    records: dict[str, list[str]] = {m: [] for m in modes}
    sizes: dict[str, list[int]] = {m: [] for m in modes}
    draws, histories = [], []
    checked = 0
    optimized_checked = 0
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x9811)
    setup_start = perf_counter()
    with ContinuationWorker(
        args.worker,
        front.optimized_prefix,
        max_width=args.max_width,
        phase=front.phase_required,
        continuation=front.template_source,
    ) as worker:
        result["worker_setup_seconds"] = perf_counter() - setup_start
        result["native_setup"] = worker.setup
        assert worker.setup["generators"] == front.generator_count
        for index in range(args.shots):
            start = perf_counter()
            history = front.model.draw(faults)
            draws.append(perf_counter() - start)
            histories.append(history)
            seed, sample_seed = quantum.getrandbits(64), quantum.getrandbits(64)
            replies = {}
            for mode in modes[index % len(modes) :] + modes[: index % len(modes)]:
                start = perf_counter()
                if mode == "fresh":
                    branch = front.rewrite(history, seed)
                    tail = branch.source[len(front.optimized_prefix) :]
                    host[mode].append(perf_counter() - start)
                    row = worker.run("fresh", tail, sample_seed, check=False)
                    header = f"fresh {sample_seed} {len(tail.splitlines())} 0\n"
                    sizes[mode].append(len(header.encode()) + len(tail.encode()))
                else:
                    payload = front.payload(history, seed)
                    host[mode].append(perf_counter() - start)
                    row = worker.instantiate(payload, sample_seed, mode=mode)
                    sizes[mode].append(row["request_bytes"])
                replies[mode] = row
                times[mode].append(row["roundtrip_seconds"])
                for key, value in row["stage_seconds"].items():
                    stages[mode][key] += value
                for key, value in row["composition_seconds"].items():
                    composition[mode][key] += value
                for key, value in row["optimizer_seconds"].items():
                    optimizer[mode][key] += value
                guard[mode] += row["squeeze_guard_seconds"]
                statuses[mode][row["squeeze_status"]] += 1
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
                verified = worker.instantiate(payload, sample_seed, reference=tail, mode=mode)
                checked += int(verified["checked"])
                optimized_checked += int(verified["optimized_checked"])
                for key in (
                    "width",
                    "t_count",
                    "measurements",
                    "detectors",
                    "observables",
                    "exp_vals",
                ):
                    assert replies[mode][key] == replies["fresh"][key], (mode, key)
                    assert verified[key] == replies["fresh"][key], (mode, key)
        stress = []
        selected = stress_histories(front.model)
        # Dense histories test the response algebra, not a truncated draw law.
        selected += [tuple((i, len(s.replacements) - 1) for i, s in enumerate(front.model.sites))]
        stress_inputs = [(h, 419, "fault_stress") for h in selected]
        stress_inputs += [((), seed, "no_fault") for seed in range(16)]
        stress_inputs += [(h, 419, "fixed_prefix_seed") for h in histories[:32]]
        for history, prefix_seed, group in stress_inputs:
            branch = front.rewrite(history, prefix_seed)
            for mode in modes[1:]:
                row = worker.instantiate(
                    front.payload(history, prefix_seed),
                    715,
                    reference=branch.source[len(front.optimized_prefix) :],
                    mode=mode,
                )
                checked += int(row["checked"])
                optimized_checked += int(row["optimized_checked"])
                stress.append(
                    {
                        "mode": mode,
                        "group": group,
                        "fault_weight": len(history),
                        "width": row["width"],
                        "squeeze_status": row["squeeze_status"],
                    }
                )
        result["native_peak_kib"] = row["native_peak_kib"]
    draw_mean = statistics.mean(draws)
    paired = {}
    for baseline, candidate in zip(modes, modes[1:]):
        saved = [
            a + b - c - d
            for a, b, c, d in zip(
                host[baseline], times[baseline], host[candidate], times[candidate]
            )
        ]
        paired[f"{baseline}_minus_{candidate}"] = {
            "mean_seconds": statistics.mean(saved),
            "median_seconds": statistics.median(saved),
            "block_mean_seconds": [
                statistics.mean(saved[i : i + 32]) for i in range(0, len(saved), 32)
            ],
        }
    result.update(
        draw_seconds_per_shot=draw_mean,
        paired_seconds_saved=paired,
        modes={
            m: {
                "seconds_per_attempt": draw_mean
                + statistics.mean(host[m])
                + statistics.mean(times[m]),
                "host_seconds_per_attempt": statistics.mean(host[m]),
                "worker_seconds_per_attempt": statistics.mean(times[m]),
                "stage_seconds_per_attempt": {k: v / args.shots for k, v in stages[m].items()},
                "composition_seconds_per_attempt": {
                    k: v / args.shots for k, v in composition[m].items()
                },
                "optimizer_seconds_per_attempt": {
                    k: v / args.shots for k, v in optimizer[m].items()
                },
                "squeeze_guard_seconds_per_attempt": guard[m] / args.shots,
                "squeeze_statuses": dict(statuses[m]),
                "mean_request_bytes": statistics.mean(sizes[m]),
                "widths": dict(widths[m]),
                "completed_shots": len(records[m]),
                "record_sha256": hashlib.sha256("".join(records[m]).encode()).hexdigest(),
            }
            for m in modes
        },
        histories_sha256=hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
        fault_weights=dict(Counter(map(len, histories))),
        exact_trace_checks=checked,
        exact_optimized_checks=optimized_checked,
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
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("fresh", "continuation", "diagonal", "squeeze"),
        default=["fresh", "continuation"],
    )
    args = parser.parse_args()
    if args.shots < 32 or not 0 <= args.max_width <= 16:
        parser.error("Require at least 32 shots and width budget at most sixteen")
    if args.modes[0] != "fresh" or len(args.modes) < 2 or len(set(args.modes)) != len(args.modes):
        parser.error("Require fresh first and at least one distinct reuse mode")
    panel = sources(args)
    result: dict[str, Any] = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("continuation_trace_reuse.py"),
                Path(__file__).with_name("prefix_trace_reuse.py"),
                Path(__file__).with_name("profile_prefix_trace_reuse.cpp"),
                Path(__file__).with_name("planning_reuse_audit.h"),
                Path(__file__).with_name("squeeze_schedule_reuse.h"),
            ]
        },
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
        "cases": {},
    }
    for name in (
        "quadcycle-noisy.stim",
        "steane-D.stim",
        "steane-E.stim",
        "bt27_direct_x",
        "bt27_scored",
        "cultivation_d3",
        "cultivation_d5",
        "unsupported_rotation",
    ):
        print(name, "starting", flush=True)
        row = result["cases"][name] = study(panel[name], args)
        print(
            name, {m: r["seconds_per_attempt"] for m, r in row.get("modes", {}).items()}, flush=True
        )
        result["host_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
