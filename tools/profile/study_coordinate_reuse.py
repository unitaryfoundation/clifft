"""Measure coordinate queries and compare planner-only reuse policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from continuation_trace_reuse import ContinuationPhase, ContinuationWorker
from prefix_trace_reuse import PreparedPhase
from study_automatic_specialization import stress_histories
from study_factory_frontend import sources

FIELDS = ("width", "t_count", "measurements", "detectors", "observables", "exp_vals")


def study(source: str, args: Any) -> dict[str, Any]:
    start = perf_counter()
    front = ContinuationPhase(source, args.exporter)
    result: dict[str, Any] = {
        "frontend_setup_seconds": perf_counter() - start,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "continuation": front.continuation_info,
        "physical_qubits": front.model.num_qubits,
    }
    if not front.continuation_info["eligible"]:
        histories = stress_histories(front.model)
        for h in histories:
            assert front.rewrite(h, 319) == PreparedPhase.rewrite(front, h, 319)
        return result | {"unchanged_fallback_checks": len(histories)}
    modes = tuple(args.modes)
    times: dict[str, list[float]] = {m: [] for m in modes}
    stages: dict[str, dict[str, float]] = {m: defaultdict(float) for m in modes}
    counters: dict[str, Counter[str]] = {m: Counter() for m in modes}
    widths: dict[str, Counter[int]] = {m: Counter() for m in modes}
    records: dict[str, list[str]] = {m: [] for m in modes}
    checks: Counter[str] = Counter()
    audit_histogram: Counter[int] = Counter()
    audit_ends: Counter[str] = Counter()
    audit_totals: Counter[str] = Counter()
    audit_by_length: dict[int, Counter[str]] = defaultdict(Counter)
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x7815)
    histories = []
    with ContinuationWorker(
        args.worker,
        front.optimized_prefix,
        max_width=12,
        phase=front.phase_required,
        continuation=front.template_source,
    ) as worker:
        result["native_setup"] = worker.setup

        def verify(history: Any, prefix_seed: int, seed: int) -> None:
            branch = front.rewrite(history, prefix_seed)
            tail = branch.source[len(front.optimized_prefix) :]
            payload = front.payload(history, prefix_seed)
            reference = None
            for mode in modes:
                row = worker.instantiate(payload, seed, mode=mode, reference=tail)
                assert row["checked"] and row["optimized_checked"]
                checks["raw_and_optimized_hir"] += 1
                if mode != "squeeze":
                    assert row["plan_checked"]
                    checks["plans"] += 1
                    diagnostic = row["coordinate_diagnostics"]
                    assert diagnostic["coordinate_checks"] == (
                        0 if mode == "coordinate-native" else diagnostic["queries"]
                    )
                    checks["coordinates"] += diagnostic["coordinate_checks"]
                values = [row[k] for k in FIELDS]
                if reference is None:
                    reference = values
                assert values == reference

        for index in range(args.shots):
            start = perf_counter()
            history = front.model.draw(faults)
            draw_time = perf_counter() - start
            histories.append(history)
            prefix_seed, seed = quantum.getrandbits(64), quantum.getrandbits(64)
            replies = {}
            for mode in modes[index % len(modes) :] + modes[: index % len(modes)]:
                start = perf_counter()
                payload = front.payload(history, prefix_seed)
                row = worker.instantiate(payload, seed, mode=mode)
                times[mode].append(draw_time + perf_counter() - start)
                replies[mode] = [row[k] for k in FIELDS]
                assert row["width"] <= 12
                records[mode].append(row["measurements"])
                widths[mode][row["width"]] += 1
                for key, value in row["stage_seconds"].items():
                    stages[mode][key] += value
                if row["coordinate_diagnostics"] is not None:
                    counters[mode].update(
                        {k: v for k, v in row["coordinate_diagnostics"].items() if k != "intervals"}
                    )
            assert all(r == replies["squeeze"] for r in replies.values())
            verify(history, prefix_seed, seed)
            if index < args.audit_shots:
                row = worker.instantiate(payload, seed, mode="coordinate-audit")
                assert row["plan_checked"]
                assert [row[k] for k in FIELDS] == replies["squeeze"]
                checks["audit_plans"] += 1
                for interval in row["coordinate_diagnostics"]["intervals"]:
                    length = interval["queries"]
                    audit_histogram[length] += 1
                    audit_ends[interval["end"]] += 1
                    values = {k: v for k, v in interval.items() if k != "end"}
                    audit_totals.update(values)
                    audit_by_length[length].update(values)
        selected = stress_histories(front.model)
        selected += [tuple((i, len(s.replacements) - 1) for i, s in enumerate(front.model.sites))]
        for history in selected:
            verify(history, 419, 715)
        for seed in range(16):
            verify((), seed, 715)
        for history in histories[:32]:
            verify(history, 419, 715)
    result.update(
        modes={
            m: {
                "seconds_per_attempt": statistics.mean(times[m]),
                "stage_seconds_per_attempt": {k: v / args.shots for k, v in stages[m].items()},
                "coordinate_totals": dict(counters[m]),
                "widths": dict(widths[m]),
                "record_sha256": hashlib.sha256("".join(records[m]).encode()).hexdigest(),
            }
            for m in modes
        },
        paired_seconds_saved={
            m: {
                "mean": statistics.mean(a - b for a, b in zip(times["squeeze"], times[m])),
                "blocks": [
                    statistics.mean(
                        a - b for a, b in zip(times["squeeze"][i : i + 32], times[m][i : i + 32])
                    )
                    for i in range(0, args.shots, 32)
                ],
            }
            for m in modes[1:]
        },
        checks=dict(checks),
        audit={
            "histogram": dict(sorted(audit_histogram.items())),
            "ends": dict(audit_ends),
            "totals": dict(audit_totals),
            "by_length": {k: dict(v) for k, v in sorted(audit_by_length.items())},
        },
        histories_sha256=hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
        fault_weights=dict(Counter(map(len, histories))),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark-dir", "merlin-checkout", "worker", "exporter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--audit-shots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=271041)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "squeeze",
            "coordinate-native",
            "coordinate-identity",
            "coordinate-columns",
            "coordinate-inverse32",
        ),
        default=[
            "squeeze",
            "coordinate-native",
            "coordinate-identity",
            "coordinate-columns",
            "coordinate-inverse32",
        ],
    )
    args = parser.parse_args()
    if args.shots < 8 or not 0 <= args.audit_shots <= args.shots:
        parser.error("Require at least eight shots and no more audits than shots")
    if args.modes[0] != "squeeze" or len(set(args.modes)) != len(args.modes):
        parser.error("Require squeeze first and distinct policies")
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        *(
            Path(__file__).with_name(n)
            for n in (
                "coordinate_reuse.cpp",
                "coordinate_reuse.h",
                "profile_prefix_trace_reuse.cpp",
                "continuation_trace_reuse.py",
                "squeeze_schedule_reuse.h",
            )
        ),
        *(
            root / n
            for n in (
                "src/clifft/sampling/planner.cc",
                "src/clifft/sampling/planner_frame.cc",
                "src/clifft/tableau/tableau.cc",
            )
        ),
    ]
    result: dict[str, Any] = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_hashes": {
            str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
        "cases": {},
    }
    panel = sources(args)
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
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
