"""Assess optimization schedules and plan dependencies across fault histories."""

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
from typing import Any

from continuation_trace_reuse import ContinuationPhase, ContinuationWorker
from prefix_trace_reuse import PreparedPhase
from study_automatic_specialization import stress_histories
from study_factory_frontend import sources

PASSES = ("PeepholeFusionPass", "RotationSimplificationPass", "StatevectorSqueezePass")


def key(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def common_prefix(values: list[list[Any]]) -> int:
    if not values:
        return 0
    for i in range(min(map(len, values))):
        if any(v[i] != values[0][i] for v in values[1:]):
            return i
    return min(map(len, values))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "histories": len(rows),
        "widths": dict(Counter(r["width"] for r in rows)),
    }
    result["passes"] = {}
    for name in PASSES:
        stages = [r["audit"][name] for r in rows]
        result["passes"][name] = {
            "changed": sum(s["changed"] for s in stages),
            "op_counts": dict(Counter(s["hir"]["ops"] for s in stages)),
            "t_counts": dict(Counter(s["hir"]["t_count"] for s in stages)),
            "distinct_origin_orders": len({key(s["hir"]["origins"]) for s in stages}),
            "distinct_unsigned_hir": len({key(s["hir"]["unsigned_rows"]) for s in stages}),
            "common_signed_prefix_ops": common_prefix([s["hir"]["signed_rows"] for s in stages]),
            "common_unsigned_prefix_ops": common_prefix(
                [s["hir"]["unsigned_rows"] for s in stages]
            ),
        }
    plans = [r["audit"]["plan"] for r in rows]
    result["plans"] = {
        "action_counts": dict(Counter(p["actions"] for p in plans)),
        "distinct_full_actions": len(
            {key((p["symbols"], p["full_actions"], p["frame"])) for p in plans}
        ),
        "distinct_without_constants": len(
            {key((p["symbols"], p["without_constants"])) for p in plans}
        ),
        "distinct_kind_width": len({key(p["kind_width"]) for p in plans}),
        "distinct_width_traces": len({key(r["audit"]["width_trace"]) for r in rows}),
        "common_full_prefix_actions": common_prefix([p["full_actions"] for p in plans]),
        "common_prefix_without_constants": common_prefix([p["without_constants"] for p in plans]),
        "common_kind_width_prefix": common_prefix([p["kind_width"] for p in plans]),
        "equal_width_variants": {},
    }
    for width in sorted({r["width"] for r in rows}):
        subset = [r["audit"]["plan"] for r in rows if r["width"] == width]
        result["plans"]["equal_width_variants"][width] = {
            "histories": len(subset),
            "distinct_without_constants": len(
                {key((p["symbols"], p["without_constants"])) for p in subset}
            ),
            "distinct_kind_width": len({key(p["kind_width"]) for p in subset}),
        }
    witness = None
    for i, left in enumerate(rows):
        for j in range(i + 1, len(rows)):
            right = rows[j]
            a, b = left["audit"]["plan"], right["audit"]["plan"]
            if left["width"] != right["width"] or a["kind_width"] == b["kind_width"]:
                continue
            index = common_prefix([a["kind_width"], b["kind_width"]])
            witness = {
                "row_indices": [i, j],
                "width": left["width"],
                "first_kind_width_difference": index,
                "left": a["full_actions"][index] if index < len(a["full_actions"]) else "end",
                "right": b["full_actions"][index] if index < len(b["full_actions"]) else "end",
                "left_faults": left["history"],
                "right_faults": right["history"],
                "prefix_seeds": [left["prefix_seed"], right["prefix_seed"]],
            }
            break
        if witness:
            break
    result["equal_width_witness"] = witness
    return result


def study(source: str, args: Any) -> dict[str, Any]:
    start = perf_counter()
    front = ContinuationPhase(source, args.exporter)
    result: dict[str, Any] = {
        "frontend_setup_seconds": perf_counter() - start,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "continuation": front.continuation_info,
    }
    if not front.continuation_info["eligible"]:
        selected = stress_histories(front.model)
        for h in selected:
            assert front.rewrite(h, 319) == PreparedPhase.rewrite(front, h, 319)
        result["unchanged_fallback_checks"] = len(selected)
        return result
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)
    original_histories = []
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x9811)
    exact = 0
    launcher = (
        ()
        if args.perf_data is None
        else (
            str(args.perf),
            "record",
            "-e",
            "cpu-clock:u",
            "-F",
            "499",
            "--call-graph",
            "dwarf",
            "-o",
            str(args.perf_data),
            "--",
        )
    )
    with ContinuationWorker(
        args.worker,
        front.optimized_prefix,
        max_width=args.max_width,
        phase=front.phase_required,
        continuation=front.template_source,
        launcher=launcher,
    ) as worker:
        result["native_setup"] = worker.setup

        def audit(history: Any, seed: int, sample_seed: int) -> dict[str, Any]:
            nonlocal exact
            payload = front.payload(history, seed)
            branch = front.rewrite(history, seed)
            row = worker.instantiate(
                payload,
                sample_seed,
                reference=branch.source[len(front.optimized_prefix) :],
                mode="audit",
            )
            exact += int(row["checked"])
            if row["width"] > args.max_width:
                raise AssertionError("Audit exceeded its width budget")
            row["history"], row["prefix_seed"] = history, seed
            return row

        for _ in range(args.shots):
            start = perf_counter()
            history = front.model.draw(faults)
            drawn = perf_counter()
            seed, sample_seed = quantum.getrandbits(64), quantum.getrandbits(64)
            original_histories.append(history)
            payload = front.payload(history, seed)
            constructed = perf_counter()
            row = worker.instantiate(payload, sample_seed, mode="diagonal")
            times["draw"].append(drawn - start)
            times["host"].append(constructed - drawn)
            times["worker"].append(row["roundtrip_seconds"])
            for name, value in row["optimizer_seconds"].items():
                times[name].append(value)
            for name, value in row["stage_seconds"].items():
                times[name].append(value)
            if args.perf_data is not None:
                continue
            checked = audit(history, seed, sample_seed)
            for name in (
                "width",
                "t_count",
                "measurements",
                "detectors",
                "observables",
                "exp_vals",
            ):
                assert checked[name] == row[name], name
            groups["ordinary"].append(checked)
        if args.perf_data is None:
            for seed in range(16):
                groups["no_faults"].append(audit((), 9137 + seed, 817))
            for history in original_histories[:32]:
                groups["fixed_prefix_seed"].append(audit(history, 419, 817))
            selected = stress_histories(front.model)
            selected += [
                tuple((i, len(s.replacements) - 1) for i, s in enumerate(front.model.sites))
            ]
            for history in selected:
                groups["stress"].append(audit(history, 419, 817))
    result["timings_seconds_per_attempt"] = {k: statistics.mean(v) for k, v in times.items()}
    result["seconds_per_attempt"] = sum(
        statistics.mean(times[k]) for k in ("draw", "host", "worker")
    )
    result["exact_trace_checks"] = exact
    result["timed_samples_matching_audit"] = len(groups["ordinary"])
    result["histories_sha256"] = hashlib.sha256(key(original_histories).encode()).hexdigest()
    if args.perf_data is not None:
        result["perf_data_sha256"] = hashlib.sha256(args.perf_data.read_bytes()).hexdigest()
        return result
    result["groups"] = {k: summarize(v) for k, v in groups.items()}
    result["all_groups"] = summarize([r for group in groups.values() for r in group])
    # Diagnostic snapshots deliberately remain outside the retained timing
    # samples and artifacts; summaries retain variation and concrete witnesses.
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark-dir", "merlin-checkout", "exporter", "worker", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=271021)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--perf", type=Path, default=Path("perf"))
    parser.add_argument(
        "--perf-data", type=Path, help="Profile one native worker without audit requests"
    )
    args = parser.parse_args()
    if args.shots < 32 or not 0 <= args.max_width <= 16:
        parser.error("Require at least 32 shots and width budget at most sixteen")
    if args.perf_data is not None and (not args.cases or len(args.cases) != 1):
        parser.error("Profiling requires exactly one selected case")
    panel = sources(args)
    result: dict[str, Any] = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("planning_reuse_audit.h"),
                Path(__file__).with_name("profile_prefix_trace_reuse.cpp"),
                Path(__file__).with_name("continuation_trace_reuse.py"),
                Path(__file__).with_name("prefix_trace_reuse.py"),
            ]
        },
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
        "cases": {},
    }
    for name in args.cases or [
        "quadcycle-noisy.stim",
        "steane-D.stim",
        "steane-E.stim",
        "bt27_direct_x",
        "bt27_scored",
        "cultivation_d3",
        "cultivation_d5",
        "unsupported_rotation",
    ]:
        print(name, "starting", flush=True)
        row = result["cases"][name] = study(panel[name], args)
        print(name, row.get("seconds_per_attempt"), flush=True)
        result["host_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
