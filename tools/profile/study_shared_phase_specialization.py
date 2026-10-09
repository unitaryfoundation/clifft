"""Measure reusable phase analysis and a conservative backend choice."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import resource
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from automatic_specialization import analyze
from shared_phase_specialization import SharedPhase
from study_automatic_specialization import candidates, sample_parities

import clifft


def features(source: str, records: np.ndarray) -> np.ndarray:
    rng = random.Random(73581)
    columns = [[i] for i in range(records.shape[1])]
    columns += [rng.sample(range(records.shape[1]), min(5, records.shape[1])) for _ in range(64)]
    columns += [
        [t.value for t in node.targets]
        for node in clifft.parse(source).nodes
        if node.gate.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}
    ]
    return np.column_stack(
        [
            1.0 - 2.0 * np.logical_xor.reduce(records[:, col], axis=1, initial=False)
            for col in columns
        ]
    )


def baseline(source: str, args: Any) -> dict[str, Any]:
    started = perf_counter()
    hir, info = analyze(source)
    if info["peak_width"] > args.max_width:
        return {**info, "status": "width_budget", "analysis_seconds": perf_counter() - started}
    program = clifft.lower(hir)
    compile_seconds = perf_counter() - started
    clifft.sample(program, shots=1, seed=73, threads=1)
    batches = {}
    for count in sorted({1, args.shots, args.expected_shots}):
        times = []
        for repeat in range(3):
            start = perf_counter()
            clifft.sample(program, shots=count, seed=731 + repeat, threads=1)
            times.append(perf_counter() - start)
        batches[str(count)] = statistics.median(times)
    return {
        **info,
        "status": "sampled",
        "compile_seconds": compile_seconds,
        "batch_seconds": batches,
    }


def fresh(shared: SharedPhase, args: Any) -> tuple[np.ndarray, dict[str, Any]]:
    rng, quantum_rng = random.Random(args.seed), random.Random(args.seed ^ 0x3938)
    times = []
    stages = dict.fromkeys(
        ("fault_draw", "prefix_and_controls", "correction_and_render", "trace_and_lower", "sample"),
        0.0,
    )
    widths, t_counts, histories, arrays = [], [], [], []
    for _ in range(args.shots):
        start = perf_counter()
        history = shared.model.draw(rng)
        drawn = perf_counter()
        controls = shared.controls(history)
        controlled = perf_counter()
        text = shared.render(controls, quantum_rng)
        rendered = perf_counter()
        hir = clifft.trace(clifft.parse(text))
        if clifft.active_width_trace(hir)["peak"] > args.max_width:
            raise ValueError("Reduced computation exceeds the execution width budget")
        program = clifft.lower(hir)
        lowered = perf_counter()
        result = clifft.sample(program, shots=1, seed=quantum_rng.getrandbits(64), threads=1)
        ended = perf_counter()
        times.append(ended - start)
        for name, duration in zip(
            stages,
            (
                drawn - start,
                controlled - drawn,
                rendered - controlled,
                lowered - rendered,
                ended - lowered,
            ),
        ):
            stages[name] += duration
        sample_parities(shared.model.source, result)
        widths.append(program.peak_active_width)
        t_counts.append(hir.num_t_gates)
        histories.append(history)
        arrays.append(result.measurements[0].copy())
    records = np.stack(arrays)
    return records, {
        "shots": args.shots,
        "mean_seconds_per_shot": statistics.mean(times),
        "median_seconds_per_shot": statistics.median(times),
        "stage_seconds": stages,
        "widths": dict(Counter(widths)),
        "t_counts": dict(Counter(t_counts)),
        "fault_counts": dict(sorted(Counter(map(len, histories)).items())),
        "unique_histories": len(set(histories)),
        "history_sha256": hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
        "record_sha256": hashlib.sha256(records.tobytes()).hexdigest(),
    }


def study(source: str, args: Any) -> dict[str, Any]:
    ordinary = baseline(source, args)
    started = perf_counter()
    try:
        shared = SharedPhase(source)
    except ValueError as error:
        return {
            "ordinary": ordinary,
            "shared": {
                "status": "unsupported",
                "reason": str(error),
                "analysis_seconds": perf_counter() - started,
            },
            "choice": "ordinary" if ordinary["status"] == "sampled" else "neither_within_budget",
        }
    setup = perf_counter() - started
    # Preview before lowering so a failed reduction cannot allocate a wide state.
    preview = shared.render(shared.controls(()), random.Random(19))
    raw = clifft.trace(clifft.parse(preview))
    width = clifft.active_width_trace(raw)["peak"]
    if width > args.max_width:
        return {
            "ordinary": ordinary,
            "shared": {"status": "width_budget", "peak_width": width, **shared.metadata()},
            "choice": "ordinary" if ordinary["status"] == "sampled" else "neither_within_budget",
        }
    records, timing = fresh(shared, args)
    result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "ordinary": ordinary,
        "shared": {"status": "sampled", "analysis_seconds": setup, **shared.metadata(), **timing},
    }
    conditional_cost = setup + args.expected_shots * timing["mean_seconds_per_shot"]
    ordinary_cost = None
    if ordinary["status"] == "sampled":
        ordinary_cost = (
            ordinary["compile_seconds"] + ordinary["batch_seconds"][str(args.expected_shots)]
        )
    result["selection"] = {
        "expected_shots": args.expected_shots,
        "ordinary_seconds": ordinary_cost,
        "conditional_seconds": conditional_cost,
        "choice": "shared"
        if ordinary_cost is None or conditional_cost < ordinary_cost
        else "ordinary",
        "reason": "ordinary_exceeds_width_budget"
        if ordinary_cost is None
        else "measured_setup_plus_sampling_cost",
        "scope": (
            "Research comparison after profiling; analysis and measurement probes "
            "are not a production policy"
        ),
    }
    merlin = importlib.import_module("merlin")
    try:
        start = perf_counter()
        sampler = merlin.CircuitSampler(source, seed=args.seed + 7)
        initialized = perf_counter()
        reference = sampler.sample(args.shots)
        elapsed = perf_counter() - initialized
        a, b = features(source, records), features(source, reference.measurements)
        variance = a.var(axis=0, ddof=1) / args.shots + b.var(axis=0, ddof=1) / args.shots
        delta = np.maximum(0, abs(a.mean(axis=0) - b.mean(axis=0)) - 4 / args.shots)
        score = delta / np.maximum(np.sqrt(variance), 1e-12)
        if max(score) > 7:
            raise AssertionError("Fresh shared/reference output moments differ")
        result["merlin"] = {
            "status": "sampled",
            "setup_seconds": initialized - start,
            "seconds_per_shot": elapsed / args.shots,
            "moment_features": a.shape[1],
            "maximum_score": float(max(score)),
            "scope": "Fresh-noise bug check, not rate equivalence",
        }
    except (ValueError, RuntimeError, merlin.MeasurementError) as error:
        result["merlin"] = {"status": "unsupported", "reason": str(error)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--expected-shots", type=int, default=1024)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=791929)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.shots < 32 or args.expected_shots < 1 or not 0 <= args.max_width <= 16:
        parser.error(
            "Require at least 32 fresh shots, positive expected shots, and width at most 16"
        )
    panel = candidates(args.merlin_checkout)
    selected = args.cases or [name for name in panel if not name.startswith("bt81")]
    if set(selected) - panel.keys():
        parser.error("Unknown panel case")
    result: dict[str, Any] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("shared_phase_specialization.py"))
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "affinity": sorted(os.sched_getaffinity(0)),
        "cases": {},
    }
    for name in selected:
        row = study(panel[name][1], args)
        result["cases"][name] = {"family": panel[name][0], **row}
        result["whole_process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "case": name,
                    "status": row["shared"]["status"],
                    "metadata": {
                        key: value
                        for key, value in row["shared"].items()
                        if key
                        in {"quantum_core_qubits", "residual_t", "mean_seconds_per_shot", "reason"}
                    },
                    "selection": row.get("selection", row.get("choice")),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
