"""Diagnose ordinary compilation after fixing BT27's pre-CCZ Pauli faults.

This does not specialize measurement outcomes or implement a production cache.
Run with a development installation built from the checkout being investigated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import resource
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import clifft
import clifft._clifft_core as core

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/merlin_bt27.stim"
LOGICAL_QUBITS = (0, 1, 2, 27, 28, 29, 54, 55, 56)
Faults = tuple[tuple[int, str], ...]


def source(faults: Faults = (), *, noise: str = "") -> str:
    lines = FIXTURE.read_text().splitlines()
    first_t = next(i for i, line in enumerate(lines) if line.startswith("T "))
    injected = [f"{pauli} {qubit}" for qubit, pauli in faults]
    if noise:
        injected.append(noise + " " + " ".join(map(str, range(81))))
    return "\n".join(lines[:first_t] + injected + lines[first_t:]) + "\n"


def fault_id(faults: Faults) -> str:
    return "_".join(f"{pauli}{qubit}" for qubit, pauli in faults) or "identity"


def cases(
    seed: int, per_weight: int, draws: int, probability: float
) -> tuple[dict[str, Faults], dict[str, list[str]]]:
    rng = random.Random(seed)
    patterns: dict[str, Faults] = {}
    groups: dict[str, list[str]] = {}

    def add(group: str, faults: Faults) -> None:
        faults = tuple(sorted(faults))
        key = fault_id(faults)
        patterns[key] = faults
        groups.setdefault(group, []).append(key)

    add("no_fault", ())
    for q in range(81):
        for pauli in "XYZ":
            add("single_fault", ((q, pauli),))
    for weight in (2, 3, 5, 10, 40, 81):
        for _ in range(per_weight):
            add(
                f"weight_{weight}",
                tuple((q, rng.choice("XYZ")) for q in rng.sample(range(81), weight)),
            )
    # Each location is one categorical event; X, Y, and Z are mutually exclusive.
    for _ in range(draws):
        faults = []
        for q in range(81):
            value = rng.random()
            if value < probability:
                faults.append((q, "XYZ"[min(2, int(value * 3 / probability))]))
        add("sampled_depolarizing", tuple(faults))
    return patterns, groups


def optimize(text: str) -> tuple[Any, dict[str, Any]]:
    start = perf_counter()
    hir = clifft.trace(clifft.parse(text))
    traced = perf_counter()
    phase = clifft.PhasePolynomialPass()
    manager = clifft.HirPassManager()
    for pass_ in (
        clifft.PeepholeFusionPass(),
        phase,
        clifft.RotationSimplificationPass(),
        clifft.StatevectorSqueezePass(),
    ):
        manager.add(pass_)
    manager.run(hir)
    optimized = perf_counter()
    width = clifft.active_width_trace(hir)
    inspected = perf_counter()
    first_wide = next((i for i, k in enumerate(width["widths"]) if k > 9), None)
    return hir, {
        "t_count": hir.num_t_gates,
        "peak_width": width["peak"],
        "final_width": width["final"],
        "first_above_nine": (
            {
                "hir_index": first_wide,
                "source_lines": hir.source_map[first_wide],
                "operation": str(hir[first_wide]),
                "effect": width["effects"][first_wide],
            }
            if first_wide is not None
            else None
        ),
        "phase": {
            name: getattr(phase, name)
            for name in ("blocks_examined", "blocks_capped", "blocks_expanded", "blocks_reduced")
        },
        "parse_trace_seconds": traced - start,
        "optimizer_seconds": optimized - traced,
        "width_inspection_seconds": inspected - optimized,
    }


def profile(
    text: str, reference: dict[str, Any], repeats: int, shots: int, max_width: int
) -> dict[str, Any]:
    rows = []
    sample_rows = []
    summary = None
    for _ in range(repeats):
        hir, info = optimize(text)
        if info["peak_width"] <= max_width:
            start = perf_counter()
            program = clifft.lower(
                hir,
                expected_detectors=reference["detectors"],
                expected_observables=reference["observables"],
            )
            info["lower_seconds"] = perf_counter() - start
            assert program.peak_active_width == info["peak_width"]
            start = perf_counter()
            sampled = clifft.sample(program, shots=shots, seed=5719, threads=1, batch_size=1)
            sample_rows.append(perf_counter() - start)
            summary = {
                "measurement_columns": sampled.measurements.shape[1],
                "detector_columns": sampled.detectors.shape[1],
                "observable_columns": sampled.observables.shape[1],
                "shots_with_detector_events": int(np.any(sampled.detectors, axis=1).sum()),
            }
        else:
            info["lower_seconds"] = None
        rows.append(info)
    result = rows[-1]
    result["timings"] = {
        name: [row[name] for row in rows]
        for name in (
            "parse_trace_seconds",
            "optimizer_seconds",
            "width_inspection_seconds",
            "lower_seconds",
        )
    }
    for name in result["timings"]:
        del result[name]
    result["timings"]["sample_seconds"] = sample_rows
    result["sample_summary"] = summary
    return result


def summarize(keys: list[str], results: dict[str, Any]) -> dict[str, Any]:
    rows = [results[key] for key in keys]
    compile_times = [
        parse + optimize + lower
        for row in rows
        for parse, optimize, lower in zip(
            row["timings"]["parse_trace_seconds"],
            row["timings"]["optimizer_seconds"],
            row["timings"]["lower_seconds"],
        )
        if lower is not None
    ]
    sampling = [t for row in rows for t in row["timings"]["sample_seconds"]]
    return {
        "cases": len(keys),
        "unique_patterns": len(set(keys)),
        "width_counts": dict(sorted(Counter(row["peak_width"] for row in rows).items())),
        "t_count_range": [min(row["t_count"] for row in rows), max(row["t_count"] for row in rows)],
        "median_compile_seconds": statistics.median(compile_times) if compile_times else None,
        "median_sample_seconds": statistics.median(sampling) if sampling else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--per-weight", type=int, default=8)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-width", type=int, default=16)
    args = parser.parse_args()
    if not (0 <= args.probability <= 1):
        parser.error("probability must be between zero and one")
    if min(args.repeats, args.shots) < 1 or min(args.per_weight, args.draws) < 0:
        parser.error("repeats and shots must be positive; case counts must be non-negative")
    if not 0 <= args.max_width <= 20:
        parser.error("max-width must be between zero and 20")
    ideal_hir, _ = optimize(source())
    reference = clifft.compute_reference_syndrome(ideal_hir)
    patterns, groups = cases(args.seed, args.per_weight, args.draws, args.probability)
    # Warm the same code paths once; high-width controls are never executed.
    profile(source(), reference, 1, args.shots, args.max_width)
    results = {
        "unspecialized_depolarizing": profile(
            source(noise=f"DEPOLARIZE1({args.probability})"),
            reference,
            args.repeats,
            args.shots,
            args.max_width,
        )
    }
    start = perf_counter()
    for i, (key, faults) in enumerate(patterns.items()):
        results[key] = profile(source(faults), reference, args.repeats, args.shots, args.max_width)
        results[key]["faults"] = faults
        if i % 32 == 0 or i + 1 == len(patterns):
            print(
                f"{i + 1}/{len(patterns)} patterns; {key}: width {results[key]['peak_width']}",
                flush=True,
            )
    summary = {group: summarize(keys, results) for group, keys in groups.items()}
    document = {
        "metadata": {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "extension_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "arguments": vars(args) | {"output": str(args.output)},
            "sweep_seconds": perf_counter() - start,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "memory_scope": "one process; programs sequential; includes diagnostic Python state",
            "timing_scope": (
                "compile excludes width inspection; sampling includes executor preparation"
            ),
        },
        "reference": reference,
        "summary": summary,
        "groups": groups,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
