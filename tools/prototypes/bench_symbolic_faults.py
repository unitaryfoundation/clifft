"""Profile symbolic noise geometry and benchmark exact noisy reference samplers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import clifft
from tools.prototypes.bench_phase_polynomial import _compile
from tools.prototypes.phase_core_study import trace
from tools.prototypes.symbolic_fault_study import (
    AffineRecordSampler,
    BranchBank,
    FaultSite,
    affine_record_flips,
    compile_frame,
    gate_noise_geometry_stats,
)


def _locations(source: str) -> tuple[int, int]:
    rotations = [
        j for j, line in enumerate(source.splitlines()) if line.startswith(("T ", "T_DAG "))
    ]
    if not rotations:
        raise ValueError("the study expects a transversal T preparation")
    return rotations[0] - 1, rotations[-1]


def scan(corpus: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    with (corpus / "circuits.csv").open(newline="") as handle:
        entries = list(csv.DictReader(handle))
    for entry in entries:
        source = (corpus / entry["circuit"]).read_text()
        before, after = _locations(source)
        n = len(trace(source).rows)
        start = time.perf_counter()
        pre = compile_frame(source, [FaultSite(before, "X", q, 0.001) for q in range(n)])
        pre_seconds = time.perf_counter() - start
        start = time.perf_counter()
        post = compile_frame(source, [FaultSite(after, "Y", q, 0.001) for q in range(n)])
        post_seconds = time.perf_counter() - start
        assert pre.core_width == post.core_width
        assert post.geometry_rank() == 0
        items.append(
            {
                "circuit": entry["circuit"],
                "physical_qubits": n,
                "affine_pre_t": affine_record_flips(pre) is not None,
                "affine_post_t": affine_record_flips(post) is not None,
                "gate_target_noise": gate_noise_geometry_stats(source),
                "before_t_x": pre.stats(),
                "after_t_y": post.stats(),
                "precompute_seconds": [pre_seconds, post_seconds],
            }
        )
        if len(items) % 100 == 0:
            print(f"symbolic frames={len(items)}/{len(entries)}", flush=True)
    summary = {
        "circuits": len(items),
        "affine_pre_t": sum(i["affine_pre_t"] for i in items),
        "affine_post_t": sum(i["affine_post_t"] for i in items),
        "gate_target_geometry_rank_histogram": dict(
            sorted(Counter(i["gate_target_noise"]["geometry_rank"] for i in items).items())
        ),
        "gate_target_fixed_frame_prepared_core_histogram": dict(
            sorted(
                Counter(
                    i["gate_target_noise"]["fixed_frame_prepared_core_width"] for i in items
                ).items()
            )
        ),
        "models": (
            "Independent X faults on every physical qubit before first T; "
            "independent Y faults on every physical qubit after last T. "
            "Not full circuit-level depolarizing noise."
        ),
        "pre_t_geometry_rank_histogram": dict(
            sorted(Counter(i["before_t_x"]["geometry_rank"] for i in items).items())
        ),
        "post_t_geometry_rank_histogram": dict(
            sorted(Counter(i["after_t_y"]["geometry_rank"] for i in items).items())
        ),
        "median_precompute_ms": 1000 * statistics.median(i["precompute_seconds"][0] for i in items),
        "maximum_precompute_seconds": max(i["precompute_seconds"][0] for i in items),
        "maximum_s_parity_terms": max(i["before_t_x"]["s_parity_terms"] for i in items),
        "maximum_selector_words_64": max(i["before_t_x"]["selector_words_64"] for i in items),
        "one_core": {
            "circuits": sum(i["before_t_x"]["core_width"] == 1 for i in items),
            "geometry_rank_histogram": dict(
                sorted(
                    Counter(
                        i["before_t_x"]["geometry_rank"]
                        for i in items
                        if i["before_t_x"]["core_width"] == 1
                    ).items()
                )
            ),
        },
    }
    (output / "scan.json").write_text(json.dumps(items, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def _timing_runners(
    runners: list[Callable[[int, int], Any]], target: float, repeats: int
) -> dict[str, Any]:
    budgets = []
    for run in runners:
        count = 256
        for calibration in range(3):
            start = time.perf_counter()
            run(count, 171 + calibration)
            elapsed = time.perf_counter() - start
            proposed = min(1000000, max(16, math.ceil(count * target / elapsed)))
            if elapsed >= target / 2 or proposed == count:
                break
            count = proposed
        budgets.append(proposed)
    timings: list[list[float]] = [[], []]
    counts: list[list[dict[str, Any]]] = [[], []]
    for repeat in range(repeats):
        for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
            start = time.perf_counter()
            result = runners[arm](budgets[arm], 181 + repeat)
            timings[arm].append(time.perf_counter() - start)
            if isinstance(result, dict):
                counts[arm].append(
                    {**result, "observable_ones": result["observable_ones"].tolist()}
                )
            else:
                counts[arm].append(
                    {
                        "total_shots": result.total_shots,
                        "passed_shots": result.passed_shots,
                        "observable_ones": result.observable_ones.tolist(),
                    }
                )
    latency = [statistics.median(values) / shots for values, shots in zip(timings, budgets)]
    return {
        "shots": budgets,
        "timings_seconds": timings,
        "counts": counts,
        "baseline_shots_per_second": 1 / latency[0],
        "candidate_shots_per_second": 1 / latency[1],
        "speedup": latency[0] / latency[1],
    }


def _time_case(
    source: str, mask: list[int], selected: int, target: float, repeats: int
) -> dict[str, Any]:
    before, _ = _locations(source)
    n = len(trace(source).rows)
    rng = random.Random(195311)
    qubits = rng.sample(range(n), min(selected, n))
    start = time.perf_counter()
    frame = compile_frame(source, [FaultSite(before, "X", q, 0.03) for q in qubits])
    symbolic_seconds = time.perf_counter() - start
    start = time.perf_counter()
    baseline = _compile(frame.original(), mask)
    baseline_compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    bank = BranchBank(frame, mask)
    bank_compile_seconds = time.perf_counter() - start
    for program in {id(p): p for p in bank.programs}.values():
        clifft.sample_survivors(program, shots=1, seed=919, batch_size=1, threads=1)
    clifft.sample_survivors(baseline, shots=1, seed=919, batch_size=1, threads=1)
    runners = [
        lambda count, seed: clifft.sample_survivors(
            baseline, shots=count, seed=seed, batch_size=1, threads=1
        ),
        lambda count, seed: bank.sample_survivors(count, seed=seed, batch_size=1),
    ]
    timing = _timing_runners(runners, target, repeats)
    return {
        "noise": {"type": "independent X before first T", "qubits": qubits, "probability": 0.03},
        "symbolic": frame.stats(),
        "fault_patterns": len(bank.programs),
        "unique_programs": bank.unique_programs,
        "baseline_peak": baseline.peak_active_width,
        "bank_peak": max(program.peak_active_width for program in bank.programs),
        "symbolic_compile_seconds": symbolic_seconds,
        "baseline_compile_seconds": baseline_compile_seconds,
        "bank_compile_seconds": bank_compile_seconds,
        **timing,
    }


def benchmark(
    corpus: Path, output: Path, names: list[str], sites: int, target: float, repeats: int
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (corpus / "circuits.csv").open(newline="") as handle:
        entries = {Path(entry["circuit"]).stem: entry for entry in csv.DictReader(handle)}
    results = []
    for name in names:
        entry = entries["circuit_" + name]
        source = (corpus / entry["circuit"]).read_text()
        result = {
            "circuit": entry["circuit"],
            **_time_case(
                source, [1] * int(entry["postselection_detectors"]), sites, target, repeats
            ),
        }
        results.append(result)
        (output / "sampling.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "circuit",
                        "baseline_peak",
                        "bank_peak",
                        "speedup",
                        "bank_compile_seconds",
                    )
                }
            ),
            flush=True,
        )
    summary = {
        "version": clifft.version(),
        "isa": clifft.runtime_isa(),
        "threads": 1,
        "batch_size": 1,
        "fault_sites": sites,
        "target_seconds": target,
        "repeats": repeats,
        "scope": (
            "Exact independent-X models restricted to selected sites before first T. "
            "Branch bank fully enumerated and compiled up front. Timings include "
            "fault draws, grouping, dispatch and native sampling, but exclude compile and warmup."
        ),
        "median_speedup": statistics.median(result["speedup"] for result in results),
        "equal_shots_total_speedup": sum(
            1 / result["baseline_shots_per_second"] for result in results
        )
        / sum(1 / result["candidate_shots_per_second"] for result in results),
        "cases": results,
    }
    (output / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def affine_benchmark(
    corpus: Path, output: Path, names: list[str], placement: str, target: float, repeats: int
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (corpus / "circuits.csv").open(newline="") as handle:
        entries = {Path(entry["circuit"]).stem: entry for entry in csv.DictReader(handle)}
    results = []
    for name in names:
        entry = entries["circuit_" + name]
        source = (corpus / entry["circuit"]).read_text()
        before, after = _locations(source)
        n = len(trace(source).rows)
        position, axis = (before, "X") if placement == "pre" else (after, "Y")
        start = time.perf_counter()
        frame = compile_frame(source, [FaultSite(position, axis, q, 0.001) for q in range(n)])
        symbolic_seconds = time.perf_counter() - start
        mask = [1] * int(entry["postselection_detectors"])
        start = time.perf_counter()
        baseline = _compile(frame.original(), mask)
        baseline_compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        sampler = AffineRecordSampler(frame, mask)
        compile_seconds = time.perf_counter() - start
        runners = [
            lambda count, seed: clifft.sample_survivors(
                baseline, shots=count, seed=seed, batch_size=1, threads=1
            ),
            lambda count, seed: sampler.sample_survivors(count, seed=seed),
        ]
        for run in runners:
            run(1, 171)
        result = {
            "circuit": entry["circuit"],
            "physical_qubits": n,
            "noise": {"placement": placement, "axis": axis, "probability": 0.001, "sites": n},
            "symbolic": frame.stats(),
            "baseline_peak": baseline.peak_active_width,
            "candidate_peak": sampler.program.peak_active_width,
            "symbolic_compile_seconds": symbolic_seconds,
            "baseline_compile_seconds": baseline_compile_seconds,
            "candidate_compile_seconds": compile_seconds,
            **_timing_runners(runners, target, repeats),
        }
        results.append(result)
        (output / "affine_sampling.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("circuit", "baseline_peak", "candidate_peak", "speedup")
                }
            ),
            flush=True,
        )
    summary = {
        "version": clifft.version(),
        "isa": clifft.runtime_isa(),
        "threads": 1,
        "batch_size": 1,
        "placement": placement,
        "target_seconds": target,
        "repeats": repeats,
        "scope": (
            "Independent Pauli faults at every physical qubit at one selected layer, "
            "not throughout "
            "the circuit. One fixed native sampler plus precomputed affine record flips. "
            "Sampling includes noise draws, NumPy parity evaluation and postselection, "
            "and excludes compile and warmup."
        ),
        "median_speedup": statistics.median(result["speedup"] for result in results),
        "cases": results,
    }
    (output / "affine_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "benchmark", "affine"))
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--circuits", nargs="+", default=["0005", "0262", "0689", "1037"])
    parser.add_argument("--placement", choices=("pre", "post"), default="post")
    parser.add_argument("--sites", type=int, default=6)
    parser.add_argument("--target-seconds", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.sites <= 8 or args.target_seconds <= 0 or args.repeats < 1:
        parser.error("invalid research benchmark limits")
    if args.mode == "scan":
        scan(args.corpus, args.output)
    elif args.mode == "affine":
        affine_benchmark(
            args.corpus,
            args.output,
            args.circuits,
            args.placement,
            args.target_seconds,
            args.repeats,
        )
    else:
        benchmark(
            args.corpus, args.output, args.circuits, args.sites, args.target_seconds, args.repeats
        )


if __name__ == "__main__":
    main()
