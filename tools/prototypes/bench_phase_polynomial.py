"""Compare the opt-in phase rewrite with the existing width scheduler."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import clifft
from tools.prototypes.phase_polynomial import reduce_phase_polynomial


def _compile(source: str, mask: list[int]) -> Any:
    manager = clifft.default_hir_pass_manager()
    manager.add(clifft.ActiveWidthSchedulePass())
    return clifft.compile(source, postselection_mask=mask, hir_passes=manager)


def _timings(programs: list[Any], target: float, repeats: int) -> dict[str, Any]:
    shots = []
    for program in programs:
        start = time.perf_counter()
        clifft.sample_survivors(program, shots=1, seed=20261001, threads=1, batch_size=1)
        elapsed = time.perf_counter() - start
        count = min(200000, max(1, math.ceil(target / max(elapsed, 1e-9))))
        start = time.perf_counter()
        clifft.sample_survivors(program, shots=count, seed=20261001, threads=1, batch_size=1)
        elapsed = time.perf_counter() - start
        shots.append(min(200000, max(1, math.ceil(count * target / max(elapsed, 1e-9)))))
    timings: list[list[float]] = [[], []]
    counts: list[list[dict[str, Any]]] = [[], []]
    for repeat in range(repeats):
        for arm in [0, 1] if repeat % 2 == 0 else [1, 0]:
            start = time.perf_counter()
            result = clifft.sample_survivors(
                programs[arm], shots=shots[arm], seed=20261001 + repeat, threads=1, batch_size=1
            )
            timings[arm].append(time.perf_counter() - start)
            counts[arm].append(
                dict(
                    total=result.total_shots,
                    passed=result.passed_shots,
                    observable_ones=result.observable_ones.tolist(),
                )
            )
    latency = [statistics.median(values) / n for values, n in zip(timings, shots)]
    return dict(
        shots=shots,
        timings_seconds=timings,
        counts=counts,
        baseline_shots_per_second=1 / latency[0],
        reduced_shots_per_second=1 / latency[1],
        speedup=latency[0] / latency[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory containing circuits.csv")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-limit", type=int, default=64, help="Zero samples every circuit")
    parser.add_argument("--target-seconds", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.sample_limit < 0 or args.target_seconds <= 0 or args.repeats < 1:
        parser.error("invalid sampling limits")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.corpus / "circuits.csv").open(newline="") as handle:
        inputs = list(csv.DictReader(handle))
    rows = []
    start = time.perf_counter()
    for item in inputs:
        source = (args.corpus / item["circuit"]).read_text()
        mask = [1] * int(item["postselection_detectors"])
        begin = time.perf_counter()
        result = reduce_phase_polynomial(source)
        rewrite_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        baseline = _compile(source, mask)
        baseline_compile = time.perf_counter() - begin
        begin = time.perf_counter()
        reduced = _compile(result.circuit, mask)
        reduced_compile = time.perf_counter() - begin
        for field in ("num_qubits", "num_measurements", "num_detectors", "num_observables"):
            assert getattr(baseline, field) == getattr(reduced, field), (item["circuit"], field)
        rows.append(
            dict(
                item,
                applied=result.applied,
                reason=result.reason,
                initial_width=result.initial_width,
                logical_width=result.logical_width,
                input_t=result.input_t_count,
                output_t=result.output_t_count,
                padded_records=result.deterministic_records,
                baseline_peak=baseline.peak_active_width,
                reduced_peak=reduced.peak_active_width,
                rewrite_seconds=rewrite_seconds,
                baseline_compile_seconds=baseline_compile,
                reduced_compile_seconds=reduced_compile,
            )
        )
        if len(rows) % 100 == 0:
            print(
                f"Scanned {len(rows)}/{len(inputs)} in {time.perf_counter() - start:.1f}s",
                flush=True,
            )
    (args.output / "scan.json").write_text(json.dumps(rows, indent=2) + "\n")
    selected = rows
    if args.sample_limit and args.sample_limit < len(rows):
        rng = random.Random(20261001)
        selected = rng.sample(rows, args.sample_limit)
        names = {row["circuit"] for row in selected}
        for name in ("0005", "0262", "0300", "0689", "1037"):
            row = next((r for r in rows if r["circuit"].endswith(f"_{name}.stim")), None)
            if row is not None and row["circuit"] not in names:
                selected.append(row)
    sampled = []
    with (args.output / "sampling.jsonl").open("w") as handle:
        for row in selected:
            source = (args.corpus / row["circuit"]).read_text()
            result = reduce_phase_polynomial(source)
            mask = [1] * int(row["postselection_detectors"])
            timing = _timings(
                [_compile(source, mask), _compile(result.circuit, mask)],
                args.target_seconds,
                args.repeats,
            )
            record = dict(row, **timing)
            sampled.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if len(sampled) % 25 == 0:
                print(
                    f"Sampled {len(sampled)}/{len(selected)} in {time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    (args.output / "sampling.json").write_text(json.dumps(sampled, indent=2) + "\n")
    summary = dict(
        version=clifft.version(),
        isa=clifft.runtime_isa(),
        threads=1,
        batch_size=1,
        baseline="default passes plus ActiveWidthSchedulePass with default settings",
        circuits=len(rows),
        rewritten=sum(r["applied"] for r in rows),
        unchanged=sum(not r["applied"] for r in rows),
        lower_peak=sum(r["reduced_peak"] < r["baseline_peak"] for r in rows),
        higher_peak=sum(r["reduced_peak"] > r["baseline_peak"] for r in rows),
        input_t=sum(r["input_t"] for r in rows),
        output_t=sum(r["output_t"] for r in rows),
        median_rewrite_ms=1000 * statistics.median(r["rewrite_seconds"] for r in rows),
        sampled=len(sampled),
        target_seconds=args.target_seconds,
        repeats=args.repeats,
        median_speedup=statistics.median(r["speedup"] for r in sampled),
        geometric_mean_speedup=math.exp(statistics.mean(math.log(r["speedup"]) for r in sampled)),
        equal_shots_total_speedup=sum(1 / r["baseline_shots_per_second"] for r in sampled)
        / sum(1 / r["reduced_shots_per_second"] for r in sampled),
        slower_over_ten_percent=sum(r["speedup"] < 1 / 1.1 for r in sampled),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
