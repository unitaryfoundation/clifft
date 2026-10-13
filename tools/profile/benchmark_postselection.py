#!/usr/bin/env python3
"""Compare postselected auto and explicit capacities with expanded detector masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

import clifft


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuits", type=Path, nargs="+")
    parser.add_argument("--shots", type=positive_int, nargs="+", default=[2048, 65536, 200000])
    parser.add_argument("--threads", type=positive_int, nargs="+", default=[1, 4])
    parser.add_argument("--batch-sizes", nargs="+", default=["auto", "1", "256", "2048"])
    parser.add_argument("--outputs", choices=["counts", "rows", "both"], default="both")
    parser.add_argument("--repetitions", type=positive_int, default=5)
    parser.add_argument("--calls", type=positive_int, default=1)
    parser.add_argument("--fixed-k", type=int)
    parser.add_argument("--herald", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fixed_k is not None and args.fixed_k < 0:
        parser.error("--fixed-k must be nonnegative")
    if args.herald is not None and not 0 <= args.herald <= 1:
        parser.error("--herald must be a probability")
    try:
        capacities: dict[str, str | int] = {
            value: "auto" if value == "auto" else positive_int(value) for value in args.batch_sizes
        }
    except (ValueError, argparse.ArgumentTypeError):
        parser.error("--batch-sizes must contain auto or positive integers")
    modes = [False, True] if args.outputs == "both" else [args.outputs == "rows"]
    output: dict[str, Any] = {
        "clifft": clifft.version(),
        "isa": clifft.runtime_isa(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in args.circuits:
        source = path.read_text()
        if args.herald is not None:
            qubit = clifft.trace(clifft.parse(source)).num_qubits
            source = (
                f"R {qubit}\nX_ERROR({args.herald}) {qubit}\nM {qubit}\nDETECTOR rec[-1]\n" + source
            )
        hir = clifft.trace(clifft.parse(source))
        if hir.num_detectors == 0:
            parser.error(f"{path} has no detectors")
        program = clifft.compile(source, postselection_mask=[1] * hir.num_detectors)
        sample: Any = clifft.sample_survivors if args.fixed_k is None else clifft.sample_k_survivors
        options = {} if args.fixed_k is None else {"k": args.fixed_k}
        for shots in args.shots:
            for threads in args.threads:
                for keep_records in modes:

                    def run(capacity: str | int, count: int, seed: int, **extra: Any) -> Any:
                        return sample(
                            program,
                            count,
                            seed=seed,
                            keep_records=keep_records,
                            thread_layout=(threads, 1),
                            batch_size=capacity,
                            **options,
                            **extra,
                        )

                    # A zero budget reports the production policy without timed calibration.
                    report = run("tune", shots, 123, tuning_budget_seconds=0).batch_tuning
                    elapsed: dict[str, list[float]] = {label: [] for label in capacities}
                    passed: dict[str, list[int]] = {label: [] for label in capacities}
                    for capacity in capacities.values():
                        run(capacity, shots, 124)
                    for repetition in range(args.repetitions):
                        order = list(capacities)
                        random.Random(repetition).shuffle(order)
                        for label in order:
                            survivors = 0
                            start = time.perf_counter()
                            for call in range(args.calls):
                                result = run(
                                    capacities[label], shots, 1000 + repetition * args.calls + call
                                )
                                survivors += result.passed_shots
                                del result
                            elapsed[label].append(time.perf_counter() - start)
                            passed[label].append(survivors)
                    for label, capacity in capacities.items():
                        effective = (
                            report.batch_size
                            if capacity == "auto"
                            else min(int(capacity), shots, 2048)
                        )
                        workers = (
                            report.shot_workers
                            if capacity == "auto"
                            else (
                                threads
                                if effective == 1
                                else min(threads, (shots + effective - 1) // effective)
                            )
                        )
                        row = {
                            "circuit": str(path),
                            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                            "width": program.peak_active_width,
                            "detectors": hir.num_detectors,
                            "shots_per_call": shots,
                            "calls": args.calls,
                            "threads": threads,
                            "keep_records": keep_records,
                            "fixed_k": args.fixed_k,
                            "herald": args.herald,
                            "requested_batch_size": capacity,
                            "effective_batch_size": effective,
                            "policy_workers": workers,
                            "seconds": elapsed[label],
                            "median_seconds": statistics.median(elapsed[label]),
                            "passed_shots": passed[label],
                        }
                        output["results"].append(row)
                        print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
