#!/usr/bin/env python3
"""Measure one reconstruction configuration in a fresh process; emit a JSON row.

Run configurations sequentially with the same interpreter, affinity, shot count,
and batch size. Times include noise generation and requested output production.
Peak RSS includes Python and temporary compiler storage, not just coefficients.
"""

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import clifft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", type=Path)
    parser.add_argument("--reconstruction", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--postselect", action="store_true", help="Select every detector")
    parser.add_argument("--keep-records", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.shots, args.repeats) < 1:
        parser.error("batch size, shots, and repeats must be positive")
    source = args.circuit.read_text()
    detectors = sum(node.gate == clifft.GateType.DETECTOR for node in clifft.parse(source).nodes)
    passes = clifft.default_hir_pass_manager()
    if args.schedule:
        passes.add(clifft.ActiveWidthSchedulePass())
    start = time.perf_counter()
    program = clifft.compile(
        source,
        hir_passes=passes,
        reconstruction=args.reconstruction,
        postselection_mask=[1] * detectors if args.postselect else None,
        normalize_syndromes=True,
    )
    compile_seconds = time.perf_counter() - start

    def peak_rss_bytes():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024

    compile_peak_rss = peak_rss_bytes()
    # Allocate and execute the same path before timing; do not retain row arrays
    # from earlier calls while measuring subsequent calls.
    clifft.sample_survivors(
        program, 65, seed=19, keep_records=args.keep_records, batch_size=args.batch_size
    )
    seconds = []
    passed = []
    errors = []
    for repeat in range(args.repeats):
        start = time.perf_counter()
        result = clifft.sample_survivors(
            program,
            args.shots,
            seed=312 + repeat,
            keep_records=args.keep_records,
            batch_size=args.batch_size,
        )
        seconds.append(time.perf_counter() - start)
        passed.append(result.passed_shots)
        errors.append(result.logical_errors)
        del result
    row = vars(args) | {
        "circuit": str(args.circuit),
        "compile_seconds": compile_seconds,
        "sample_seconds": seconds,
        "median_us_per_attempt": statistics.median(seconds) * 1e6 / args.shots,
        "passed_shots": passed,
        "logical_errors": errors,
        "compile_peak_rss_bytes": compile_peak_rss,
        "peak_rss_bytes": peak_rss_bytes(),
        "plan": program.reconstruction_info(args.keep_records),
        "num_measurements": program.num_measurements,
        "num_detectors": program.num_detectors,
        "num_observables": program.num_observables,
    }
    print(json.dumps(row))


if __name__ == "__main__":
    main()
