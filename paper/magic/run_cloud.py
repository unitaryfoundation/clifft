#!/usr/bin/env python3
"""Run a single Table III data point to convergence.

Designed for cloud execution on a multi-core instance. Uses Sinter's
save_resume_filepath for crash resilience (spot interruptions).

Usage:
    uv run python paper/magic/run_cloud.py --noise 0.002
    uv run python paper/magic/run_cloud.py --noise 0.001 --max-errors 50
    uv run python paper/magic/run_cloud.py --noise 0.0005 --max-errors 8
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sinter  # noqa: E402
import stim  # noqa: E402

from paper.magic.ucc_soft_sampler import UccSoftSampler  # noqa: E402

CIRCUIT_DIR = ROOT / "paper" / "magic" / "circuits"

PAPER_DATA: dict[float, dict[str, float]] = {
    0.002: {"rate": 3.41e-8, "discard": 0.9792},
    0.001: {"rate": 4.59e-9, "discard": 0.8560},
    0.0005: {"rate": 1.57e-10, "discard": 0.6210},
}


def build_all_detector_mask(num_dets: int) -> np.ndarray:
    """Build a postselection mask that flags ALL detectors."""
    num_bytes = math.ceil(num_dets / 8)
    mask = bytearray([0xFF] * num_bytes)
    remainder = num_dets % 8
    if remainder != 0:
        mask[-1] = (1 << remainder) - 1
    return np.array(mask, dtype=np.uint8)


def sanitize_for_stim(text: str) -> str:
    """Replace T/T_DAG gates with I so Stim can parse the circuit."""
    import re

    return re.sub(r"^(T_DAG|T)\b", "I", text, flags=re.MULTILINE)


def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym Zs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single SOFT Table III data point")
    parser.add_argument(
        "--noise",
        type=float,
        required=True,
        choices=[0.0005, 0.001, 0.002],
        help="Noise strength (Table III data point)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="Stop after this many errors (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers (default: physical cores = logical/2)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=60.0,
        help="Seconds between progress updates (default: 60)",
    )
    args = parser.parse_args()

    noise = args.noise
    info = PAPER_DATA[noise]

    # Auto-detect worker count: use physical cores (half of logical)
    if args.workers:
        num_workers = args.workers
    else:
        logical = os.cpu_count() or 2
        num_workers = max(1, logical // 2)

    # Build task
    circuit_file = CIRCUIT_DIR / f"circuit_d5_p{noise}.stim"
    raw_text = circuit_file.read_text()
    circuit = stim.Circuit(sanitize_for_stim(raw_text))
    mask = build_all_detector_mask(circuit.num_detectors)
    label = f"d5_p{noise}"

    task = sinter.Task(
        circuit=circuit,
        decoder="ucc",
        detector_error_model=stim.DetectorErrorModel(),
        postselection_mask=mask,
        json_metadata={
            "label": label,
            "d": 5,
            "p": noise,
            "circuit_text": raw_text,
        },
        skip_validation=True,
        _unvalidated_strong_id=f"ucc:{label}",
    )

    # Resume file
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    resume_path = results_dir / f"resume_d5_p{noise}.csv"
    final_path = results_dir / f"results_d5_p{noise}.csv"

    # Progress tracking
    cumulative_shots = 0
    cumulative_errors = 0
    cumulative_discards = 0
    last_print_time = time.monotonic()
    wall_start = time.monotonic()

    print("=== SOFT Table III Reproduction ===")
    print(f"Noise:     p={noise}")
    print(f"Circuit:   {circuit_file.name}")
    print(f"Workers:   {num_workers}")
    print(f"Target:    {args.max_errors} errors")
    print(f"Resume:    {resume_path}")
    print("")
    print("Paper reference:")
    print(f"  Discard rate:  {info['discard']*100:.2f}%")
    print(f"  Error rate:    {info['rate']:.2e}")
    survivors_needed = args.max_errors / info["rate"]
    total_shots_est = survivors_needed / (1 - info["discard"])
    print(f"  Est. shots needed: {total_shots_est:,.0f}")
    print("")
    print(f"{'='*90}")
    sys.stdout.flush()

    def progress_callback(progress: sinter.Progress) -> None:
        nonlocal cumulative_shots, cumulative_errors, cumulative_discards
        nonlocal last_print_time

        for stats in progress.new_stats:
            cumulative_shots += stats.shots
            cumulative_errors += stats.errors
            cumulative_discards += stats.discards

        now = time.monotonic()
        if now - last_print_time < args.progress_interval:
            return
        last_print_time = now

        wall = now - wall_start
        survivors = cumulative_shots - cumulative_discards
        discard_pct = 100.0 * cumulative_discards / cumulative_shots if cumulative_shots > 0 else 0
        error_rate = cumulative_errors / survivors if survivors > 0 else 0
        shots_per_s = cumulative_shots / wall if wall > 0 else 0

        if cumulative_errors > 0:
            remaining = args.max_errors - cumulative_errors
            if remaining > 0:
                secs_per_error = wall / cumulative_errors
                eta = format_duration(remaining * secs_per_error)
            else:
                eta = "done!"
        else:
            eta = "waiting for 1st error..."

        print(
            f"[{format_duration(wall)}] "
            f"shots={cumulative_shots:>14,} "
            f"surv={survivors:>12,} "
            f"err={cumulative_errors:>4} "
            f"disc={discard_pct:.2f}% "
            f"rate={error_rate:.2e} "
            f"({shots_per_s:,.0f} shots/s) "
            f"| ETA: {eta}"
        )
        sys.stdout.flush()

    results = sinter.collect(
        num_workers=num_workers,
        tasks=[task],
        custom_decoders={"ucc": UccSoftSampler()},
        max_errors=args.max_errors,
        max_shots=10**18,
        max_batch_seconds=30,
        progress_callback=progress_callback,
        save_resume_filepath=str(resume_path),
    )

    wall_total = time.monotonic() - wall_start

    # Save final results
    with open(final_path, "w") as f:
        print(sinter.CSV_HEADER, file=f)
        for stat in results:
            print(stat.to_csv_line(), file=f)

    print(f"\n{'='*90}")
    print("=== FINAL RESULTS ===")
    print(f"Wall time: {format_duration(wall_total)}")
    for stat in results:
        total = stat.shots
        errors = stat.errors
        discards = stat.discards
        survivors = total - discards
        discard_pct = 100.0 * discards / total if total > 0 else 0
        error_rate = errors / survivors if survivors > 0 else 0
        print(f"  Shots:      {total:>16,}")
        print(f"  Survivors:  {survivors:>16,}")
        print(f"  Discards:   {discards:>16,} ({discard_pct:.2f}%)")
        print(f"  Errors:     {errors:>16,}")
        print(f"  Error rate: {error_rate:>16.4e}")
        print(f"  Paper rate: {info['rate']:>16.4e}")
        if errors > 0:
            print(f"  Ratio:      {error_rate/info['rate']:>16.2f}x")
    print(f"\nResults saved to: {final_path}")
    print(f"Resume file: {resume_path}")


if __name__ == "__main__":
    main()
