#!/usr/bin/env python3
"""Local correctness validation: observe logical errors on d=5 MSC circuits.

Runs d=5 p=0.002 (the cheapest Table III data point) until at least
max_errors logical errors are observed, with periodic progress logging
and Sinter resume-file support for crash resilience.

Paper reference (Table III, p=0.002):
    Total shots: 28.9B, Preserved: 0.60B, Discard: 97.92%
    Detected errors: 22, Logical error rate: 3.41e-8

To see ~1 error we need ~30M survivors => ~1.4B total shots.
At ~68k shots/s (97.92% discard) => ~5-6 hours single-core.

Usage:
    uv run python paper/magic/validate_local.py
    uv run python paper/magic/validate_local.py --max-errors 3 --workers 2
    uv run python paper/magic/validate_local.py --resume results/validate_resume.csv
"""

from __future__ import annotations

import argparse
import math
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

# Paper Table III reference for p=0.002
PAPER_ERROR_RATE = 3.41e-8
PAPER_DISCARD_RATE = 97.92  # percent


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
    parser = argparse.ArgumentParser(
        description="Local correctness validation for d=5 MSC circuits"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.002,
        help="Noise strength (default: 0.002, cheapest Table III point)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=5,
        help="Stop after this many logical errors (default: 5)",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=None,
        help="Maximum total shots (default: unlimited)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of Sinter workers (default: 1)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="results/validate_resume.csv",
        help="Sinter resume file path (default: results/validate_resume.csv)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between progress updates (default: 30)",
    )
    args = parser.parse_args()

    # Find circuit file
    circuit_file = CIRCUIT_DIR / f"circuit_d5_p{args.noise}.stim"
    if not circuit_file.exists():
        print(f"Circuit not found: {circuit_file}")
        return

    raw_text = circuit_file.read_text()
    circuit = stim.Circuit(sanitize_for_stim(raw_text))
    mask = build_all_detector_mask(circuit.num_detectors)
    label = f"d5_p{args.noise}"

    dummy_dem = stim.DetectorErrorModel()
    strong_id = f"ucc:{label}"
    task = sinter.Task(
        circuit=circuit,
        decoder="ucc",
        detector_error_model=dummy_dem,
        postselection_mask=mask,
        json_metadata={
            "label": label,
            "d": 5,
            "p": args.noise,
            "circuit_text": raw_text,
        },
        skip_validation=True,
        _unvalidated_strong_id=strong_id,
    )

    # Ensure resume directory exists
    resume_path = pathlib.Path(args.resume)
    resume_path.parent.mkdir(parents=True, exist_ok=True)

    # Accumulator for progress tracking
    cumulative_shots = 0
    cumulative_errors = 0
    cumulative_discards = 0
    cumulative_seconds = 0.0
    last_print_time = time.monotonic()
    wall_start = time.monotonic()

    print("=== Local Correctness Validation ===")
    print(f"Circuit:   {circuit_file.name}")
    print(f"Qubits:    {circuit.num_qubits}")
    print(f"Detectors: {circuit.num_detectors}")
    print(f"Target:    {args.max_errors} logical errors")
    print(f"Workers:   {args.workers}")
    print(f"Resume:    {resume_path}")
    print("")
    print(f"Paper reference (p={args.noise}):")
    print(f"  Discard rate:      {PAPER_DISCARD_RATE}%")
    print(f"  Logical error rate: {PAPER_ERROR_RATE:.2e}")
    print("")
    print(f"Progress updates every {args.progress_interval:.0f}s...")
    print(f"{'='*80}")
    sys.stdout.flush()

    def progress_callback(progress: sinter.Progress) -> None:
        nonlocal cumulative_shots, cumulative_errors, cumulative_discards
        nonlocal cumulative_seconds, last_print_time

        for stats in progress.new_stats:
            cumulative_shots += stats.shots
            cumulative_errors += stats.errors
            cumulative_discards += stats.discards
            cumulative_seconds += stats.seconds

        now = time.monotonic()
        if now - last_print_time < args.progress_interval:
            return
        last_print_time = now

        wall_elapsed = now - wall_start
        survivors = cumulative_shots - cumulative_discards
        discard_pct = 100.0 * cumulative_discards / cumulative_shots if cumulative_shots > 0 else 0
        error_rate = cumulative_errors / survivors if survivors > 0 else 0
        shots_per_sec = cumulative_shots / wall_elapsed if wall_elapsed > 0 else 0
        survivors_per_sec = survivors / wall_elapsed if wall_elapsed > 0 else 0

        # Estimate time to next error (if we have survivors but no errors yet)
        if cumulative_errors == 0 and survivors > 0:
            # At paper rate, how many more survivors needed for 1 error?
            survivors_for_1 = 1.0 / PAPER_ERROR_RATE
            remaining_survivors = survivors_for_1 - survivors
            if remaining_survivors > 0 and survivors_per_sec > 0:
                eta_next_error = remaining_survivors / survivors_per_sec
                eta_str = f"~{format_duration(eta_next_error)} to 1st error (at paper rate)"
            else:
                eta_str = "estimating..."
        elif cumulative_errors > 0:
            # Estimate time to target errors
            remaining_errors = args.max_errors - cumulative_errors
            if remaining_errors > 0:
                secs_per_error = wall_elapsed / cumulative_errors
                eta_remaining = format_duration(remaining_errors * secs_per_error)
                eta_str = f"~{eta_remaining} to {args.max_errors} errors"
            else:
                eta_str = "target reached!"
        else:
            eta_str = "warming up..."

        print(
            f"[{format_duration(wall_elapsed)}] "
            f"shots={cumulative_shots:>12,} "
            f"survivors={survivors:>10,} "
            f"errors={cumulative_errors} "
            f"discard={discard_pct:.2f}% "
            f"err_rate={error_rate:.2e} "
            f"({shots_per_sec:,.0f} shots/s, {survivors_per_sec:,.0f} surv/s) "
            f"| {eta_str}"
        )
        sys.stdout.flush()

    # Sinter requires max_shots even when using max_errors (it does
    # min(errors_left, shots_left) internally). Use a large default.
    max_shots = args.max_shots if args.max_shots is not None else 10**15

    collect_kwargs: dict[str, object] = {
        "num_workers": args.workers,
        "tasks": [task],
        "custom_decoders": {"ucc": UccSoftSampler()},
        "max_errors": args.max_errors,
        "max_shots": max_shots,
        "max_batch_seconds": 30,
        "progress_callback": progress_callback,
        "save_resume_filepath": str(resume_path),
    }

    results = sinter.collect(**collect_kwargs)

    wall_total = time.monotonic() - wall_start

    print(f"\n{'='*80}")
    print("=== FINAL RESULTS ===")
    print(f"Wall time: {format_duration(wall_total)}")
    for stat in results:
        total = stat.shots
        errors = stat.errors
        discards = stat.discards
        survivors = total - discards
        discard_pct = 100.0 * discards / total if total > 0 else 0
        error_rate = errors / survivors if survivors > 0 else 0

        print("")
        print(f"  Total shots:       {total:>15,}")
        print(f"  Survivors:         {survivors:>15,}")
        print(f"  Discards:          {discards:>15,} ({discard_pct:.2f}%)")
        print(f"  Logical errors:    {errors:>15,}")
        print(f"  Error rate:        {error_rate:>15.2e}")
        print("")
        print("  Paper reference:")
        print(f"    Discard rate:    {PAPER_DISCARD_RATE}%")
        print(f"    Error rate:      {PAPER_ERROR_RATE:.2e}")
        print("")

        if errors > 0:
            ratio = error_rate / PAPER_ERROR_RATE
            print(f"  UCC/Paper ratio:   {ratio:.2f}x")
            if 0.1 <= ratio <= 10.0:
                print("  PASS: Within 10x of paper value")
            else:
                print("  WARNING: Outside 10x range")
        else:
            print(f"  No errors observed. Error rate is <= {1.0/survivors:.2e} (upper bound)")
            print(f"  This is consistent if true rate is near {PAPER_ERROR_RATE:.2e}")
            print(f"  (would need ~{int(1.0/PAPER_ERROR_RATE):,} survivors to expect 1 error)")


if __name__ == "__main__":
    main()
