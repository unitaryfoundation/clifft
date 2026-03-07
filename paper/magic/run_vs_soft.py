#!/usr/bin/env python3
"""Run UCC sampling on Magic State Cultivation circuits via Sinter.

This script registers UCC as a Sinter sampler and runs d=5 (and optionally
d=3) MSC circuits with all-detector postselection, matching the SOFT paper
protocol where ANY fired detector triggers discard.

Usage:
    uv run python paper/magic/run_vs_soft.py [--shots N] [--workers N]
    uv run python paper/magic/run_vs_soft.py --circuit d5_p0.001 --shots 1000000
    uv run python paper/magic/run_vs_soft.py --max-errors 50 --save results.csv
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

# Ensure the repo root is on sys.path so paper.magic is importable.
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sinter  # noqa: E402
import stim  # noqa: E402

from paper.magic.ucc_soft_sampler import UccSoftSampler  # noqa: E402

CIRCUIT_DIR = ROOT / "paper" / "magic" / "circuits"

# All available circuits: (label, filename, distance, noise_strength)
ALL_CIRCUITS: list[tuple[str, str, int, float]] = [
    ("d3_p0.001", "circuit_d3_p0.001.stim", 3, 0.001),
    ("d5_p0.0005", "circuit_d5_p0.0005.stim", 5, 0.0005),
    ("d5_p0.001", "circuit_d5_p0.001.stim", 5, 0.001),
    ("d5_p0.002", "circuit_d5_p0.002.stim", 5, 0.002),
    ("d5_p0.003", "circuit_d5_p0.003.stim", 5, 0.003),
    ("d5_p0.004", "circuit_d5_p0.004.stim", 5, 0.004),
    ("d5_p0.005", "circuit_d5_p0.005.stim", 5, 0.005),
]

# Paper Table IV reference discard rates for d=5
TABLE_IV: dict[float, float] = {
    0.0005: 62.10,
    0.001: 85.60,
    0.002: 97.92,
    0.003: 99.70,
    0.004: 99.96,
    0.005: 99.99,
}


def build_all_detector_mask(num_dets: int) -> np.ndarray:
    """Build a postselection mask that flags ALL detectors.

    The MSC protocol postselects on every detector: if any detector fires,
    the shot is discarded. This matches the SOFT paper's compiled format
    where all detectors map to CHECK instructions.
    """
    num_bytes = math.ceil(num_dets / 8)
    mask = bytearray([0xFF] * num_bytes)
    # Clear unused high bits in the last byte
    remainder = num_dets % 8
    if remainder != 0:
        mask[-1] = (1 << remainder) - 1
    return np.array(mask, dtype=np.uint8)


def sanitize_for_stim(text: str) -> str:
    """Replace T/T_DAG gates with I so Stim can parse the circuit.

    Stim is a Clifford simulator and does not support non-Clifford gates
    like T/T_DAG. We only need the stim.Circuit object for metadata
    (num_detectors, coordinates). UCC's own parser handles the real
    T gates via the raw text passed in json_metadata.
    """
    import re

    return re.sub(r"^(T_DAG|T)\b", "I", text, flags=re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="UCC Magic State Cultivation benchmark")
    parser.add_argument("--shots", type=int, default=1_000_000, help="Max shots per task")
    parser.add_argument("--workers", type=int, default=1, help="Number of Sinter workers")
    parser.add_argument(
        "--circuit",
        type=str,
        default=None,
        help="Run only this circuit (e.g. 'd5_p0.001'). Default: all d=5.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=None,
        help="Stop after this many errors (for convergence-based runs)",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save CSV results")
    parser.add_argument(
        "--save-resume",
        type=str,
        default=None,
        help="Sinter resume file for crash resilience",
    )
    args = parser.parse_args()

    # Select circuits
    if args.circuit:
        selected = [c for c in ALL_CIRCUITS if c[0] == args.circuit]
        if not selected:
            valid = [c[0] for c in ALL_CIRCUITS]
            print(f"Unknown circuit '{args.circuit}'. Valid: {valid}")
            return
    else:
        # Default: all d=5 circuits
        selected = [c for c in ALL_CIRCUITS if c[2] == 5]

    tasks: list[sinter.Task] = []
    for label, filename, d, p in selected:
        path = CIRCUIT_DIR / filename
        if not path.exists():
            print(f"Skipping {label}: {path} not found")
            continue
        raw_text = path.read_text()
        # Stim can't parse T/T_DAG (non-Clifford); sanitize for metadata only.
        circuit = stim.Circuit(sanitize_for_stim(raw_text))
        mask = build_all_detector_mask(circuit.num_detectors)
        # Sinter derives a DEM for strong_id computation, which fails on
        # sanitized circuits (T->I makes observables non-deterministic).
        # Provide a dummy DEM and explicit strong_id to bypass this.
        # Note: _unvalidated_strong_id is Sinter's only API for this;
        # there is no public strong_id parameter.
        dummy_dem = stim.DetectorErrorModel()
        strong_id = f"ucc:{label}"
        tasks.append(
            sinter.Task(
                circuit=circuit,
                decoder="ucc",
                detector_error_model=dummy_dem,
                postselection_mask=mask,
                json_metadata={
                    "label": label,
                    "d": d,
                    "p": p,
                    "circuit_text": raw_text,
                },
                skip_validation=True,
                _unvalidated_strong_id=strong_id,
            )
        )

    if not tasks:
        print("No circuits found. Exiting.")
        return

    print(f"Running {len(tasks)} task(s) with {args.shots} shots, {args.workers} worker(s)...")

    collect_kwargs: dict[str, object] = {
        "num_workers": args.workers,
        "tasks": tasks,
        "custom_decoders": {"ucc": UccSoftSampler()},
        "max_shots": args.shots,
        "print_progress": True,
    }
    if args.max_errors is not None:
        collect_kwargs["max_errors"] = args.max_errors
    if args.save_resume is not None:
        collect_kwargs["save_resume_filepath"] = args.save_resume

    t_start = time.monotonic()
    results = sinter.collect(**collect_kwargs)
    t_elapsed = time.monotonic() - t_start

    print(f"\n=== Results (wall time: {t_elapsed:.1f}s) ===")
    print(
        f"{'Circuit':<15} {'Shots':>12} {'Passed':>12} {'Discards':>12} "
        f"{'Discard%':>9} {'Errors':>8} {'Error Rate':>12} {'Table IV':>9}"
        f" {'Time(s)':>8} {'us/shot':>8}"
    )
    print("-" * 120)
    for stat in results:
        meta = stat.json_metadata or {}
        label = meta.get("label", "unknown")
        p = meta.get("p", 0.0)
        total = stat.shots
        errors = stat.errors
        discards = stat.discards
        passed = total - discards
        discard_pct = 100.0 * discards / total if total > 0 else 0
        error_rate = errors / passed if passed > 0 else 0
        ref = TABLE_IV.get(p, float("nan"))
        task_secs = stat.seconds
        us_per_shot = task_secs / total * 1e6 if total > 0 else 0
        print(
            f"{label:<15} {total:>12,} {passed:>12,} {discards:>12,} "
            f"{discard_pct:>8.2f}% {errors:>8,} {error_rate:>12.2e} {ref:>8.2f}%"
            f" {task_secs:>8.1f} {us_per_shot:>8.1f}"
        )

    print(f"\nTotal wall time: {t_elapsed:.1f}s" f" ({t_elapsed/60:.1f}m, {t_elapsed/3600:.2f}h)")

    if args.save:
        save_path = pathlib.Path(args.save)
        with open(save_path, "w") as f:
            print(sinter.CSV_HEADER, file=f)
            for stat in results:
                print(stat.to_csv_line(), file=f)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
