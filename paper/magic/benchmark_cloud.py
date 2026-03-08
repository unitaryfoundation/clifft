#!/usr/bin/env python3
"""Benchmark UCC throughput on cloud hardware.

Runs short sampling bursts at each noise level to measure real shots/s,
both single-core and multi-core. Use results to estimate wall time and
cost for the full Table III reproduction.

Usage:
    uv run python paper/magic/benchmark_cloud.py
    uv run python paper/magic/benchmark_cloud.py --duration 30 --workers 12
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

NOISE_LEVELS = [0.002, 0.001, 0.0005]

# Paper Table III reference data
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


def make_task(noise: float) -> sinter.Task:
    """Build a Sinter task for the given noise level."""
    path = CIRCUIT_DIR / f"circuit_d5_p{noise}.stim"
    raw_text = path.read_text()
    circuit = stim.Circuit(sanitize_for_stim(raw_text))
    mask = build_all_detector_mask(circuit.num_detectors)
    label = f"d5_p{noise}"
    return sinter.Task(
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


def run_benchmark(noise: float, num_workers: int, duration_s: float) -> dict[str, float]:
    """Run a timed benchmark and return throughput stats."""
    task = make_task(noise)
    # Estimate shots for the duration based on a conservative 50k shots/s/worker
    est_shots = int(50_000 * num_workers * duration_s * 1.5)

    t0 = time.monotonic()
    results = sinter.collect(
        num_workers=num_workers,
        tasks=[task],
        custom_decoders={"ucc": UccSoftSampler()},
        max_shots=est_shots,
        max_errors=10**9,
        max_batch_seconds=int(max(duration_s // 4, 5)),
    )
    t1 = time.monotonic()
    wall = t1 - t0

    stat = results[0]
    shots = stat.shots
    survivors = shots - stat.discards
    discard_pct = 100.0 * stat.discards / shots if shots > 0 else 0

    return {
        "noise": noise,
        "workers": num_workers,
        "wall_s": wall,
        "shots": shots,
        "survivors": survivors,
        "discard_pct": discard_pct,
        "shots_per_s": shots / wall,
        "survivors_per_s": survivors / wall,
    }


def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UCC cloud throughput")
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Seconds per benchmark run (default: 60)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Multi-core worker count (default: physical cores)",
    )
    parser.add_argument(
        "--spot-price",
        type=float,
        default=1.50,
        help="Spot price $/hr for cost estimates (default: 1.50)",
    )
    parser.add_argument(
        "--target-errors",
        type=int,
        default=100,
        help="Target error count for cost estimates (default: 100)",
    )
    args = parser.parse_args()

    num_physical = os.cpu_count() or 2
    # Use half of logical CPUs as estimate of physical cores
    default_workers = max(1, num_physical // 2)
    multi_workers = args.workers if args.workers else default_workers

    print("=== UCC Cloud Benchmark ===")
    print(f"CPUs detected: {num_physical} logical")
    print(f"Workers: 1 (single-core) and {multi_workers} (multi-core)")
    print(f"Duration: {args.duration:.0f}s per test")
    print(f"Target errors for estimates: {args.target_errors}")
    print()

    results: list[dict[str, float]] = []

    for noise in NOISE_LEVELS:
        print(f"--- p={noise} ---")

        print(f"  Single-core ({args.duration:.0f}s)...", end="", flush=True)
        r1 = run_benchmark(noise, 1, args.duration)
        print(
            f" {r1['shots_per_s']:,.0f} shots/s, "
            f"{r1['survivors_per_s']:,.0f} surv/s, "
            f"discard={r1['discard_pct']:.2f}%"
        )
        results.append(r1)

        if multi_workers > 1:
            print(
                f"  {multi_workers}-core ({args.duration:.0f}s)...",
                end="",
                flush=True,
            )
            rm = run_benchmark(noise, multi_workers, args.duration)
            scaling = rm["shots_per_s"] / r1["shots_per_s"]
            print(
                f" {rm['shots_per_s']:,.0f} shots/s, "
                f"{rm['survivors_per_s']:,.0f} surv/s, "
                f"scaling={scaling:.1f}x/{multi_workers}w"
            )
            results.append(rm)

    # Summary table
    print(f"\n{'='*90}")
    print("=== THROUGHPUT SUMMARY ===")
    print(
        f"{'p':>8} {'Workers':>8} {'Shots/s':>12} {'Surv/s':>10} " f"{'Discard%':>9} {'Scaling':>8}"
    )
    print("-" * 60)
    single_core: dict[float, float] = {}
    multi_core: dict[float, float] = {}
    for r in results:
        noise = r["noise"]
        w = int(r["workers"])
        if w == 1:
            single_core[noise] = r["shots_per_s"]
            scaling_str = "--"
        else:
            sc = r["shots_per_s"] / single_core.get(noise, r["shots_per_s"])
            multi_core[noise] = r["shots_per_s"]
            scaling_str = f"{sc:.1f}x"
        print(
            f"{noise:>8.4f} {w:>8} {r['shots_per_s']:>12,.0f} "
            f"{r['survivors_per_s']:>10,.0f} {r['discard_pct']:>8.2f}% "
            f"{scaling_str:>8}"
        )

    # Cost projections
    print(f"\n{'='*90}")
    print(f"=== COST PROJECTIONS (target={args.target_errors} errors) ===")
    tput_source = multi_core if multi_core else single_core
    print(
        f"{'p':>8} {'Surv needed':>14} {'Total shots':>16} "
        f"{'Wall time':>12} {'Cost (spot)':>12}"
    )
    print("-" * 70)
    total_wall_h = 0.0
    total_cost = 0.0
    for noise in NOISE_LEVELS:
        info = PAPER_DATA[noise]
        survivors_needed = args.target_errors / info["rate"]
        total_shots = survivors_needed / (1 - info["discard"])
        tput = tput_source.get(noise, single_core.get(noise, 80_000))
        wall_s = total_shots / tput
        wall_h = wall_s / 3600
        cost = wall_h * args.spot_price
        total_wall_h += wall_h
        total_cost += cost
        print(
            f"{noise:>8.4f} {survivors_needed:>14,.0f} {total_shots:>16,.0f} "
            f"{format_duration(wall_s):>12} ${cost:>10.0f}"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':>8} {'':>14} {'':>16} "
        f"{format_duration(total_wall_h * 3600):>12} ${total_cost:>10.0f}"
    )
    print(f"\n(Spot price: ${args.spot_price:.2f}/hr)")

    # Also show cost for matching paper error counts
    paper_errors = {0.002: 22, 0.001: 49, 0.0005: 8}
    print("\n=== COST TO MATCH PAPER ERROR COUNTS (22, 49, 8) ===")
    total_wall_paper = 0.0
    total_cost_paper = 0.0
    for noise in NOISE_LEVELS:
        info = PAPER_DATA[noise]
        survivors_needed = paper_errors[noise] / info["rate"]
        total_shots = survivors_needed / (1 - info["discard"])
        tput = tput_source.get(noise, single_core.get(noise, 80_000))
        wall_s = total_shots / tput
        wall_h = wall_s / 3600
        cost = wall_h * args.spot_price
        total_wall_paper += wall_h
        total_cost_paper += cost
        print(
            f"{noise:>8.4f} errors={paper_errors[noise]:>3} "
            f"{format_duration(wall_s):>12} ${cost:>10.0f}"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':>8} {'':>14} "
        f"{format_duration(total_wall_paper * 3600):>12} ${total_cost_paper:>10.0f}"
    )


if __name__ == "__main__":
    main()
