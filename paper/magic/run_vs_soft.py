#!/usr/bin/env python3
"""Run UCC sampling on Magic State Cultivation circuits via Sinter.

This script registers UCC as a Sinter sampler and runs the d=3 MSC
cultivation circuit, producing a CSV of discard/error statistics.

Usage:
    uv run python paper/magic/run_vs_soft.py [--shots N] [--workers N]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

# Ensure the repo root is on sys.path so paper.magic is importable.
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sinter  # noqa: E402
import stim  # noqa: E402

from paper.magic.ucc_soft_sampler import UccSoftSampler  # noqa: E402

# Circuits to benchmark. Each entry is (label, path).
# The SOFT T-gate circuit is vendored but requires Stim >= 1.16 for T_DAG
# parsing, so it is excluded from the default list for now.
CIRCUIT_PATHS = [
    ("d3_ucc_target", ROOT / "tools" / "bench" / "target_qec.stim"),
]


def build_postselection_mask(
    circuit: stim.Circuit,
) -> np.ndarray:
    """Build a bit-packed postselection mask from detector coordinates.

    Detectors with a 5th coordinate of -9 are flagged for postselection,
    following the convention from the SOFT paper.
    """
    num_dets = circuit.num_detectors
    mask = bytearray(math.ceil(num_dets / 8))
    for k, coord in circuit.get_detector_coordinates().items():
        if len(coord) >= 5 and coord[4] == -9:
            mask[k // 8] |= 1 << (k % 8)
    return np.array(mask, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="UCC Magic State Cultivation benchmark")
    parser.add_argument("--shots", type=int, default=10_000, help="Max shots per task")
    parser.add_argument("--workers", type=int, default=1, help="Number of Sinter workers")
    parser.add_argument("--save", type=str, default=None, help="Path to save CSV results")
    args = parser.parse_args()

    tasks: list[sinter.Task] = []
    for label, path in CIRCUIT_PATHS:
        if not path.exists():
            print(f"Skipping {label}: {path} not found")
            continue
        circuit = stim.Circuit.from_file(str(path))
        mask = build_postselection_mask(circuit)
        tasks.append(
            sinter.Task(
                circuit=circuit,
                decoder="ucc",
                postselection_mask=mask,
                json_metadata={"label": label, "d": 3, "p": 0.001},
            )
        )

    if not tasks:
        print("No circuits found. Exiting.")
        return

    print(f"Running {len(tasks)} task(s) with {args.shots} shots, {args.workers} worker(s)...")

    results = sinter.collect(
        num_workers=args.workers,
        tasks=tasks,
        custom_decoders={"ucc": UccSoftSampler()},
        max_shots=args.shots,
        print_progress=True,
    )

    print("\n=== Results ===")
    for stat in results:
        meta = stat.json_metadata or {}
        label = meta.get("label", "unknown")
        total = stat.shots
        errors = stat.errors
        discards = stat.discards
        passed = total - discards
        discard_rate = discards / total if total > 0 else 0
        error_rate = errors / passed if passed > 0 else 0
        print(
            f"  {label}: {total} shots, {passed} passed, "
            f"{discards} discards ({discard_rate:.2%}), "
            f"{errors} errors (rate={error_rate:.6e})"
        )

    if args.save:
        save_path = pathlib.Path(args.save)
        with open(save_path, "w") as f:
            print(sinter.CSV_HEADER, file=f)
            for stat in results:
                print(stat.to_csv_line(), file=f)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
