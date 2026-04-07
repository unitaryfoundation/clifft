"""Clifford-limit (surface code) benchmark.

Compares UCC, Stim, and tsim on rotated surface code memory-Z circuits
with depolarizing noise, sweeping physical error rate.

Usage
-----
    python run_benchmark.py
    python run_benchmark.py --distances 3,5,7 --simulators ucc,stim
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import stim
from bench_common import run_benchmark_loop

_HERE = Path(__file__).resolve().parent

_DEFAULT_SIMULATORS = "ucc,stim,tsim"
_DEFAULT_ERROR_RATES = "1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1"


# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------


def generate_circuit(distance: int, rounds: int, phys_error_rate: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=phys_error_rate,
        after_reset_flip_probability=phys_error_rate,
        before_measure_flip_probability=phys_error_rate,
        before_round_data_depolarization=phys_error_rate,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Clifford-limit surface code benchmarks.",
    )
    p.add_argument(
        "--distances",
        type=str,
        default="7",
        help="Comma-separated code distances (default: 7).",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Fixed number of rounds (default: rounds = distance).",
    )
    p.add_argument(
        "--error-rates",
        type=str,
        default=_DEFAULT_ERROR_RATES,
        help=f"Comma-separated physical error rates (default: {_DEFAULT_ERROR_RATES}).",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=1_000_000,
        help="Number of shots per run (default: 1000000).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repetitions per (distance, p, simulator) combo (default: 3).",
    )
    p.add_argument(
        "--simulators",
        type=str,
        default=_DEFAULT_SIMULATORS,
        help=f"Comma-separated simulators (default: {_DEFAULT_SIMULATORS}).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_HERE / "results.csv"),
        help="Output CSV path (default: results.csv in this directory).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    distances = [int(d.strip()) for d in args.distances.split(",")]
    error_rates = [float(p.strip()) for p in args.error_rates.split(",")]
    simulators = [s.strip() for s in args.simulators.split(",") if s.strip()]

    # Build (metadata, circuit) pairs for each (distance, error_rate).
    circuits: list[tuple[dict[str, object], stim.Circuit]] = []
    for d in distances:
        rounds = args.rounds if args.rounds is not None else d
        for p in error_rates:
            meta: dict[str, object] = {
                "distance": d,
                "rounds": rounds,
                "phys_error_rate": p,
                "circuit": f"d={d} r={rounds} p={p:.0e}",
            }
            circuits.append((meta, generate_circuit(d, rounds, p)))

    csv_path = Path(args.output)
    run_benchmark_loop(
        circuits=circuits,
        simulators=simulators,
        shots=args.shots,
        repeats=args.repeats,
        output_csv=csv_path,
    )

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
