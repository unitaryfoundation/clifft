"""Coherent noise (surface code) benchmark.

Benchmarks UCC and tsim on rotated surface code memory-Z circuits with
coherent R_Z over-rotation noise.  All depolarizing noise channels
(DEPOLARIZE1 and DEPOLARIZE2) are replaced with R_Z(alpha) on the
same qubit targets, modeling coherent noise after every gate and
during idle periods.  Measurement bit-flip errors (X_ERROR) are
preserved.

This follows the circuit-level coherent noise model of:

    Tuloup & Ayral, arXiv:2603.14670 (2025).

The default angle (alpha=0.02, i.e. ~0.063 rad) is a small
non-Clifford rotation suitable for benchmarking.

Stim cannot simulate non-Clifford gates.  tsim supports R_Z but its
compilation currently hangs on surface code circuits with rounds > 1
(tracked separately).  Default simulators: UCC only.

Usage
-----
    python run_benchmark.py
    python run_benchmark.py --distances 3,5
    python run_benchmark.py --rz-angle 0.01
    python run_benchmark.py --simulators ucc,tsim  # if tsim compilation is fixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import stim  # noqa: E402

from paper.bench_common import run_benchmark_loop  # noqa: E402

_HERE = Path(__file__).resolve().parent

_DEFAULT_SIMULATORS = "ucc"
_DEFAULT_RZ_ANGLE = 0.02
_DEFAULT_DEPOL_RATE = "1e-3"


# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------


def generate_coherent_circuit(
    distance: int,
    rounds: int,
    phys_error_rate: float,
    rz_angle: float,
) -> str:
    """Generate a surface code memory circuit with coherent R_Z noise.

    Starts from a stim-generated rotated surface code memory-Z circuit,
    then replaces every depolarizing noise channel with R_Z(alpha) on
    the same targets.  Measurement bit-flip errors are preserved.

    Returns raw program text (not a stim.Circuit) because the output
    contains R_Z gates that stim cannot parse.
    """
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=phys_error_rate,
        after_reset_flip_probability=phys_error_rate,
        before_measure_flip_probability=phys_error_rate,
        before_round_data_depolarization=phys_error_rate,
    )

    lines = []
    for line in str(c).split("\n"):
        s = line.strip()
        if s.startswith("DEPOLARIZE1(") or s.startswith("DEPOLARIZE2("):
            targets = s.split(")")[1].strip()
            lines.append(f"R_Z({rz_angle}) {targets}")
        else:
            lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run coherent R_Z noise surface code benchmarks.",
    )
    p.add_argument(
        "--distances",
        type=str,
        default="3,5",
        help="Comma-separated code distances (default: 3,5).",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Fixed number of rounds (default: rounds = distance).",
    )
    p.add_argument(
        "--rz-angle",
        type=float,
        default=_DEFAULT_RZ_ANGLE,
        help=f"R_Z rotation angle in units of pi (default: {_DEFAULT_RZ_ANGLE}).",
    )
    p.add_argument(
        "--depol-rate",
        type=str,
        default=_DEFAULT_DEPOL_RATE,
        help=f"Comma-separated base depolarizing error rates (default: {_DEFAULT_DEPOL_RATE}). "
        "Controls reset/measure flip probabilities.",
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
        help="Repetitions per (circuit, simulator) combo (default: 3).",
    )
    p.add_argument(
        "--simulators",
        type=str,
        default=_DEFAULT_SIMULATORS,
        help=f"Comma-separated simulators (default: {_DEFAULT_SIMULATORS}).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Fixed tsim batch size (skip autotuning). Useful once the "
        "optimal size is known for a given GPU.",
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
    depol_rates = [float(p.strip()) for p in args.depol_rate.split(",")]
    simulators = [s.strip() for s in args.simulators.split(",") if s.strip()]
    rz_angle = args.rz_angle

    circuits: list[tuple[dict[str, object], str]] = []
    for d in distances:
        rounds = args.rounds if args.rounds is not None else d
        for p in depol_rates:
            circuit_text = generate_coherent_circuit(d, rounds, p, rz_angle)

            meta: dict[str, object] = {
                "circuit": f"d={d} r={rounds} p={p:.0e} rz={rz_angle}",
                "distance": d,
                "rounds": rounds,
                "phys_error_rate": p,
                "rz_angle": rz_angle,
            }
            circuits.append((meta, circuit_text))

    results = run_benchmark_loop(
        circuits=circuits,
        simulators=simulators,
        shots=args.shots,
        repeats=args.repeats,
        batch_size=args.batch_size,
    )

    import pandas as pd

    csv_path = Path(args.output)
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
