"""Near-Clifford benchmark: magic state cultivation.

Benchmarks UCC and tsim on magic state cultivation circuits (d=3, d=5)
at various physical error rates.  Stim cannot simulate non-Clifford
circuits and is not included.

The template circuits in ``circuits/`` were generated at p=0.005.
Other error rates are produced by text substitution of the noise
parameter.

Usage
-----
    python run_benchmark.py
    python run_benchmark.py --distances 3,5 --simulators ucc,tsim
    python run_benchmark.py --distances 3 --error-rates 1e-3,5e-3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

from bench_common import run_benchmark_loop

_HERE = Path(__file__).resolve().parent
_CIRCUITS_DIR = _HERE / "circuits"

# Template noise rate baked into the vendored circuit files.
_TEMPLATE_ERROR_RATE = 0.005

_DEFAULT_SIMULATORS = "ucc,tsim"
_DEFAULT_ERROR_RATES = "1e-3,5e-3"


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------

# Matches all noise parameters: gate errors, depolarizing channels, and
# noisy measurements like M(0.005), MX(0.005), etc.
_NOISE_RE = re.compile(
    r"(?P<prefix>(?:X_ERROR|Y_ERROR|Z_ERROR|DEPOLARIZE1|DEPOLARIZE2|MX?|MY|MZ)\()"
    + re.escape(str(_TEMPLATE_ERROR_RATE))
    + r"(?P<suffix>\))"
)


def _load_cultivation_circuit(distance: int, error_rate: float) -> str:
    """Load a cultivation circuit template and substitute the noise rate.

    Returns raw program text (not a stim.Circuit) because these circuits
    contain non-Clifford gates (T, T_DAG) that stim cannot parse.
    """
    path = _CIRCUITS_DIR / f"cultivation_d{distance}.stim"
    text = path.read_text()
    if error_rate != _TEMPLATE_ERROR_RATE:
        text = _NOISE_RE.sub(rf"\g<prefix>{error_rate}\g<suffix>", text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run near-Clifford magic state cultivation benchmarks.",
    )
    p.add_argument(
        "--distances",
        type=str,
        default="3,5",
        help="Comma-separated code distances (default: 3,5).",
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
        help="Repetitions per (circuit, simulator) combo (default: 3).",
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

    circuits: list[tuple[dict[str, object], str]] = []
    for d in distances:
        for p in error_rates:
            meta: dict[str, object] = {
                "circuit": f"cultivation d={d} p={p:.0e}",
                "distance": d,
                "phys_error_rate": p,
            }
            circuits.append((meta, _load_cultivation_circuit(d, p)))

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
