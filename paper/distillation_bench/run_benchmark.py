"""Magic state distillation benchmark.

Benchmarks UCC and tsim on a logical magic state distillation circuit
(85 qubits, [17,1,5] color code) from Rodriguez et al. (2025).

The vendored circuit in ``circuits/`` was extracted from the tsim
demo notebook (``magic_state_distillation.ipynb``) using
``tsim.utils.encoder.ColorEncoder5``.  It encodes a 5-qubit
distillation protocol into the [17,1,5] color code with Z-basis
measurement.  The circuit has two noise scales:

  - **prep noise** (``p``): depolarizing noise on the 5 input magic
    states (default 0.05).
  - **circuit noise** (``p / 5``): depolarizing noise on the
    transversal distillation gates (default 0.01).

Other noise rates are produced by text substitution of both parameters.

Usage
-----
    python run_benchmark.py
    python run_benchmark.py --prep-noise 0.05,0.01 --simulators ucc,tsim
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

from bench_common import run_benchmark_loop

_HERE = Path(__file__).resolve().parent
_CIRCUITS_DIR = _HERE / "circuits"

# Template noise rates baked into the vendored circuit.
_TEMPLATE_PREP_NOISE = 0.05
_TEMPLATE_CIRCUIT_NOISE = 0.01  # prep / 5

_DEFAULT_SIMULATORS = "ucc,tsim"
_DEFAULT_PREP_NOISE = "0.05"

# Circuit noise / prep noise ratio (fixed by the protocol).
_NOISE_RATIO = 5


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------

_PREP_NOISE_RE = re.compile(
    r"(?P<prefix>DEPOLARIZE1\()" + re.escape(str(_TEMPLATE_PREP_NOISE)) + r"(?P<suffix>\))"
)

_CIRCUIT_NOISE_RE = re.compile(
    r"(?P<prefix>(?:DEPOLARIZE1|DEPOLARIZE2)\()"
    + re.escape(str(_TEMPLATE_CIRCUIT_NOISE))
    + r"(?P<suffix>\))"
)


def _load_distillation_circuit(prep_noise: float) -> str:
    """Load the distillation circuit and substitute noise rates.

    Returns raw program text (not a stim.Circuit) because these circuits
    contain non-Clifford gates (T_DAG, R_X) that stim cannot parse.
    """
    path = _CIRCUITS_DIR / "distillation.stim"
    text = path.read_text()

    circuit_noise = prep_noise / _NOISE_RATIO

    # Replace circuit noise first — its template value (0.01) could collide
    # with a user-specified prep noise, so it must be rewritten before the
    # prep noise substitution introduces the same literal.
    if circuit_noise != _TEMPLATE_CIRCUIT_NOISE:
        text = _CIRCUIT_NOISE_RE.sub(rf"\g<prefix>{circuit_noise}\g<suffix>", text)
    if prep_noise != _TEMPLATE_PREP_NOISE:
        text = _PREP_NOISE_RE.sub(rf"\g<prefix>{prep_noise}\g<suffix>", text)

    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run magic state distillation benchmark " "(85-qubit [17,1,5] color code).",
    )
    p.add_argument(
        "--prep-noise",
        type=str,
        default=_DEFAULT_PREP_NOISE,
        help=f"Comma-separated prep noise rates (default: {_DEFAULT_PREP_NOISE}). "
        f"Circuit noise is prep_noise / {_NOISE_RATIO}.",
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

    prep_rates = [float(p.strip()) for p in args.prep_noise.split(",")]
    simulators = [s.strip() for s in args.simulators.split(",") if s.strip()]

    circuits: list[tuple[dict[str, object], str]] = []
    for p in prep_rates:
        meta: dict[str, object] = {
            "circuit": f"distillation p={p:.0e}",
            "prep_noise": p,
            "circuit_noise": p / _NOISE_RATIO,
        }
        circuits.append((meta, _load_distillation_circuit(p)))

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
