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
import os
import time
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import stim

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
# Per-simulator runners
# ---------------------------------------------------------------------------


def run_stim(circuit: stim.Circuit, shots: int) -> dict[str, float]:
    t0 = time.perf_counter()
    sampler = circuit.compile_detector_sampler()
    compile_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    sampler.sample(shots, separate_observables=True)
    sample_s = time.perf_counter() - t1

    return {"compile_s": compile_s, "sample_s": sample_s}


def run_ucc(circuit: stim.Circuit, shots: int) -> dict[str, float]:
    # NOTE: ucc.sample() always materializes measurements, detectors, and
    # observables, whereas Stim/tsim compile_detector_sampler() only produces
    # detectors + observables.  For d=7 this is 385 vs 337 values per shot
    # (~6% difference in Stim's own timing), so UCC's numbers are slightly
    # conservative relative to the other backends.
    import ucc

    stim_text = str(circuit)

    t0 = time.perf_counter()
    prog = ucc.compile(
        stim_text,
        hir_passes=ucc.default_hir_pass_manager(),
        bytecode_passes=ucc.default_bytecode_pass_manager(),
    )
    compile_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    ucc.sample(prog, shots)
    sample_s = time.perf_counter() - t1

    return {"compile_s": compile_s, "sample_s": sample_s}


def run_tsim(circuit: stim.Circuit, shots: int) -> dict[str, float]:
    # Prevent JAX from pre-allocating the entire GPU memory.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # CPU vs GPU is controlled by setting JAX_PLATFORMS=cpu before launching
    # the script.  JAX reads this env var once at import time, so it cannot
    # be toggled within a single process.
    import tsim

    stim_text = str(circuit)

    t0 = time.perf_counter()
    tc = tsim.Circuit(stim_text)
    sampler = tc.compile_detector_sampler()
    compile_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    sampler.sample(shots, separate_observables=True)
    sample_s = time.perf_counter() - t1

    return {"compile_s": compile_s, "sample_s": sample_s}


RUNNERS: dict[str, Callable[[stim.Circuit, int], dict[str, float]]] = {
    "stim": run_stim,
    "ucc": run_ucc,
    "tsim": run_tsim,
}


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
    shots = args.shots
    repeats = args.repeats

    csv_path = Path(args.output)

    results: list[dict[str, object]] = []
    total = len(distances) * len(error_rates) * len(simulators) * repeats
    done = 0

    for d in distances:
        rounds = args.rounds if args.rounds is not None else d
        for p in error_rates:
            circuit = generate_circuit(d, rounds, p)
            for sim in simulators:
                runner = RUNNERS.get(sim)
                if runner is None:
                    print(f"Unknown simulator '{sim}', skipping.")
                    continue

                for rep in range(repeats):
                    done += 1
                    tag = (
                        f"[{done}/{total}] d={d} r={rounds} p={p:.0e} "
                        f"{sim} (rep {rep + 1}/{repeats})"
                    )
                    print(f"{tag} -> ", end="", flush=True)

                    row: dict[str, object] = {
                        "distance": d,
                        "rounds": rounds,
                        "phys_error_rate": p,
                        "shots": shots,
                        "simulator": sim,
                        "repeat": rep,
                    }

                    try:
                        timings = runner(circuit, shots)
                        row["status"] = "SUCCESS"
                        row["compile_s"] = round(timings["compile_s"], 6)
                        row["sample_s"] = round(timings["sample_s"], 6)
                        print(
                            f"SUCCESS (compile {timings['compile_s'] * 1e3:.1f}ms, "
                            f"sample {timings['sample_s']:.3f}s)"
                        )
                    except Exception as exc:  # noqa: BLE001
                        row["status"] = "ERROR"
                        row["compile_s"] = ""
                        row["sample_s"] = ""
                        print(f"ERROR ({type(exc).__name__}: {exc})")

                    results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
