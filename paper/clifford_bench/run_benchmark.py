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
from typing import Any, Sequence

# Must be set before JAX is imported (tsim depends on JAX).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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


_WARMUP_SHOTS = 64


class StimRunner:
    """Stim detector-sampler benchmark runner."""

    def compile(self, circuit: stim.Circuit, shots: int) -> None:
        self._sampler = circuit.compile_detector_sampler()
        # Warmup: populate CPU caches / branch predictors.
        self._sampler.sample(min(_WARMUP_SHOTS, shots), separate_observables=True)

    def sample(self, shots: int) -> None:
        self._sampler.sample(shots, separate_observables=True)


class UccRunner:
    """UCC benchmark runner.

    NOTE: ucc.sample() always materializes measurements, detectors, and
    observables, whereas Stim/tsim compile_detector_sampler() only produces
    detectors + observables.  For d=7 this is 385 vs 337 values per shot
    (~6% difference in Stim's own timing), so UCC's numbers are slightly
    conservative relative to the other backends.
    """

    def compile(self, circuit: stim.Circuit, shots: int) -> None:
        import ucc

        self._prog = ucc.compile(
            str(circuit),
            hir_passes=ucc.default_hir_pass_manager(),
            bytecode_passes=ucc.default_bytecode_pass_manager(),
        )
        # Warmup: populate CPU caches / branch predictors.
        ucc.sample(self._prog, min(_WARMUP_SHOTS, shots))
        self._ucc = ucc

    def sample(self, shots: int) -> None:
        self._ucc.sample(self._prog, shots)


def _autotune_tsim_batch_size(sampler: Any, shots: int, separate_observables: bool = True) -> int:
    """Find the batch size that maximizes tsim throughput.

    Starts from a small batch and doubles until throughput stops improving
    or the batch no longer fits in device memory, mirroring the strategy
    described in the tsim paper.  As a side-effect, the winning batch size
    also serves as the JIT warmup.

    Raises on the first probe if the sampler fails for any reason (including
    OOM), since there is no valid batch size to fall back to.
    """
    batch = min(64, shots)
    best_throughput = 0.0
    best_batch = batch

    first = True
    while batch <= shots:
        try:
            t0 = time.perf_counter()
            sampler.sample(batch, separate_observables=separate_observables)
            elapsed = time.perf_counter() - t0
        except Exception:
            if first:
                # No successful batch yet — propagate so the caller sees
                # the real error (OOM, API mismatch, backend bug, etc.).
                raise
            # Later iterations may hit device memory limits; keep the
            # previous best.
            break

        first = False
        throughput = batch / elapsed
        if throughput > best_throughput:
            best_throughput = throughput
            best_batch = batch
        else:
            # Throughput stopped improving — done.
            break
        batch *= 2

    return best_batch


class TsimRunner:
    """Tsim benchmark runner with batch-size autotuning."""

    def __init__(self, batch_size: int | None = None) -> None:
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    def compile(self, circuit: stim.Circuit, shots: int) -> None:
        # CPU vs GPU is controlled by setting JAX_PLATFORMS=cpu before
        # launching the script.  JAX reads this env var once at import time,
        # so it cannot be toggled within a single process.
        import tsim

        tc = tsim.Circuit(str(circuit))
        self._sampler = tc.compile_detector_sampler()
        if self._batch_size is not None:
            # User-specified batch size — just warm up the JIT.
            self._sampler.sample(min(self._batch_size, shots), separate_observables=True)
        else:
            # Autotune batch size (also warms up the JIT).
            self._batch_size = _autotune_tsim_batch_size(self._sampler, shots)

    def sample(self, shots: int) -> None:
        self._sampler.sample(shots, batch_size=self._batch_size, separate_observables=True)


RUNNERS: dict[str, type] = {
    "stim": StimRunner,
    "ucc": UccRunner,
    "tsim": TsimRunner,
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
    error_rates = [float(p.strip()) for p in args.error_rates.split(",")]
    simulators = [s.strip() for s in args.simulators.split(",") if s.strip()]
    shots = args.shots
    repeats = args.repeats

    csv_path = Path(args.output)
    batch_size: int | None = args.batch_size

    results: list[dict[str, object]] = []
    total = len(distances) * len(error_rates) * len(simulators) * repeats
    done = 0

    for d in distances:
        rounds = args.rounds if args.rounds is not None else d
        for p in error_rates:
            circuit = generate_circuit(d, rounds, p)
            for sim in simulators:
                factory = RUNNERS.get(sim)
                if factory is None:
                    print(f"Unknown simulator '{sim}', skipping.")
                    continue

                # Compile once per (circuit, simulator).  For tsim this
                # also autotunes the batch size and warms up the JIT;
                # for Stim/UCC it warms CPU caches.
                runner = factory(batch_size=batch_size) if factory is TsimRunner else factory()
                header = f"d={d} r={rounds} p={p:.0e} {sim}"
                print(f"  {header}: compiling ...", end="", flush=True)
                try:
                    t0 = time.perf_counter()
                    runner.compile(circuit, shots)
                    compile_s = time.perf_counter() - t0
                except Exception as exc:  # noqa: BLE001
                    print(f" ERROR ({type(exc).__name__}: {exc})")
                    for rep in range(repeats):
                        done += 1
                        results.append(
                            {
                                "distance": d,
                                "rounds": rounds,
                                "phys_error_rate": p,
                                "shots": shots,
                                "simulator": sim,
                                "repeat": rep,
                                "status": "ERROR",
                                "compile_s": "",
                                "sample_s": "",
                            }
                        )
                    continue

                batch_info = ""
                if isinstance(runner, TsimRunner) and runner.batch_size is not None:
                    batch_info = f", batch_size={runner.batch_size}"
                print(f" {compile_s * 1e3:.1f}ms{batch_info}")

                for rep in range(repeats):
                    done += 1
                    tag = f"[{done}/{total}] {header} (rep {rep + 1}/{repeats})"
                    print(f"  {tag} -> ", end="", flush=True)

                    row: dict[str, object] = {
                        "distance": d,
                        "rounds": rounds,
                        "phys_error_rate": p,
                        "shots": shots,
                        "simulator": sim,
                        "repeat": rep,
                    }

                    try:
                        t1 = time.perf_counter()
                        runner.sample(shots)
                        sample_s = time.perf_counter() - t1

                        row["status"] = "SUCCESS"
                        row["compile_s"] = round(compile_s, 6)
                        row["sample_s"] = round(sample_s, 6)
                        print(f"SUCCESS ({sample_s:.3f}s)")
                    except Exception as exc:  # noqa: BLE001
                        row["status"] = "ERROR"
                        row["compile_s"] = round(compile_s, 6)
                        row["sample_s"] = ""
                        print(f"ERROR ({type(exc).__name__}: {exc})")

                    results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
