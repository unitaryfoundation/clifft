"""Orchestrator for Quantum Volume benchmarks.

Spawns isolated subprocess workers for each (simulator, N, seed)
combination and collects results into a CSV file.

Usage
-----
    python run_benchmark.py --min-q 6 --max-q 26 --repeats 3
    python run_benchmark.py --qubits 10,14,18,22 --simulators ucc,qiskit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SIMULATORS: str = "ucc,qiskit,qulacs,qsim,qrack"

_HERE: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    p = argparse.ArgumentParser(
        description="Run Quantum Volume benchmarks across simulators.",
    )
    p.add_argument(
        "--min-q",
        type=int,
        default=6,
        help="Minimum qubit count (default: 6).",
    )
    p.add_argument(
        "--max-q",
        type=int,
        default=26,
        help="Maximum qubit count (default: 26).",
    )
    p.add_argument(
        "--qubits",
        type=str,
        default=None,
        help=(
            "Explicit comma-separated list of N values, "
            "e.g. --qubits 10,14,18,22. Overrides --min-q/--max-q."
        ),
    )
    p.add_argument(
        "--step",
        type=int,
        default=2,
        help="Step size for qubit range when using --min-q/--max-q (default: 2).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per (N, simulator) combo (default: 3).",
    )
    p.add_argument(
        "--simulators",
        type=str,
        default=_DEFAULT_SIMULATORS,
        help=f"Comma-separated list of simulators to benchmark (default: {_DEFAULT_SIMULATORS}).",
    )
    p.add_argument(
        "--mem-limit-gb",
        type=float,
        default=6.0,
        help="Per-worker RLIMIT_AS memory cap in GB (default: 6.0).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-worker timeout in seconds (default: 300).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_HERE / "results.csv"),
        help="Output CSV path (default: results.csv in this directory).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed (default: 42). Each repeat uses seed+i.",
    )
    return p


def _parse_qubit_list(args: argparse.Namespace) -> list[int]:
    """Derive the list of qubit counts from parsed CLI arguments."""
    if args.qubits is not None:
        return [int(q.strip()) for q in args.qubits.split(",") if q.strip()]
    return list(range(args.min_q, args.max_q + 1, args.step))


# ---------------------------------------------------------------------------
# Worker invocation
# ---------------------------------------------------------------------------


def _make_env(mem_limit_gb: float) -> dict[str, str]:
    """Build an environment dict for the subprocess worker."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["UCC_BENCH_MEM_LIMIT_GB"] = str(mem_limit_gb)
    return env


def _classify_failure(
    returncode: int | None,
    stderr: str,
) -> str:
    """Classify a worker failure as OOM, TIMEOUT, or ERROR."""
    if returncode is not None and returncode in (-9, 137):
        return "OOM"
    if "MemoryError" in stderr or "bad_alloc" in stderr:
        return "OOM"
    return "ERROR"


def _run_worker(
    simulator: str,
    n_qubits: int,
    seed: int,
    timeout: int,
    env: dict[str, str],
) -> dict[str, Any]:
    """Spawn a single benchmark worker and return the parsed result dict.

    Returns a dict with at least ``status``.  On success the dict also
    contains ``exec_s``, ``compile_s``, ``sample_s``, and ``peak_mb``.
    """
    cmd: list[str] = [
        sys.executable,
        str(_HERE / "worker.py"),
        simulator,
        str(n_qubits),
        str(seed),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}

    if proc.returncode != 0:
        status = _classify_failure(proc.returncode, proc.stderr)
        return {"status": status}

    # Find the JSON result line in stdout.  Some backends (e.g. Qrack's native
    # library) print non-JSON diagnostics to stdout, so scan in reverse for
    # the first parseable JSON object.
    stdout_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    for ln in reversed(stdout_lines):
        try:
            data: dict[str, Any] = json.loads(ln)
            data.setdefault("status", "SUCCESS")
            return data
        except (json.JSONDecodeError, ValueError):
            continue
    return {"status": "ERROR"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the benchmark orchestrator."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    qubit_list = _parse_qubit_list(args)
    simulators = [s.strip() for s in args.simulators.split(",") if s.strip()]
    base_seed: int = args.seed
    repeats: int = args.repeats
    timeout: int = args.timeout
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    env = _make_env(args.mem_limit_gb)

    _CSV_COLUMNS = [
        "N",
        "simulator",
        "repeat",
        "seed",
        "status",
        "exec_s",
        "compile_s",
        "sample_s",
        "peak_mb",
    ]
    # Write header (overwrites any existing file).
    pd.DataFrame(columns=_CSV_COLUMNS).to_csv(csv_path, index=False)

    total = len(qubit_list) * len(simulators) * repeats
    done = 0

    for n in qubit_list:
        for sim in simulators:
            for rep in range(repeats):
                seed = base_seed + rep
                done += 1
                tag = f"[{done}/{total}] [N={n}] {sim} (repeat {rep + 1}/{repeats})"
                print(f"{tag} -> ", end="", flush=True)

                result = _run_worker(sim, n, seed, timeout, env)
                status: str = result.get("status", "ERROR")

                row: dict[str, object] = {
                    "N": n,
                    "simulator": sim,
                    "repeat": rep,
                    "seed": seed,
                    "status": status,
                    "exec_s": result.get("exec_s", ""),
                    "compile_s": result.get("compile_s", ""),
                    "sample_s": result.get("sample_s", ""),
                    "peak_mb": result.get("peak_mb", ""),
                }

                # Append incrementally so partial results survive interruptions.
                pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)

                if status == "SUCCESS":
                    try:
                        print(
                            f"SUCCESS ({float(row['exec_s']):.2f}s"  # type: ignore[arg-type]
                            f" | {float(row['peak_mb']):.1f}MB)"  # type: ignore[arg-type]
                        )
                    except (ValueError, TypeError):
                        print("SUCCESS")
                else:
                    print(status)

    print(f"\nResults written to {csv_path}")

    # Print summary table
    df = pd.read_csv(csv_path)
    success = df[df["status"] == "SUCCESS"].copy()
    if not success.empty:
        success["exec_s"] = success["exec_s"].astype(float)
        summary = (
            success.groupby(["N", "simulator"])["exec_s"].median().unstack(fill_value=float("nan"))
        )
        print("\n  Quantum Volume Benchmark Summary (median exec_s)")
        print(summary.to_string(float_format=lambda x: f"{x:.2f}s"))
        print()


if __name__ == "__main__":
    main()
