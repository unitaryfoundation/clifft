"""Orchestrator for Quantum Volume benchmarks.

Spawns isolated subprocess workers for each (simulator, N, seed)
combination and collects results into a CSV file.

Usage
-----
    python -m paper.qv_benchmark.run_benchmark --min-q 6 --max-q 26 --repeats 3
    python -m paper.qv_benchmark.run_benchmark --qubits 10,14,18,22 --simulators ucc,qiskit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CSV_HEADER: List[str] = [
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

_DEFAULT_SIMULATORS: str = "ucc,qiskit,qulacs,qsim"

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


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
        help=(
            "Comma-separated list of simulators to benchmark " f"(default: {_DEFAULT_SIMULATORS})."
        ),
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
        default="paper/qv_benchmark/results.csv",
        help="Output CSV path (default: paper/qv_benchmark/results.csv).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed (default: 42). Each repeat uses seed+i.",
    )
    return p


def _parse_qubit_list(args: argparse.Namespace) -> List[int]:
    """Derive the list of qubit counts from parsed CLI arguments."""
    if args.qubits is not None:
        return [int(q.strip()) for q in args.qubits.split(",") if q.strip()]
    return list(range(args.min_q, args.max_q + 1, args.step))


# ---------------------------------------------------------------------------
# Worker invocation
# ---------------------------------------------------------------------------


def _make_env(mem_limit_gb: float) -> Dict[str, str]:
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
    env: Dict[str, str],
) -> Dict[str, Any]:
    """Spawn a single benchmark worker and return the parsed result dict.

    Returns a dict with at least ``status``.  On success the dict also
    contains ``exec_s``, ``compile_s``, ``sample_s``, and ``peak_mb``.
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "paper.qv_benchmark.worker",
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
            cwd=str(_PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}

    if proc.returncode != 0:
        status = _classify_failure(proc.returncode, proc.stderr)
        return {"status": status}

    # Parse the last non-empty line of stdout as JSON.
    stdout_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not stdout_lines:
        return {"status": "ERROR"}

    try:
        data: Dict[str, Any] = json.loads(stdout_lines[-1])
    except (json.JSONDecodeError, IndexError):
        return {"status": "ERROR"}

    data.setdefault("status", "SUCCESS")
    return data


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _ensure_csv_header(csv_path: Path) -> None:
    """Write the CSV header if the file does not exist or is empty."""
    write_header = False
    if not csv_path.exists():
        write_header = True
    elif csv_path.stat().st_size == 0:
        write_header = True

    if write_header:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(_CSV_HEADER)


def _append_row(csv_path: Path, row: Sequence[Any]) -> None:
    """Append a single data row to the CSV."""
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(csv_path: Path) -> None:
    """Read the CSV and print a human-readable summary table to stdout."""
    if not csv_path.exists():
        return

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        return

    # Collect unique N and simulator values (preserve order).
    n_values: List[int] = []
    sim_values: List[str] = []
    for r in rows:
        n = int(r["N"])
        s = r["simulator"]
        if n not in n_values:
            n_values.append(n)
        if s not in sim_values:
            sim_values.append(s)

    n_values.sort()

    # Build lookup: (N, sim) -> list of status strings
    from collections import defaultdict

    lookup: Dict[tuple[int, str], List[str]] = defaultdict(list)
    for r in rows:
        key = (int(r["N"]), r["simulator"])
        status = r["status"]
        if status == "SUCCESS":
            exec_s = r.get("exec_s", "")
            try:
                summary_val = f"{float(exec_s):.2f}s"
            except (ValueError, TypeError):
                summary_val = "OK"
        else:
            summary_val = status
        lookup[key].append(summary_val)

    # Print table
    col_w = 16
    header_parts = [f"{'N':>4}"] + [f"{s:>{col_w}}" for s in sim_values]
    sep = "-" * len("  ".join(header_parts))

    print()
    print("=" * len(sep))
    print("  Quantum Volume Benchmark Summary")
    print("=" * len(sep))
    print("  ".join(header_parts))
    print(sep)

    for n in n_values:
        parts = [f"{n:>4}"]
        for s in sim_values:
            entries = lookup.get((n, s), ["-"])
            cell = ",".join(entries)
            parts.append(f"{cell:>{col_w}}")
        print("  ".join(parts))

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the benchmark orchestrator."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    qubit_list: List[int] = _parse_qubit_list(args)
    simulators: List[str] = [s.strip() for s in args.simulators.split(",") if s.strip()]
    base_seed: int = args.seed
    repeats: int = args.repeats
    timeout: int = args.timeout
    csv_path = Path(args.output)
    if not csv_path.is_absolute():
        csv_path = _PROJECT_ROOT / csv_path

    env = _make_env(args.mem_limit_gb)

    _ensure_csv_header(csv_path)

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

                exec_s = result.get("exec_s", "")
                compile_s = result.get("compile_s", "")
                sample_s = result.get("sample_s", "")
                peak_mb = result.get("peak_mb", "")

                row = [
                    n,
                    sim,
                    rep,
                    seed,
                    status,
                    exec_s,
                    compile_s,
                    sample_s,
                    peak_mb,
                ]
                _append_row(csv_path, row)

                if status == "SUCCESS":
                    try:
                        print(f"SUCCESS ({float(exec_s):.2f}s | {float(peak_mb):.1f}MB)")
                    except (ValueError, TypeError):
                        print("SUCCESS")
                else:
                    print(status)

    _print_summary(csv_path)


if __name__ == "__main__":
    main()
