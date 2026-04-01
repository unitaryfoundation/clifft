"""Benchmark runner for the Three Exponential Walls comparison.

Usage:
    # Full production sweep (requires GPU for tsim, 30GB+ RAM)
    python runner.py

    # Local validation on a small VM (UCC + Qiskit only, small params)
    python runner.py --local
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Any

from generator import generate_boundary_circuit

TIMEOUT_S = 120
RESULTS_FILE = "results_walls.csv"
CSV_HEADER = ["Panel", "N", "t", "k", "Tool", "Status", "Exec_s", "PeakMem_MB"]

# Sweep definitions: (Panel_Name, N_list, t_list, k_list, Tools)
SweepDef = tuple[str, list[int], list[int], list[int], list[str]]

FULL_SWEEPS: list[SweepDef] = [
    ("Panel_A", [20, 24, 28, 30, 32, 36, 40], [20], [15], ["qiskit", "ucc", "tsim"]),
    ("Panel_B", [50], [40], [10, 15, 20, 25, 30, 32], ["ucc", "tsim"]),
    ("Panel_C", [50], [10, 30, 50, 70, 90, 120], [10], ["ucc", "tsim"]),
]

# Scaled-down sweeps for local testing on limited hardware.
# Panel A: Qiskit hits 2^N memory wall; UCC stays at 2^k=2^12.
# Panel B: Qiskit flat at 2^24=256MB; UCC climbs as k grows.
# Panel C: Both scale linearly in t; UCC has smaller constant (2^k vs 2^N).
LOCAL_SWEEPS: list[SweepDef] = [
    ("Panel_A", [16, 20, 24, 26, 28, 29], [20], [12], ["qiskit", "ucc"]),
    ("Panel_B", [24], [40], [8, 12, 16, 20, 22, 24, 25], ["qiskit", "ucc"]),
    ("Panel_C", [26], [10, 100, 500, 1000, 5000], [12], ["qiskit", "ucc"]),
]


def check_theoretical_oom(tool: str, N: int, k: int, max_ram_gb: float) -> bool:
    """Short-circuit simulations that mathematically exceed physical RAM."""
    # 16 bytes per complex<double>. Leave 2GB headroom for OS/Python.
    headroom_gb = max(max_ram_gb - 2, 0)
    max_elements = int((headroom_gb * 1024**3) // 16)

    if tool == "qiskit":
        return bool((2**N) > max_elements)
    elif tool == "ucc":
        return bool((2**k) > max_elements)
    return False


def run_worker_process(
    tool: str, stim_file: str, qasm_file: str, mem_limit_gb: float
) -> dict[str, Any]:
    """Run the simulation in an isolated subprocess."""
    cmd = [sys.executable, __file__, "--internal-worker", tool, stim_file, qasm_file]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["UCC_BENCH_MEM_LIMIT_GB"] = str(mem_limit_gb)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env)

        if (
            result.returncode in (-9, 137)
            or "MemoryError" in result.stderr
            or "bad_alloc" in result.stderr
        ):
            return {"status": "OOM", "exec_s": 0.0, "peak_mb": 0.0}

        if result.returncode != 0:
            return {
                "status": "ERROR",
                "exec_s": 0.0,
                "peak_mb": 0.0,
                "msg": result.stderr.strip()[:200],
            }

        lines = result.stdout.strip().split("\n")
        try:
            return json.loads(lines[-1])  # type: ignore[no-any-return]
        except (json.JSONDecodeError, IndexError):
            return {
                "status": "ERROR",
                "exec_s": 0.0,
                "peak_mb": 0.0,
                "msg": "Bad output",
            }

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "exec_s": float(TIMEOUT_S), "peak_mb": 0.0}


def execute_internal(tool: str, stim_file: str, qasm_file: str) -> None:
    """Payload executed inside the isolated subprocess."""
    import resource

    # Apply RLIMIT_AS memory cap on Linux before importing heavy libraries.
    if sys.platform.startswith("linux"):
        mem_limit_str = os.environ.get("UCC_BENCH_MEM_LIMIT_GB")
        if mem_limit_str:
            mem_limit_bytes = int(float(mem_limit_str) * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))

    start_t = time.perf_counter()

    if tool == "qiskit":
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit.from_qasm_file(qasm_file)
        sim = AerSimulator(method="statevector", max_parallel_threads=1)
        sim.run(transpile(qc, sim), shots=1).result()

    elif tool == "tsim":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        import tsim

        with open(stim_file) as f:
            circuit = tsim.Circuit(f.read())
        sampler = circuit.compile_sampler()
        sampler.sample(shots=1)

    elif tool == "ucc":
        import ucc

        with open(stim_file) as f:
            program = ucc.compile(f.read())
        ucc.sample(program, shots=1)

    exec_s = time.perf_counter() - start_t

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS
    peak_mb = (peak_kb / 1024) if sys.platform.startswith("linux") else (peak_kb / (1024 * 1024))

    print(json.dumps({"status": "SUCCESS", "exec_s": exec_s, "peak_mb": peak_mb}))
    sys.exit(0)


def run_sweep(sweeps: list[SweepDef], max_ram_gb: float, mem_limit_gb: float) -> None:
    """Iterate through the parameter phase space."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, mode="w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    for panel, n_list, t_list, k_list, tools in sweeps:
        print(f"\n--- Starting {panel} ---")
        for n_val in n_list:
            for t_val in t_list:
                for k_val in k_list:
                    base = f"circuits/bench_{panel}_N{n_val}_t{t_val}_k{k_val}"
                    stim_f, qasm_f = generate_boundary_circuit(n_val, t_val, k_val, base)

                    for tool in tools:
                        label = f"[{panel}] {tool:<7} (N={n_val:<3}, t={t_val:<3}, k={k_val:<2})"
                        print(f"{label} -> ", end="", flush=True)

                        if check_theoretical_oom(tool, n_val, k_val, max_ram_gb):
                            print("SKIPPED (Math OOM)")
                            res: dict[str, Any] = {
                                "status": "OOM",
                                "exec_s": 0.0,
                                "peak_mb": 0.0,
                            }
                        else:
                            res = run_worker_process(tool, stim_f, qasm_f, mem_limit_gb)
                            status = res["status"]
                            secs = res["exec_s"]
                            mb = res["peak_mb"]
                            print(f"{status} ({secs:.2f}s | {mb:.1f}MB)")

                        with open(RESULTS_FILE, mode="a", newline="") as f:
                            csv.writer(f).writerow(
                                [
                                    panel,
                                    n_val,
                                    t_val,
                                    k_val,
                                    tool,
                                    res["status"],
                                    res["exec_s"],
                                    res["peak_mb"],
                                ]
                            )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Three Walls benchmark runner")
    parser.add_argument(
        "--internal-worker",
        nargs=3,
        metavar=("TOOL", "STIM_FILE", "QASM_FILE"),
        help="(internal) Run a single simulation in subprocess isolation.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use scaled-down sweeps for local testing on limited hardware.",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=None,
        help="Max RAM budget in GB (default: 7 for --local, 30 for full).",
    )
    parser.add_argument(
        "--mem-limit-gb",
        type=float,
        default=6.5,
        help="Per-worker RLIMIT_AS memory cap in GB (default: 6.5).",
    )
    args = parser.parse_args()

    if args.internal_worker:
        execute_internal(args.internal_worker[0], args.internal_worker[1], args.internal_worker[2])
    else:
        if args.local:
            sweeps = LOCAL_SWEEPS
            max_ram = args.max_ram_gb if args.max_ram_gb is not None else 7.0
        else:
            sweeps = FULL_SWEEPS
            max_ram = args.max_ram_gb if args.max_ram_gb is not None else 30.0
        print(
            f"Mode: {'LOCAL' if args.local else 'FULL'}"
            f" | RAM budget: {max_ram}GB"
            f" | Worker mem limit: {args.mem_limit_gb}GB"
        )
        run_sweep(sweeps, max_ram, args.mem_limit_gb)


if __name__ == "__main__":
    main()
