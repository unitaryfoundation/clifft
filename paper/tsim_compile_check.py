#!/usr/bin/env python3
"""Smoke-test tsim compilation on all benchmark circuits.

Reports which circuits tsim can compile (with default and cutting
strategies) and which time out.  Each compilation runs in a separate
subprocess so that hung compiles are fully killed on timeout.

Requires: bloqade-tsim >= 0.1.2, stim, clifft

Usage:
    JAX_PLATFORMS=cpu uv run --with 'bloqade-tsim>=0.1.2' \
        python tsim_compile_check.py
    JAX_PLATFORMS=cpu uv run --with 'bloqade-tsim>=0.1.2' \
        python tsim_compile_check.py --timeout 120
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

# Subprocess script that compiles a circuit and prints JSON result.
_COMPILE_SCRIPT = textwrap.dedent("""\
    import json, os, sys, time
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import tsim
    circuit_text = sys.stdin.read()
    strategy = sys.argv[1]
    tc = tsim.Circuit(circuit_text)
    t0 = time.time()
    sampler = tc.compile_detector_sampler(strategy=strategy)
    elapsed = time.time() - t0
    total = sum(
        csg.num_graphs
        for comp in sampler._program.components
        for csg in comp.compiled_scalar_graphs
    )
    json.dump({"time_s": round(elapsed, 1), "num_graphs": total}, sys.stdout)
""")


def _try_compile(
    label: str,
    circuit_text: str,
    strategy: str,
    timeout: int,
) -> dict:
    """Attempt tsim compilation in a subprocess with timeout."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _COMPILE_SCRIPT, strategy],
            input=circuit_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode == 0:
            result = json.loads(proc.stdout)
            print(
                f"  {label:45s} [{strategy:7s}]  "
                f"OK {result['time_s']:6.1f}s  graphs={result['num_graphs']}"
            )
            return {
                "circuit": label,
                "strategy": strategy,
                "status": "OK",
                "time_s": result["time_s"],
                "num_graphs": result["num_graphs"],
            }
        else:
            err_msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown"
            print(f"  {label:45s} [{strategy:7s}]  ERROR: {err_msg}")
            return {
                "circuit": label,
                "strategy": strategy,
                "status": "ERROR",
                "time_s": None,
                "num_graphs": None,
            }

    except subprocess.TimeoutExpired:
        print(f"  {label:45s} [{strategy:7s}]  TIMEOUT ({timeout}s)")
        return {
            "circuit": label,
            "strategy": strategy,
            "status": "TIMEOUT",
            "time_s": None,
            "num_graphs": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test tsim compilation.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-circuit timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="default,cutting",
        help="Comma-separated strategies to test (default: default,cutting).",
    )
    args = parser.parse_args()
    timeout = args.timeout
    strategies = [s.strip() for s in args.strategies.split(",")]

    import stim

    circuits: list[tuple[str, str]] = []

    # --- Clifford bench (match exact circuit from clifford_bench/run_benchmark.py) ---
    for d in [3, 5, 7]:
        c = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d,
            distance=d,
            after_clifford_depolarization=1e-3,
            after_reset_flip_probability=1e-3,
            before_measure_flip_probability=1e-3,
            before_round_data_depolarization=1e-3,
        )
        circuits.append((f"clifford d={d} r={d}", str(c)))

    # --- Near-Clifford (cultivation) ---
    circuits_dir = Path(__file__).resolve().parent / "near_clifford_bench" / "circuits"
    for f in sorted(circuits_dir.glob("*.stim")):
        circuits.append((f"cultivation {f.stem}", f.read_text()))

    # --- Distillation ---
    dist_path = (
        Path(__file__).resolve().parent / "distillation_bench" / "circuits" / "distillation.stim"
    )
    if dist_path.exists():
        circuits.append(("distillation 85q", dist_path.read_text()))

    # --- Coherent noise ---
    from coherent_noise_bench.run_benchmark import generate_coherent_circuit

    for d, r in [(3, 1), (3, 3), (5, 1), (5, 5)]:
        ct = generate_coherent_circuit(d, r, 1e-3, 0.02)
        circuits.append((f"coherent d={d} r={r}", ct))

    # --- Run ---
    print(
        f"Testing {len(circuits)} circuits x {len(strategies)} strategies "
        f"(timeout={timeout}s)\n"
    )

    results = []
    for label, text in circuits:
        for strategy in strategies:
            results.append(_try_compile(label, text, strategy, timeout))

    # --- Summary ---
    ok = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] != "OK"]
    print(f"\n{'=' * 70}")
    print(
        f"  {len(ok)} passed, {len(fail)} failed/timed out "
        f"({len(circuits)} circuits x {len(strategies)} strategies)"
    )


if __name__ == "__main__":
    main()
