"""QV-20 Fusion Pass A/B Benchmark.

Compares UCC VM throughput with and without SingleAxisFusionPass on the
QV-20 reference circuit. Runs each configuration for multiple shots and
reports per-shot timing.

Usage:
    uv run python paper/qv_benchmark/bench_fusion.py
"""

import json
import os
import subprocess
import time

CIRCUITS = [
    ("QV-10", "tests/fixtures/qv10.stim", [100, 500]),
    ("QV-20", "tests/fixtures/qv20.stim", [1, 3, 5]),
    ("Cultivation d=5", "tests/fixtures/cultivation_d5.stim", [1000, 5000]),
]
RESULTS_FILE = "paper/qv_benchmark/results_fusion.json"


def run_cpp_profiler(circuit_file: str, shots: int, binary: str) -> dict[str, float]:
    """Run the C++ profile_svm binary and parse its output."""
    env = os.environ.copy()
    env["UCC_CIRCUIT_FILE"] = circuit_file
    env["UCC_SHOTS"] = str(shots)
    env["OMP_NUM_THREADS"] = "1"

    result = subprocess.run(
        [binary],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[:500]}")
        return {"total_ms": 0.0, "per_shot_ms": 0.0, "instructions": 0}

    out: dict[str, float] = {}
    for line in result.stdout.split("\n"):
        if "Total:" in line and "ms" in line:
            out["total_ms"] = float(line.split()[-2])
        if "Per shot:" in line:
            val = float(line.split()[-2])
            # Could be in us or ms, normalize to ms
            if val > 1e4:
                val /= 1000.0  # was in us
            out["per_shot_ms"] = val
        if "->" in line and "instructions" in line:
            # "8987 -> 3089 instructions"
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "->":
                    out["instructions_before"] = float(parts[i - 1])
                    out["instructions_after"] = float(parts[i + 1])
        if "Peak rank:" in line:
            out["peak_rank"] = float(line.split()[2])
        if "Compilation total:" in line:
            out["compile_ms"] = float(line.split()[-2])

    return out


def run_python_benchmark(circuit_file: str, shots: int, use_fusion: bool) -> dict[str, float]:
    """Run the benchmark using Python bindings with controlled pipeline."""
    import ucc
    import ucc._ucc_core as core

    with open(circuit_file) as f:
        text = f.read()

    # Compile
    t0 = time.perf_counter()
    circuit = ucc.parse(text)
    hir = ucc.trace(circuit)
    hpm = ucc.default_hir_pass_manager()
    hpm.run(hir)
    mod = ucc.lower(hir)

    bpm = ucc.BytecodePassManager()
    bpm.add(ucc.NoiseBlockPass())
    bpm.add(ucc.MultiGatePass())
    bpm.add(ucc.ExpandTPass())
    bpm.add(ucc.ExpandRotPass())
    bpm.add(ucc.SwapMeasPass())
    if use_fusion:
        bpm.add(core.SingleAxisFusionPass())
    bpm.run(mod)
    compile_time = time.perf_counter() - t0

    n_instructions = mod.num_instructions

    # Sample
    t0 = time.perf_counter()
    ucc.sample(mod, shots, 0)
    sample_time = time.perf_counter() - t0

    return {
        "compile_ms": compile_time * 1000,
        "total_ms": sample_time * 1000,
        "per_shot_ms": (sample_time * 1000) / shots,
        "instructions": n_instructions,
        "peak_rank": mod.peak_rank,
    }


def main() -> None:
    os.makedirs("paper/qv_benchmark", exist_ok=True)

    print("=" * 70)
    print("SingleAxisFusionPass A/B Benchmark")
    print("=" * 70)
    print()

    all_results: list[dict[str, object]] = []

    for circuit_name, circuit_file, shots_list in CIRCUITS:
        print(f"\n{'='*70}")
        print(f"Circuit: {circuit_name} ({circuit_file})")
        print(f"{'='*70}")

        for shots in shots_list:
            for use_fusion in [False, True]:
                label = "WITH fusion" if use_fusion else "WITHOUT fusion"
                print(f"\n  {label}, {shots} shot(s)...", end="", flush=True)

                r_raw = run_python_benchmark(circuit_file, shots, use_fusion)
                r: dict[str, object] = dict(r_raw)
                r["circuit"] = circuit_name
                r["shots"] = shots
                r["fusion"] = use_fusion
                all_results.append(r)

                print(
                    f" {r['per_shot_ms']:.2f} ms/shot"
                    f" ({r['instructions']} instrs, {r['total_ms']:.0f} ms total)"
                )

    # Summary table
    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"{'Circuit':<18} {'Shots':>6}  {'Config':<16}  {'Instrs':>7}"
        f"  {'ms/shot':>10}  {'Speedup':>8}"
    )
    print("-" * 78)

    for circuit_name, circuit_file, shots_list in CIRCUITS:
        for shots in shots_list:
            no_fuse = [
                r
                for r in all_results
                if r["circuit"] == circuit_name and r["shots"] == shots and not r["fusion"]
            ][0]
            with_fuse = [
                r
                for r in all_results
                if r["circuit"] == circuit_name and r["shots"] == shots and r["fusion"]
            ][0]

            no_ms = float(no_fuse["per_shot_ms"])  # type: ignore[arg-type]
            w_ms = float(with_fuse["per_shot_ms"])  # type: ignore[arg-type]
            speedup = no_ms / w_ms if w_ms > 0 else 0

            print(
                f"{circuit_name:<18} {shots:>6}  {'no fusion':<16}  "
                f"{no_fuse['instructions']:>7}  {no_ms:>10.2f}  {'':>8}"
            )
            print(
                f"{'':18} {'':>6}  {'WITH fusion':<16}  "
                f"{with_fuse['instructions']:>7}  {w_ms:>10.2f}  "
                f"{speedup:>7.2f}x"
            )
        print("-" * 78)

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
