"""Benchmark compilation time for .ucc circuits."""

import statistics
import sys
import time

import ucc


def bench_compile(path: str, warmup: int = 2, trials: int = 10) -> dict:
    text = open(path).read()

    # Warmup
    for _ in range(warmup):
        ucc.compile(text)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        ucc.compile(text)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "path": path,
        "median_s": statistics.median(times),
        "mean_s": statistics.mean(times),
        "min_s": min(times),
        "max_s": max(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "trials": trials,
    }


if __name__ == "__main__":
    circuits = sys.argv[1:] or [
        "/home/exedev/d3_t_gate_e2e_expval_p0.001.ucc",
        "/home/exedev/d3_s_proxy_e2e_expval_p0.001.ucc",
    ]
    for path in circuits:
        r = bench_compile(path)
        print(f"\n{r['path']}:")
        print(
            f"  median: {r['median_s']:.4f}s  mean: {r['mean_s']:.4f}s  "
            f"min: {r['min_s']:.4f}s  max: {r['max_s']:.4f}s  "
            f"stdev: {r['stdev_s']:.4f}s  (n={r['trials']})"
        )
