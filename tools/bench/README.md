# Performance Benchmarks

pytest-benchmark tests for tracking Clifft performance over time.

## Running

```bash
just bench
```

Or directly:

```bash
uv run pytest tools/bench/ --benchmark-sort=name --benchmark-columns=Mean,StdDev,Ops
```

## Benchmarks

| File | Circuit | What it measures |
|------|---------|-----------------|
| `test_bench_qec.py` | d=3 surface code (`tests/fixtures/target_qec.stim`) | Compile and sample latency vs Stim |
| `test_bench_deep_clifford.py` | 50-qubit, 5000 random Cliffords | Pure Clifford compile/sample throughput |
| `test_bench_qv.py` | 20-qubit Quantum Volume (`fixtures/qv20_seed42.stim`) | Large statevector (peak rank 20) per-shot throughput |

## Fixtures

Pre-generated circuit files live in `fixtures/`:

- **`qv20_seed42.stim`** — 20-qubit Quantum Volume circuit (seed=42) in Stim-superset
  format. Peak rank 20 (2^20 = 1M complex amplitudes, 16 MB statevector).
  Useful for profiling SVM array operations and benchmarking multi-core scaling.
