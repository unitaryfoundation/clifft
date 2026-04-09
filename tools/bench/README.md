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
