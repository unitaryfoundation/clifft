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
| `test_bench_qv.py` | 20-qubit Quantum Volume (`fixtures/qv20_seed42.stim`) | Large statevector (peak active width 20) per-shot throughput |
| `test_bench_noncomp.py` | d=17, r=5 repetition-code memory with a hooked leak/loss layer | Noncomputational pipeline overhead and trap/continuation cost vs plain sampling |
| `test_bench_sinter.py` | Clifford S-gate cultivation at p=0.001 | Counts-only postselection through the Sinter adapter |

## Sinter postselection

The Sinter benchmark uses the existing fully Clifford S-gate cultivation fixture,
with every detector postselected. It measures 16384 attempts per call after
compilation and warmup, using one native thread and capacity 2048. The output is
aggregate attempted-shot, discard, and logical-error counts; this is error
detection with zero observable prediction, not a decoded memory task. It runs
with `just bench`, or on its own:

```bash
uv run pytest tools/bench/test_bench_sinter.py
```

This is a throughput regression case, not a complete Sinter job or a general
speed comparison with Stim. The existing QEC and deep-Clifford cases provide
other Clifford workload coverage.

## Fixtures

Pre-generated circuit files live in `fixtures/`:

- **`qv20_seed42.stim`** — 20-qubit Quantum Volume circuit (seed=42) in Stim-superset
  format. Peak active width 20 (2^20 = 1M complex amplitudes, 16 MB statevector).
  Useful for profiling dense active-state kernels.
