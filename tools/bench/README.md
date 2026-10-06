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
| `test_bench_adder.py` | Fixed-input 16-bit Draper adder on 32 qubits | Compile and sample latency with and without rotation simplification |
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

## Fixed-input arithmetic

`test_bench_adder.py` uses `fixtures/draper_adder_16_basis.stim` and compares
the default pipeline with an explicit PeepholeFusion, PhasePolynomial,
StatevectorSqueeze control. Sampling uses one native thread, 1024 shots, and both
scalar and automatic batching. Setup checks every output bit against classical
addition.

The circuit prepares a=37449 and b=18724, then computes b=(a+b) modulo 65536,
leaving a unchanged. Its 32 terminal records follow physical-qubit order; the
first 16 encode a and the next 16 encode b=56173, least significant bit first.
Observable 0 is the parity of the two most significant sum bits and is zero.
There is no noise or postselection. This tests state-dependent simplification
for a fixed computational-basis input, not general superposition arithmetic.

The generator uses MQT Bench 2.3.0's Draper circuit and Qiskit 2.5.2 with
`optimization_level=0`. Rotation arguments in the fixture are half-turns.
MQT Bench is needed only to regenerate the fixture:

```bash
uv run --with mqt-bench==2.3.0 --with qiskit==2.5.2 python tools/bench/generate_draper_adder.py
uv run pytest tools/bench/test_bench_adder.py
```

The pinned fixture SHA-256 is
`c62c0d7210be01315fbaad3ff6114c14524a40477c5246fe0ae65fae36b31289`.
The separate corpus addition is tracked in
[clifft-bench issue 66](https://github.com/unitaryfoundation/clifft-bench/issues/66).
