# Paper Benchmarks

Performance benchmarks comparing UCC against Stim and tsim on quantum
error correction circuits.

## Benchmarks

| Directory | Circuit | Simulators | Notes |
|-----------|---------|-----------|-------|
| `clifford_bench/` | Surface code memory (Clifford) | UCC, Stim, tsim | Pure stabilizer; all simulators work |
| `near_clifford_bench/` | Magic state cultivation d=3,5 | UCC, tsim | 29 T gates (d=3); tsim needs `strategy="cutting"` |
| `distillation_bench/` | Magic state distillation 85q | UCC, tsim | [17,1,5] color code; needs `UCC_MAX_QUBITS>=128` |
| `coherent_noise_bench/` | Surface code + R_Z noise | UCC | Coherent over-rotation per Tuloup & Ayral (arXiv:2603.14670) |

## Dependencies

All benchmarks require `stim` and `pandas` (for CSV output).  tsim
benchmarks additionally require `bloqade-tsim`.  These are not declared
as project dependencies since they are only needed for benchmarking:

```bash
# UCC + Stim benchmarks
uv run --with pandas python paper/clifford_bench/run_benchmark.py ...

# With tsim
JAX_PLATFORMS=cpu uv run --with 'pandas,bloqade-tsim>=0.1.2' \
    python paper/clifford_bench/run_benchmark.py --simulators ucc,stim,tsim ...
```

## Quick start

```bash
# Run a single benchmark (UCC only, quick test)
uv run --with pandas python paper/clifford_bench/run_benchmark.py \
    --distances 3 --shots 1000 --repeats 1 --simulators ucc

# Run with tsim (CPU)
JAX_PLATFORMS=cpu uv run --with 'pandas,bloqade-tsim>=0.1.2' \
    python paper/clifford_bench/run_benchmark.py \
    --distances 3 --simulators ucc,stim,tsim
```

## tsim compile check

tsim compilation can hang on certain circuits.  Use the smoke-test
script to quickly check which circuits compile on your machine.
Each circuit is compiled in a separate subprocess, so hung compiles
are fully killed on timeout.

```bash
JAX_PLATFORMS=cpu uv run --with 'bloqade-tsim>=0.1.2' \
    python paper/tsim_compile_check.py

JAX_PLATFORMS=cpu uv run --with 'bloqade-tsim>=0.1.2' \
    python paper/tsim_compile_check.py --timeout 120
```

This tests all benchmark circuits with both the default and cutting
decomposition strategies, reporting compile time and stabilizer term
count (`num_graphs`) for each.

### Known tsim results (v0.1.2, CPU, 60s timeout)

| Circuit | Default | Cutting |
|---------|---------|---------|
| clifford d=3,5,7 | OK | OK |
| cultivation d=3 | TIMEOUT | OK (29s, 3903 graphs) |
| cultivation d=5 | TIMEOUT | TIMEOUT |
| distillation 85q | OK (51s, 600 graphs) | TIMEOUT |
| coherent d=3 r=1 | OK (0.2s, 18 graphs) | OK |
| coherent d=3 r=3 | TIMEOUT | TIMEOUT |
| coherent d=5 r=1 | TIMEOUT | OK (15s, 50 graphs) |
| coherent d=5 r=5 | TIMEOUT | TIMEOUT |

Neither strategy works universally.  The default strategy handles
Clifford circuits and distillation; cutting handles some non-Clifford
circuits that default cannot.
