# Paper Benchmarks

Performance benchmarks comparing Clifft against Stim and tsim on quantum
error correction circuits.

## Benchmarks

| Directory | Circuit | Simulators | Notes |
|-----------|---------|-----------|-------|
| `clifford_bench/` | Surface code memory (Clifford) | Clifft, Stim, tsim | Pure stabilizer; all simulators work |
| `near_clifford_bench/` | Magic state cultivation d=3,5 | Clifft, tsim | 29 T gates (d=3); tsim needs `strategy="cutting"` |
| `distillation_bench/` | Magic state distillation 85q | Clifft, tsim | [17,1,5] color code; needs `CLIFFT_MAX_QUBITS>=128` |
| `coherent_noise_bench/` | Surface code + R_Z noise | Clifft | Coherent over-rotation per Tuloup & Ayral (arXiv:2603.14670) |
| `qv_benchmark/` | Quantum Volume (random) | Clifft, Qiskit, Qulacs, qsim, Qrack | Fully non-Clifford; worst-case exponential scaling |

## Setup

The `paper/` directory is a standalone Python package.  Install it
(in editable mode) to make all benchmark modules importable:

```bash
cd paper
uv sync          # creates .venv and installs stim, pandas, numpy
```

**Additional runtime dependencies** (not on PyPI):
- **clifft** — install from the parent repo: `uv pip install -e ..`
- **tsim** — `uv pip install bloqade-tsim>=0.1.2`

Benchmarks that select only available simulators (e.g.
`--simulators stim`) work without clifft or tsim.

## Quick start

All commands below run from the `paper/` directory.

```bash
# Smoke test (Stim only, fast)
uv run python -m clifford_bench.run_benchmark \
    --distances 3 --shots 1000 --repeats 1 --simulators stim
```

## Recommended benchmark plan

The commands below cover the full set of paper benchmarks at
`p=1e-3`.  Each benchmark writes its own `results.csv`.  tsim
uses the default compilation strategy for Clifford/distillation
and cutting for non-Clifford circuits.

```bash
# 1. Clifford surface code d=3,5,7 — Stim, Clifft, tsim (default strategy)
JAX_PLATFORMS=cpu uv run python -m clifford_bench.run_benchmark \
    --distances 3,5,7 --error-rates 1e-3 \
    --simulators stim,clifft,tsim

# 2. Near-Clifford cultivation d=3 — Clifft, tsim (cutting strategy)
JAX_PLATFORMS=cpu uv run python -m near_clifford_bench.run_benchmark \
    --distances 3 --error-rates 1e-3 \
    --simulators clifft,tsim

# 3. Magic state distillation 85q — Clifft, tsim (default strategy)
JAX_PLATFORMS=cpu uv run python -m distillation_bench.run_benchmark \
    --simulators clifft,tsim

# 4. Coherent noise d=5 r=1 — Clifft, tsim (cutting strategy)
JAX_PLATFORMS=cpu uv run python -m coherent_noise_bench.run_benchmark \
    --distances 5 --rounds 1 \
    --simulators clifft,tsim

# 5. Coherent noise d=5 r=5 — Clifft only (tsim cannot compile)
uv run python -m coherent_noise_bench.run_benchmark \
    --distances 5 --rounds 5 \
    --simulators clifft
```

Notes:
- `JAX_PLATFORMS=cpu` is only needed when tsim is included.
- The near-Clifford and coherent benchmarks default to
  `--tsim-strategy cutting`; Clifford and distillation default to
  `--tsim-strategy default`.
- Distillation uses its own noise model (`--prep-noise 0.05` by
  default), not `--error-rates`.

## Quantum Volume benchmark

The QV benchmark compares Clifft against statevector simulators
(Qiskit-Aer, Qulacs, qsim, Qrack) on fully random circuits that
scale exponentially in memory.  Each (simulator, N, seed) runs in
an isolated subprocess with memory and timeout limits.

**Recommended run (24-core, 96 GB RAM instance):**

```bash
uv run python -m qv_benchmark \
    --min-q 6 --max-q 32 --step 2 \
    --mem-limit-gb 80 --timeout 600 \
    --repeats 3
```

Memory scaling for statevector simulators is 2^N × 16 bytes:

| N  | Statevector RAM |
|----|-----------------|
| 28 | 4 GB            |
| 30 | 16 GB           |
| 32 | 64 GB           |

N=32 is the practical ceiling for 96 GB — beyond that nothing fits.
Clifft on random QV circuits has similar exponential scaling, so expect
OOM at comparable N values.  The benchmark gracefully records OOM
and timeout as status codes in the CSV.

For a faster first pass, use `--repeats 1` to find the OOM
boundaries, then re-run with `--repeats 3` on a narrower range.

See [`qv_benchmark/README.md`](qv_benchmark/README.md) for full
CLI options and simulator-specific details.

## tsim compile check

tsim compilation can hang on certain circuits.  Use the smoke-test
script to quickly check which circuits compile on your machine.
Each circuit is compiled in a separate subprocess, so hung compiles
are fully killed on timeout.

```bash
JAX_PLATFORMS=cpu uv run python -m tsim_compile_check

JAX_PLATFORMS=cpu uv run python -m tsim_compile_check --timeout 120
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
