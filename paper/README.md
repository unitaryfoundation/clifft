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

## Setup

The `paper/` directory is a standalone Python package.  Install it
(in editable mode) to make all benchmark modules importable:

```bash
cd paper
uv sync          # creates .venv and installs stim, pandas, numpy
```

**Additional runtime dependencies** (not on PyPI):
- **ucc** — install from the parent repo: `uv pip install -e ..`
- **tsim** — `uv pip install bloqade-tsim>=0.1.2`

Benchmarks that select only available simulators (e.g.
`--simulators stim`) work without ucc or tsim.

## Quick start

All commands below run from the `paper/` directory:

```bash
# Run a single benchmark (Stim only, quick test)
uv run python -m clifford_bench.run_benchmark \
    --distances 3 --shots 1000 --repeats 1 --simulators stim

# Run with UCC + tsim (CPU)
JAX_PLATFORMS=cpu uv run python -m clifford_bench.run_benchmark \
    --distances 3 --simulators ucc,stim,tsim
```

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
