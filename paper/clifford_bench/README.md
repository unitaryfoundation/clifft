# Clifford-Limit Surface Code Benchmark

Benchmarks UCC vs Stim vs tsim on **rotated surface code memory-Z** circuits
in the fully Clifford (stabilizer) regime, sweeping depolarizing error rate.

Default configuration targets **d=7, 7 rounds, 1M shots** to match the
comparison point in the tsim paper (Hoshino et al.).

## Dependencies

```
stim>=1.10
pandas
matplotlib
ucc
tsim
```

For GPU runs, JAX must be installed with GPU support.

## Running

```bash
# Full benchmark (d=7, 1M shots, ucc + stim + tsim, 3 repeats)
python run_benchmark.py

# Quick local test
python run_benchmark.py --distances 3 --shots 1000 --repeats 1 --simulators ucc,stim

# Multiple distances
python run_benchmark.py --distances 3,5,7

# Fixed rounds (instead of rounds = distance)
python run_benchmark.py --rounds 10
```

Results are written to `results.csv` in this directory.

### tsim CPU vs GPU

JAX reads the `JAX_PLATFORMS` env var once at import time, so CPU vs GPU
must be selected per invocation — they cannot be mixed in a single run:

```bash
# tsim on CPU
JAX_PLATFORMS=cpu python run_benchmark.py --simulators tsim --output results_tsim_cpu.csv

# tsim on GPU (auto-detects)
python run_benchmark.py --simulators tsim --output results_tsim_gpu.csv
```

## Plotting

The plot script accepts multiple input CSVs.  Use `path:label` to rename
the simulator column, so separate tsim CPU/GPU runs can be combined:

```bash
# Single file
python plot_clifford.py

# Combine multiple runs
python plot_clifford.py \
    --input results.csv results_tsim_cpu.csv:tsim-cpu results_tsim_gpu.csv:tsim-gpu
```

Produces a log-log plot of sample time vs error rate.

## CLI options

### run_benchmark.py

| Flag | Default | Description |
|------|---------|-------------|
| `--distances` | `7` | Comma-separated code distances |
| `--rounds` | `= distance` | Fixed round count (overrides per-distance default) |
| `--error-rates` | `1e-7,...,1e-1` | Comma-separated physical error rates |
| `--shots` | `1000000` | Shots per run |
| `--repeats` | `3` | Repetitions per configuration |
| `--simulators` | `ucc,stim,tsim` | Comma-separated simulator backends |
| `--output` | `results.csv` | Output CSV path |

### plot_clifford.py

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `results.csv` | One or more input CSVs (supports `path:label` syntax) |
| `--output` | `clifford_bench.png` | Output plot path |

## Caveats

**Sampling output asymmetry.** Stim and tsim use `compile_detector_sampler()`,
which returns only detectors and observables. UCC's `sample()` always
materializes the full measurement record in addition to detectors and
observables. For d=7 this is 385 measurements vs 337 detector+observable values
per shot. Empirically this adds ~6% overhead in Stim's own timing, so UCC's
sample-time numbers are slightly conservative (i.e. UCC is doing more work per
shot than the other backends).
