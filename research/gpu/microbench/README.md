# clifft active-block GPU microbenchmark

A standalone, dependency-free benchmark that answers the questions the
dense-statevector GPU literature leaves open for clifft (see
`../README.md`, open questions #1, #2, #4):

1. **Single-statevector crossover** — at what active rank *k* does a GH200 GPU
   (FP64) beat clifft's CPU kernels for each op type, in **complex128**, against
   a *faithful* reimplementation of clifft's own kernels (not a stock simulator)?
2. **Batched-across-shots throughput** — the central hypothesis: many small
   (k<20) statevectors run in parallel on the GPU. Does batching thousands of
   shots saturate the GPU and win big, where a single small state cannot?
3. **Transfer cost** — host↔device bandwidth, to quantify the NVLink-C2C
   advantage on Grace-Hopper vs the PCIe figures in the literature.

## What it is / isn't

- The CPU kernels in `include/cpu_kernels.hpp` are faithful to clifft's **scalar
  reference paths** in `src/clifft/svm/svm_kernels.inl` (same butterfly /
  bit-scatter / fold / phase logic), compiled `-O3 -march=native` so they
  autovectorize (NEON on ARM/Grace/Apple Silicon, AVX on x86). On ARM this is
  representative of what clifft actually runs, since clifft's hand-written AVX
  paths are x86-only. It is **not** clifft's literal AVX-512 path — for the
  truest x86 comparison, wire in clifft's real kernels later.
- The GPU kernels (`src/bench_gpu.cu`) are hand-written CUDA in **complex128**,
  the same ops in single and batched form. This is the "custom CUDA" arm of the
  design space — it does **not** use cuStateVec/CUDA-Q (those are the next step
  if these numbers justify it; the non-standard ops EXPAND/EXPAND_T/INTERFERE
  have no native cuStateVec primitive anyway).

## Ops benchmarked

`H` (butterfly), `T` (phase waterfall), `CZ`, `CNOT` (permutation), `EXPAND`,
`EXPAND_T` (fused expand+phase), `MEAS_DIAG` (Z-basis: reduce+sample+compact),
`MEAS_INTERFERE` (X-basis fold). The batched workload uses only the
rank-preserving ops (H/T/CZ/CNOT) so every shot stays at the same *k* — honoring
cuStateVec's uniform-qubit-count constraint and isolating raw batched throughput.

## Build & run

### Locally (CPU only — macOS / Linux)
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bench_cpu                 # defaults: k=10..26, batched k={12,16,18,20}, 4 layers
./build/bench_cpu 10 28 12,16,18,20 4 > cpu.csv
```
On macOS, `brew install libomp` first for a multithreaded baseline (otherwise it
runs single-threaded and prints a warning). The Grace CPU on GH200 has OpenMP, so
build there for a fair multicore CPU baseline.

### On the GH200 (CPU + GPU)
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=90
cmake --build build -j
./build/bench_cpu 10 30 12,16,18,20 4 > cpu.csv
./build/bench_gpu 10 30 12,16,18,20 4 > gpu.csv
```
`bench_gpu` first runs a **CPU-vs-GPU correctness check** (`[validate]`, expects
`max|cpu-gpu| < 1e-9`), then transfer-bandwidth, single-op, and batched sweeps.

## Output

CSV on stdout, human-readable table on stderr.

| tag           | columns                                                    |
|---------------|------------------------------------------------------------|
| `single`      | `k, op, dim, ns/op, Gamp/s`           (CPU single-state)   |
| `singlegpu`   | `k, op, dim, ns/op, Gamp/s`           (GPU single-state)   |
| `batched`     | `k, B, shots, shots/s, Mshot-layer/s` (CPU batched)        |
| `batchedgpu`  | `k, B, shots, shots/s, Mshot-layer/s` (GPU batched)        |
| `transfer`    | `k, bytes, H2D GB/s, D2H GB/s`        (GPU only)           |

## How to read the results

- **Crossover**: compare `single` vs `singlegpu` `Gamp/s` at each k. The k where
  GPU overtakes CPU is the single-statevector crossover *for clifft's op mix in
  FP64* — the number the literature couldn't give us.
- **Batched win**: compare `batched` vs `batchedgpu` `shots/s` at fixed small k
  (e.g. 16). If the GPU's shots/s keeps climbing with B while CPU saturates, the
  batch-across-shots hypothesis holds. Watch for the B where GPU shots/s plateaus
  (full occupancy).
- **Transfer**: H2D/D2H GB/s ≈ PCIe (~25 GB/s) vs NVLink-C2C (hundreds of GB/s)
  tells you how cheap it is to keep clifft's CPU-side branch control in the loop.

Save `cpu.csv` / `gpu.csv` next to this README and summarize findings back into
`../README.md`.
