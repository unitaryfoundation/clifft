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

`H` (butterfly), `T` (phase waterfall), `CZ`, `CNOT` (permutation), `U2`, `U4`
(fused dense 1q/2q matrices — clifft's `OP_ARRAY_U2`/`OP_ARRAY_U4`, the opcodes
the optimizer's fusion passes actually emit, so the mix matches compiled
programs), `EXPAND`, `EXPAND_T` (fused expand+phase), `MEAS_DIAG` (Z-basis:
reduce+sample+compact), `MEAS_INTERFERE` (X-basis fold).

Two batched workloads:
- **gates-only** (`batched`/`batchedgpu`): rank-preserving ops so every shot
  stays at the same *k* — the idealized upper bound on batched throughput.
- **real shot shape** (`batchedmeas`/`batchedmeasgpu`): each layer additionally
  runs one **completed measurement** — on GPU that is per-shot reduce → D2H of
  branch probabilities → host sampling → H2D of outcomes → outcome-selected
  collapse — followed by a per-shot expand that restores the rank (mirroring
  clifft's static, shot-invariant k trajectory, where all shots move through
  the same k schedule in lockstep). **The gap between the two workloads is the
  decisive go/no-go number**: it prices the host round-trip that real
  mid-circuit-measurement shots pay and that the literature never measures.
  The `measround` tag times one gate-free measurement round alone (latency of
  the reduce→D2H→sample→H2D→collapse chain).

Per-shot measurement outcomes differ across the batch; the collapse kernels
take the outcome/sign as a **per-shot scalar** (predicated lanes, no divergent
code paths) — the same shape a production batched backend would use.

`Gamp/s` counts **amplitudes actually touched** per op (H: `2^k`, T: `2^k/2`,
CZ: `2^k/4`, EXPAND/MEAS: `2·2^k`, …; see `amps_touched()` in
`bench_common.hpp`), so per-op comparisons are meaningful. CPU and GPU use the
same definition, so crossovers are unaffected by the normalization.

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
`bench_gpu` first runs the **CPU-vs-GPU correctness checks** — `[validate]`
(single-state: gate layers incl. U2/U4, EXPAND_T, forced MEAS_DIAG and
MEAS_INTERFERE, with the GPU reduce sums cross-checked against CPU
probabilities) and `[validate-batched]` (all `kb_*` kernels plus a completed
batched measurement round with alternating forced outcomes, so both arms of
the outcome-predicated collapse are exercised). Both expect `< 1e-9`. Then
transfer-bandwidth, single-op, batched, and batched-with-measurement sweeps.

## Output

CSV on stdout, human-readable table on stderr.

| tag              | columns                                                        |
|------------------|----------------------------------------------------------------|
| `single`         | `k, op, dim, ns/op, Gamp/s`           (CPU single-state)       |
| `singlegpu`      | `k, op, dim, ns/op, Gamp/s`           (GPU single-state)       |
| `batched`        | `k, B, shots, shots/s, Mshot-layer/s` (CPU batched, gates-only)|
| `batchedgpu`     | `k, B, shots, shots/s, Mshot-layer/s` (GPU batched, gates-only)|
| `batchedmeas`    | `k, B, shots, shots/s, Mshot-layer/s` (CPU, + completed meas)  |
| `batchedmeasgpu` | `k, B, shots, shots/s, us/meas-round` (GPU, + completed meas)  |
| `measround`      | `k, B, shots, us/round, us/round/shot` (GPU meas round alone)  |
| `transfer`       | `k, bytes, H2D GB/s, D2H GB/s`        (GPU only)               |

## How to read the results

- **Crossover**: compare `single` vs `singlegpu` `Gamp/s` at each k. The k where
  GPU overtakes CPU is the single-statevector crossover *for clifft's op mix in
  FP64* — the number the literature couldn't give us.
- **Batched win**: compare `batched` vs `batchedgpu` `shots/s` at fixed small k
  (e.g. 16). If the GPU's shots/s keeps climbing with B while CPU saturates, the
  batch-across-shots hypothesis holds. Watch for the B where GPU shots/s plateaus
  (full occupancy).
- **The decisive number**: `batchedmeasgpu` vs `batchedgpu` shots/s at the same
  (k, B). If completed measurements erase most of the batched win, the whole
  GPU route needs on-device sampling or CUDA-graphs work before it is viable;
  if the gap is modest (NVLink-C2C should make the D2H/H2D legs cheap on
  GH200), the go case is strong. Compare `batchedmeas` (CPU) the same way —
  on CPU the measurement adds only a few percent (no round-trip), so the GPU
  delta isolates the round-trip cost.
- **Transfer**: H2D/D2H GB/s ≈ PCIe (~25 GB/s) vs NVLink-C2C (hundreds of GB/s)
  tells you how cheap it is to keep clifft's CPU-side branch control in the loop.

## Quoting rules

Do **not** quote `batchedgpu ÷ batched` as "the GPU speedup for clifft" — it is
a gates-only, sync-free ceiling. Quote `batchedmeasgpu ÷ batchedmeas` (same
schedule, both sides pay a completed measurement per layer) as the honest
batched comparison, and report the gates-only figure as the ceiling. Known
remaining gaps even after a GH200 run: no `SWAP_MEAS_INTERFERE`, CZ/CNOT/U4
axes fixed at (0,1)-adjacent (most coalescing-friendly), no per-shot noise-op
modeling, no x86/AVX-512 CPU baseline (Grace/NEON only).

Save `cpu.csv` / `gpu.csv` next to this README and summarize findings back into
`../README.md`.
