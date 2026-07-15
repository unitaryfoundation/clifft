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

Two further GPU arms (added after studying
[clifft-cuda](https://github.com/haoliri0/clifft-cuda), a third-party FP32
MIMD sampler for clifft-compiled circuits):

- **on-device sampling** (`batcheddevgpu`): the same real-shot-shape workload,
  but the measurement outcome is drawn **on the device** by a stateless
  counter-based RNG (`kb_sample_dev`) — zero host legs, one sync per batch.
  The (`batchedmeasgpu` vs `batcheddevgpu`) delta prices the host round-trip
  itself; clifft-cuda's design shows production code never needs to pay it.
  `measrounddev` times one gate-free device round (with a per-round sync, so
  read it as latency; the whole-batch figure is the honest throughput).
- **MIMD per-shot interpreter** (`mimdgpu`): clifft-cuda's architecture in
  FP64 — one block per shot walks the entire layered run (gates, in-block
  reduce, in-block sampling, collapse, re-expand) in a **single kernel
  launch**, with the slice in shared memory when it fits (metric2 = 1) and in
  global memory otherwise. Same math, initial state, RNG, and schedule as
  `batcheddevgpu`, so the pair isolates the execution-architecture choice:
  lockstep SoA (coalesced across shots, one launch per op) vs MIMD
  (launch-free, shot-local, no cross-shot coalescing).

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
the outcome-predicated collapse are exercised), plus `[validate-devmeas]`
(on-device sampling: device outcomes must match the host recomputation of the
same stateless hash bit-for-bit, and the collapsed state must match a CPU
reference forced to those outcomes) and `[validate-mimd]` (the per-shot
interpreter, global and shared variants, against the identical forced CPU
trajectory). All expect `< 1e-9`. Then transfer-bandwidth, single-op, batched,
batched-with-measurement, on-device-sampling, and MIMD sweeps.

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
| `batcheddevgpu`  | `k, B, shots, shots/s, us/meas-round` (GPU, meas sampled on device) |
| `measrounddev`   | `k, B, shots, us/round, us/round/shot` (GPU device round alone)|
| `mimdgpu`        | `k, B, shots, shots/s, shared?`       (GPU, one block/shot)    |
| `transfer`       | `k, bytes, H2D GB/s, D2H GB/s`        (GPU only)               |

## How to read the results

- **Crossover**: compare `single` vs `singlegpu` `Gamp/s` at each k. The k where
  GPU overtakes CPU is the single-statevector crossover *for clifft's op mix in
  FP64* — the number the literature couldn't give us.
- **Batched win**: compare `batched` vs `batchedgpu` `shots/s` at fixed small k
  (e.g. 16). If the GPU's shots/s keeps climbing with B while CPU saturates, the
  batch-across-shots hypothesis holds. Watch for the B where GPU shots/s plateaus
  (full occupancy).
- **The decisive number**: `batcheddevgpu` vs `batchedmeas` (CPU) shots/s at
  the same (k, B) — both sides run the identical real-shot-shape schedule, and
  the GPU side pays no host legs (the production design, per clifft-cuda). The
  older `batchedmeasgpu` (host round-trip) stays as a comparison arm; its gap
  to `batcheddevgpu` prices the round-trip that on-device sampling removes.
- **Architecture fork**: `mimdgpu` vs `batcheddevgpu` at the same (k, B).
  clifft-cuda achieved only ~1.6× over a 40-thread CPU with the MIMD design
  (FP32, workstation card); if lockstep SoA wins this head-to-head clearly,
  the gap was architectural headroom, and our design is the one to build. If
  MIMD ties or wins (shot-local shared memory beating cross-shot coalescing),
  the cheapest production path is upgrading clifft-cuda to FP64 rather than a
  new backend. Note MIMD also wins by construction on early-discard
  (postselection-heavy) workloads, which this schedule does not model.
- **Transfer**: H2D/D2H GB/s ≈ PCIe (~25 GB/s) vs NVLink-C2C (hundreds of GB/s)
  tells you how cheap it is to keep clifft's CPU-side branch control in the loop.

## Quoting rules

Do **not** quote `batchedgpu ÷ batched` as "the GPU speedup for clifft" — it is
a gates-only, sync-free ceiling. Quote `batcheddevgpu ÷ batchedmeas` (same
schedule, both sides pay a completed measurement per layer; GPU samples on
device, the production design) as the honest batched comparison, and report
the gates-only figure as the ceiling. Known remaining gaps even after a GH200
run: no `SWAP_MEAS_INTERFERE`, CZ/CNOT/U4 axes fixed at (0,1)-adjacent (most
coalescing-friendly), no per-shot noise-op modeling, no early-discard
(postselection) modeling, no x86/AVX-512 CPU baseline (Grace/NEON only).

## External baseline: clifft-cuda

For an end-to-end reference point on the same machine, build and run
[clifft-cuda](https://github.com/haoliri0/clifft-cuda) (Apache-2.0; see its
`instruction.md`) on its bundled `circuit_d5_p=0.001.stim`, plus its
`tools/run_msc.py` for the clifft CPU number on the identical circuit. Caveats
when comparing: it is FP32 (`GpuComplex` is `float` pairs), MIMD per-shot, and
its workload includes noise + postselection early-discard, which this
microbenchmark's synthetic schedule does not model — treat it as context, not
as a same-ruler comparison. Its published calibration: ~1.18M shots/s (RTX PRO
5000, 300 W) vs ~727K shots/s (40-thread Xeon 5218R, 125 W) on d=5 cultivation
at p=0.1%.

Save `cpu.csv` / `gpu.csv` next to this README and summarize findings back into
`../README.md`.
