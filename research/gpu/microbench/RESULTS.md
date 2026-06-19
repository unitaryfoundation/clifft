# Microbenchmark results

## Run 1 — local CPU baseline (2026-06-19)

**Machine:** Apple Silicon (this Mac), 38 GB RAM, **single-threaded** (no libomp
installed → `OpenMP: DISABLED`). complex128. `-O3 -march=native` (NEON).
This is a *per-core* baseline; the Grace CPU on GH200 (ARM, ~72 cores, OpenMP)
will be much higher for the parallelizable cases — see caveats.

### Single-statevector per-op throughput (Gamp/s = amplitudes touched/sec)

Throughput is essentially **flat in k** (memory-bandwidth bound), as expected for
element-wise/permutation kernels. Representative values (steady state, k≥14):

| op              | Gamp/s (1 core) | notes                                        |
|-----------------|-----------------|----------------------------------------------|
| CZ / CNOT       | ~3.1–4.9        | pure permutation/negation, fastest           |
| EXPAND          | ~3.0–5.2        | straight copy                                 |
| MEAS_DIAG       | ~4.0–4.5        | reduce + compact                              |
| T               | ~3.3–4.2        | phase waterfall                               |
| H               | ~2.3–3.7        | butterfly                                     |
| EXPAND_T        | ~2.8–3.0        | copy + complex multiply                       |
| MEAS_INTERFERE  | ~2.0            | two reductions + fold, most memory traffic   |

At k=22–26 everything settles to ~2–3 Gamp/s (≈32–48 GB/s effective for a single
core — one core's share of the memory bus). Sub-k=13 numbers run hotter (cache
residency) with some small-size timing noise.

### Batched-across-shots throughput (4 layers/shot)

| k  | B (256→16384)        | shots/s        |
|----|----------------------|----------------|
| 12 | flat across all B    | ~7,500         |
| 16 | 256 / 1024           | ~340           |
| 18 | 256                  | ~74            |

**Key baseline observation:** on a single CPU core, batched throughput is **flat
in B** — batching is just more independent serial work, no throughput gain. So
*any* scaling-with-B the GH200 shows in `batchedgpu` is pure GPU upside. shots/s
falls ~2× per +1 k (the 2^k per-shot cost), as expected.

### How to read this against the eventual GPU run
- **Single-state crossover:** GPU `singlegpu` Gamp/s vs CPU `single` Gamp/s. The
  CPU ceiling here is one core; on GH200 multiply the *parallelizable* CPU ops
  (k≥18, where clifft threads) by ~core-count for the fair Grace baseline — the
  `bench_cpu` run on GH200 does this automatically via OpenMP.
- **Batched win:** watch whether GPU shots/s *rises* with B (CPU's is flat).
  The B where GPU plateaus = full-occupancy; the plateau height vs CPU×cores is
  the real batched speedup.

## Run 2 — GH200 (PENDING)

Not yet run — no GH200 access at time of writing. When available:
```sh
./run.sh 10 30 12,16,18,20 4   # writes cpu.csv (multicore Grace) + gpu.csv
```
Then fill in: single-state crossover k, batched shots/s vs B curve + plateau,
H2D/D2H bandwidth (expect NVLink-C2C hundreds of GB/s, not PCIe ~25), and the
CPU/GPU `[validate]` max-error (expect <1e-9).
