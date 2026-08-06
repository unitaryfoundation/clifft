# Microbenchmark results

## Run 1 — local CPU baseline (2026-07-07)

Harness coverage: the fused `U2`/`U4` opcodes, the completed-measurement
batched workload (`batchedmeas`), and the amplitudes-actually-touched `Gamp/s`
metric. Raw CSV: `cpu.csv` (`./build/bench_cpu 10 26 12,16,18 4`).

**Machine:** Apple Silicon (this Mac), 38 GB RAM, **single-threaded** (no libomp
→ `OpenMP: DISABLED`). complex128, `-O3 -march=native` (NEON). Note this is a
*per-core model baseline*; the installed clifft wheel itself runs 11 threads
via its own pool.

### Cross-check against real clifft (2026-07-06)

The model was validated against the actual installed clifft on this machine:
real clifft runs `R_PAULI(θ) Z` rotation sweeps at k=20 (unoptimized compile;
the optimizer eliminates naive test circuits) at **3.64 Gamp/s single-thread**
— within ~10% of the model's T/H band below — and **9.6 Gamp/s on all 11
cores** (2.6× scaling: memory-bandwidth-bound). Two consequences:
1. The model's absolute throughput is trustworthy for the scalar/NEON path.
2. The multicore CPU ceiling is the memory bus, not core count — on GH200 the
   fair Grace baseline will similarly be bandwidth-limited (~0.5 TB/s), which
   bounds any GPU win at roughly the HBM/LPDDR bandwidth ratio (~6–8×).

### Single-statevector per-op throughput (Gamp/s = amplitudes touched/sec)

Representative steady-state values at k=20–22 (see `cpu.csv` for the sweep;
sub-k=13 numbers run hotter from cache residency):

| op              | Gamp/s (1 core, k=20/22) | notes                                  |
|-----------------|--------------------------|----------------------------------------|
| H               | 3.3 / 2.3                | butterfly (touches 2^k)                |
| T               | 2.1 / 1.6                | phase (touches 2^k/2)                  |
| CZ              | 1.2 / 0.9                | negate (touches 2^k/4)                 |
| CNOT            | 2.4 / 1.7                | swap (touches 2^k/2)                   |
| **U2**          | 2.6 / 1.9                | fused dense 1q (post-optimizer opcode) |
| **U4**          | **0.76 / 0.76**          | fused dense 4×4 — **compute-bound**, flat in k: the one op where GPU FP64 FLOPs (not bandwidth) will matter |
| EXPAND          | 9.2 / 6.5                | copy (touches 2·2^k; streams well)     |
| EXPAND_T        | 5.9 / 5.7                | copy+phase                             |
| MEAS_DIAG       | 15.4 / 13.5              | reduce+compact (touches 2·2^k; forced compact — same work as the GPU row) |
| MEAS_INTERFERE  | 9.1 / 9.0                | two reductions + fold                  |

Key structural read: everything except **U4** is memory-bandwidth-bound
(Gamp/s falls with k as the working set leaves cache, and scales weakly with
cores). U4 is flat in k and ~3–4× slower per amplitude — arithmetic-bound —
so it is the op where the GH200's FP64 throughput can beat the bandwidth
ceiling, and it now anchors the op mix.

### Batched-across-shots (4 layers/shot, layer now incl. U2+U4)

| k  | B (256→16384)      | shots/s (gates-only) | shots/s (+1 completed meas/layer) |
|----|--------------------|----------------------|-----------------------------------|
| 12 | flat across all B  | ~6,250               | ~6,050 (−3%)                      |
| 16 | flat               | ~294                 | ~287 (−3%)                        |
| 18 | flat               | ~65                  | ~64 (−2%)                         |

Two baseline observations for reading the GH200 run:
- **CPU batched throughput is flat in B** (batching is just more serial work);
  any GPU scaling-with-B is pure GPU upside.
- **On CPU, a completed measurement per layer costs only ~3%** — there is no
  round-trip. On GPU the same schedule pays reduce→D2H→host-sample→H2D→
  collapse per layer; `batchedmeasgpu / batchedgpu` at the same (k,B) is
  therefore the decisive number: it isolates exactly the cost the literature
  never measures. If it erases the batched win, the design needs on-device
  sampling / CUDA graphs before it is viable.

## Run 2 — GH200 (PENDING — access unblocked, budget now ~$200 at ~$3/hr)

Lambda or Vultr, self-serve; H200 SXM is a valid FP64 stand-in except for
NVLink-C2C transfer behavior.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=90
cmake --build build -j
./build/bench_cpu 10 30 12,16,18,20 4 > cpu.csv    # multicore Grace baseline
./build/bench_gpu 10 30 12,16,18,20 4 > gpu.csv
```

Note: `bench_gpu.cu` has passed clang CUDA host+device syntax checking but has
never been through nvcc — budget the first session for compile fixes before
the timed runs.

### Trust gate (no number counts until these pass)

1. All validators OK (< 1e-9): `[validate]`, `[validate-batched]`,
   `[validate-devmeas]`, `[validate-mimd]`, `[validate-real]` (incl. zero rng
   and frame-word mismatches).
2. Grace `bench_cpu` runs multicore (OpenMP enabled) — that is the honest
   opponent, not the single-thread numbers from Run 1.

### Decision numbers (precommitted)

3. **Go/no-go ratio**: `batchedrealgpu ÷ batchedreal` shots/s at the same
   (k, B), k ∈ {16, 18, 20}, best B — the census-calibrated real op mix on
   both sides (see `../opcode_census.md`; the older gate-heavy pair
   `batcheddevgpu ÷ batchedmeas` is the rotation-heavy secondary, and
   gates-only is the ceiling, per the README quoting rules).
4. **Architecture fork**: `mimdrealgpu` vs `batchedrealgpu` at the same (k, B).

Verdicts (written down before the run so the result can't be argued with):
- **Build the backend** if SoA-real ≥ 2× MIMD-real AND SoA-real ≥ 3× the
  Grace-multicore `batchedreal` baseline.
- **Tie → upgrade clifft-cuda to FP64 instead** if SoA-real is within 1.3× of
  MIMD-real (whichever side is ahead) while either clearly beats the CPU.
- **Stop ("CPU wins per watt")** if the best GPU arm is < 2× the
  Grace-multicore CPU on the real mix.
- Between these bands: run the borderline protocol — replay a real compiled
  trace before deciding (see `../opcode_census.py` for the trace tooling).

### Context numbers

5. Single-state crossover k per op (incl. U2/U4 — expect U4 to cross
   earliest).
6. Batched gates-only curve: shots/s vs B, plateau B (occupancy), plateau
   height vs Grace-multicore CPU.
7. `measround` vs `measrounddev` vs `measroundreal` µs (host round-trip vs
   device sampling vs real interfere round, per batch size) — plus the
   `batchedrealgpu` vs `batcheddevgpu` gap, which prices the frame-tick launch
   tax on real instruction streams.
8. H2D/D2H bandwidth (NVLink-C2C; expect hundreds of GB/s, not PCIe ~25).
9. Optional with spare budget: build clifft-cuda on the same box (end-to-end
   real-circuit anchor) and rent a many-core x86 node for the AVX-512 CPU
   baseline the model lacks.
