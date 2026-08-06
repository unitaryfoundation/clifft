# GPU acceleration of clifft's dense active block — scoping research

**Updated:** 2026-07-07
**Branch:** `research/gpu-cudaq-custatevec`
**Question:** Can NVIDIA cuStateVec / custom CUDA accelerate clifft's dense active block `|φ⟩_A`, and what is the most promising GPU approach overall?

Status: scoping complete, microbenchmark built and CPU-validated; the decisive GH200 run (~$20 of self-serve cloud compute) is the remaining step.

---

## TL;DR recommendation

GPU acceleration is **technically feasible and a representation match** (the active block is a genuine dense 2^k complex128 statevector), but the **single-large-statevector route is the wrong target** for clifft's profile. The promising route is **batching across Monte-Carlo shots** — many small (k<20) statevectors run in parallel on the GPU.

**Recommended architecture:** custom CUDA kernels over self-managed device buffers, optionally using cuStateVec's classic-API batched primitives (`custatevecApplyMatrixBatched`, `MeasureBatched`, `abs2sumArrayBatched`, `collapseByBitStringBatched`, `computeExpectationBatched`) for standard gates. clifft's non-standard ops (`OP_EXPAND`, `OP_EXPAND_T`, X-basis `INTERFERE`) have no library primitive and need custom kernels regardless. Notes:

- The batched API family exists **only in the classic cuStateVec API**; the Ex API has no batched equivalents (Ex targets single large/distributed statevectors). Ex is optionally useful for `custatevecExStateVectorAddWires()` — resizable statevectors, relevant to `OP_EXPAND` — and its interop entry point `custatevecExStateVectorGetResourcesFromDeviceSubSV` exposes raw device pointers/streams for mixing library calls with custom kernels. Blackwell GPUs require each batched statevector to be 128-byte aligned.
- **CUDA-Q is not a viable integration layer** — it is an end-user kernel DSL/platform. Its extension interface points the opposite direction (you plug a simulator *into* CUDA-Q via `nvqir::CircuitSimulatorBase`); state access from outside a kernel is read/init-only; the statevector backend defaults to fp32 (fp64 via `--target-option fp64`). Direct cuStateVec + custom CUDA is the right layer.

**Honest expectation:** these kernels are memory-bandwidth-bound, so the achievable GPU/CPU ratio is capped by the bandwidth ratio (~3.3 TB/s HBM3 vs ~0.5 TB/s Grace ≈ **6–8×** once both sides saturate; the best published apples-to-apples figure is NVIDIA's own 5.9× vs qsim-CPU at 32q). Expect **~5–10× vs a tuned CPU at 28–32q FP64, single node** — batching's job is to *reach* saturation at small k (where a single statevector cannot occupy the GPU), plus cache-residency effects, not to exceed the bandwidth ceiling. The one exception is the fused `OP_ARRAY_U4` opcode, which is compute-bound (measured 0.76 Gamp/s flat in k on CPU) — the op where GPU FP64 FLOPs, not bandwidth, can pay.

---

## Why batching across shots fits clifft structurally

1. **Shots run in lockstep k — no re-bucketing needed.** clifft's active-k trajectory is a compile-time-static schedule, identical for every shot of a program: every k-moving opcode (`OP_EXPAND*`, `OP_MEAS_ACTIVE_*`) moves k unconditionally, and the compiler materializes the per-instruction k history (`SourceMap`, `backend/source_map.h`). Noise ops (including correlated Pauli channels, #158) mutate only the symbolic frame — never k. A batch of same-program shots is therefore a dense `B × 2^k` tensor throughout, automatically satisfying cuStateVec's uniform-qubit-count-per-batch constraint.
2. **Per-shot divergence is scalar-shaped.** Measurement outcomes, noise draws, and feedforward differ across shots, but each is expressible as a per-shot *scalar* (an outcome bit selecting keep-low/keep-high or a fold sign, a channel index, a frame word) parameterizing shared kernels — predicated lanes, no divergent code paths. Post-selection is a per-shot active mask. Frame/record bookkeeping between array ops is bulk-batchable (the same op on B frame words), so there is no Amdahl serial tail.
3. **Memory fits.** Batch footprint is `B × 2^peak_rank × 16` bytes with `peak_rank` known at compile time: k=14 → 256 MiB at B=1024; k=18 → 4 GiB; k=22 → 64 GiB (B must shrink). The k<20 target regime batches comfortably on one GPU.
4. **Integration cost: medium.** ~30 `exec_*` kernels become SoA-batched device kernels; measurements become batched segmented reductions + per-shot-scalar collapses (the existing kernels already have exactly this reduce-then-scalar-branch shape); per-shot RNG streams are cheap (xoshiro's 256-bit state); the hugetlb/NUMA host machinery is replaced by device allocation. The main semantic change is per-shot RNG streams (today one stream runs across all shots), which redefines seeding/reproducibility contracts.
5. **The niche is unoccupied.** Qiskit Aer's `batched_shots_gpu` (default cap 16 qubits) batches stochastic divergence (measure/reset/noise/legacy `c_if`) but not dynamic control flow (`jump`/`mark` excluded), and is mutually exclusive with cuStateVec; Aer is under reduced maintenance (PR #2427, 2026-05-21). CUDA-Q ships batched trajectories (`CUDAQ_BATCHED_SIM_MAX_QUBITS`, default 20) but deactivates batching for circuits with mid-circuit measurement + conditional branching (sequential fallback). Batched small statevectors under per-shot-divergent mid-circuit measurement — exactly clifft's profile — is served by no one.

## Why the single-statevector route is weak

1. **Narrow win band.** clifft's sweet spot is small k; GPUs only beat a tuned CPU at large k; the memory wall (~k=30, 16 GB) caps the single-state slice at k≈22–30, where the bandwidth-ceiling argument above bounds the win at low single digits to ~10×. Large published GPU-vs-CPU wins are baseline artifacts (single-threaded NumPy, stock simulators; a 24-simulator benchmark found up to 1000× spread between CPU packages).
2. **Monte-Carlo, not one big state.** clifft runs many shots with data-dependent mid-circuit branching decided on the CPU — the batched-shots profile, not the single-big-state profile.
3. **complex128 everywhere.** FP64 runs at 1:2 of FP32 on H100/GH200 but 1:64 on consumer RTX (Ada and Blackwell) — datacenter parts are effectively required, which the GH200 target satisfies.

## Measured CPU anchors (Apple Silicon, complex128)

Real clifft executes rotation sweeps at k=20 at **3.64 Gamp/s single-thread** (unoptimized compile; the optimizer eliminates naive benchmark circuits — measure against `hir_passes=None` or real workloads) and **9.6 Gamp/s on all 11 cores** — 2.6× scaling, i.e. memory-bandwidth-bound, so multicore CPU baselines are bus-limited, not core-limited. The microbenchmark's portable CPU model reproduces the per-op numbers within ~10% (see `microbench/RESULTS.md`), so its GH200 CPU baseline (Grace/NEON, OpenMP) can be trusted.

## Landscape (verified July 2026)

- **cuQuantum 26.06** (cuStateVec v1.14): batched classic-API family current; uniform-qubit-count constraint documented; Ex API no longer experimental. New in SDK 25.11: **cuStabilizer** (Clifford-only GPU Pauli-frame simulation, ~1000× vs Stim — a watch item if it ever grows T-gate support) and cuPauliProp. No library-level dynamic-circuit support anywhere in cuQuantum (measurement returns to host; classical control is the caller's loop).
- **QuEra Tsim** (arXiv:2604.01059, `bloqade-tsim`): GPU stabilizer-rank QEC *sampler* — batches decomposition terms and shots on GH200/RTX, but FP32-only (complex64 hard-coded) and T-count-exponential (~30–40 T practical ceiling). Different regime: clifft's GPU pitch is exact/FP64/k-exponential strong+weak simulation, not QEC sampling throughput.
- **quEStab** (ICS '26): multi-GPU extended-stabilizer simulation parallelizing over decomposition *terms*, not shots — an orthogonal axis. Details paywalled (no preprint); pull the PDF before any publication comparison.
- **GH200 cloud access is self-serve and cheap**: Lambda $2.29/GPU/hr, Vultr $1.99/hr, CoreWeave $6.50/hr (mid-2026). H200 SXM is a valid FP64 stand-in (identical 34/67/67 TFLOPS) for everything except NVLink-C2C transfer behavior.
- **clifft-cuda** (github.com/haoliri0/clifft-cuda, July 2026): third-party CUDA sampler that interprets clifft's *own compiled bytecode* on-device — an existence proof for the batch-across-shots axis and for the small integration surface (~2000-line `.cu`). Design: MIMD one-block-per-shot (peak_rank ≤ 19), **FP32** amplitudes, on-device per-shot RNG (no measurement round-trip at all), hazard skip-ahead noise, early-exit postselection discards. Honest calibration from its own README: ~1.18M shots/s (RTX PRO 5000, 300 W) vs ~727K shots/s (40-thread Xeon, 125 W) = **1.6× per node, a loss per watt** on d=5 cultivation — consistent with this report's skeptical prior, and well under the bandwidth ratio, leaving the MIMD-vs-lockstep-SoA question open. It does not occupy the exact-FP64 niche.

## Refuted claims (killed in verification — do NOT cite these)

- ❌ "GPU beats CPU at n=16 (5.7x), >64x at n≥20 on A100."
- ❌ "20–28 qubits: 64–146x speedup over NumPy." — NumPy is not a real baseline.
- ❌ "complex64 gives an extra 1.7–1.9x over complex128."
- ❌ "Distributed: comms ~99%, GPU benefit negligible."
- ❌ "32-qubit GHZ: cuStateVec 5.9x over qsim 64-core." — the verified figure is 5.9x vs qsim's *CPU version* on one A100, the honest anchor for the ~5–10x prior.
- ❌ "Single dense statevector GPU crossover at 14–18 qubits."

## The decisive experiment — built, pending the GH200 run

`microbench/` implements clifft's op mix (including the fused `U2`/`U4` opcodes the optimizer emits) as faithful portable CPU kernels and hand-written complex128 CUDA, single-state and batched, with full CPU-vs-GPU validation including all batched kernels and completed measurements. It measures, per `microbench/README.md`:

1. single-statevector per-op crossover k (FP64, vs the faithful CPU baseline);
2. batched gates-only throughput vs B (the occupancy/saturation curve — the ceiling);
3. **batched throughput with one completed measurement per layer** (per-shot reduce → D2H → host sample → H2D → outcome-selected collapse → rank-restoring expand) — the number that decides the go/no-go, since it prices the host round-trip real mid-circuit-measurement shots pay. On CPU the same schedule costs only ~3% extra, so the GPU-side gap isolates the round-trip. On GH200, NVLink-C2C (900 GB/s, 450/direction) should make the D2H/H2D legs cheap — that is the bet being tested;
4. host↔device bandwidth.

Two arms added after the clifft-cuda find (2026-07-15): **on-device outcome sampling** (`batcheddevgpu` — the production design; the host round-trip variant stays as a priced comparison) and a **FP64 MIMD one-block-per-shot interpreter** (`mimdgpu` — clifft-cuda's architecture on the identical schedule/state/RNG).

Two more arms added after the opcode census (2026-08-06, `opcode_census.py` / `opcode_census.md` — a walk of real clifft-compiled bytecode): the original synthetic layer overweights gates ~3–30× (real gates-per-measurement band 1.7–15.3 vs the synthetic 50; measurement+expand is 30–73% of real amp-weighted work, almost entirely `MEAS_INTERFERE` + `EXPAND_T`; frame-only ops are 23–77% of real instruction streams). The **census-calibrated real mix** (`batchedreal`/`batchedrealgpu`/`mimdrealgpu`) fixes all three: ~8 gates per completed X-basis measurement + `EXPAND_T`, plus frame-tick ops that are ~free on CPU/MIMD but one kernel launch each on SoA — pricing the launch tax a real bytecode stream imposes. `mimdrealgpu` vs `batchedrealgpu` settles the architecture fork on the realistic workload: if SoA wins clearly, clifft-cuda's 1.6× was architectural headroom and our design is the one to build; if MIMD ties, the cheap path is upgrading clifft-cuda to FP64.

Quoting rule: the honest batched comparison is `batchedrealgpu / batchedreal` (census-calibrated mix, both sides pay completed measurements; GPU samples on device); `batcheddevgpu / batchedmeas` is the gate-heavy secondary (bounds rotation-heavy workloads); the gates-only ratio is a ceiling, not "the speedup". Precommitted numeric verdicts live in `microbench/RESULTS.md`.

Run on the GH200: `./microbench/run.sh 10 30 12,16,18,20 4` → `cpu.csv` + `gpu.csv`, then fill in the Run-2 checklist in `microbench/RESULTS.md` and summarize here.

## If the numbers clear the bar: integration order

1. **Batched forced-replay** (`record_probabilities`): zero per-shot host divergence (all outcomes forced — no sampling round-trips), embarrassingly parallel across records, and it accelerates an API users call for likelihood/probability workloads. The cleanest first deliverable.
2. **Noisy sampling**: the most shot-hungry workload; noise is frame-only and k-invariant, so it batches cleanly (per-shot fire/skip predication on frame XORs).
3. General `sample()` batching (per-shot RNG semantics change; reproducibility contract needs redefinition).

## Sources

Primary (NVIDIA docs):
- cuStateVec API reference — https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/api-reference/custatevec/index.html
- cuStateVec Ex overview — https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/custatevecex/overview.html
- cuQuantum SDK release notes (26.06; cuStabilizer in 25.11) — https://docs.nvidia.com/cuda/cuquantum/latest/cuquantum-sdk-release-notes.html
- cuStateVec acceleration blog (the 5.9x-vs-qsim-CPU figure; vendor) — https://developer.nvidia.com/blog/accelerating-quantum-circuit-simulation-with-nvidia-custatevec/
- Grace-Hopper architecture (NVLink-C2C 900 GB/s) — https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/
- Qiskit Aer `batched_shots_gpu` — https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
- CUDA-Q simulators / NVQIR extension — https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html , https://nvidia.github.io/cuda-quantum/latest/using/extending/nvqir_simulator.html

Academic / competitive:
- van Niekerk et al. 2024, QV crossover bands — https://arxiv.org/pdf/2412.20518
- Faj et al., IEEE eScience 2023, Aer Thrust+cuQuantum FP64 on A100 — https://arxiv.org/pdf/2307.14860
- QuEra Tsim — https://arxiv.org/abs/2604.01059 , https://github.com/QuEraComputing/tsim
- quEStab, ICS '26 — DOI 10.1145/3797905.3816723 (paywalled, no preprint)

Full raw findings (verified-claim list with vote tallies, refuted claims, caveats) archived in `findings-raw.json`.
