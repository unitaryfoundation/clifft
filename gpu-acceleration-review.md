# GPU Acceleration for Clifft: Review and Initial Analysis

*Working analysis document, 2026-06-10. CPU-only desk study: all numbers are from
the Clifft paper (arXiv:2604.27058), the clifft-paper benchmark repo, the
clifft-cuda repository, the SOFT paper, and the NVIDIA cuQuantum documentation.
No new profiling or GPU experiments were run.*

This document addresses three questions:

1. What do we already know about Clifft's scaling and measured performance,
   including the third-party clifft-cuda port?
2. In which regimes and for which features is a GPU plausibly a win?
3. For each candidate, do NVIDIA's cuStateVec and cuStabilizer libraries fit,
   based on an API-level study (Appendices A–C)?

---

## 1. What we know: Clifft's execution model and existing numbers

### 1.1 Execution model recap

Clifft compiles a Stim-format circuit once, then samples many shots:

- **Compile time (offline):** all Clifford tableau work — Heisenberg mapping,
  HIR optimization, Pauli localization, active-space planning — costs
  `O(CN + EN + (M+T)N²)` and runs once. Measured compile times are
  milliseconds (e.g. ~6.4 ms for surface_d7_r7 after the compile-perf work).
  **There is no tableau work left at sample time.** This is the single most
  important fact for the GPU discussion: the thing GPU stabilizer libraries
  accelerate is the thing Clifft already removed from the hot path.
- **Sample time (online, per shot):**

  ```
  O( (T + M + E)·N  +  (T + M_active)·2^k_max )
     └── packed-bit Pauli-frame /      └── dense active-array sweeps
         noise / record updates            (SIMD kernels, complex<double>)
  ```

  The SVM interprets a fixed bytecode schedule. By the decoupling theorem
  (paper Thm. 1), the instruction sequence and the active-set geometry are
  **identical for every shot** — only frame bits, amplitudes, RNG draws, and
  measurement outcomes differ. Shots are embarrassingly parallel.

Two structural consequences for GPUs:

- **No structural divergence across shots.** Unlike SOFT/PFSR-style sparse
  simulators (whose per-shot tableau eliminations are data-dependent), every
  Clifft shot executes the same ops on the same-sized arrays. This is unusually
  GPU-friendly: a warp of shots stays in lockstep by construction. Residual
  divergence comes only from data (noise realizations, measurement branches,
  postselect early exit).
- **Per-shot state is tiny in the design sweet spot.** At `k_max = 10` the
  active array is 1024 `complex<double>` = 16 KB; the Pauli frame is ~2N bits.
  A whole d=5 MSC shot takes ~7 µs on one modern CPU core — less than the
  latency of a single CUDA kernel launch. Any GPU mapping must therefore batch
  thousands-to-millions of shots per launch.

### 1.2 Published CPU numbers (paper, Tables 1–2 and QV benchmark)

Throughput on one AWS c8i.8xlarge (16 physical cores) unless noted:

| Circuit | N | k_max | Clifft | Reference points |
|---|---|---|---|---|
| Surface code d=7 r=7 (pure Clifford) | 118 | 0 | 2.2 M shots/s | Stim 20.1 M, Tsim 315 k (GH200) |
| MSC cultivation d=3 | 15 | 4 | 10.4 M shots/s | Tsim 27.9 k |
| MSC cultivation d=5 | 42 | 10 | 314 k shots/s | Tsim DNC |
| Distillation | 85 | 5 | 1.5 M shots/s | Tsim 1.5 M |
| Coherent-noise surface d=3 r=3 | 26 | 8 | 1.7 M shots/s | Tsim DNC |
| Coherent-noise surface d=5 r=1 | 64 | 13 | 133 k shots/s | Tsim 571 k |
| Coherent-noise surface d=5 r=5 | 64 | 24 | **0.7 shots/s** | Tsim DNC |

Production-run numbers (c6i.8xlarge, 16 cores, d=5 inject+cultivation with
postselect early termination and stratified importance sampling):
**2.17 M shots/s per machine ≈ 135 k shots/s per core ≈ 7.4 µs/shot/core.**
The same workload on SOFT: ~10.7 k shots/s per H800 GPU, i.e. **one CPU core ≈
13× one H800** under SOFT's generalized-stabilizer approach.

Dense limit (QV benchmark, k_max = N, 16 cores, single shot): Clifft is
neck-and-neck with leading CPU statevector simulators — at N=28:
Clifft 34.1 s, qsim 10.6 s, Qulacs 360 s. Throughput here is governed by
memory bandwidth over a 2^N array, the classic GPU statevector regime.

Key derived facts:

- **Small k (≤ ~10–12):** per-shot cost is microseconds and dominated by a mix
  of tiny dense sweeps (≤16 KB, cache-resident) and integer/bit bookkeeping.
- **Crossover (k ≈ 18):** the CPU implementation switches from single-threaded
  cache-resident execution to OpenMP-parallel array kernels.
- **Large k (≥ ~20):** throughput falls off the 2^k cliff (0.7 shots/s at
  k=24); this is where Clifft is bandwidth-bound like any statevector sim.

### 1.3 What clifft-cuda actually is

From reading its source (`src/cuda_sampler.cu`, ~2,000 lines):

- A **megakernel bytecode interpreter**: the compiled Clifft program is
  flattened and shipped to the GPU; three kernel variants execute whole shots
  end-to-end — thread-per-shot (`sample_kernel`), block-per-shot with the
  active array in shared memory (`sample_kernel_coop`), and block-per-shot
  with global-memory arrays for larger k (`sample_kernel_global_coop`).
  This is the right architectural shape for the small-k regime (see §3) —
  notably it does *not* use cuStateVec or cuStabilizer at all.
- **Restrictions** (vs. Clifft's full bytecode, see Appendix C):
  `peak_rank ≤ 19`, `≤ 128` qubits, `≤ 8` observables, `≤ 1024` measurement
  slots; **fp32 amplitudes** (Clifft CPU is `complex<double>`; probabilities
  accumulated in double); **aggregate counts only — no per-shot detector
  records**; no arbitrary-angle rotations (`OP_ARRAY_ROT`, `OP_EXPAND_ROT`,
  `OP_ARRAY_U2/U4`), no expectation probes (`OP_EXP_VAL`), and none of the
  `*_FORCED` ops, so **no stratified importance sampling**. The unsupported-op
  error message calls it "the d=3 GPU VM".
- The no-records restriction matters most: it rules out every
  decoder-in-the-loop workflow (e.g. the end-to-end MSC study, decoder-gap
  post-selection), restricting it to circuits whose answer is a handful of
  aggregate counters.

### 1.4 Is clifft-cuda's 19.7× apples-to-apples? No — it's a GPU vs. roughly one CPU core

clifft-cuda's comparison (d=5 MSC, p=10⁻³, postselection): 59.8 k shots/s for
Clifft on a Xeon Gold 5218R — **core count unstated** — vs. 1.18 M shots/s on
an RTX PRO 5000.

The paper's numbers for the same workload class calibrate that baseline: one
modern CPU core sustains ~135 k shots/s, a 16-core node ~2.17 M shots/s. So:

- Their 59.8 k shots/s baseline ≈ **one core** of a 2019 Xeon (the 5218R has
  20 cores) — a single core, not a machine.
- Their 1.18 M shots/s GPU result ≈ **half of one modern 16-core CPU node**,
  and roughly parity with the full 20-core socket they sampled one slice of.
- Per watt the CPU node also wins (~9–11 k vs. ~4 k shots/s/W), and the GPU
  ran fp32 while the CPU baseline runs fp64.

Caveat: their exact circuit, postselection accounting, and thread count are
not fully documented, so this is inference from published numbers, not a
measurement. The cheap follow-up: run mainline Clifft on their
`circuit_d5_p=0.001.stim` on one modern CPU node, and their binary on matched
hardware once GPU access exists.

Bottom line: **the published evidence shows machine-level parity, not a GPU
win, in the small-k regime — at reduced precision and a much smaller feature
set.** That is itself a useful result: it sharpens the question of *where*
GPUs do win.

---

## 2. Candidate regimes and features for GPU acceleration

To directly answer "does only the SVM benefit, or also localization/lowering?":
**only sample-time execution is a candidate.** Localization, lowering, and all
tableau operations are compile-time, run once, and cost milliseconds-to-seconds
even for the largest paper circuits; GPU-accelerating them has no payoff. All
candidates below are sample-time.

| # | Candidate | Where it shows up | Evidence we have |
|---|---|---|---|
| A | Small-k mass sampling (k ≤ ~12) | MSC inject+cultivation, rare-event QEC; Clifft's headline regime | §1.2, §1.4: CPU does 0.3–10 M shots/s; GPU parity at best so far |
| B | Mid-k batched sampling (k ≈ 13–19) | Coherent-noise d=5 r=1 (k=13); larger cultivation variants | CPU drops to ~133 k shots/s as arrays leave cache; clifft-cuda's own cap is k=19 |
| C | Large-k execution (k ≥ ~20) | Coherent-noise circuits (k=24 → 0.7 shots/s), dense/QV limit | CPU hits the 2^k cliff; classic GPU statevector territory |
| D | Pure-Clifford bulk segments | MSC escape stage (N=463, Clifford-only), surface-code memory | Frame-update term `(M+E)·N` dominates when k=0; Stim is 10× faster than Clifft here |
| E | Stratified importance sampling (`*_FORCED` ops) | Rare-event estimation; gave 2.5× shot reduction in the d=5 IC study | Feature, not regime; absent from clifft-cuda |
| F | Expectation-value probes (`OP_EXP_VAL`) | T-state fidelity estimator in end-to-end MSC | Per-shot array reduction; batched analog exists in cuStateVec |

Cross-cutting design constraint (any GPU backend): **output bandwidth.**
Decoder workflows need per-shot detector records. At 1 M shots/s × ~10³
detector bits ≈ 125 MB/s — well within PCIe, but it must be designed for
(pinned-memory streaming, shot compaction after postselect, downstream decoder
throughput). clifft-cuda sidestepped this by only returning aggregates; a
general backend cannot.

---

## 3. Verdicts: GPU or not, and library fit

Summary table; the candidates with a real opportunity (B, C, E, F) get
detailed sections below. Library-fit reasoning is grounded in the API study of
Appendices A–B.

| Candidate | GPU worth it? | cuStateVec | cuStabilizer | What you'd actually build |
|---|---|---|---|---|
| A. Small-k sampling | Unproven; parity so far | ✗ | ✗ | Custom megakernel (clifft-cuda shape), done properly |
| B. Mid-k sampling | Most promising open question | Partial (batched APIs for array ops) | ✗ | Custom megakernel + possibly batched cuStateVec calls |
| C. Large-k execution | Yes — clearest win | ✓ strong fit | ✗ | cuStateVec-backed SVM array backend |
| D. Clifford segments | Not as a standalone | ✗ | Partial but it's "GPU Stim" | Nothing now; hybrid idea only |
| E. Importance sampling | Required feature of any GPU backend; GPU-friendly | ✗ (no support) | ✗ | Custom kernels (uniform forced schedules reduce divergence) |
| F. Expectation probes | Rides along with B/C | ✓ (`ComputeExpectationBatched`) | ✗ | Library call |

**Why A and D get no detailed section — no opportunity seen.** Small-k
sampling (A) is the one regime with published GPU evidence, and that evidence
shows socket-level parity (§1.4): per-shot work is ~7 µs of cache-resident
sweeps, bit bookkeeping, RNG, and branchy control — latency/integer-bound work
that is CPU home turf. The only viable GPU mapping is a clifft-cuda-style
megakernel; the NVIDIA library APIs cannot hold a 16 KB array resident across
thousands of instructions (Appendix A.4), and a win from better megakernel
engineering would likely be small multiples, not 20×. Pure-Clifford segments
(D) are Stim's domain, which Clifft deliberately concedes (10× gap in the
pure-Clifford limit); using cuStabilizer there would re-introduce the runtime
Clifford representation the compiler exists to eliminate. Appendix B records
why, plus one speculative GF(2) hybrid idea kept for the back pocket.

### 3.1 Candidate B — mid-k (≈13–19): the most interesting open question

Here the active array (64 KB–4 MB fp64) falls out of CPU L1/L2 and CPU
throughput drops sharply (133 k shots/s at k=13 already), while it's still far
too small to use a GPU efficiently one shot at a time. Batching changes that:
a few thousand resident shots give the GPU real bandwidth-bound work per
instruction, and the shot-uniform schedule means a single
`ApplyMatrixBatched`-style sweep per instruction is semantically exact.

- **cuStateVec batched APIs partially fit** (Appendix A.2): targets/controls
  are shared across the batch — fine, since Clifft's schedule is
  shot-independent; per-shot variation (frame-sign of a rotation, measurement
  branch) maps onto `CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED` (per-SV matrix
  selection) and `CollapseByBitStringBatched` (per-SV bitstring + norm).
  Active-dimension growth maps onto a fixed `svStride = 2^k_max` buffer with
  per-call `nIndexBits` = current k. Measurements decompose into
  `Abs2SumArrayBatched` → per-shot draw → `CollapseByBitStringBatched`.
- **But** noise sampling, frame bits, records, forced ops, and postselect
  compaction still require custom kernels, and per-instruction dispatch makes
  instruction fusion (which the bytecode optimizer relies on) impossible
  through the library boundary.

Realistic shape: a custom megakernel in the clifft-cuda style but with
global-memory arrays (their `sample_kernel_global_coop` variant already gestures
at this), with cuStateVec batched calls as an optional backend for the heavy
array ops. This is the regime where neither CPU nor published GPU numbers
exist — the natural first experiment once GPU hardware is available.

### 3.2 Candidate C — large k (≥ ~20): clear GPU win, cuStateVec is the right tool

At k=24, Clifft does 0.7 shots/s on 16 cores; at k=28 (QV) a single dense pass
takes ~34 s. Per-shot arrays are 16 MB–16 GB; execution is a sequence of full
sweeps over one large array — precisely what cuStateVec is built for
(`ApplyMatrix`, `ApplyPauliRotation`, samplers, expectation; plus the Ex API's
SVUpdater gate-fusion and multi-GPU distribution for k beyond single-device
memory). A modern GPU's ~10× HBM-vs-DDR bandwidth advantage translates
directly here, with batching across shots when memory allows. The glue costs
(frame bits, noise, records — all O(N) bits per shot) are negligible relative
to 2^k sweeps, so the per-instruction library-call granularity that rules out
the small-k regime (Appendix A.4) is irrelevant here.

This is also the most actionable implementation: it accelerates
**coherent-noise QEC simulation**, where Clifft is currently
throughput-limited (the paper's own k=24 row is nearly unusable on CPU), so it
*extends Clifft's reach* rather than re-running an already-solved workload
faster. It requires no precision compromise (cuStateVec is natively fp64) and
no rewrite of the CPU code for fairness comparisons.

### 3.3 Candidates E & F — features any GPU backend must carry

- **Importance sampling** is arguably *more* GPU-friendly than brute-force
  sampling: within a stratum, every shot has the same number of forced faults,
  making the noise path's work uniform across the warp and removing the main
  source of data divergence. It's also central to the rare-event workflow
  (2.5× shot reduction in the d=5 study). No NVIDIA library covers it.
  clifft-cuda's omission of it (and of per-shot records) is a large part of
  why it's not a general solution.
- **Expectation probes** map directly onto `custatevecComputeExpectationBatched`
  / `ComputeExpectation`, and matter for fidelity-estimator workflows.

### 3.4 Precision note

clifft-cuda's fp32 amplitudes (with double-precision probability
accumulations) make its numbers incomparable to Clifft's fp64 CPU numbers in
both speed and exactness claims. Options for a fair story without templating
the CPU code: (a) build the GPU backend fp64-first (cuStateVec supports it
natively; candidate C needs nothing else), and treat fp32 as a clearly-labeled
2× bandwidth bonus; (b) validate fp32 statistically — at k ≤ 12 a shot sees
only ~10²–10³ array operations, so fp32 error is plausibly negligible for
aggregate rate estimation, but that's a claim to verify against CPU
ground-truth rates, not assume.

---


## Appendix A — cuStateVec API review vs. Clifft bytecode

### A.1 Library shape

cuStateVec (cuQuantum) operates on dense state vectors in GPU global memory
through a host-side handle. Two API layers: the standard API (single SV +
"batched" variants over many same-shape SVs) and the **Ex** API
(`custatevecex*`: SVUpdater with gate fusion, multi-GPU/multi-node
distribution, state migration). Supported precisions include fp32 and fp64
complex. Relevant function groups:

- Gate application: `ApplyMatrix`, `ApplyMatrixBatched`,
  `ApplyPauliRotation` (exp(iθP) directly — matches Clifft's Pauli-rotation
  primitive), `ApplyGeneralizedPermutationMatrix`, diagonal matrices.
- Probabilities/measurement: `Abs2SumOnZBasis`, `Abs2SumArray(+Batched)`,
  `CollapseOnZBasis`, `CollapseByBitString(+Batched)`, `MeasureOnZBasis`,
  `BatchMeasure` (multi-qubit, single SV), `MeasureBatched` (across SVs).
- Expectation: `ComputeExpectation(+Batched)`, expectations on Pauli bases.
- Sampling: `SamplerCreate/Preprocess/Sample` (multinomial output sampling).
- Distribution: `SwapIndexBits`, multi-device variants, communicator APIs.

### A.2 Batched-API constraints (verified against NVIDIA's samples)

From `batched_gate_application.cu` / `batched_collapse.cu`:

- SVs are packed contiguously: `nSVs × svStride`, same `nIndexBits` per call
  (the call's `nIndexBits` is a parameter, so a fixed-stride buffer can be
  treated as growing-k SVs across the program — matches Clifft's
  expand-on-schedule model).
- `ApplyMatrixBatched`: **targets and controls are shared across the whole
  batch**; per-SV variation only via `matrixIndices` selecting among
  `nMatrices` uploaded matrices (`MATRIX_MAP_TYPE_MATRIX_INDEXED`).
  For Clifft this is workable *because* the schedule is shot-independent:
  the virtual axis is the same for all shots; the per-shot ±θ frame sign is a
  2-entry indexed matrix set.
- `CollapseByBitStringBatched`: per-SV bitstring and per-SV norm — covers
  per-shot measurement branches.

### A.3 Mapping Clifft's bytecode op classes

| Clifft op class | cuStateVec mapping | Fit |
|---|---|---|
| `OP_FRAME_*` (H, S, S†, CNOT, CZ, SWAP) | none — O(1) frame-bit updates | custom kernel (trivial) |
| `OP_ARRAY_*` (H, S, T, CNOT, CZ, SWAP, MULTI_*) | `ApplyMatrixBatched` (indexed matrices for frame-sign variants) | good (mid/large k) |
| `OP_ARRAY_ROT`, `OP_ARRAY_U2/U4` | `ApplyMatrixBatched` / `ApplyPauliRotation` (single-SV) | good |
| `OP_EXPAND`, `OP_EXPAND_T`, `OP_EXPAND_ROT` | no direct analog; write \|+⟩ factor into next axis | custom kernel (trivial) |
| `OP_MEAS_DORMANT_STATIC/RANDOM` | none — frame parity + RNG, no array touch | custom kernel |
| `OP_MEAS_ACTIVE_DIAGONAL/INTERFERE`, `OP_SWAP_MEAS_INTERFERE` | `Abs2SumArrayBatched` → per-shot draw → `CollapseByBitStringBatched` (+`ApplyMatrixBatched` for basis fold) | good, 2–3 calls per measurement |
| `OP_APPLY_PAULI` | none — frame multiplication + γ phase | custom kernel |
| `OP_NOISE`, `OP_NOISE_BLOCK`, `OP_READOUT_NOISE` | none — hazard-jump sampling | custom kernel |
| `OP_DETECTOR`, `OP_OBSERVABLE`, `OP_POSTSELECT` | none — bit records, early exit needs stream compaction | custom kernel + CUB |
| `*_FORCED` (importance sampling) | none | custom kernel |
| `OP_EXP_VAL` | `ComputeExpectationBatched` | good |

Reading of the table: cuStateVec covers the **array column** of Clifft's cost
model well, and covers none of the **bit/bookkeeping column**. The array
column dominates only at mid-to-large k — hence the regime verdicts in §3.

### A.4 Dispatch-granularity problem (small k)

Each batched call is a host-side library invocation and at least one kernel
over `nSVs × 2^k` global memory. For a several-thousand-instruction program
this is fine when 2^k is large (work per call dominates) and fatal when 2^k is
16 KB (the megakernel alternative holds the array in shared memory across the
entire program and pays zero inter-instruction traffic). There is no public
mechanism to fuse cuStateVec batched calls or capture them into a
megakernel — worth raising as a feature request (CUDA Graphs reduces launch
overhead but not the HBM round-trips).

## Appendix B — cuStabilizer API review vs. Clifft

cuStabilizer (new in cuQuantum) implements **batched Pauli-frame trajectory
simulation of Clifford circuits**: an ensemble of frames F is propagated by
conjugation `F ← G†FG` through a circuit's gates, with Pauli (and
amplitude-damping) noise injected stochastically per trajectory; outputs are
measurement-flip records relative to a noise-free reference. API surface:
circuit construction from a string format
(`custabilizerCreateCircuitFromString`), frame-simulator lifecycle with batch
size (`custabilizerCreateFrameSimulator`), execution
(`custabilizerFrameSimulatorApplyCircuit`), probability/sampling utilities
(`custabilizerSampleProbArray*`), and batched GF(2) sparse matrix products
(`custabilizerGF2Sparse{Dense,Sparse}MatrixMultiply`).

Relation to Clifft:

- It is the GPU analog of **Stim's frame simulator** — same compute model,
  same Clifford-only restriction. It cannot express non-Clifford rotations or
  amplitude-bearing state, so it cannot run Clifft's active subspace.
- The work it accelerates (conjugating frames through Clifford gates) is work
  Clifft performs **once at compile time** via Stim's tableau code in
  milliseconds, not per shot. Clifft's *runtime* frame updates are sparse
  event-driven Pauli multiplications (noise realizations, feed-forward),
  already O(realized faults · N/64) per shot — there is no per-gate frame
  conjugation left to accelerate.
- Residual relevance: (a) the GF(2) sparse-dense multiply could batch
  detector/observable evaluation from realized-fault vectors for *purely
  Clifford segments* of a program (e.g. the MSC escape stage, with the active
  core running elsewhere — a speculative hybrid, not a near-term candidate);
  (b) if NVIDIA's interest is pure-Clifford QEC sampling at scale, cuStabilizer
  competes with Stim, not with Clifft — a boundary worth keeping clear.

## Appendix C — clifft-cuda restrictions vs. mainline Clifft

| Dimension | Clifft (CPU) | clifft-cuda |
|---|---|---|
| Precision | complex<double> | fp32 amplitudes, double accumulations |
| Peak active dimension | limited by memory (k=28+ demonstrated) | k ≤ 19 hard cap |
| Qubits | no practical cap (463 demonstrated) | ≤ 128 |
| Observables / meas slots | unbounded | ≤ 8 / ≤ 1024 |
| Per-shot records (detectors) | yes (decoder workflows) | no — aggregate counters only |
| Arbitrary rotations (`R_X`, `U3`, …) | yes | no (`OP_ARRAY_ROT/U2/U4` unsupported) |
| Expectation probes | yes (`OP_EXP_VAL`) | no |
| Stratified importance sampling | yes (`*_FORCED` ops) | no |
| Postselect early exit | yes | yes |
| Kernel design | bytecode interpreter, SIMD over active array, OpenMP at k>18 | megakernel interpreter; thread-per-shot, block-per-shot (shared mem), block-per-shot (global mem) |

The supported-opcode subset is exactly what the d=3/d=5 MSC postselection
circuits need — hence "not a general solution": it's a point port of Clifft's
easiest GPU-shaped workload, and (per §1.4) even there the published numbers
show socket-level parity rather than a GPU win.
