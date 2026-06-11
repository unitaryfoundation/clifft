# Review: clifft-amd (AMD ROCm/HIP backend for clifft)

**Date:** 2026-06-11
**Subject:** [randreshg/clifft-amd](https://github.com/randreshg/clifft-amd), a fork of
[haoliri0/clifft-cuda](https://github.com/haoliri0/clifft-cuda)
**Reviewed against:** clifft `main` @ `83d8d0d` (2026-06-08)

## TL;DR

- `clifft-amd` is `clifft-cuda` plus exactly **one commit** that adds a HIP backend. The HIP
  kernel (`src/hip_sampler.hip`, 1960 lines) is a **byte-for-byte mechanical transform** of the
  CUDA kernel (`src/cuda_sampler.cu`): every changed line is a runtime-API rename
  (`cudaMalloc` → `hipMalloc`, etc.) or an error-message string. Zero algorithmic divergence.
- The simulation semantics are a **faithful port of clifft's SVM** for the opcodes it supports.
  I diffed every kernel against the current CPU implementations (frame ops, array ops, T/T†,
  EXPAND, all five measurement variants, gap-based noise sampling, dust clamping, conditional
  Paulis): the logic matches, including subtle details like the in-place interfere-fold aliasing
  and the `upper_bound` hazard search.
- The principal differences from the CPU SVM are deliberate scope cuts: **fp32 amplitudes, no
  gamma (global phase / deferred-normalization scalar), aggregate counts only**. These are sound
  for sampling statistics, but two numerical-robustness gaps follow from them (dust-epsilon
  calibration and unbounded norm drift; see findings R1/R2).
- It vendors a **~2026-05-01 snapshot of clifft** (57 commits behind main, pre-PR #52). The
  host-side flattening code uses APIs that no longer exist (`kMaxInlineWords`, inline
  `PauliMask.w[]`, `NoiseChannel.destab_mask`), so **it will not compile against current main**
  without a small porting pass.
- Performance: MI300X (750 W) reaches ~1.2 M shots/s on the d5 MSC circuit — parity with an
  RTX PRO 5000 (300 W), and roughly what one modern CPU socket does running shot streams on all
  cores. The "~20×" speedup in their README is against a single sequential CPU sampler process.
  For our k ≤ 10 workloads, skepticism remains justified; the current kernels leave the obvious
  k=5–10 optimization (multiple shots per block) on the table.

---

## 1. Correctness and quality of the AMD implementation

### 1.1 What it is

The runner (`tools/run_msc_hip.cpp`) compiles a Stim circuit **in-process with the vendored
clifft AOT compiler** (parse → HIR → HIR passes → reference-syndrome probe → lower with a
postselection mask → bytecode passes), then hands the resulting `clifft::CompiledModule` to
`sample_survivors_hip()`, which flattens it into POD device arrays (`GpuProgram`: instructions,
Pauli masks, noise sites/channels/hazards, readout noise, detector/observable target lists) and
launches one of **three kernels selected by `peak_rank`** (the compiled program's maximum active
statevector rank k):

| Regime | peak_rank | Kernel | State location |
|---|---|---|---|
| per-thread | ≤ 4 | one shot per thread, `ShotState` in registers/local | 16 fp32 amplitudes per thread |
| shared-block | 5–10 | one shot per block, cooperative | 2^k amplitudes in LDS (≤ 8 KB) + 16 KB reduction scratch |
| global-block | 11–19 | persistent blocks pulling shots off an atomic work counter | 2^k amplitudes in HBM per worker block |

Output is aggregate-only: `passed_shots`, `logical_errors`, `observable_ones[]` (plus derived
rates), reduced via shared-memory trees (per-thread kernel) or global atomics (coop kernels).

The Python package (`clifft_cuda/`) is scaffolding: CPU-reference plumbing and CUDA/ROCm
*diagnostics* (toolkit/driver discovery). Both `sample_survivors_cuda()` and
`sample_survivors_hip()` in Python unconditionally raise `BackendUnavailable` — the only real
GPU entry point is the C++ CLI.

### 1.2 Semantics verified against the CPU SVM

I checked each device function against the current `src/clifft/svm/` implementations:

| GPU code | CPU counterpart | Verdict |
|---|---|---|
| `frame_{cnot,cz,h,s,swap}` | `exec_frame_*` | identical bit algebra |
| `array_{cnot,cz,swap,h,s}`, `apply_phase`, scatter-bit indexing | `exec_array_*` waterfall loops | same index math and phases (CPU adds AVX tiling, same result) |
| `array_multi_{cnot,cz}` | `exec_array_multi_*` | same popcount-parity formulation |
| `array_t`, `expand_t` (incl. `p_x` conjugation: T ↔ T† when the frame has X on the axis) | `exec_array_t`, `exec_expand_t` | match, **except gamma** (below) |
| `meas_dormant_{static,random}`, `meas_active_{diagonal,interfere}`, `swap_meas_interfere` | `exec_meas_*` | outcome algebra (`m_abs = b ^ p_x`, `m_phys = m_abs ^ sign`), branch probabilities, fold normalization (×1/√2) all match. The per-thread in-place swap-interfere fold is aliasing-safe sequentially (reads hit only not-yet-written or ≥2^to slots); the coop variant correctly goes through a scratch buffer. |
| `sample_branch` with `kDustEpsilon = 1e-18` | `svm_internal.h::sample_branch` | identical, minus the `dust_clamps` telemetry |
| `draw_next_noise` (exponential gap + binary search over cumulative hazards) | `SchrodingerState::draw_next_noise` | binary search is exactly `upper_bound`; sentinel handling equivalent |
| `exec_noise` / `exec_noise_block` channel selection (`u · prob_sum` cumulative scan) | `exec_noise` | identical |
| `apply_pauli` (conditional on a measurement record slot) | `exec_apply_pauli` | identical, **except mask sign** (global phase, dropped — fine for aggregates) |
| `OP_POSTSELECT` / `OP_OBSERVABLE` / `OP_READOUT_NOISE` / expected-observable XOR | `exec_*` + survivor accounting | match |

Things the GPU deliberately does **not** track, all sound for aggregate sampling:

- **gamma** (global phase × deferred normalization scalar). All measurement statistics are
  computed as ratios of branch norms, so phases and the missing 1/√2 in `OP_EXPAND` cancel.
  Consequence: this VM is *measurement-statistics equivalent*, not amplitude-equivalent — it can
  never serve `EXP_VAL`, `probability_of`, or any amplitude-level API without adding gamma back.
- Per-shot records (`meas`/`det` arrays live only for the duration of the shot).

Good defensive design: unsupported opcodes and limit violations (`peak_rank > 19`,
`total_meas_slots > 1024`, `num_observables > 8`, `num_qubits > 128`, `keep_records`) are
**rejected loudly on the host** before launch — no silent wrong answers.

Determinism: the RNG is clifft's own xoshiro256++/splitmix64, but seeded **per shot**
(`seed ⊕ golden·(shot_id+1)` → splitmix64 → state). Results are therefore reproducible for a
fixed seed even in the dynamically scheduled global-block kernel (counts are order-independent
integer atomics), and independent of block size and batch splits. The trade-off: the CPU sampler
draws one sequential stream across shots, so **CPU and GPU can never match shot-for-shot**
as-is; only statistical comparison is possible today (see §4 for how to fix that).

### 1.3 Findings

**Correctness / robustness (would block upstreaming as-is):**

- **R1 — dust epsilon miscalibrated for fp32.** `kDustEpsilon = 1e-18` was tuned (per the
  comment in `svm_internal.h`) so that double-precision interference dust (norms ~1e-30…1e-24)
  is clamped to a deterministic branch. With fp32 amplitudes, rounding dust in branch norms sits
  around 1e-14 relative — **above** the clamp. A branch that is analytically zero can therefore
  survive as ~1e-12 probability and occasionally be *sampled*. At the regime this fork brags
  about (100 G shots, 51 logical errors, 3.5e-9), ~1e-12 spurious-branch probability per active
  measurement × O(10–100) measurements × 1e11 shots is the **same order as the signal**. Needs a
  precision-scaled epsilon (≈1e-12 relative for fp32) or fp64 mode (cheap on MI300X, see §5).
- **R2 — no normalization safeguard.** The CPU clamps gamma drift via `scale_magnitude()`
  (renormalizes `v` when |gamma| leaves [1e-100, 1e100]). The GPU tracks nothing: `EXPAND`
  doubles the array norm, measurement folds shrink it by the branch probability, and the running
  norm random-walks. In fp32 the representable range is ~1e±38; the d5/d7 circuits stay inside
  it, but deeper circuits (more T events / active measurements per shot) can underflow amplitudes
  to denormals/zero and silently corrupt branch statistics. Needs a per-shot running scale (the
  measurement reductions already compute `total`, so renormalizing at folds is nearly free).
- **R3 — verification is thinner than the README claims.** "Matches within 5σ across all three
  kernel regimes" rests on three circuits and three scalar rates each. The global-block (d7) row
  compares logical-error rates of 0.5002 (CPU) vs 0.5124 (HIP) — an observable sitting at 0.5 is
  maximally *uninformative* (a completely scrambled frame also yields 0.5), and the delta only
  fits inside a 5σ band because the surviving-shot count is small. The per-thread row is a
  pure-Clifford circuit with rate ≈ 0.4996 — same problem. Only the d5 row (survival 0.1439 vs
  0.1435) actually constrains the implementation. See §4 for what adequate testing looks like.

**Compatibility (blocks use with current clifft):**

- **C1 — built against a ~2026-05-01 snapshot; host flattening won't compile on main.**
  `flatten_mask`/`flatten_channel_mask` use `clifft::kMaxInlineWords`, inline `PauliMask.x.w[]`,
  and `NoiseChannel.destab_mask/stab_mask`. PRs #49/#52/#53 replaced all of this with
  runtime-width Pauli arenas (`MaskView`, `noise_channel_masks` + handles) and removed the
  128-qubit inline ceiling entirely. The fix is localized (a few dozen lines in the flattening
  layer) but mandatory.

**Quality (non-blocking but worth requiring):**

- **Q1 — 2×1960-line duplicated kernel.** CUDA and HIP sources are identical modulo API names.
  Fine for a proof-of-port; a maintenance hazard the moment either side changes. Should be one
  source compiled twice through a ~30-line runtime shim (the 124-line mechanical diff *proves*
  this works). §5 assumes this.
- **Q2 — Python layer is vestigial.** `backends.py` (392 lines) is mostly environment probing;
  the GPU samplers raise unconditionally. Harmless, but the package name promises more than it
  delivers; tests only cover the diagnostics dataclasses.
- **Q3 — per-thread kernel state pressure.** `ShotState` carries `uint8_t meas[1024]` +
  amplitudes per thread → ~1 KB of thread-local memory traffic. Works, but is a perf cliff;
  sizing `meas` to `total_meas_slots` (or bit-packing) would help.
- **Q4 — no kernel for "several shots per block".** The k = 5–10 regime runs one shot per block
  (256 threads cooperating on ≤1024 amplitudes — most threads idle in the array loops, and a
  30 KB LDS footprint of which 16 KB is a fixed 1024-entry reduction scratch). MI300X has 64 KB
  LDS per CU; 4–6 concurrent fp32 shots per block (one wavefront each) is the obvious win for
  exactly the k ≤ 10 workloads we care about. Its absence is the main reason to doubt the
  current numbers say anything about the achievable ceiling.
- **Q5 — positives worth keeping:** every runtime call goes through `check_hip` (including
  `hipGetLastError` after launches); buffers freed on the exception path; limits validated
  before launch; `compare_oracle.py` reuses clifft's own `binomial_tolerance` conftest
  primitive byte-for-byte and includes a determinism gate (same seed → byte-identical JSON,
  timing fields excluded). The engineering hygiene is genuinely decent.

---

## 2. How it differs from current clifft; what is and isn't supported

Two separate axes: the **snapshot drift** (their vendored clifft vs our main) and the **GPU VM's
feature subset** (what their kernels execute vs what our SVM executes).

### 2.1 Snapshot drift

The vendored tree matches clifft main as of ~2026-05-01 (after PR #25, before #52) — **57
commits behind**. Material changes since, none of which the fork has:

- **Runtime-width Pauli arenas** (#49/#52/#53): removed the compile-time qubit ceiling
  (`CLIFFT_MAX_QUBITS=128` is now a fork-only restriction; main supports up to 65,536 axes) and
  reshaped the `ConstantPool` APIs the fork's flattening code consumes (→ finding C1).
- **Forced-outcome measurement opcodes + kernels** (#76–#79): the replay/conditioning machinery
  behind importance sampling and `probability_of`.
- **`probability_of` / exact basis & record probabilities APIs** (#60, #73, #80, #82) with the
  gray-code strong-simulation speedup.
- AVX dispatch hardening (#94), stim symbol hiding (#131), lockfile/CI/docs work.

### 2.2 GPU VM feature subset

Supported (33 opcodes): all frame ops, all array Clifford ops (incl. fused multi-CNOT/CZ),
T/T†, EXPAND/EXPAND_T/EXPAND_T†, all five *sampling* measurement variants, conditional Pauli,
noise/noise-block/readout-noise, detector/postselect/observable.

**Not supported** (host-side validation rejects the program):

| Missing | Consequence |
|---|---|
| `OP_ARRAY_ROT`, `OP_ARRAY_U2`, `OP_ARRAY_U4`, `OP_EXPAND_ROT` | Clifford+T circuits only — no arbitrary-angle rotations, no coherent-noise circuits, none of the fused-unitary CISC path |
| `OP_EXP_VAL` | no expectation values (would also require gamma, which the GPU dropped) |
| the five `*_FORCED` measurement opcodes | **no importance sampling, no `probability_of`, no record replay** |
| `keep_records` | aggregate counts only — no per-shot detector/measurement records, hence **no decoder workflows** (sinter/PyMatching-style sampling is impossible) |
| limits | peak_rank ≤ 19, ≤ 128 qubits, ≤ 1024 measurement slots, ≤ 8 observables; postselection is all-detectors-or-none at the CLI |
| Python bindings | GPU is CLI-only; clifft's `import clifft; sample_survivors(...)` surface doesn't reach it |

The importance-sampling gap deserves emphasis: their flagship result (3.54e-9 after-postselection
error rate from 10^11 brute-force shots in 24 h) is the kind of rare-event estimate clifft's
forced-fault importance sampling targets with orders of magnitude fewer shots on a CPU. Brute
GPU throughput and smarter CPU sampling should be benchmarked against each other before
investing in GPU throughput for its own sake.

---

## 3. How the AMD implementation differs from CUDA

Smaller question than it sounds: **one commit, and the kernel logic is unchanged.**

- `src/hip_sampler.hip` vs `src/cuda_sampler.cu`: 124 differing lines, every one a runtime API
  rename (`cuda*` → `hip*`, `<cuda_runtime.h>` → `<hip/hip_runtime.h>`), an error-string change
  ("clifft-cuda" → "clifft-amd"), or the diagnostics printing `gcnArchName`. No kernel, RNG,
  reduction, atomic, indexing, or dispatch changes. I verified this by diff, not by trusting the
  README (which says the same thing — accurately).
- **Wavefront width:** CUDA assumes 32-lane warps, MI300X is 64-lane. The README's claim that
  this needs no changes is correct *for this code*: there are no shuffles/ballots/lane masks,
  and every reduction is a full-block `__syncthreads` tree, which is width-agnostic. Any future
  warp-level optimization breaks this property and must be parameterized on `warpSize`.
- **Build:** new default-OFF `CLIFFT_AMD_ENABLE_HIP` option and a parallel `clifft_hip` target
  (CUDA target untouched; either/both/neither configure cleanly). Two genuinely useful AMD
  lessons encoded in CMake: link the host runner against `hip::host` (linking `hip::device`
  leaks `--offload-arch` flags into host compiles), and build the device library with
  `-ffp-contract=off` so fp32 amplitude math rounds reproducibly.
- **Floating point:** note the asymmetry — HIP gets `-ffp-contract=off`, the CUDA side keeps
  nvcc's default FMA contraction. So CUDA and HIP runs are *each* deterministic for a fixed seed
  but are **not bit-identical to each other**; cross-vendor comparison must stay statistical (or
  both sides must pin contraction).
- **Fork-only additions** that the CUDA repo lacks (all on the verification side):
  `scripts/compare_oracle.py` (5σ cross-binomial harness + determinism gate), ROCm diagnostics
  in `backends.py`, a `clifford_peakrank1.stim` test circuit, `--cpu-reference` documented as
  the oracle flow.
- **Performance:** ~1.2 M shots/s (MI300X, 750 W) vs ~1.18 M shots/s (RTX PRO 5000, 300 W) on
  d5/p=0.001 — the port achieves parity but exploits nothing MI300X-specific: its strong fp64
  (unused; everything is fp32), its 64 KB LDS (underused in the one-shot-per-block regime), or
  its CU count (the k=5–10 kernel's occupancy is LDS-bound). Their README's CPU baseline
  (Xeon Gold 5218R, 59.8 k shots/s) is a *sequential* sampler process; shot sampling is
  embarrassingly parallel, so the honest CPU comparison is per-socket (×20 cores ≈ 1.2 M
  shots/s on that old Xeon — i.e., the MI300X currently ties one 2019 CPU socket at 6× the
  power). The "~20×" headline should not survive into any document of ours.

---

## 4. Testing for correctness without a local GPU

The core problem: aggregate 5σ statistics on three circuits is a weak oracle (finding R3), and
we have no GPU on this VM. The plan below separates *logic verification* (no GPU needed, runs in
CI on every PR) from *device verification* (needs hardware, runs nightly/on-demand), and is
designed so confidence is *retained* as the implementation iterates.

### Tier 0 — every PR, no GPU required

1. **Compile-only jobs.** Build the flattening layer + kernels with `hipcc` (ROCm container)
   and `nvcc` (CUDA container), no device present. Catches exactly the class of breakage in
   finding C1 (host API drift) and all kernel-language errors. We already run containerized
   cross-toolchain CI for wasm/emsdk; this is the same pattern.
2. **Run the kernel logic on the CPU.** The device functions are scalar C++ over POD state —
   nothing intrinsically GPU about them. Compile the same kernel source as plain C++ with a
   ~50-line shim (`threadIdx.x = 0`, `blockDim.x = 1`, `__syncthreads()` = no-op, `atomicAdd` =
   plain add) and execute shots single-threaded. (The AMD-maintained header-only **HIP-CPU**
   library is an off-the-shelf version of this shim if we'd rather not write it.) Now the *GPU
   semantics* are testable with gtest/pytest on any machine: per-opcode unit tests, golden
   measurement records, the works.
3. **Differential testing against the CPU SVM, bit-exact.** Make the per-shot RNG substream
   scheme part of the backend contract and add it to the CPU SVM as an opt-in mode (a few lines:
   reseed per shot from `(seed, shot_id)` exactly as the GPU does). Then run the Tier-0 CPU-shim
   kernel in **fp64** against the real SVM on the same program: every integer output
   (measurement record, detector, observable, discard flag) must match **per shot, exactly** —
   FP only enters through branch *decisions*, and at fp64 with identical RNG draws those agree
   except on knife-edge ties. This is the test that actually pins the VM semantics; the
   statistical tier then only needs to certify "device execution ≡ shim execution".
4. **Flattening unit tests.** `CompiledModule → DeviceProgram` is pure host code: golden-compare
   flattened programs for a fixture set of circuits, so compiler-side changes that should (or
   shouldn't) alter the device contract are caught in review.
5. **Negative controls (harness calibration).** A tiny mutation suite: flip a sign in
   `frame_s`, drop the `p_x` conjugation in `array_t`, off-by-one a scatter bit — each mutant
   must fail Tier 0. Without this we don't know the tests have teeth (their 5σ harness was never
   shown to *fail* on a broken kernel).

### Tier 1 — nightly / on-demand, needs hardware

Hardware path: AMD Developer Cloud MI300X instances (AMD has offered access in these
engagements — make it a concrete ask), or a partner-hosted self-hosted runner with a
`gpu-amd` label, triggered by `workflow_dispatch` + nightly cron. The CUDA twin runs on any
NVIDIA cloud box.

1. **Determinism gate** (theirs, kept): same seed twice → byte-identical JSON; plus invariance
   checks determinism alone misses: results identical across `--block-size` values and across
   batch splits (N shots in one run ≡ merged counts of two N/2 runs with the right offsets), and
   `--postselection none` ⇒ `passed == shots`.
2. **Shim-vs-device equivalence:** the Tier-0 CPU shim and the real device, same seed, fp64:
   aggregate counts must match exactly (integer outputs, order-independent reductions). This
   replaces "GPU vs CPU-SVM statistics" as the primary device test — it's exact, not 5σ.
3. **Statistical suite vs the CPU SVM (fp32 mode):** keep `compare_oracle.py`, but fix its
   power problems: (a) circuits chosen so the observables are *informative* — analytic
   micro-circuits with closed-form rates (Bell/GHZ states, repetition codes, single-T magic
   injection with known acceptance probability — the same ladder we sketched for noncomp
   validation), plus the d5 MSC circuit; (b) compare **per-detector fire rates** and pairwise
   detector correlations, not three scalars — the GPU already counts observables, adding a
   per-detector ones-counter is one atomic per detector; (c) state the shot count needed for
   target resolution up front instead of letting the band absorb whatever delta appears.
4. **CUDA-vs-HIP cross-check:** same seed, statistical agreement on the full metric vector
   (exact agreement isn't expected while FP contraction differs; see §3).

### Tier 2 — release / claims

- Rare-event cross-validation: the GPU's brute-force postselected error rate vs clifft CPU
  importance sampling on the same circuit — validates both the GPU at depth and the
  forced-fault machinery, and answers the "is GPU throughput even the right tool" question with
  data.
- Matched-hardware throughput benchmark (GPU vs all-core CPU socket) before any performance
  claim goes in a README of ours.

---

## 5. Design for AMD (and CUDA) backend support in clifft

Agreed on the framing: **the GPU backend is an alternate executor of compiled programs — "the
SVM on a device" — and the AOT compiler stays on the CPU.** Compilation is milliseconds (~6.4 ms
for surface_d7_r7 after the #67 work) and amortizes over millions of shots; there is no reason
to move any compiler stage to the GPU.

### 5.1 Architecture

```
                        clifft core (CPU, unchanged)
 stim → parse → HIR → passes → lower → bytecode passes → CompiledModule
                                                             │
                                              flatten_for_device()        ← new, host-only,
                                                             │               vendor-neutral
                                                       DeviceProgram          (POD arrays)
                                                             │
                  ┌──────────────────────────┬───────────────┴──────────────┐
            SVM (cpu)                 gpu kernel × CUDA               gpu kernel × HIP
      (existing, oracle)            (single source, shim)          (same source, shim)
```

1. **`DeviceProgram` + `flatten_for_device()`** in a new `src/clifft/gpu/` (or `device/`)
   directory: essentially the fork's `GpuProgram` cleaned up — instruction array, Pauli-mask
   words, noise sites/channels/hazards, readout entries, detector/observable CSR lists, limits,
   and a header (version, peak_rank, slot counts, precision). Host-only C++, **always compiled
   and unit-tested** regardless of GPU options, and kept out of the `clifft_core` source list
   (also avoids the wasm dual-source-list trap entirely).
2. **One kernel source, two vendors.** The fork's 124-line mechanical diff is the proof: a
   ~30-line `gpu_runtime.h` shim (`GPU_CHECK`, `gpuMalloc`, … mapped to `cuda*`/`hip*` at
   compile time) lets a single `gpu_sampler.inl` build as both targets. I'd pick the shim over
   the alternatives: HIP-as-single-source (hipcc can target NVIDIA, but it drags ROCm into the
   NVIDIA build) and Orochi-style runtime dispatch (single binary, but a heavier dependency than
   we need). Same shim also compiles the source as plain C++ for the Tier-0 CPU tests (§4).
3. **Backend selection at the existing API seam.** `sample_survivors(program, shots, seed,
   keep_records, backend=…)` in C++ and Python, mirroring how the SVM already dispatches
   scalar/AVX2/AVX-512. CMake: `CLIFFT_ENABLE_CUDA` / `CLIFFT_ENABLE_HIP`, default OFF; when
   off, the backend registers a stub that raises with the fork's (good) diagnostics text.
4. **Precision is a parameter, not a constant.** Template the kernel on the amplitude type.
   **fp64 is the default** — it's the validation anchor (bit-exact differential testing, §4) and
   MI300X's 1:2 fp64 rate makes it cheap there, unlike consumer NVIDIA parts. fp32 is the
   opt-in fast mode with a precision-scaled dust epsilon (R1) and a per-shot norm guard (R2),
   certified statistically against the fp64 mode rather than against the CPU.
5. **The RNG contract is part of the spec:** per-shot xoshiro256++ substreams seeded
   `splitmix64(seed ⊕ φ·(shot_id+1))`, implemented identically in the CPU SVM (opt-in), the CPU
   shim, and both device backends. This is what turns cross-backend testing from statistics into
   equality.
6. **Scope ladder** (each step shippable):
   - **P0:** aggregate survivor sampling, Clifford+T, k ≤ ~20 — the fork's scope, upstreamed.
   - **P1:** per-shot records (`keep_records`): bit-packed measurement/detector words streamed
     to global memory with discard compaction. This unlocks decoder workflows, which is where
     GPU throughput would actually matter to users.
   - **P2:** a multi-shot-per-block kernel for k = 5–10 (one wavefront per shot, 4–6 shots per
     CU in LDS) — the experiment that decides whether GPU ever beats a CPU socket on *our*
     workloads.
   - **P3 (maybe never):** forced ops / importance sampling on device; `EXP_VAL` (requires
     reintroducing gamma and fp64). Don't promise these.

### 5.2 Should the SVM bytecode get a serializable format?

Split the question in two:

**Don't serialize `CompiledModule`/the bytecode itself (yet).** It's a young IR that we
actively churn — the #52 arena migration alone would have been a breaking format rev, and the
noncomp branch is adding opcodes. Freezing it taxes every optimizer and constant-pool change
with format/versioning work, for no consumer that needs it: clifft core is plain CPU C++ that
builds on any GPU host, so the compiler can always run in-process next to the executor (exactly
what the fork does, and it works).

**Do version-and-dump `DeviceProgram`.** A flat POD blob (header + arrays) is nearly free to
serialize, and it has three real consumers:

1. **Golden test fixtures** — frozen flattened programs make backend tests independent of
   compiler drift (and compiler tests independent of backends).
2. **Compile-once / run-anywhere farm workflows** — compile on a workstation, fan the blob out
   to GPU nodes that need zero clifft build (`run_device --program d5.cdp --shots 1e9`).
3. **The partner interface.** This is the answer to "how do we support AMD's work without
   merging it tomorrow": publish the `DeviceProgram` schema + a conformance suite
   (fixtures + the §4 statistical harness + the RNG contract), and AMD/NVIDIA backends can be
   developed and certified against a frozen contract instead of vendoring a clifft snapshot —
   precisely the failure mode that left clifft-amd 57 commits behind with a broken host API.

Mark the format explicitly experimental (`version` byte in the header, no stability promise
before clifft 1.0). It's deliberately *less* expressive than the bytecode — sampling-only, no
forced ops, no exp-val masks — which is what makes it cheap to keep stable.

### 5.3 Recommended next steps

1. Reply to AMD: the port is faithful and well-checked for what it is; flag R1/R2/C1 and the
   verification gaps (R3) as the things we'd want addressed; share the §4 test-tier sketch and
   ask about MI300X access for a nightly runner.
2. Land the vendor-neutral pieces in clifft main first — `DeviceProgram` + flattening + CPU
   shim + per-shot-RNG mode in the SVM + Tier-0 tests. All of it is testable without any GPU and
   immediately useful to both their fork and any future in-tree backend.
3. Keep clifft-amd a fork until P2 data exists: if multi-shot-per-block on MI300X can't beat a
   modern CPU socket on k ≤ 10 workloads, the in-tree backend ships as "supported, for
   throughput at k ≳ 11" and we say so honestly.
