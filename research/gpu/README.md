# GPU acceleration of clifft's dense active block — scoping research

**Date:** 2026-06-19
**Branch:** `research/gpu-cudaq-custatevec`
**Question:** Can NVIDIA cuStateVec / CUDA-Q accelerate clifft's dense active block `|φ⟩_A`, and what is the most promising GPU approach overall?

Status: **scoping only** — no code written yet. This is a go/no-go and direction-setting document.

---

## TL;DR recommendation

GPU acceleration is **technically feasible and a representation match** (the active block is a genuine dense 2^k complex128 statevector, exactly what cuStateVec operates on), but the **single-large-statevector route is the wrong target** for clifft's profile. The order-of-magnitude win, if it exists, lives in **batching across Monte-Carlo shots** — many small (k<20) statevectors run in parallel on the GPU.

**Recommended path:** prototype a **batched-across-shots** backend, and architect it as a **hybrid** — cuStateVec batched primitives for standard gates/measurement, custom CUDA kernels (via cuStateVec **Ex** interop, which exposes raw device pointers) for clifft's non-standard ops (`OP_EXPAND`, `OP_EXPAND_T`, X-basis `INTERFERE`). Custom-CUDA-first is also defensible since the non-standard ops have no library primitive anyway.

**But:** every relevant benchmark in the literature is an optimistic upper bound for clifft (dense circuits, FP32 or unstated precision, stock-simulator CPU baselines — not clifft's hand-tuned AVX-512+OpenMP complex128 kernels). The honest single-statevector k=22–30 win against clifft's *actual* CPU baseline is **unquantified and likely low single-digit, not order-of-magnitude.** A small benchmark must be run before committing engineering effort.

---

## Why the single-statevector route is weak for clifft

Three structural mismatches between clifft and the GPU-statevector literature:

1. **Narrow win band.** clifft's sweet spot is small k (single-threaded below k=18). GPUs only beat a tuned CPU at large k. With the ~k=30 memory wall (16 GB), the GPU-favorable single-state slice is just k≈22–30 — and the directly relevant crossover benchmarks against an *optimized* CPU baseline were **refuted in verification** (see Refuted claims). Surviving evidence puts single-node k~30 GPU wins at low single-digit multiples (e.g. ~2.7–3.6x for 30-qubit QAOA/VQE), not 50–100x.

2. **Monte-Carlo, not one big state.** clifft runs many shots with data-dependent mid-circuit branching decided on the CPU. This is the profile that batched-shots GPU execution is *built* for (independently validated by Qiskit Aer's `batched_shots_gpu`, which targets ≤16 qubits per shot by default — squarely clifft's k<20 active block).

3. **complex128 + non-standard ops.** FP64 is supported by cuStateVec (`CUDA_C_64F`, FP64 compute by default) but **throttled** on hardware: consumer GPUs run FP64 at ~1/32–1/64 of FP32; even datacenter A100/H100 are ~1/2. And `OP_EXPAND`/`OP_EXPAND_T`/`INTERFERE` have no native primitive.

---

## Findings by sub-question

### 1. cuStateVec API model — does it fit?
- **Standard gates: yes, directly.** `custatevecApplyMatrix()` applies arbitrary user unitaries; CNOT/CZ via the `controls` parameter; dense / diagonal / generalized-permutation / Pauli-matrix-exponential forms all documented. *(high confidence)*
- **Batched primitives: yes, a full family.** `custatevecApplyMatrixBatched`, `MeasureBatched`, `abs2sum...Batched`, `collapseByBitStringBatched`, `computeExpectationBatched`. Statevectors live in one device buffer addressed by `nSVs` + `svStride`; `mapType` (BROADCAST vs INDEXED) allows shared or per-SV matrices. Docs explicitly recommend this "when computing with many small state vectors." *(high confidence)*
- **Non-standard ops: no native primitive.** cuStateVec operates on a fixed-dimension statevector — there is no resize/grow API (`OP_EXPAND`) and no user-defined element-wise kernel API (X-basis fold). These **must drop to raw CUDA.** *(high confidence)*
- **Escape hatch: cuStateVec Ex interop.** `custatevecExGetResourcesFromDeviceSubSV/...View` expose underlying device pointers, CUDA streams, and handles, so custom kernels run on the resident state vector alongside library calls (NVIDIA ships a cuBLAS-on-resident-SV example). **This is what makes a hybrid design viable.** *(high confidence)*

### 2. Single-statevector crossover (complex128)
- General band: GPU advantageous ~16–24 qubits, recommended 24–32; below 16, CPU (e.g. Qulacs CPU) wins. clifft's k=22–30 sits in/above this band. *(medium-high; arXiv:2412.20518)*
- FP64 wins are real but **only large**: ~13–14x at the max tractable ~31–34 qubits for QV/QFT — **and that figure used two A100s**; single-GPU Thrust was only ~4x at 20 qubits, with large fixed cuQuantum allocation/pinning overhead dominating 10–22 qubit runs. *(high; arXiv:2307.14860)* — directly concerning for k=22–30.
- **Caveat that undercuts all of these:** baselines are stock simulators, not clifft's hand-tuned AVX-512 kernels; precision often FP32 or unstated; circuits are dense (not butterfly/permutation/element-wise).

### 3. PCIe transfer & control flow
- Keep the active block **resident on the GPU**: single-GPU host↔device transfer is negligible in profiling; transfer only dominates (>90% of GPU time) in the multi-GPU regime. *(medium; arXiv:2307.14860, FP32/Aer)*
- **Unaddressed gap:** no benchmark models clifft's per-shot, data-dependent, mid-circuit-measurement control flow where the CPU decides branches. CUDA graphs / on-device control logic are the textbook mitigations but are **unbenchmarked for this profile.**

### 4. Batching across shots — the most promising route
- cuStateVec batched APIs (see #1) support it natively. **Independently validated:** Qiskit Aer's `batched_shots_gpu` "can greatly accelerate" statevector sims **with intermediate (mid-circuit) measurements** — exactly clifft's case — and defaults to ≤16 qubits/shot (`batched_shots_gpu_max_qubits=16`). *(high confidence)*
- **Hard structural constraint:** all statevectors in a cuStateVec batch must share the same qubit count. clifft's `OP_EXPAND` changes k mid-shot and k varies per shot ⇒ **shots must be grouped/re-bucketed by current k.** This is the central engineering problem of the batched design.
- Memory: K-way batching needs ~K × one statevector. For k<20 that's cheap; this is why batching pairs naturally with clifft's *small* active block.

### 5. CUDA-Q (cudaq)
- **Not resolved by surviving evidence.** No verified claim established whether CUDA-Q is a usable C++ integration layer vs. an end-user circuit DSL, nor its complex128/FP64 story. **Open question — do not assume it fits.** (Working prior from the docs sweep: CUDA-Q is the higher-level programming model / kernel DSL; cuStateVec is the lower-level library you'd actually integrate against. Verify before relying on this.)

### 6. Multi-GPU / breaking the k>30 wall
- Multi-GPU cuStateVec can exceed the 16 GB single-node wall with reported 50–90x (vs 2× 64-core EPYC) and 4.5–7x on 8 GPUs vs 1 GPU — but this is a **vendor marketing blog**, cherry-picked algorithms, **precision not stated** (a corroborating appliance benchmark used complex64), needs NVLink-class bandwidth (~600 GB/s). For complex128 these are optimistic. *(medium confidence)*
- Relevance to clifft is secondary: per the stabilizer-rank research, k>30 circuits are mostly refused/OOM today; breaking the wall is a different project than speeding up the existing regime.

### 7. Alternatives to cuStateVec
- **Custom CUDA / Thrust is a serious contender.** Thrust and cuQuantum were found "comparable" at the ~13–14x FP64 max. clifft's ops are exactly the simple element-wise/permutation kernels hand-written CUDA handles well — **and the non-standard ops have no cuStateVec primitive regardless** — strengthening the case for custom-CUDA-first (or custom + Ex-interop hybrid) over a pure cuStateVec dependency. *(medium confidence)*
- Other GPU simulators (qsim-GPU, Aer-GPU, QuEST, Lightning-GPU) exist but the surviving benchmarks don't isolate FP64 crossovers cleanly for clifft's op mix.

### 8. Engineering cost & maturity
- complex128 is supported across the stack. NVIDIA-only; CUDA-version pinning; actively versioned API (cited 24.11 → 26.03.x); Ex interop & batched-measurement APIs are relatively recent.
- **clifft currently has zero BLAS/GPU dependencies** — adding CUDA is a real maintenance/build-integration burden (CMake/C++20 + CUDA toolkit, NVIDIA-only CI). Must be justified by a measured win.

---

## Refuted claims (killed in 3-vote verification — do NOT cite these)

These attractive-sounding numbers failed verification; flagged so we don't accidentally rely on them:
- ❌ "GPU beats CPU at n=16 (5.7x), >64x at n≥20 on A100." (0-3)
- ❌ "20–28 qubits: 64–146x speedup over NumPy, peak 146x at 22q." (0-3) — NumPy is not a real baseline anyway.
- ❌ "complex64 gives an extra 1.7–1.9x over complex128." (0-3)
- ❌ "Distributed: comms ~99%, GPU benefit negligible." (0-3)
- ❌ "32-qubit GHZ: cuStateVec 5.9x over qsim 64-core." (1-2)
- ❌ "Single dense statevector GPU crossover at 14–18 qubits." (1-2)

Net effect: **concrete single-statevector crossover numbers and the exact FP64 penalty for clifft's op mix are NOT established by surviving evidence.**

---

## Open questions (must answer before/while prototyping)

1. **Real FP64 penalty for clifft's kernels** (butterfly/permutation/element-wise) on a target GPU (consumer RTX vs A100/H100) — does it erase the batched-shots win?
2. **Does batching survive clifft's control flow?** Can shots be bucketed by k and re-batched as `OP_EXPAND` changes k, and can branch logic move on-device (CUDA graphs) without host round-trips?
3. **CUDA-Q**: integration layer or end-user DSL? FP64 support? (unresolved)
4. **Honest k=22–30 single-state speedup vs clifft's actual AVX-512+OpenMP kernels** (not stock simulators).
5. **End-to-end engineering/build cost** vs. the measured gain, given zero current GPU deps.

---

## Next step (cheap, decisive) — BUILT, see `microbench/`

A **standalone microbenchmark** implementing clifft's actual kernels (H butterfly,
CZ/CNOT permutation, EXPAND/EXPAND_T phase, diagonal + INTERFERE measurement) as
(a) faithful portable CPU kernels (`-O3 -march=native`, NEON on ARM/Grace) and
(b) hand-written CUDA in **complex128**, single-statevector and batched-across-shots.
See [`microbench/`](microbench/) — built and CPU-validated locally; the CUDA half
builds on the GH200 with `-DCUDA_ARCH=90`. No cuStateVec dependency (that is the
follow-on if these numbers justify it; the non-standard ops have no native
cuStateVec primitive regardless).

It measures, answering open questions 1, 2, 4:
- single-statevector crossover k for clifft's op mix, FP64, vs a faithful CPU baseline;
- batched-across-shots throughput at k=12,16,18,20 for B = 256…65536;
- host↔device transfer bandwidth (to quantify NVLink-C2C vs PCIe).

**Target hardware: GH200 — near-ideal for clifft.** Full-rate FP64 (Hopper),
and the 900 GB/s cache-coherent NVLink-C2C link largely dissolves the
PCIe-transfer concern that dominates the skeptical literature (sub-question 3).
The Grace CPU is ARM, so the relevant CPU baseline is clifft's NEON path (which
the local Mac approximates), not AVX-512 — the report's "vs AVX-512" framing is
moot on this box.

Run on the GH200: `./microbench/run.sh 10 30 12,16,18,20 4` → `cpu.csv` + `gpu.csv`,
then summarize the crossover and batched-win numbers back into this file.

---

## Sources

Primary (NVIDIA docs):
- cuStateVec API functions — https://docs.nvidia.com/cuda/cuquantum/24.11.0/custatevec/api/functions.html
- cuStateVec overview — https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/custatevec/overview.html
- cuStateVec Ex (interop) overview — https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/custatevecex/overview.html
- NVIDIA cuStateVec acceleration blog (multi-GPU figures; vendor, cherry-picked) — https://developer.nvidia.com/blog/accelerating-quantum-circuit-simulation-with-nvidia-custatevec/
- Qiskit Aer `batched_shots_gpu` — https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html

Academic benchmarks (note: none use clifft's profile):
- van Niekerk et al. 2024, QV crossover bands — https://arxiv.org/pdf/2412.20518
- Faj et al., IEEE eScience 2023, Aer Thrust+cuQuantum FP64 on A100 — https://arxiv.org/pdf/2307.14860

Full raw findings (verified-claim list with vote tallies, refuted claims, caveats) archived in `findings-raw.json`.
