# UCC SVM Architecture Refactor & SIMD Runtime Dispatch Plan

## Status: Complete (Steps 1-6 done)

The UCC Virtual Machine currently suffers from a monolithic `svm.cc` file (~1900 lines) that interleaves OS-level memory allocation, PRNG state, bit-weaving mathematics, and the execution dispatch loop. Furthermore, profiling reveals two distinct bottlenecks depending on circuit scale:
1. **QV-20 (Rank 20):** Bound by memory contiguity and 3-cycle `pdep` latencies that defeat auto-vectorization on simple 2-qubit gates.
2. **Cultivation (Rank 10):** Bound by L1 cache branch mispredictions (`if (parity)`) in heavily-used multi-gates.

**Goal:** Break apart `svm.cc` to enable Translation Unit (TU) splitting, introduce a fat-wheel runtime CPU dispatcher for Linux/macOS, and safely inject targeted AVX2/AVX-512 and branchless optimizations. Windows (MSVC) and Wasm will safely use the generic scalar fallback.

## Machine Specifications

- Intel Xeon Platinum 8259CL @ 2.50GHz (AWS Cascade Lake), 2 cores, 1 thread/core
- Supports: SSE4.2, AVX2, BMI2, FMA3, **AVX-512** (AVX512F, AVX512DQ)
- L1d: 32KB, L2: 1MB, L3: 35.8MB
- OS: Linux (Ubuntu) for SIMD, Windows for Scalar fallback
- Compiler: GCC/Clang with C++20, `-O2 -march=native -ffast-math`

## Benchmark Circuits

| Circuit | Qubits | peak_rank | Bytecode | Baseline/shot | Bottleneck |
|---------|--------|-----------|----------|---------------|------------|
| QV-10 | 10 | 10 | 814 | ~0.79ms | Base overhead |
| Cultivation d=5 | 42 | 10 | 1518 | ~0.089ms | Branch mispredictions in `multi_cnot` |
| QV-20 (ref) | 20 | 20 | 3089 | ~3.40s | Instruction latency & scalar memory limits |

---

## Instructions for the LLM Agent

1. **Implement strictly ONE step at a time.**
2. After completing a step, run `pytest tests/` and the C++ test suite (`ctest --test-dir build --output-on-failure`) to verify correctness. Run the benchmark harness on all three circuits (QV-10, QV-20, Cultivation d=5) and report performance results.
3. Stop and ask the user to review the results. **Do not proceed** to the next step until the user explicitly confirms.

---

## Step 1: Architectural Clean-Up (De-monolith `svm.cc`)

Separate concerns into manageable files without changing *any* mathematical logic or loop structures. Performance must remain exactly identical to the baseline.

- **Tasks:**
  1. **Keep `SchrodingerState` in `svm.h`:** The struct declaration is part of the public API and stays in `svm.h`. Move the *method implementations* (constructor, destructor, `alloc_state`, `dealloc_state`, etc.) to a new `svm_state.cc`.
  2. **Extract Internal Helpers (`svm_internal.h`):** Move `aligned_alloc_portable`, `aligned_free_portable`, `Xoshiro256PlusPlus`, `kDustEpsilon`, and `sample_branch` here. These are private implementation details used by the hot loop and measurement code.
  3. **Extract Math Helpers (`svm_math.h`):** Move `bit_get`, `bit_set`, `bit_xor`, `bit_swap`, `scatter_bits_1`, `scatter_bits_2`, `insert_zero_bit`, and the `UCC_HAS_PDEP` macros here. **Do not** put `cmul_m256d` here (it uses AVX2 intrinsics and would fail to compile in the scalar TU); move it into `svm_kernels.inl` instead, guarded by `#if defined(__AVX2__)`.
  4. **Extract Kernels (`svm_kernels.inl`):** Move **all** `exec_*` inline functions (including `cmul_m256d`), frame helpers, and the `execute()` dispatch loop here. Rename the internal dispatch loop to `execute_internal()`.
  5. **Clean `svm.cc`:** Should now only contain `sample()`, `sample_survivors()`, `get_statevector()`, and a wrapper `execute()` that currently just calls `execute_internal()`. (Temporarily include `"svm_kernels.inl"` here).
  6. **Update CMake:** Add `svm_state.cc` to the `ucc_core` target.
- **Files:** `src/ucc/svm/svm.cc`, `src/ucc/svm/svm_state.cc`, `src/ucc/svm/svm_internal.h`, `src/ucc/svm/svm_math.h`, `src/ucc/svm/svm_kernels.inl`, `src/ucc/CMakeLists.txt`
- **Difficulty:** Low (pure refactoring)
- **Impact:** Zero performance impact. Improves maintainability.
- **Validation:** All tests must pass identically. Benchmark all three circuits and confirm no regression.

---

## Step 2: Translation Unit Splitting & CPUID Dispatcher

Set up the fat-wheel architecture. We will compile the exact same kernel logic multiple times with different compiler flags. Windows/MSVC will strictly use the generic scalar fallback.

- **Tasks:**
  1. **Namespace Wrapping:** Wrap the `exec_*` functions and `execute_internal()` in `svm_kernels.inl` inside `namespace ucc { namespace UCC_SIMD_NAMESPACE { ... } }`. Generic math helpers (`bit_get`, `scatter_bits`, etc.) in `svm_math.h` remain in the outer `ucc` namespace and are **not** duplicated per-ISA.
  2. **Create Translation Units:**
     - `svm_scalar.cc`: `#define UCC_SIMD_NAMESPACE scalar`, `#include "svm_kernels.inl"`.
     - `svm_avx2.cc`: `#define UCC_SIMD_NAMESPACE avx2`, `#include "svm_kernels.inl"`.
  3. **Update CMake (`src/ucc/CMakeLists.txt`):** Add `svm_scalar.cc` and `svm_avx2.cc` to `ucc_core`. Add a guard: `if((CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)") AND NOT MSVC)`. Inside this guard, use `set_source_files_properties(svm/svm_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mbmi2;-mfma")` and define `UCC_ENABLE_RUNTIME_DISPATCH`.
  4. **Implement CPUID Dispatcher (`svm.cc`):**
     - Declare signatures for `ucc::scalar::execute_internal` and `ucc::avx2::execute_internal`.
     - Write `resolve_dispatcher()` guarded by both `#if defined(__GNUC__) || defined(__clang__)` **and** `#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)`. Use `__builtin_cpu_supports("avx2")` and `__builtin_cpu_supports("bmi2")` (only active if `UCC_ENABLE_RUNTIME_DISPATCH` is defined).
     - Include an environment override: `if (const char* env = std::getenv("UCC_FORCE_ISA")) { ... }` routing to `"scalar"` or `"avx2"`.
     - Update the public `ucc::execute()` to invoke the static function pointer returned by `resolve_dispatcher()`.
- **Files:** `src/ucc/CMakeLists.txt`, `src/ucc/svm/svm_scalar.cc`, `src/ucc/svm/svm_avx2.cc`, `src/ucc/svm/svm.cc`, `src/ucc/svm/svm_kernels.inl`
- **Difficulty:** Medium
- **Impact:** Enables fat wheels and safe SIMD development.
- **Validation:** Verify via `UCC_FORCE_ISA=scalar pytest tests/` and `UCC_FORCE_ISA=avx2 pytest tests/`. Windows builds should automatically use the scalar path. Benchmark all three circuits under both ISA paths.

---

## Step 3: Cultivation & U2 Latency Optimizations (Branchless & Broadcast)

Fix the structural bottlenecks inside `svm_kernels.inl` that don't require heavy loop rewrites.

- **Tasks:**
  1. **Branchless Multi-Gates (`exec_array_multi_cnot` & `exec_array_multi_cz`):**
     - **ILP trick:** Compute `mapped_cm = _pext_u64(ctrl_mask, pdep_mask)` **outside** the loop (when `UCC_HAS_PDEP` is true). Inside the loop, compute parity as `std::popcount(idx & mapped_cm) & 1` where `idx` is the raw loop counter. This breaks the data dependency chain: `idx` is instantly available as the loop counter, so the CPU executes `popcount` and the `pdep` (for address generation) **simultaneously** in parallel, completely hiding the 3-cycle `pdep` latency.
     - Replace the inner `if (parity)` branch with a branchless swap pattern using `CMOV`-friendly code (e.g., `v[actual] = parity ? val_b : val_a;` and `v[actual_t] = parity ? val_a : val_b;`).
  2. **U2 Broadcast-from-Memory (`exec_array_u2`):**
     - In the AVX2 `axis == 0` block, replace the two `_mm256_permute2f128_pd` cross-lane shuffles.
     - Use `_mm256_broadcast_pd` casting the array pointers directly to `__m128d const*` to broadcast 128-bits from memory. This uses the Load Ports (not the ALU shuffle port), freeing ALU capacity for the actual FMA math. `vbroadcastf128` has equivalent or better latency and avoids consuming a shuffle port.
- **Files:** `src/ucc/svm/svm_kernels.inl`
- **Difficulty:** Medium
- **Impact:** Cultivation d=5 should show a substantial speedup due to branch elimination. U2 axis==0 path benefits from freed ALU ports.
- **Validation:** Benchmark Cultivation d=5 and QV-20. Report speedup vs. Step 2 baseline.

---

## Step 4: AVX2 Waterfall for Heavy 2-Qubit Gates

Saturate memory bandwidth for large arrays without falling into the `inner_len == 1` scalar trap.

- **Tasks:**
  1. **AVX2 Fast-Path (`exec_array_cz`, `exec_array_swap`, `exec_array_cnot`):**
     - Introduce a fast-path guarded by `#if defined(__AVX2__)`.
     - Guard condition: `if (min_axis >= 1 && state.active_k >= 2)`. This evades the scalar trap because `min_axis >= 1` guarantees contiguous chunks of at least 32 bytes (2 complex doubles).
     - Implement explicit 3D nested loops: outer block (stride `2 * step2`), middle block (stride `2 * step1`), inner contiguous chunk (`k += 2`) using `_mm256_load_pd` and `_mm256_store_pd`.
     - The existing `pdep` loop remains below as the fallback for `min_axis == 0` or scalar targets.
- **Files:** `src/ucc/svm/svm_kernels.inl`
- **Difficulty:** High
- **Impact:** QV-20 should show a massive throughput jump.
- **Validation:** Benchmark QV-20. Report speedup vs. Step 3 baseline.

---

## Step 5: AVX-512 Specialization (Data-Center Throughput)

Add AVX-512 support for high-end Linux compute nodes to double ALU throughput and vectorize branchless opmasks. The dev machine (Xeon Platinum 8259CL, AWS Cascade Lake) natively supports AVX-512F and AVX-512DQ, so all AVX-512 code can be tested directly via `UCC_FORCE_ISA=avx512`.

- **Tasks:**
  1. **Create AVX-512 Unit:** Create `src/ucc/svm/svm_avx512.cc`. Define `UCC_SIMD_NAMESPACE avx512`.
  2. **Update CMake & Dispatcher:** Apply `-mavx512f -mavx512dq` to `svm_avx512.cc` in CMake (inside the x86_64 non-MSVC block). Add `"avx512"` to the `UCC_FORCE_ISA` check and `__builtin_cpu_supports` logic in `svm.cc`.
  3. **Upgrade Kernels (`svm_kernels.inl`):**
     - Wrap AVX-512 logic in `#if defined(__AVX512F__) && defined(__AVX512DQ__)`.
     - **Simple Gates (`CZ`/`SWAP`/`CNOT`):** Add a new waterfall tier for `if (min_axis >= 2)` loading 4 complex doubles (`__m512d`).
     - **Multi-Gates (`multi_cnot`/`multi_cz`):** Vectorize the branchless `CMOV` fallback using hardware opmask registers (`__mmask8`) and `_mm512_mask_blend_pd`.
     - **`exec_array_u2`:** Upgrade the matrix math to `_mm512_fmaddsub_pd`.
- **Files:** `src/ucc/svm/svm_avx512.cc`, `src/ucc/svm/svm_kernels.inl`, `src/ucc/CMakeLists.txt`, `src/ucc/svm/svm.cc`
- **Difficulty:** High
- **Impact:** Further throughput improvements on supported hardware.
- **Validation:** Test natively on dev machine with `UCC_FORCE_ISA=avx512`. Benchmark all three circuits. Report final speedups vs. original baseline.

---

## Step 6: Single-Axis Vectorization

Vectorize single-axis phase gates and EXPAND fused gates using 2D waterfall loops (phase/H/S) and flat 1D loops (EXPAND). Profiling data from Step 5 confirms these are the dominant remaining hotspots:

- **QV-20**: `exec_phase_rot` accounts for ~14% of total runtime. 81% of its invocations (482/592) have `axis >= 2` at `active_k = 20`, making them directly eligible for AVX-512 vectorization. Each such invocation performs 524K scalar `complex *= phase` operations.
- **Cultivation d=5**: `exec_phase_t` + `exec_phase_t_dag` total ~23% of runtime. Only 15% of invocations have `axis >= 2` (most land on axis 0), but the EXPAND_T/EXPAND_T_DAG fused gates (24 total, always targeting `active_k`) have perfectly contiguous memory.

**Rejected optimization**: Lowering the `kMinRankFor3DLoop` threshold for multi-gates. Bytecode analysis shows 86% of `OP_ARRAY_MULTI_CNOT` invocations in Cultivation land on axis 0 or 1, making the `axis >= 2` contiguity guard — not the rank threshold — the binding constraint. Zero additional invocations would be captured.

- **Tasks:**
  1. **2D Waterfall for Phase Gates (`exec_phase_rot`, `exec_phase_t`, `exec_phase_t_dag`):**
     - These gates multiply `arr[idx | v_bit]` by a complex phase for all indices where bit `v` is set. When `v >= 2`, the set-bit indices form contiguous chunks of size `step = 1 << v`.
     - Implement a 2D loop: outer stride `2 * step`, inner contiguous chunk processing `step` complex doubles.
     - AVX-512 path (`v >= 2 && active_k >= kMinRankFor3DLoop`): inner loop steps by 4 using `cmul_m512d`.
     - AVX2 path (`v >= 1 && active_k >= kMinRankFor3DLoop`): inner loop steps by 2 using `cmul_m256d`.
     - Broadcast `phase_re` and `phase_im` into `__m512d`/`__m256d` registers outside all loops.
  2. **2D Waterfall for Hadamard (`exec_array_h`):**
     - Butterfly transform: load from both quadrants `q0` and `q1`, compute `(q0 + q1) * inv_sqrt2` and `(q0 - q1) * inv_sqrt2` using `_mm512_add_pd` / `_mm512_sub_pd` + `_mm512_mul_pd`.
     - Same guards as phase gates: `v >= 2` for AVX-512, `v >= 1` for AVX2, `active_k >= kMinRankFor3DLoop`.
  3. **2D Waterfall for S/S_dag (`exec_array_s`, `exec_array_s_dag`):**
     - S gate multiplies by `i` (swap re/im, negate new re). S_dag multiplies by `-i`.
     - Same 2D structure as phase gates but with the fixed `i` / `-i` phase.
     - Same guards as above.
  4. **Flat 1D Loop for EXPAND Gates (`exec_expand`, `exec_expand_t`, `exec_expand_t_dag`, `exec_expand_rot`):**
     - Because `v == active_k`, memory is perfectly contiguous. No outer loop needed.
     - Read from lower half `arr[0..half)`, write to upper half `arr[half..2*half)` with optional phase multiplication.
     - Aggressive thresholds: `active_k >= 2` for AVX-512 (half = 4 complex doubles = one `__m512d`), `active_k >= 1` for AVX2 (half = 2 complex doubles = one `__m256d`).
     - `exec_expand` (pure copy + scale by `1/sqrt(2)`) can fuse the copy and scale into one pass using `_mm512_mul_pd` with a broadcast `inv_sqrt2`.
- **Files:** `src/ucc/svm/svm_kernels.inl`
- **Difficulty:** Medium
- **Impact:** QV-20 should see ~10-14% improvement from phase rotation vectorization. Cultivation should see modest gains from EXPAND vectorization.
- **Validation:** All tests pass on avx512/avx2/scalar. Benchmark all three circuits. Report speedups vs. Step 5 baseline.
