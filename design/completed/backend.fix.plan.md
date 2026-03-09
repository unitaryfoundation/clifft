
# Backend Fix Plan — Progress Tracker

This document tracks the architectural, performance, and code-quality improvements
applied to the UCC backend compiler.

**Branch:** `feat/backend-fix-plan`

**CRITICAL CONSTRAINTS (observed throughout):**

1. No changes to 64-bit shift limitations, PRNG unsequenced UB, or 64-bit hardware cap docs.
2. No changes to `V_cum` tracking for `CLIFFORD_PHASE` (compiler correctly skips S absorption).
3. No Rule of Five boilerplate added to `SchrodingerState`.

---

## Phase 1: Correctness (Math & Logic) — DONE

**Commit:** `4c126be`

* **Global Phase Loss:** Fixed. When `phase_flip` is true, T_GATE now multiplies
  `global_weight` by `e^{i*pi/4}` and CLIFFORD_PHASE multiplies by `i`.
  Added `constexpr double kInvSqrt2 = std::numbers::sqrt2 / 2.0;` in backend.cc.
* **Precision Collapse:** Fixed. Noise hazard uses `std::log1p(-x)` instead of `std::log(1-x)`.
* **Uninitialized Read Risk:** Fixed. `last_meas_idx` initialized to `UINT32_MAX` with assert guard.

## Phase 2: Testing Framework Integrity — DONE

**Commit:** `46e3f84`

* **ARRAY vs FRAME Semantics:** `verify_bytecode_compression` now enforces axis < `active_k()`
  for ARRAY opcodes and axis >= `active_k()` for FRAME opcodes.
* **Inverse Phase Blind Spot:** Added `OP_FRAME_S_DAG` / `OP_ARRAY_S_DAG` cases to
  `bytecode_to_tableau` (S_DAG = S^3 via three `append_S` calls).
* **Blind Sequential Verification:** `verify_sequential_compression` now calls
  `verify_bytecode_compression` on the emitted bytecode slice.
* **Modernize Fuzzing (Catch2 GENERATE):** CANCELLED — Catch2 GENERATE with random
  generators would lose reproducibility and complicate seed-based debugging.

## Phase 3: Performance & Cache Locality — DONE

* **CSR source_map** (commit `af77260`): `CompiledModule` and `CompilerContext` use
  `source_map_data` + `source_map_offsets` (CSR) instead of `vector<vector<uint32_t>>`.
  Python bindings reconstruct list-of-lists on demand. All 4 optimizer passes updated.
* **Branchless Z-bit** (commit `bd420fa`): `z_bits ^= (-((z_bits >> q) & 1)) & (1ULL << pivot)`
  replaces branching conditional in compress_pauli X-compression loop.
* **Per-instruction k tracking** (commit `cefb003`): `CompilerContext::emit()` helper pushes
  instruction + current `active_k()` atomically; replaces batch fill loop.
* **Hoist tableau transpositions:** DEFERRED — cascading signature and test changes
  across compress_pauli, emit_cnot, emit_cz, emit_s make this a high-risk refactor
  for marginal gain. The current design already hoists the O(n^2) transpose within
  compress_pauli itself.

## Phase 4: Code Quality & API Polish — DONE

* **Extract routing helper** (commit `fd15907`): Extracted `route_to_active_z()` from the
  duplicated dormant-X expansion / active-X rotation blocks in T_GATE and CLIFFORD_PHASE.
  Eliminates ~40 lines of identical code.
* **Remove vacuous comments** (commit `fd15907`): Removed comments that restate the next line
  (e.g., "Map the t=0 Pauli to virtual frame", "Store in constant pool"). Kept domain-specific
  comments explaining gap sampling, phase algebra, and non-obvious invariants.
* **`std::span` for postselection_mask** (commit `6292c61`): `lower()` now accepts
  `std::span<const uint8_t>` instead of `const vector<uint8_t>&`. Non-breaking: vectors
  implicitly convert to spans.
* **`[[nodiscard]]`** (commit `6292c61`): Added to `parse()`, `trace()`, `lower()`, and
  `compress_pauli()` to catch accidental discard of return values at compile time.

**Skipped items (deliberate):**
* **Delete factory boilerplate:** Skipped — ~219 call sites across production and test code
  would require touching union members through designated initializers, which C++20 does not
  support for nested anonymous unions. The factories provide a clean, type-safe API.
* **Abstract bit-clearing duplication in compress_pauli:** Skipped — the three bit-clearing
  loops differ in gate type (CNOT vs CZ), argument order, and side effects (z-bit propagation).
  A shared lambda would obscure the distinct compression algebra.
* **Clean up union padding:** Skipped — the explicit `_pad_*` arrays document the 32-byte
  layout at each variant and prevent accidental aggregate-init misalignment. The `static_assert`
  on `sizeof(Instruction) == 32` already guards correctness.
