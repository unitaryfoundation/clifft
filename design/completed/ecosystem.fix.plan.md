# Ecosystem Fix Plan

Correctness fixes, performance optimizations, and code-quality improvements
across the UCC codebase based on architecture review.

## Status: COMPLETED

All items implemented or resolved across 5 commits on `feat/ecosystem-fixes`.

---

### 1. Python Bindings & GIL Management (`src/python/bindings.cc`)

* **Zero-Copy Template Refactor:** DONE. Created `vec_to_numpy<T>()` template
  replacing duplicated capsule patterns for `uint8_t`, `uint64_t`, and
  `complex<double>`. Unified `sample`, `sample_survivors`, and
  `get_statevector`.
* **Fix GIL Blocking:** ALREADY DONE (PR #109). `execute` and
  `get_statevector` already had `gil_scoped_release`.
* **Fix Python Polymorphism (NB_TRAMPOLINE):** SKIPPED. Deferred until Python
  subclass passes are needed. Current `BorrowedPass` wrapper handles
  C++-defined passes correctly.
* **Clean up Comments:** DONE. Removed vacuous section headers that restated
  code structure.

### 2. C++ Core API & Refactoring

* **Opcode classifiers:** DONE. Moved `is_two_axis_opcode`,
  `is_one_axis_opcode`, `is_meas_opcode` to header as
  `[[nodiscard]] constexpr noexcept` inline functions.
* **`[[nodiscard]]` on `trace`/`lower`:** ALREADY DONE. Both already had
  `[[nodiscard]]` in their headers.

### 3. WebAssembly Endpoint (`src/wasm/bindings.cc`, `src/wasm/test_wasm.mjs`)

* **Fix PRNG Lobotomization:** DONE. Replaced hardcoded seed `0` with
  `std::nullopt` to use OS entropy via existing `seed_from_entropy()`.
* **Fix Heap Fragmentation:** DONE. `std::map` -> `std::unordered_map`,
  hoisted `std::string key` outside shot loop with `reserve`/`clear`.
* **Enforce Physics in Wasm Tests:** DONE. Added 50/50 distribution assertion
  for H|0> simulation (350-650 bounds for 1000 shots).

### 4. Python Test Suite & Oracles

* **Fix Invalid Fidelity Oracle:** DONE. Changed to
  `abs(fidelity - 1.0) > rtol` to catch fidelity > 1.0.
* **Silently Ignored Qiskit Gates:** DONE. `stim_to_qiskit_noiseless` now
  raises `ValueError` on M/R gates.
* **Tighten Statistical Bounds:** DONE. Replaced `4000 < ones < 6000` with
  5-sigma `binomial_tolerance` math.
* **Remove Tautological Assertions:** DONE. Removed redundant `base_m == opt_m`
  assertions from `test_small` and `test_large` in
  `TestHirPeepholeUncomputationLadder`.

### 5. Documentation (`design/data_structs.md`)

* **Sync Structs to Code:** DONE. Updated `base_phase_idx` -> `_reserved`,
  `multi` -> `multi_gate`, removed explicit `_pad` arrays from doc.
