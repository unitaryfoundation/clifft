
Please act as a Staff C++ Systems Engineer. I need you to implement a strict set of bug fixes, memory safety corrections, and cleanups in this C++ codebase.

Please address the following items. Work through them logically and provide the updated code for each affected file.

It's possible some of these are already fixed. If you disagree with any, please stop and let me review and understand why. Please also commit each as standalone commits to make it easier to review. Please stop and open a PR after each phase (1,2,3) so I can review. As part of that PR, you should update this document to reflect your progress


### 1. SVM Execution & Math (`src/ucc/svm/svm.cc`, `src/ucc/svm/svm.h`)

* **Fix `scale_magnitude` Underflow:** In `SchrodingerState::scale_magnitude`, the check `std::norm(gamma_) < 1e-200` triggers catastrophic IEEE-754 underflow because `std::norm` computes the *squared* magnitude. Change the underflow logic to check `std::abs(gamma_) < 1e-100` before any squaring occurs.
* **Fix Move Semantics Segfault:** Update the move constructor and move assignment operator in `SchrodingerState` to explicitly set `other.active_k = 0` and `other.peak_rank_ = 0`. Leaving them intact causes `.reset()` to segfault when it attempts to `memset` a null array based on an outdated `active_k`.
* **Fix Allocation UB:** In the `SchrodingerState` constructor, throw a `std::invalid_argument` if `peak_rank >= 63` to prevent Undefined Behavior during the `1ULL << peak_rank` bitshift and subsequent `std::aligned_alloc` integer overflow.
* **Fix NaN Poisoning:** In `exec_meas_active_diagonal` and `exec_meas_active_interfere`, if a statevector branch evaluates to exactly `0.0` norm, `prob_b` becomes `0`. Guard the deferred normalization division (`std::sqrt(total / prob_b)`) so we don't divide by zero and poison the global scalar with `NaN`s.
* **Fix AVX-512 Breakage:** In `apply_pauli_to_frame`, change `Bitword err_x = ps.xs.ptr_simd[0];` to `uint64_t err_x = ps.xs.u64[0];`. Calling `ptr_simd` fails to cast to a 64-bit int on AVX-512 builds. Do the same for `ps.zs`.
* **Optimize Renormalization FLOPs:** In `exec_meas_active_diagonal` and `exec_meas_active_interfere`, decrement `state.active_k--` *before* calling `state.scale_magnitude(...)`. This avoids wasting 50% of our FLOPs multiplying the upper half of the array right before we discard it.
* **Remove Redundant Zeroing:** Delete all calls to `std::memset(arr + half, 0, ...)` inside the array compaction logic (`exec_meas_active_diagonal`, `exec_meas_active_interfere`, `exec_swap_meas_interfere`). The VM never reads from dormant indices, and `exec_expand` safely overwrites them.
* **Use PDEP Abstraction:** In `exec_array_multi_cnot`, replace the hand-rolled bit-insertion `(idx & lo_mask) | ((idx & ~lo_mask) << 1)` with a call to the existing `scatter_bits_1` helper to leverage hardware BMI2 acceleration.
* **Apply `__restrict`:** Inside the VM hot loops (e.g., `exec_array_h`, `exec_array_cnot`, `exec_phase_t`), apply `__restrict` to the array pointer (e.g., `auto* __restrict arr = state.v();`) so the compiler can aggressively vectorize the complex math.

### 2. Compiler Back-End & Optimizers (`src/ucc/backend/backend.cc` & `src/ucc/optimizer/*.cc`)

* **Fix Gap Sampler Overshoot:** In `backend.cc` (near `OpType::NOISE`), the hazard clamp for deterministic noise is currently `1.0 - 1e-15`. Update this to `1.0 - 0x1.0p-53` to precisely match the maximum possible value that `random_double()` can generate, ensuring deterministic events are never skipped.
* **Fix `MultiGatePass` State Corruption:** In `src/ucc/optimizer/multi_gate_pass.cc`, CNOT/CZ masks are currently accumulated using bitwise OR (`|=`). Because these gates are self-inverting, contiguous identical gates on the same target cancel out. Change the accumulator to XOR (`^=`). If the resulting mask evaluates to `0`, drop the instruction entirely instead of emitting a vacuous array loop.
* **DRY Vector Mutations:** Across all passes in `src/ucc/optimizer/bytecode_pass.cc`, stop manually synchronizing `bytecode`, `source_map`, and `active_k_history` using 20+ lines of brittle `if (has_src) ... push_back()` code. Create a zipping mechanism or a small struct/wrapper (e.g. `InstructionStream`) to make these structural mutations transactional and clean.

### 3. Python Bindings (`src/python/bindings.cc`)

* **Prevent GIL Starvation:** Add `nb::gil_scoped_release release;` blocks around the C++ function calls inside the `ucc::compile`, `ucc::lower`, and `ucc::execute` bindings.
* **True Zero-Copy Statevectors:** In `get_statevector`, avoid `new std::complex<double>[n]` and the manual `std::copy` loop. Instead, heap-allocate the resulting `std::vector` (or take ownership of it directly) and pass it to an `nb::capsule`, mirroring how the `uint8_t` zero-copy arrays are handled.

### 4. Headers & Hygiene

* **Enforce `[[nodiscard]]`:** Add `[[nodiscard]]` to `SchrodingerState::v_size()`, `SchrodingerState::random_double()`, and all AST instruction factory functions (e.g., `make_frame_cnot`, `make_expand` in `src/ucc/backend/backend.h`).
* **Delete Vacuous Comments:** Remove line-noise comments in the compiler backend and VM that just translate C++ into English (e.g., `// Determine classical output index`, `// Plain qubit index`, `// Check for rec[-k] reference`, `// Keep chosen branch, compact array`).

### 5. Testing Enhancements (`tests/`)

* **Fix Fuzzer Blind Spots:** In `tests/python/utils_fuzzing.py`, update `generate_star_graph_honeypot` so it occasionally emits identical contiguous gates (algebraic collisions) instead of strictly using `replace=False`. This ensures the `MultiGatePass` XOR logic is actually fuzzed.
* **Extreme Normalization Fuzzer:** In `tests/test_svm_risc.cc`, update the compaction fuzzer to intentionally inject extreme magnitudes ($10^{\pm 150}$) rather than just states with a norm of `1.0`. This will prove the IEEE-754 deferred normalization rescue math works perfectly during compaction.
* **Stop Rubber-Stamping:** In `tests/test_bytecode_passes.cc`, update the basic structural checks (e.g., `REQUIRE(m.bytecode.size() == 1)`) to actually verify mathematical invariants where appropriate (e.g., by executing the state and asserting correctness) so regressions are caught.
