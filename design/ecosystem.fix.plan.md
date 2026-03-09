Please act as an expert C++ and Python developer. I need you to implement a series of correctness fixes, performance optimizations, and code-quality improvements across my quantum compiler codebase based on a recent architecture review.

 It's possible some of these are already fixed. If you disagree with any, please stop and let me review and understand why. Please also commit each as standalone commits to make it easier to review. Please stop and open a PR after each phase (1,2,3) so I can review. As part of that PR, you should update this document to reflect your progress


**CRITICAL CONSTRAINT:** Do **not** make any changes related to 64-bit hardcoding limits, physical scaling constraints, `TableauTransposedRaii`, or C++ undefined behavior (e.g., bit-shift bounds). Do not attempt to add destructors to `SchrodingerState`. Focus strictly on the following items grouped by domain:

### 1. Python Bindings & GIL Management (`src/python/bindings.cc`)

* **Zero-Copy Template Refactor:** The ownership transfer of `std::vector` to a `nanobind::capsule` (to avoid deep copies) is currently duplicated for `uint8_t`, `uint64_t`, and `complex<double>`. Create a reusable C++ template (e.g., `template <typename T> nb::ndarray<nb::numpy, T, nb::c_contig> make_numpy_array(std::vector<T> vec, std::initializer_list<size_t> shape)`) and refactor `sample`, `sample_survivors`, and `get_statevector` to use it. In `get_statevector`, eliminate the `new std::complex<double>[n]` allocation and blocking `std::copy`, moving the `std::vector` directly into the zero-copy template.
* **Fix GIL Blocking:** Add `nb::gil_scoped_release release;` to the `execute` and `get_statevector` bindings so that dense matrix operations don't freeze multithreaded Python workloads (like Sinter).
* **Fix Python Polymorphism:** Apply nanobind trampoline macros (`NB_TRAMPOLINE`) to `ucc::Pass` and `ucc::BytecodePass`. Create trampoline wrapper classes (e.g., `PyPass` and `PyBytecodePass`) so Python subclasses can properly override the virtual `run()` methods. Change the bindings to bind the wrappers instead.
* **Clean up Comments:** Delete vacuous comments that just restate the code (e.g., `// GateType enum`, `// Circuit class`, `// Bytecode Optimization Passes`).

### 2. C++ Core API & Refactoring

* **Modern C++ Attributes:**
* In `src/ucc/util/introspection.h` and `src/ucc/util/introspection.cc`, add `[[nodiscard]] constexpr noexcept` to the opcode classifiers (`is_two_axis_opcode`, `is_one_axis_opcode`, `is_meas_opcode`).
* Add `[[nodiscard]]` to the pipeline functions `ucc::trace` (`src/ucc/frontend/frontend.h`) and `ucc::lower` (`src/ucc/backend/backend.h`).


### 3. WebAssembly Endpoint (`src/wasm/bindings.cc`, `src/wasm/test_wasm.mjs`)

* **Fix PRNG Lobotomization:** In `simulate_wasm`, replace the hardcoded `0` seed in `ucc::sample(prog, shots, 0)` with a dynamically generated seed from the device random (not sure how to do this for WASM?)
* **Fix Heap Fragmentation:** In `simulate_wasm`, change `std::map<std::string, uint32_t> histogram;` to `std::unordered_map`. Hoist the `std::string key;` declaration outside the `for (shot)` loop, call `key.reserve(n_meas);` once, and use `key.clear();` inside the loop to eliminate memory allocation churn.
* **Enforce Physics in Wasm Tests:** In `test_wasm.mjs`, update the simulation verification for `H 0; M 0`. Instead of just asserting `total == 1000`, assert that the histogram contains roughly 500 counts for `'0'` and 500 counts for `'1'`.

### 4. Python Test Suite & Oracles

* **Fix Invalid Fidelity Oracle:** In `tests/python/conftest.py` (`assert_statevectors_equal`), change the strict less-than check `if fidelity < 1.0 - rtol:` to `if abs(fidelity - 1.0) > rtol:`.
* **Silently Ignored Qiskit Gates:** In `tests/python/utils_qiskit.py` (`stim_to_qiskit_noiseless`), instead of using `continue` to silently skip measurement and reset gates (`M`, `R`, `MX`, etc.), explicitly raise a `ValueError("Measurements and resets are not supported in the noiseless statevector oracle")`.
* **Tighten Statistical Bounds:** In `tests/python/test_sample.py` (`test_observable_ones_counts_errors`), change the wildly loose `4000 < ones < 6000` assertion. Use the existing `binomial_tolerance` helper (or equivalent $5\sigma$ math) to tighten the bounds.
* **Remove Tautological Assertions:** In `tests/python/test_optimization_invariants.py` (`TestHirPeepholeUncomputationLadder`), delete the `np.testing.assert_array_equal(base_m, opt_m)` assertion immediately following the code that asserts both arrays are strictly equal to `np.zeros_like`.

### 5. Documentation (`design/data_structs.md`)

* **Sync Structs to Code:** Update the documentation for the `Instruction` struct. Change the union struct name `multi` to `multi_gate` to match the C++ implementation. Remove the references to the manual `_pad` arrays to match the updates in `backend.h`.
