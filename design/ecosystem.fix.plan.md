You are an Expert C++20 Systems Engineer and Compiler Architect. I need you to perform a strict refactoring of this quantum compiler codebase based on a recent architectural review.

*CRITICAL INSTRUCTION: Do not modify the 64-bit mask logic, `uint16_t` axis scaling, or bitshift operations. Those are being handled in a separate 512-bit architectural update. Focus strictly on the items listed below.*

Please implement the following fixes, grouped by domain. Work through them step by step, with one commit per bullet. Please open a PR for review after each section (e.g. 1, 2, 3).

If you disagree with a suggestion, stop and ask for feedback.  Some of them may have already been addressed, in which case also stop and ask for feedback.

### 1. C++ Core & Memory Safety

* **Fix `SchrodingerState` Memory Leak:** Convert the raw `std::complex<double>* v` pointer to `std::unique_ptr<std::complex<double>[]> v`. This ensures Python's garbage collector correctly frees the statevector array when a `State` object is destroyed via `nanobind`. Update constructors and `reset()` logic accordingly.
* **Use `std::span`:** Modify `ucc::compile` and `ucc::lower` to accept `std::span<const uint8_t> postselection_mask` instead of passing `std::vector<uint8_t>` by value. Update the Python bindings to pass nanobind array views to avoid $O(N)$ deep copies.
* **Optimize Bit-Scanning:** In `src/ucc/util/introspection.cc`, refactor `format_pauli_mask` to `#include <bit>` and use `std::countr_zero` on the bitmasks to jump directly to active bits instead of scanning all bits sequentially with a `for` loop.
* **Remove C-style Union Padding:** In `design/data_structs.md` and the `Instruction` struct, delete all manual padding arrays (e.g., `uint8_t _pad_a[8];`). The outer union natively sizes itself to its largest member, making explicit padding brittle and unnecessary.
* **Modernization:** Apply `[[nodiscard]]` to query and pipeline functions (`max_sim_qubits`, `trace`, `lower`). Apply `constexpr noexcept` to the opcode classifiers. Delete `kInvSqrt2` in `test_helpers.h` and use `std::numbers::inv_sqrt2` from `<numbers>`.

### 2. Python Bindings (`nanobind`)

* **Release the GIL:** Add `nb::gil_scoped_release release;` to the Python bindings for `execute` and `get_statevector` to prevent hard-locking the Python interpreter during exponential array evaluations.
* **Fix Python Polymorphism:** Wrap the `ucc::Pass` and `ucc::BytecodePass` bindings with the `NB_TRAMPOLINE` macro. Without this, C++ cannot execute overridden `run()` methods defined in Python subclasses.
* **Zero-Copy Abstraction:** The `nb::capsule` ownership transfer boilerplate is duplicated in `make_numpy_array`, `sample_survivors`, and `get_statevector`. Abstract this into a reusable C++ template function (e.g., `template <typename T> wrap_vector_to_numpy(...)`).
* **Eliminate Statevector Deep Copy:** In `get_statevector`, eliminate the `new[]` allocation and blocking `std::copy`. Heap-allocate the returned `std::vector` directly and use the new zero-copy template to transfer ownership to NumPy.

### 3. Wasm / WebAssembly Fixes

* **Fix PRNG Lobotomization:** In `src/wasm/bindings.cc`, `simulate_wasm` currently hardcodes a seed of `0` via `ucc::sample(prog, shots, 0)`. Change this to use a proper random seed (e.g., via `<random>`) or accept an optional seed from JavaScript so it produces actual stochastic distributions.
* **Fix WASM Heap Fragmentation:** In `simulate_wasm`, the histogram generation loop appends to a string dynamically (`key += ...`) per measurement per shot. Pre-allocate the string with `key.resize(n_meas)` and write to the characters directly by index. Replace `std::map` with `std::unordered_map` to prevent $O(N \log M)$ pointer-chasing.
* **Stop WASM Rubber-Stamping:** In `test_wasm.mjs`, the simulation test only checks `total == 1000`. Add assertions that actually check the physics (e.g., a Hadamard circuit should yield ~500 for '0' and ~500 for '1').

### 4. Python Testing & Validation Oracles

* **Fix Fidelity Oracle (Critical):** In `tests/python/conftest.py`, `assert_statevectors_equal` checks `fidelity < 1.0 - rtol` without enforcing normalization. This will silently pass unnormalized vectors (e.g., an array with norm 2.0 yielding fidelity 4.0). Add explicit assertions that `actual` is normalized ($L^2$ norm == 1.0) before computing fidelity.
* **Strict Qiskit Oracle:** In `utils_qiskit.py` (`stim_to_qiskit_noiseless`), instead of silently `continue`-ing when hitting unsupported measurements/resets (`M`, `R`, `MX`, etc.), explicitly `raise ValueError`. This prevents uncollapsed circuits from silently passing tests.
* **Tighten Statistical Bounds:** Fix loose bounds in `cross_binomial_tolerance` (scaling $5\sigma$ by $\sqrt{2}$ creates a vacuous $7.07\sigma$ bound). In the test suite (e.g., `test_sample_survivors`), replace hardcoded assertions like `4000 < ones < 6000` for 10,000 shots with strict binomial math.
* **Fix Tautological Assertions:** In `TestHirPeepholeUncomputationLadder`, remove `np.testing.assert_array_equal(base_m, opt_m)` where they have both just been independently asserted to equal `np.zeros_like`.

### 5. Documentation Cleanup

* **Fix Tableau Misinformation:** Update `design/architecture.md`. Transposing a tableau makes *prepending* fast. Appending (row operations) is natively fast in Stim's default memory layout.
* **Fix Struct Documentation:** Update `design/data_structs.md` so the multi-gate member is correctly named `multi_gate` (to match the actual C++ code in `bindings.cc`), not `multi`.
* **Remove Vacuous Comments:** Strip out comments that merely restate the code (e.g., `// GateType enum`, `// Circuit class`). Remove the contradictory payload union comments (`up to 12 bytes` vs `exactly 12 bytes`).
