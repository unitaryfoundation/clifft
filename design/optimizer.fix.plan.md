
You are an Expert C++20 Systems Engineer and Compiler Architect. I need you to perform a strict refactoring of this quantum compiler codebase based on a recent architectural review.

*CRITICAL INSTRUCTION: Do not modify the 64-bit mask logic, `uint16_t` axis scaling, or bitshift operations. Those are being handled in a separate 512-bit architectural update. Focus strictly on the items listed below.*

Please implement the following fixes, grouped by domain. Work through them step by step, with one commit per bullet. Please open a PR for review after each section (e.g. 1, 2, 3).

If you disagree with a suggestion, stop and ask for feedback.  Some of them may have already been addressed, in which case also stop and ask for feedback.

### 1. Correctness & Memory Safety

* **Implement the Rule of Five (`SchrodingerState`):** The struct dynamically allocates a heap array (`std::complex<double>* v`) but lacks a destructor, copy constructor, move constructor, and assignment operators. Implement these to prevent memory leaks and double-free segfaults (especially critical when bridged to Python via `nanobind`).
* **Fix Instruction Data Erasure (`ExpandTPass`):** When creating the `fused` instruction, explicitly copy over the `flags` and `base_phase_idx` from the original `OP_PHASE_T` instruction. Currently, they are discarded, which permanently destroys quantum parity/conditional state.
* **Preserve Frame Sign (`PeepholeFusionPass`):** When fusing $T$ and $T^\dagger$ into an $S$ gate, the code blindly hardcodes `/*sign=*/false`. Update this to mathematically calculate and preserve the resulting algebraic sign based on the incoming gates.
* **Prevent Out-of-Bounds Segfaults (`is_blocked`):** Add a strict bounds check for `site_idx` against `hir.noise_sites.size()` before indexing the array to protect against malformed compiler IR.
* **Release the Python GIL:** For Python bindings, ensure entry points like `ucc.sample()` that execute dense statevector array math explicitly drop the lock using `nanobind::gil_scoped_release` to prevent starving multithreaded Python hosts.

### 2. Architecture & Algorithmic Complexity

* **Fix $O(N^3)$ Bottleneck (`PeepholeFusionPass`):** The nested $O(N^2)$ loop combined with manual $O(N)$ array compaction on *every single fusion* will freeze the compiler on deep circuits. Refactor this to use a single-pass active-qubit frontier array or a dependency DAG. At an absolute minimum, defer compaction to a single `std::erase_if` pass at the end of the loop.
* **Deduplicate Bytecode Pass Boilerplate:** `ExpandTPass`, `MultiGatePass`, `NoiseBlockPass`, and `SwapMeasPass` identically duplicate ~30 lines of variable initialization, vector capacity reservations, and `while` loop iteration. Unify this behind a base `InstructionRewriter` visitor class or a shared sliding-window pipeline utility.
* **Flatten `source_map` for Cache Locality:** Change `CompiledModule::source_map` from a cache-thrashing `std::vector<std::vector<uint32_t>>` to a contiguous 1D array (`std::vector<uint32_t>`) backed by a `{uint32_t offset, uint32_t length}` span table (CSR format). Update all passes accordingly.
* **Fix Symmetry Blindspot (`SwapMeasPass`):** `OP_ARRAY_SWAP(a, b)` is mathematically symmetric. Update the pass to check for both `SWAP(k-1, tgt)` and `SWAP(tgt, k-1)` so valid commutative optimizations aren't missed.

### 3. Modern C++20 Best Practices

* **Eliminate Accidental Deep Copies:** In the fall-through/skip pathways of all bytecode passes, `new_src.push_back(old_src[i]);` deep-copies a vector for every unmodified instruction. Change this to `std::move(old_src[i])`.
* **Optimize Vector Insertions:** Replace piecewise insertions (`for (uint32_t line : old_src[i]) merged.push_back(line);`) with bulk pre-sized inserts (`merged.insert(merged.end(), old_src[i].begin(), old_src[i].end());`).
* **Remove `std::vector<bool>` Overhead:** In `PeepholeFusionPass`, change `std::vector<bool> deleted` to `std::vector<uint8_t> deleted` to avoid bit-proxy decoding/branching overhead in the hot loop.
* **Fast Bitwise Parity:** In `anti_commute` and `anti_commute_raw`, change the signed modulo `.popcount() % 2 != 0` to pure bitwise math (`.popcount() & 1`).
* **Apply `[[nodiscard]]`:** Add `[[nodiscard]]` to pure mathematical queries (`anti_commute`, `anti_commute_raw`, `effective_angle`, `is_blocked`, `pauli_masks`) to ensure bounds/commutation checks are never accidentally dropped.
* **Standard Constants:** In `test_helpers.h`, replace the manually typed `kInvSqrt2` with C++20's `<numbers>` library (`std::numbers::inv_sqrt2`).
* **Clean Struct Layout:** Remove the explicit padding members (e.g., `uint8_t _pad_a[8]`) inside the `Instruction` payload union. Rely entirely on the struct-level `alignas(32)`.
* **Delete Vacuous Comments:** Remove comments that literally narrate code execution (e.g., `// Pass through all other instructions`, `// Start of a noise run`, `// Compact: remove deleted ops`).

### 4. Test Suite Hardening

* **Fix Vacuous Rubber-Stamping (`test_bytecode_passes.cc`):** Under `MULTI_CNOT`, replace the conditional `if (cnot_count >= 2) { CHECK(...); }` with a strict `REQUIRE(cnot_count >= 2);` so the test immediately fails if the parser breaks.
* **Fix Brittle RNG Assertions:** Tests asserting equivalence via `REQUIRE(res_orig.measurements == res_opt.measurements)` with a fixed RNG seed are brittle when passes like `NoiseBlockPass` legitimately alter RNG draw sequences. Rewrite these to compare structural bounds or pure statevectors.
* **Strengthen State Matchers:** In `test_optimizer.cc` (e.g., "T_dag plus T_dag fuses to S_dag"), add explicit assertions that the target virtual axes (`stab_mask` and `destab_mask`) survive the fusion completely untouched.
* **Fix Absolute Tolerance Hazards:** Update `check_complex` (`test_helpers.h`) to evaluate using `Catch::Matchers::WithinRel` alongside `Catch::Matchers::WithinAbs` to safely handle relative drift in exponentially decaying statevector amplitudes.
