Please implement the following targeted fixes, performance optimizations, and refactors in this C++ quantum compiler codebase. Focus strictly on the files and functions mentioned.

It's possible some of these are already fixed. If you disagree with any, please stop and let me review and understand why. Please also commit each as standalone commits to make it easier to review. Please stop and open a PR after each phase (1,2,3) so I can review. As part of that PR, you should update this document to reflect your progress


Treat each bullet point as a strict requirement. Do not make any changes related to union memory initialization, 64-bit bounds/truncations, or bit-shift undefined behavior.

### 1. `src/ucc/frontend/frontend.cc` -- DONE

* **Math Error in Y-Basis Corrections:** DONE. Changed `extract_rewound_x` to `extract_rewound_z` in RY and MRY correction paths. Added multi-seed regression tests in `test_statevector.cc`.
* **Duplicated Reset Decompositions:** DONE. Unified all 6 cases (R, RX, RY, MR, MRX, MRY) into a single switch case with `Basis` enum and two lambdas (`extract_meas`, `extract_corr`). Reduced frontend.cc by ~87 lines.

### 2. `src/ucc/circuit/parser.cc` & `src/ucc/circuit/parser.h` -- DONE

* **Anti-Hermitian MPP Phase Loss:** DONE. Added `vector<bool> seen_qubits` per product; throws `ParseError("Duplicate qubit in MPP product")` on duplicates.
* **Parser Denial of Service (Empty Loops):** DONE. Scans body for non-whitespace/non-comment content before the unroll loop; skips iteration entirely if empty.
* **Unbounded Recursion Stack Overflow:** DONE. Threaded `uint32_t depth` through `parse_block`, `parse_line`, and `parse_repeat`. Throws at depth > 100. Renamed local brace-tracking variable to `brace_depth` to avoid shadowing.
* **Cache-Destroying Memory Zeroing:** DONE. Replaced `std::string(size, '\0')` with `std::make_unique<char[]>` + `std::string_view`.

### 3. `src/python/bindings.cc` -- DONE

* **Python GIL Starvation:** DONE. Added `nb::gil_scoped_release` to `parse` (both overloads), `parse_file` (both overloads), `trace`, `lower`, and `compile`. PassManager and BytecodePassManager are pure C++ so the entire `compile` lambda releases the GIL. Added `test_gil_release.py` with 4 threading tests.

### 4. `src/ucc/circuit/gate_data.h` -- DONE

* **Massive Switch Statement Duplication:** DONE. Replaced 8 switch statements with a single `constexpr GateTraits kGateTraitsData[]` lookup table indexed by `GateType` enum value. Each helper function is now a one-liner. A `static_assert` ensures the table has exactly one entry per `GateType`. Added 12 Catch2 tests covering every gate category.

### 5. `tests/test_frontend.cc` -- DONE

* **Vacuous Tautologies in Tests:** DONE. Added `REQUIRE(hir_cx.num_ops() > 0)` before the comparison loop in the ZCX alias test.
