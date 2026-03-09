Please implement the following targeted fixes, performance optimizations, and refactors in this C++ quantum compiler codebase. Focus strictly on the files and functions mentioned.

It's possible some of these are already fixed. If you disagree with any, please stop and let me review and understand why. Please also commit each as standalone commits to make it easier to review. Please stop and open a PR after each phase (1,2,3) so I can review. As part of that PR, you should update this document to reflect your progress


Treat each bullet point as a strict requirement. Do not make any changes related to union memory initialization, 64-bit bounds/truncations, or bit-shift undefined behavior.

### 1. `src/ucc/frontend/frontend.cc`

* **Math Error in Y-Basis Corrections:** In `trace()`, within the switch cases for `GateType::RY` and `GateType::MRY`, the conditional correction extracts the rewound $X$ mask (`extract_rewound_x`). This is mathematically incorrect as applying $X$ to $|-Y\rangle$ injects an unphysical $-i$ global phase. Change this to extract the rewound $Z$ mask (`extract_rewound_z`) instead, which correctly maps $|-Y\rangle \to |+Y\rangle$ perfectly.
* **Duplicated Reset Decompositions:** The AST tracing logic inside the switch block for `R`, `RX`, `RY`, `MR`, `MRX`, and `MRY` is heavily copy-pasted (extracting observables, emitting measurements, extracting corrections, emitting conditionals, handling readout noise). Extract this sequence into a generic helper lambda inside `trace()` (e.g., `emit_reset_sequence(...)`) to drastically reduce duplicated code.

### 2. `src/ucc/circuit/parser.cc` & `src/ucc/circuit/parser.h`

* **Anti-Hermitian MPP Phase Loss:** In `Parser::parse_mpp`, users can submit malformed products with duplicate qubits (e.g., `MPP X0*Z0`). Track the seen qubits for the current product being parsed (e.g., using a local `std::vector<bool>` or a small set). If a duplicate qubit index is encountered within the *same* product, throw a `ParseError("Duplicate qubit in MPP product")` to prevent silent phase loss.
* **Parser Denial of Service (Empty Loops):** In `Parser::parse_repeat`, a massive empty loop like `REPEAT 4000000000 {}` bypasses the AST `max_ops` limit and spins infinitely. Right before the `for (uint32_t i = 0; i < repeat_count; i++)` unrolling loop, check if the `body` string contains only whitespace/comments. If it does, break or return early.
* **Unbounded Recursion Stack Overflow:** Deeply nested `REPEAT` blocks will blow the C++ call stack. Add a `uint32_t recursion_depth = 0` parameter to `parse_block` and `parse_repeat`. Increment it on recursive calls. If `recursion_depth > 100`, throw a `ParseError("Max recursion depth exceeded")`. Update the `parser.h` signature accordingly.
* **Cache-Destroying Memory Zeroing:** In `parse_file`, `std::string buffer(size, '\0')` eagerly forces the OS to page-fault and zero-fill gigabytes of RAM, which `file.read()` immediately overwrites. Replace the `std::string` buffer allocation with C++23's `std::string::resize_and_overwrite`, or manually allocate an uninitialized `std::unique_ptr<char[]>` to read into before passing a `std::string_view` of it to `parse()`.

### 3. `src/python/bindings.cc`

* **Python GIL Starvation:** Massive circuit compilations block multi-threaded Python workloads. Wrap the core C++ execution inside `parse`, `parse_file`, `trace`, and `lower` with `nb::gil_scoped_release release;` so the C++ engine releases the Python Global Interpreter Lock while executing. (For the `compile` function, be careful to release the GIL only around the intensive C++ calls, ensuring you reacquire or avoid releasing it while interacting with Python `PassManager` objects).

### 4. `src/ucc/circuit/gate_data.h`

* **Massive Switch Statement Duplication:** Replace the 8 massive, duplicated switch statements (`gate_arity`, `is_clifford`, `is_measurement`, `is_reset`, `is_measure_reset`, `is_identity_noop`, `is_noise_gate`, `gate_name`) with a single `struct GateTraits`. Create a `constexpr std::array<GateTraits, static_cast<size_t>(GateType::UNKNOWN) + 1>` to centralize all gate properties. Update the 8 inline helper functions to simply return `kGateTraits[static_cast<size_t>(g)].property`.

### 5. `tests/test_frontend.cc`

* **Vacuous Tautologies in Tests:** In `TEST_CASE("Frontend: alias ZCX produces same HIR as CX")`, add `REQUIRE(hir_cx.num_ops() > 0);` before the `for` loop that iterates over `hir_cx.num_ops()`. This prevents the test from trivially passing if parsing failed and produced 0 nodes.
