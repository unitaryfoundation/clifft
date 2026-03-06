# UCC Implementation Plan: WebAssembly Compiler Explorer

## Executive Summary & Constraints

The **UCC Compiler Explorer** is a WebAssembly-powered, client-side React application that visualizes the multi-level AOT compilation pipeline. It demonstrates how UCC natively absorbs Cliffords, optimizes the phase polynomial, and geometrically compresses multi-qubit entanglement to physically shrink the Virtual Machine's active memory footprint.

To ensure tight integration and avoid building UI against static mocks, we will execute this plan **"C++ First"**. We will instrument the pipeline, build the Wasm JSON bridge, and test it natively before writing any React.

**Strict Constraints:**
1. **Preserve the 32-Byte Invariant:** Do NOT add a `source_line` field to `Instruction` or `HeisenbergOp`. Source mapping MUST use parallel `source_map` vectors in the container modules (`HirModule` and `CompiledModule`).
2. **AstNode is Exempt:** Because `AstNode` is only used during parsing and is not constrained by L1 cache requirements, add `source_line` directly to it for simplicity.
3. **The Optimizer Desync Trap:** The Middle-End optimizer deletes and fuses operations using an `erase-remove` idiom. Any pass that mutates `hir.ops` MUST identically mutate `hir.source_map` to prevent catastrophic array desynchronization and UI crashes.
4. **Wasm & SIMD Isolation:** WebAssembly fundamentally lacks AVX-512 support. The Emscripten build must strictly target the 64-qubit template path (`W=64`) and exclude the `-mavx512f` flags being introduced in the Magic State benchmark.
5. **Modern JS Stack:** Use `bun`, `Vite`, `React`, `TypeScript`, and `@monaco-editor/react`.

---

## Phase 1: Introspection DRY & AST Source Tracking

**Goal:** Extract existing string-formatting logic for reuse, and tag the parser outputs with source lines.

* **Task 1.1 (Introspection Refactor):** Extract the string-formatting functions (`format_pauli_mask`, `op_type_to_str`, `opcode_to_str`, `format_instruction`) out of `src/python/bindings.cc` and into a new shared header/source pair: `src/ucc/util/introspection.h` and `introspection.cc`. Update the Python bindings to use this new file.
* **Task 1.2 (AST Line Tracking):** Add `uint32_t source_line = 0;` directly to the `AstNode` struct in `circuit.h`.
* **Task 1.3 (Parser Updates):** In `parser.cc`, update the line-by-line parsing loop to assign the current `line_num` to the emitted `AstNode`s. Note: During `REPEAT` unrolling, the `line_num` should correctly reflect the line inside the block being replayed (which the current logic already supports).
* **DoD:** The project compiles, the Python bindings format strings correctly using the shared utility, and `AstNode` natively knows its origin line.

## Phase 2: Pipeline Propagation & The Optimizer Trap

**Goal:** Thread source maps and Active Dimension ($k$) history through the Front-End, Middle-End, and Back-End without altering the hot-path 32-byte structs.

* **Task 2.1 (Parallel Maps & History):** Add `std::vector<std::vector<uint32_t>> source_map` to `HirModule` (an inner vector is needed because a fused optimization node maps to multiple original source lines). Add `std::vector<std::vector<uint32_t>> source_map` and `std::vector<uint32_t> active_k_history` to `CompiledModule`.
* **Task 2.2 (Front-End Trace):** In `frontend.cc`, update `trace()` to push `[node.source_line]` into `HirModule::source_map` for every emitted `HeisenbergOp`.
* **Task 2.3 (Middle-End Sync - CRITICAL):** In `peephole.cc`, update the `PeepholeFusionPass`. When fusing $T+T \to S$, concatenate the source lines of both $T$ gates into the surviving node's `source_map`. When executing the `erase-remove` compaction loop at the end of the pass, you MUST synchronously shift and truncate `hir.source_map` to perfectly match `hir.ops`.
* **Task 2.4 (Back-End Lowering):** In `backend.cc` (`lower()`), whenever an `Instruction` is emitted, push the current HIR node's source line vector to `CompiledModule::source_map`. Push the current `reg_manager.active_k()` to `active_k_history`.
* **DoD:** A native C++ Catch2 test verifies that compiling `H 0\nT 0\nT 0` correctly propagates source lines, and that the optimizer perfectly maintains `hir.ops.size() == hir.source_map.size()` after fusing the $T$ gates.

## Phase 3: WebAssembly Build System & JSON Bridge

**Goal:** Create an Emscripten target exposing the compiler and SVM to JS, returning real AOT data protected by safe memory bounds.

* **Task 3.1 (Embind API):** Create `src/wasm/bindings.cc` using `#include <emscripten/bind.h>`. Expose:
    * `compile_to_json(source)`: Calls `ucc::parse(source, /*max_ops=*/10000)` to prevent malicious `REPEAT` blocks from crashing the browser. Traces, optimizes (via `PassManager`), and lowers the circuit. Uses the introspection utilities to serialize the HIR, Bytecode, Source Maps, and Active $k$ history into a single JSON string.
    * `simulate_wasm(source, shots)`: Parses and compiles the circuit. **Safety Guard:** If `prog.peak_rank > 20` (requiring ~16 MB of continuous RAM), instantly abort and return a JSON error (`{"error": "MemoryLimitExceeded"}`). If safe, execute the VM natively in Wasm, aggregate the measurement histogram, and return it.
* **Task 3.2 (CMake Wasm Target):** Create `src/wasm/CMakeLists.txt` configured for Emscripten (`emcc`). Ensure it overrides native architecture flags (`-march=native`, `-mavx512f`) to prevent Wasm compile errors.
* **Task 3.3 (Build Automation):** Add a `just build-wasm` recipe that utilizes an official `emscripten/emsdk` docker container to build the target and output `ucc_wasm.js` and `ucc_wasm.wasm` directly into a new `explorer/public/` directory.
* **DoD:** Running `just build-wasm` works. A 10-line Node.js/Bun script can require the Wasm module, call `compile_to_json("H 0\nT 0\nM 0")`, and print the JSON payload showing localized RISC opcodes and source maps.

## Phase 4: Frontend Scaffolding, Layout, & Highlighting

**Goal:** Setup the modern React architecture and build the interactive UI layout using synchronized Monaco editors.

* **Task 4.1 (Initialization & Dependencies):** Scaffold the app using `bun create vite explorer --template react-ts`. Run `bun add @monaco-editor/react allotment recharts lz-string lucide-react`.
* **Task 4.2 (Wasm Interop Hook):** Create `src/hooks/useUccWasm.ts`. This hook asynchronously loads the Emscripten module and safely exposes `compile_to_json` and `simulate_wasm` to React.
* **Task 4.3 (The IDE Layout):** Use `<Allotment>` to construct a resizable workspace:
    * **Top Split (3 Columns):**
        1. **Source Input:** Editable Monaco instance.
        2. **HIR View:** Read-only Monaco instance for the abstract phase polynomial.
        3. **VM Bytecode:** Read-only Monaco instance for the localized RISC instructions.
    * **Bottom Split (Telemetry & Output):**
        1. **Active Dimension Graph:** (`recharts` step-line graph).
        2. **Simulation Results:** (Placeholder for the Monte Carlo histogram).
* **Task 4.4 (Live Compilation):** Wire a debounced effect (e.g., 200ms) to the Source Input. On change, call `compile_to_json`. Update the text models of the HIR and Bytecode Monaco instances, and store the returned `source_map` arrays in React state.
* **Task 4.5 (Bidirectional Highlighting):** Track a `hoveredSourceLine` in React state.
    * Wire the Source editor's `onDidChangeCursorPosition` event to update this state.
    * When it changes, scan the JSON `source_map` arrays returned by the compiler.
    * Use Monaco's `editor.createDecorationsCollection()` on the HIR and Bytecode editors to apply a subtle background highlight CSS class to the exact lines that correspond to the active source code.

## Phase 5: The Active Dimension ($k$) Timeline

**Goal:** Visualize the literal memory expansion and compaction of the Virtual Machine over time.

* **Task 5.1 (Graphing):** Use `recharts` to plot the JSON `active_k_history` array on the Y-axis against the Bytecode PC (Instruction Index) on the X-axis using a `stepAfter` digital line style.
* **Task 5.2 (Visual Feedback):** Users can literally watch the graph step up when an un-absorbed non-Clifford gate executes (`OP_EXPAND`) and watch it fold back down during array compaction (`OP_MEAS_ACTIVE_INTERFERE`). Add a red reference dashed line at $Y=20$ labeled "Browser Memory Limit (~16MB)".

## Phase 6: The Simulation UI Sandbox & Deep Linking

**Goal:** Allow users to execute fast Monte Carlo simulations directly in the browser and share them.

* **Task 6.1 (UI Controls & Execution):** Add a "Simulate (10k Shots)" button above the Source Editor. When clicked, invoke `simulate_wasm(source, 10000)` via the custom hook.
    * If it returns `"MemoryLimitExceeded"` (because $k > 20$), render a prominent red alert banner in the Simulation Results pane advising the user to use the native Python CLI.
* **Task 6.2 (Output Visualization):** If successful, parse the returned histogram and render a `recharts` BarChart or stylized list in the Simulation Results pane showing the probability distribution of the measurements.
* **Task 6.3 (URL Hydration):** In the main `App.tsx` mount effect, check the window location for a `?code=` query parameter. If present, decode and decompress it using `LZString.decompressFromEncodedURIComponent` and set it as the initial string for the Monaco editor.
* **Task 6.4 (State Sync & Share):** Add a "Share Link" button with a link icon to the toolbar. On click, capture the current editor text, compress it using `LZString.compressToEncodedURIComponent`, append it to the current URL, and copy the full URL to the user's clipboard. Disable the button visually if the circuit exceeds ~5,000 characters.
