# UCC Implementation Plan: Compiler Explorer

## Executive Summary & Constraints
You are building the **UCC Compiler Explorer**: a WebAssembly-powered, client-side React application that visualizes the multi-level Ahead-of-Time (AOT) compilation pipeline of the Unitary Compiler Collection (UCC).

Because UCC fundamentally decouples the $\mathcal{O}(n^2)$ topological tracking (Heisenberg picture) from the $\mathcal{O}(1)$ probabilistic execution (Schrödinger picture), standard 2D circuit diagrams are insufficient for understanding program flow. The Explorer provides an interactive environment allowing algorithm and hardware designers to write `.stim` code and instantly visualize how it lowers into the Heisenberg IR (HIR) and final hardware-aligned bytecode.

Crucially, it features bidirectional highlighting and a **Memory Profiler Timeline** that graphs the literal GF(2) memory rank of the Virtual Machine over the chronological execution of opcodes, allowing researchers to watch their quantum circuit "breathe" memory.

### Architectural Prerequisites
This plan relies on the core AOT architecture. Do not begin this plan until the following pipeline stages are complete and tested natively in C++:
1. **The Circuit AST & Parser:** Can parse `.stim` superset syntax into `ucc::Circuit`.
2. **The Front-End:** Drives `stim::TableauSimulator`, absorbs Cliffords, and emits `HeisenbergOp` nodes.
3. **The Back-End:** Tracks the GF(2) basis $V$, computes spatial shifts ($\beta$), and emits 32-byte `Instruction` bytecode and `peak_rank`.
4. **The SVM Runtime:** Can execute bytecode and manipulate the dense `v[]` array mathematically.

*(Note: The Middle-End Optimizer is NOT a strict prerequisite. If it is currently stubbed out, the "Optimized HIR" pane will simply mirror the "Raw HIR" pane until the optimizer is fully implemented).*

### Strict Constraints (Overrides general guidelines if in conflict)
1. **Preserve the 32-Byte Invariant:** You MUST NOT add a `source_line` field to the `ucc::AstNode`, `ucc::HeisenbergOp`, or `ucc::Instruction` structs. Source mapping MUST be implemented via parallel `source_map` arrays owned by their respective container modules (`Circuit`, `HirModule`, `Program`).
2. **Zero Runtime Overhead:** Tableau snapshotting and source tracking must be guarded by boolean flags in a new `CompilerOptions` struct. When false, the native C++ AOT pipeline must execute exactly as fast as it did before, with zero memory or compute penalty.
3. **Wasm Isolation:** The C++ core (`src/ucc/`) must remain completely unaware of JSON, Emscripten, or WebAssembly. All Wasm bindings and serialization logic must live exclusively in a new `src/wasm/` directory.
4. **Browser Rank Protection:** The "Simulate" feature running in the browser must hard-abort and return an error if `peak_rank > 20` to prevent WebAssembly Out-Of-Memory (OOM) browser crashes.
5. **Modern JS Stack:** Use `bun` (as the JS package manager and runtime), `Vite`, `React`, `TypeScript`, `@monaco-editor/react`, `recharts` (for the timeline), and `allotment` (for pane splitting).

---

## Part 1: Core C++ Source Mapping & Snapshots
**Goal:** Instrument the AOT pipeline to track source lines, rank history, and extract Tableau states without altering hot-path data structures.
*   **Task 1.1 (Compiler Options):** Create `src/ucc/util/options.h` containing `struct CompilerOptions { bool enable_snapshots = false; bool track_source_lines = false; };`. Plumb this into the compilation pipeline interfaces (parser, frontend, backend).
*   **Task 1.2 (Parallel Data Vectors):**
    *   Add `std::vector<uint32_t> source_map` to `ucc::Circuit` (maps AST node index $\to$ 1-based source line).
    *   Add `std::vector<std::vector<uint32_t>> source_map` to `ucc::HirModule` and `ucc::Program` (maps operation/instruction index $\to$ list of originating source lines). *(It is a vector of vectors because Middle-End passes will eventually fuse operations from multiple lines).*
    *   Add `std::vector<uint32_t> rank_history` to `ucc::Program`.
*   **Task 1.3 (Pipeline Propagation):**
    *   Update `parser.cc` to track `line_num` and populate `Circuit::source_map` if tracking is enabled.
    *   Update the Front-End to pass the `source_map` from the AST to the `HirModule` when emitting `HeisenbergOp`s.
    *   Update the Back-End to pass the `source_map` from the `HirModule` to the `Program` when emitting `Instruction`s.
    *   As the Back-End iterates through the HIR, it must push the current `active_rank` of the `GF2Basis` into `Program::rank_history` for *every* emitted instruction.
*   **Task 1.4 (Tableau Snapshots):** Add `std::map<uint32_t, std::string> tableau_snapshots` to `ucc::HirModule`. In the Front-End, if `options.enable_snapshots == true`, stringify the current $X$ and $Z$ stabilizers (using `sim.inv_state.str()` or equivalent) after processing the operations for a given line, and store it keyed by `source_line`.
*   **Definition of Done (DoD):** C++ Catch2 tests verify that parsing a multi-line circuit correctly populates the source maps and rank history, and that `tableau_snapshots` contains valid stringified matrices only when `enable_snapshots` is true. `sizeof(Instruction)` remains exactly 32 bytes.

## Part 2: WebAssembly Build System & Serialization
**Goal:** Create an Emscripten build target that compiles the UCC core to WebAssembly, wrapping it with a JSON API.
*   **Task 2.1 (CMake & Dependencies):** Create `cmake/FetchNlohmannJson.cmake` to fetch `nlohmann/json`. Create `src/wasm/CMakeLists.txt` configured to build a Wasm module. It must link `ucc_core` and `nlohmann_json`, compile using Emscripten flags (e.g., `-lembind`), and only be included in the top-level CMake if `EMSCRIPTEN` is true.
*   **Task 2.2 (The Wasm API):** Create `src/wasm/serialize.h/.cc` and `src/wasm/bindings.cc` using `<emscripten/bind.h>`. Expose:
    *   `std::string compile_to_json(std::string source)`: Runs the pipeline with `enable_snapshots = true` and `track_source_lines = true`. Serializes the `Circuit`, `HirModule`, `Program` (including their source maps and `rank_history`), and `tableau_snapshots` into a single JSON string.
    *   `std::string simulate_wasm(std::string source, int shots, int seed)`: Checks if `peak_rank > 20`. If so, returns `{"error": "Rank too high for browser. Limit is 20."}`. If not, runs `SchrodingerState` natively and returns a measurement histogram (e.g., `{"results": {"00": 498, "11": 502}}`).
*   **Task 2.3 (Justfile Automation):** Add two recipes to `justfile`:
    *   `build-wasm-local`: Runs `emcmake cmake` and `cmake --build` using a local `emsdk`.
    *   `build-wasm-docker`: Runs the exact same commands inside the official `emscripten/emsdk:latest` Docker container.
    *   *Both* commands must include a post-build step to copy `ucc_wasm.js` and `ucc_wasm.wasm` into the `explorer/public/wasm/` directory.
*   **DoD:** Running `just build-wasm-docker` successfully produces `ucc_wasm.js` and `.wasm` in the target directory without requiring local toolchains.

## Part 3: Frontend Scaffolding & Layout
**Goal:** Set up a modern React/Vite/Bun web project.
*   **Task 3.1 (Initialization):** Create the `explorer/` directory. Run `bun create vite . --template react-ts` inside it.
*   **Task 3.2 (Dependencies):** Run `bun add @monaco-editor/react allotment lucide-react recharts`.
*   **Task 3.3 (Wasm Interop):** Write a React hook `useUccWasm.ts` that asynchronously loads `ucc_wasm.js` from the `/wasm/` public directory, initializes the Emscripten module, and exposes the C++ functions to React.
*   **Task 3.4 (Layout Scaffolding):** In `App.tsx`, use `<Allotment>` to create the master layout:
    *   **Top Section (Horizontal Split):** Editor (Input), HIR View (Output), Bytecode View (Output).
    *   **Bottom Section (Horizontal Split):** Memory Profiler Timeline (Wide bottom pane), Tableau Frame Inspector (Collapsible pane).
*   **DoD:** Running `bun install` and `bun run dev` opens the multi-pane layout on `localhost:5173`. The browser console confirms the Wasm module loads successfully.

## Part 4: Data Rendering & Bidirectional Highlighting
**Goal:** Make the Explorer fully interactive, linking source code to compiled artifacts.
*   **Task 4.1 (Monaco Integration):** Embed Monaco `<Editor>` instances in the top panes. Set up a React `useEffect` that calls `compile_to_json(code)` whenever the input editor text changes (debounced by 200ms). Parse the JSON and format the HIR and Bytecode panes as text.
*   **Task 4.2 (Hover & Decorators):** Use Monaco's `editor.deltaDecorations` API. When the user clicks or hovers over line $X$ in the Input Editor, highlight all corresponding lines in the HIR and Bytecode panes by checking their `source_map` JSON arrays. Ensure highlighting works in reverse (clicking an opcode highlights the source).
*   **Task 4.3 (Tableau Pane):** Wire the Tableau Frame Inspector to display the stringified `tableau_snapshots` matrix for the currently active line $X$ in the Input Editor.
*   **DoD:** Typing a $T$ gate in Pane 1 instantly shows `OP_BRANCH` in the Bytecode pane. Clicking the $T$ gate highlights the `OP_BRANCH` line. Clicking a preceding $H$ gate updates the Tableau Inspector to show the flipped $X$ and $Z$ stabilizers.

## Part 5: Memory Profiler Timeline & Sandbox Simulation
**Goal:** Visualize the continuous expansion and contraction of the state vector and allow safe, in-browser Monte Carlo simulation.
*   **Task 5.1 (Timeline Graph):** In the bottom-left pane, render an `<AreaChart>` or `<LineChart>` using `recharts`.
    *   **Data Mapping:** Plot the `rank_history` array on the Y-axis against the **Bytecode Instruction Index** on the X-axis.
    *   **Style:** Use the `type="stepAfter"` line property so the graph renders as discrete digital steps reflecting exact memory allocations.
    *   **Visual Warning:** Add a horizontal red `<ReferenceLine y={20} stroke="red" label="Browser OOM Limit" />`.
*   **Task 5.2 (Timeline Interactivity):** Wire the chart's custom `<Tooltip>` and `onMouseMove` events. Hovering over data point $i$ must look up `bytecode_source_map[i]`, display the source line and opcode name in the tooltip, and instantly use the Monaco API to highlight the originating source line in the Input Editor.
*   **Task 5.3 (Simulation Sandbox):** Add a "Simulate 10k Shots" button above the Input Editor. When clicked, call `simulate_wasm`. Display the returned JSON `"results"` histogram as a simple list or HTML bar chart.
*   **Task 5.4 (Safety Guard):** If the JSON contains an `"error"` (rank exceeded), display it prominently in red, instructing the user to use the native C++ CLI for massive simulations.
*   **DoD:** A user can type a sequence of $T$ gates and watch the chart climb in a step-function format. Hovering over a jump in the graph highlights the exact $T$ gate in the source code that caused the expansion.

---

## Appendix: Demonstration Use Cases to Verify

Once implementation is complete, use these exact scenarios in the UI to manually verify the tool satisfies the core design requirements:

### Use Case 1: Defeating the Memory Wall (Rank Profiling)
Paste the following code into the Editor:
```stim
H 0 1 2
T 0
T 1
T 2
```

*Verification:* The Bytecode pane emits three OP\_BRANCH instructions. The Rank Timeline steps up from 0 to 3\.

Now, add M 0 to the end of the file.

*Verification:* The final instruction emitted is OP\_MEASURE\_MERGE. The Rank Timeline should show a step *down* from 3 to 2, visually proving to the user that parity measurements physically collapse the active rank.

### **Use Case 2: 1-to-N Expansion Interactivity**

Paste the following code:

```stim
H 0 1 2
T 0 1 2
```

*Verification:* Line 2 (T 0 1 2\) expands into three distinct OP\_BRANCH instructions. Hovering over the second step in the Timeline graph highlights *both* the middle OP\_BRANCH instruction and Line 2 of the source code.

### **Use Case 3: The "Where did my Cliffords go?" Epiphany**

Paste the following code:

```stim
H 0
CX 0 1
T 1
```

*Verification:* The HIR and Bytecode panes are completely empty for the first two lines (Cliffords are absorbed, they emit zero opcodes). The X-axis of the timeline does not move. However, clicking line 2 (CX 0 1\) updates the Tableau Inspector pane, showing entanglement forming in the $X\_0X\_1$ stabilizers. This proves to the user that the compiler mathematically absorbed the Cliffords rather than skipping them.

### **Use Case 4: Browser Safety Limits**

Paste the following code:

```stim
H 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
T 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
```
*Verification:* The Rank Timeline graph bursts through the red $Y=20$ reference line, peaking at 22\. Pressing the "Simulate" button immediately shows an error message: "Rank too high for browser memory. Limit is 20" without crashing the browser tab.
