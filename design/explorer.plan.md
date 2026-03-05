# UCC Implementation Plan: WebAssembly Compiler Explorer

## Executive Summary & Constraints

The **UCC Compiler Explorer** is a WebAssembly-powered, client-side React application that visualizes the multi-level AOT compilation pipeline.

**Strict Constraints:**
1. **Preserve the 32-Byte Invariant:** Do NOT add a `source_line` field to `Instruction` or `HeisenbergOp`. Source mapping MUST use parallel `source_map` arrays in the container modules.
2. **Zero Runtime Overhead:** Source tracking must be guarded by a `CompilerOptions` flag.
3. **Wasm Isolation:** All Emscripten and JSON logic MUST live exclusively in `src/wasm/`. The core C++ must remain pristine.
4. **Modern JS Stack:** Use `bun`, `Vite`, `React`, `TypeScript`, and `@monaco-editor/react`.

---

## Part 1: Core C++ Source Mapping

**Goal:** Instrument the AOT pipeline to track source lines and the Active Dimension ($k$) without altering hot-path structs.

* **Task 1.1 (Parallel Maps):** Add `std::vector<uint32_t> source_map` to `Circuit` and `std::vector<std::vector<uint32_t>> source_map` to `HirModule` and `Program`.
* **Task 1.2 (Active Dimension Tracking):** Add `std::vector<uint32_t> active_k_history` to `Program`. The Back-End must push the current $k$ dimension into this array for every emitted opcode.
* **Task 1.3 (Pipeline Propagation):** Update the parser, Front-End, and Back-End to pass source maps through the pipeline when `options.track_source_lines = true`.

## Part 2: WebAssembly Build System

**Goal:** Create an Emscripten target exposing the compiler to JS.

* **Task 2.1 (Embind API):** Create `src/wasm/bindings.cc`. Expose:
    * `compile_to_json(source)`: Returns the HIR, Bytecode, Source Maps, and Active $k$ history as a JSON string.
    * `simulate_wasm(source, shots)`: Executes the VM natively in Wasm. Hard-abort and return an error if the peak $k > 16$ to prevent browser Out-Of-Memory (OOM) crashes.
* **Task 2.2 (Build Automation):** Add a `just build-wasm-docker` recipe that compiles the module inside an official Emscripten docker container and outputs to the React `public/` folder.

## Part 3: Frontend Scaffolding, Layout, & Highlighting

**Goal:** Setup the modern React architecture and build the interactive UI layout.

* **Task 3.1 (Initialization & Dependencies):** Scaffold the app using `bun create vite explorer --template react-ts`. Run `bun add @monaco-editor/react allotment recharts lz-string lucide-react` to install the core UI packages.
* **Task 3.2 (Wasm Interop Hook):** Create `src/hooks/useUccWasm.ts`. This hook is responsible for asynchronously loading the Emscripten `.js` glue code and `.wasm` binary, initializing the module, and safely exposing the `compile_to_json` and `simulate_wasm` C++ functions to React state.
* **Task 3.3 (Pane Layout):** Use `<Allotment>` to construct a resizable multi-pane layout:
    * **Top Split:** Input Editor (Monaco), HIR View (Read-only Text/JSON), and Bytecode View (Read-only Text).
    * **Bottom Split:** The Active Dimension Timeline graph.
* **Task 3.4 (Bidirectional Highlighting):** Wire Monaco's `editor.deltaDecorations`. When the user clicks a line in the Input Editor, use the JSON `source_map` to highlight the corresponding abstract Paulis in the HIR pane and the localized RISC opcodes in the Bytecode pane.

## Part 4: The Active Dimension ($k$) Timeline

**Goal:** Visualize the literal memory expansion and compaction of the Virtual Machine.

* **Task 4.1 (Graphing):** Use `recharts` to plot `active_k_history` on the Y-axis against the Bytecode PC on the X-axis using a `stepAfter` digital line style.
* **Task 4.2 (Visual Feedback):** Users can literally watch the graph step up when a `T` gate executes (`OP_EXPAND`) and watch it fold back down during array compaction (`OP_MEAS_ACTIVE_INTERFERE`). Add a red reference line at $Y=16$ labeled "Browser Memory Limit".

## Part 5: The Simulation UI Sandbox

**Goal:** Allow users to execute fast, safe Monte Carlo simulations directly in the browser and visualize the output.

* **Task 5.1 (UI Controls):** Add a "Simulate (10k Shots)" button to a toolbar above the Input Editor.
* **Task 5.2 (Execution & Safety Guard):** When clicked, invoke `simulate_wasm` via the custom hook. If the returned JSON contains an `"error"` (e.g., peak $k$ exceeded 16), render a prominent red alert banner warning the user about browser memory limitations and advising them to use the native C++ CLI for heavy workloads.
* **Task 5.3 (Results Visualization):** If successful, parse the returned JSON results and display the measurement histogram. Use a simple `recharts` BarChart or a stylized HTML list to show the distribution of classical states.

## Part 6: Deep Linking & URL State

**Goal:** Allow users to share specific circuit snippets via URL without a backend database.

* **Task 6.1 (URL Hydration):** In the main `App.tsx` mount effect, check the window location for a `?code=` query parameter. If present, decode and decompress it using `LZString.decompressFromEncodedURIComponent` and set it as the initial string for the Monaco editor.
* **Task 6.2 (State Sync & Share):** Add a "Share Link" button with a link icon to the toolbar. On click, capture the current editor text, compress it using `LZString.compressToEncodedURIComponent`, append it to the current URL, and copy the full URL to the user's clipboard with a temporary "Copied!" toast notification.
* **Task 6.3 (Abuse Limits):** Track the character count in the Monaco editor. If the code exceeds 5,000 characters, visually disable the "Share Link" button and add a tooltip explaining that the circuit is too large for URL encoding.

## Part 7: (Optional) The Virtual Frame Inspector

**Goal:** Teach users how $\mathcal{O}(n)$ geometric compression works.

* **Task 7.1:** If enabled, the Back-End records stringified snapshots of the $V_{cum}$ Pauli tracker.
* **Task 7.2:** When a user hovers over a heavy multi-qubit operator in the HIR, the inspector pane visually displays how the Back-End emitted `OP_FRAME_CNOT` gates to dynamically fold that global operator into a single virtual axis.

> **STATUS: DEFERRED (POST-PAPER 1)**
> This plan is explicitly excluded from the debut UCC paper to maintain narrative focus and bound engineering time. The core narrative of Paper 1 relies on the fact that FTQC measurements naturally cool the active dimension ($k_{\max}$). Coherent noise introduces continuous rank explosion that contradicts this clean narrative. This will form the basis of a dedicated follow-up paper.
