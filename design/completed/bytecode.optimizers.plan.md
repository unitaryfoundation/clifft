# UCC Implementation Plan: AOT Bytecode Optimizations

## Executive Summary & Constraints

This phase targets the runtime Virtual Machine (VM) execution bottlenecks. Profiling shows that >60% of execution cycles are spent in memory-bandwidth-bound array operations (`OP_ARRAY_MULTI_CNOT` and `OP_PHASE_T`), while CPU pipeline branch mispredictions and opcode dispatch overhead drag down the rest.

Because UCC maps the multi-qubit physical routing to the virtual coordinate frame Ahead-Of-Time (AOT), the VM executes over a dense $2^k$ array (where $k_{\max} < 30$ in practical FTQC simulation). This allows us to use 32-bit masks inside the bytecode payload to completely flatten loop structures.

**Strict Constraints & Architectural Rules:**

1. **The 32-Byte Invariant:** Every new fused opcode MUST fit perfectly within the existing 32-byte `Instruction` struct. Utilize the 24-byte payload union to pack multiple parameters inline.
2. **Strict Sequential Pass Order (No Fixed-Point Loops):** Bytecode optimizations are localized peephole passes over a flat array, not deep algebraic DAG solvers. Do **not** use `while(changed)` loops. Each pass must run exactly once in a strictly defined $\mathcal{O}(N)$ sequence (e.g., Compaction $\to$ Bubbler $\to$ Fusion).
3. **Memory Domain Commutation (The "Bubbler" Principle):** `OP_FRAME_*` instructions strictly mutate the classical 64-bit Pauli trackers (`p_x` and `p_z`). `OP_ARRAY_*` instructions strictly mutate the complex active statevector `v[]`. Because they operate on physically disjoint memory domains, they mathematically commute. This allows us to safely "bubble" frame updates past array operations to group math operations together, provided we don't cross a stochastic barrier (like a measurement).

---

## Phase 1: Micro-Optimization — The Branchless Parity Loop

**Goal:** Eliminate the catastrophic ~50% branch misprediction penalty in the VM's heaviest hotspot (`exec_array_multi_cnot` and array swaps) by forcing the compiler to emit $\mathcal{O}(1)$ Conditional Move (`cmov`) instructions.

* **Task 1.1 (Remove `if` branches):** In `svm.cc`, locate the inner loop of `OP_ARRAY_MULTI_CNOT`. Remove the `if (parity) { std::swap(...); }` block.
* **Task 1.2 (Branchless Ternary):** Implement the swap using unconditional loads and ternary assignments based on the $\mathcal{O}(1)$ hardware `popcount`:
```cpp
// 1-instruction hardware parity
bool parity = std::popcount(idx_0 & ctrl_mask) & 1;

// Unconditional loads
auto val0 = v[idx_0];
auto val1 = v[idx_1];

// Ternary forces compiler to emit branchless CMOV
auto new_val0 = parity ? val1 : val0;
auto new_val1 = parity ? val0 : val1;

v[idx_0] = new_val0;
v[idx_1] = new_val1;

```


* **DoD:** Profiling the circuit shows CPU branch misses drop to near-zero in the hot loop, and the overall execution time of `exec_array_multi_cnot` strictly decreases.

## Phase 2: Classical Feedforward Compaction

**Goal:** Eliminate opcode dispatch overhead by fusing 1-to-1 classical syndrome tracking chains.

* **Task 2.1 (Opcode Definition):** In `svm.h` (or your opcode enum file), define `OP_MEAS_DORMANT_RANDOM_AND_APPLY_PAULI` and `OP_MEAS_DORMANT_STATIC_AND_APPLY_PAULI`.
* **Task 2.2 (Payload Union):** Add a struct to the 24-byte `Instruction` union to hold both operands:
```cpp
struct {
    uint32_t rec_idx;
    uint32_t cp_mask_idx;
    uint8_t _pad[16]; // Explicit padding to 24 bytes
} meas_and_pauli;

```


* **Task 2.3 (Compiler Pass):** Implement `pass_compact_classical(std::vector<Instruction>& ops)`. Sweep the bytecode linearly. If an `OP_MEAS_DORMANT_*` is immediately followed by an `OP_APPLY_PAULI`, and the Pauli's condition directly matches the measurement's `rec_idx`, replace both with the fused opcode.
* **Task 2.4 (VM Execution):** In `svm.cc`, implement the fused opcode handler. Generate the measurement result, write it to `meas_record[rec_idx]`, and if the bit is `1`, immediately fetch the mask from the Constant Pool and XOR it into `p_x` and `p_z` without returning to the dispatch loop.
* **DoD:** Feedforward sequences in the trace collapse into single instructions, reducing the total instruction count by ~10-15%.

## Phase 3: Code Motion (The Bubbler)

**Goal:** Group heavy array operations together by sliding zero-cost frame updates upwards. This ensures the Pauli frame is correctly updated before phase math is applied, while unlocking adjacent fusion for array ops.

* **Task 3.1 (Upward Bubbling):** Implement `pass_bubble_frames(std::vector<Instruction>& ops)`. Iterate linearly through the bytecode from left to right. When you encounter an `OP_FRAME_*` instruction (e.g., `FRAME_CNOT`), look at the instruction immediately preceding it.
* **Task 3.2 (Commutation Rules):**
* If the preceding instruction is an array mutator (like `OP_ARRAY_MULTI_CNOT`), they commute because of disjoint memory domains. Swap their positions in the bytecode.
* Continue bubbling the `OP_FRAME_*` instruction upward until it hits an instruction it cannot safely commute past (e.g., a stochastic barrier like `MEASURE`, `NOISE`, `APPLY_PAULI`, or another frame operation).


* **DoD:** Given the bytecode sequence `[ARRAY_MULTI_CNOT target=0, FRAME_CNOT target=0, PHASE_T target=0]`, the bubbler cleanly transforms it to `[FRAME_CNOT target=0, ARRAY_MULTI_CNOT target=0, PHASE_T target=0]`, making the two array operations perfectly adjacent.

## Phase 4: Phase Gadget Memory Squeezing

**Goal:** Halve the memory bandwidth of phase gadgets by evaluating the multi-controlled parity and the $T$-phase rotation in a single, branchless pass over the array.

* **Task 4.1 (Opcode Definition):** Define `OP_FUSED_MULTI_CNOT_PHASE`. The target axis goes in `axis_1`. Encode the phase type (`T` vs `T_DAG`) in the instruction `flags`. The `ctrl_mask` goes in the payload union's `uint32_t`.
* **Task 4.2 (Compiler Fusion):** Implement `pass_fuse_array_phases(std::vector<Instruction>& ops)`. Find adjacent `OP_ARRAY_MULTI_CNOT` and `OP_PHASE_T` (or `DAG`) instructions targeting the same axis. Fuse them into the new opcode.
* **Task 4.3 (Single-Pass Execution):** In `svm.cc`, implement the execution loop.
```cpp
// 1. Pre-calculate phase direction based on Pauli frame tracker p_x[target]
complex<double> active_phase = ...;

// 2. Run the branchless inner loop from Phase 1
// ... (calculate parity, val0, val1, new_val0, new_val1)

// 3. Apply the phase strictly to the |1> target branch before writing:
v[idx_0] = new_val0;
v[idx_1] = new_val1 * active_phase;

```


* **DoD:** Profiling shows `exec_array_multi_cnot` and `exec_phase_t` disappear, replaced by a single fused execution block taking ~50% of their combined historical execution time.

## Phase 5: Active Star-Graph Fan-Out

**Goal:** Collapse sequential 2-qubit CNOTs sharing a control into a single $\mathcal{O}(2^k)$ loop pass.

* **Task 5.1 (Opcode Definition):** Define `OP_ARRAY_MULTI_TARGET_CNOT`. The control axis goes in `axis_1`. Use the payload union to store `uint32_t target_mask`.
* **Task 5.2 (Compiler Fusion):** Implement `pass_fuse_star_graphs(std::vector<Instruction>& ops)`. Sweep the bytecode for sequential `OP_ARRAY_CNOT` instructions that share the exact same control (`axis_1`). Accumulate `(1 << axis_2)` into a `target_mask`. Replace the block with the single multi-target opcode.
* **Task 5.3 (HSB Execution Trick):** In `svm.cc`, implement the loop using the Highest Set Bit (HSB) trick to guarantee elements are swapped exactly once:
```cpp
uint32_t hsb = 31 - std::countl_zero(target_mask);
for (uint32_t i = 0; i < v_size; i++) {
    if ((i & (1 << hsb)) == 0) { // Only process bottom half of the HSB dimension
        if (i & (1 << ctrl)) {   // Only swap if control is 1
            std::swap(v[i], v[i ^ target_mask]);
        }
    }
}

```


* **DoD:** A sequence of 9 `ARRAY_CNOT` instructions originating from the same control compiles down to 1 instruction, requiring only a single pass over the complex array.

## Phase 6: Pipeline Integration & Validation

**Goal:** Enforce strict sequential execution of the passes and validate mathematical equivalence.

* **Task 6.1 (Pipeline Assembly):** In `backend.cc` (or where the bytecode vector is finalized), explicitly call the passes in strict order:
```cpp
void optimize_bytecode(std::vector<Instruction>& ops) {
    pass_compact_classical(ops); // 1. Clean up scalar feedforward
    pass_bubble_frames(ops);     // 2. Clear gaps between Array/Math ops
    pass_fuse_array_phases(ops); // 3. Squeeze Phase Gadgets
    pass_fuse_star_graphs(ops);  // 4. Compress Star-Graphs
}

```


* **Task 6.2 (Validation):** Run the complete Catch2 mathematical equivalence test suite. The dense statevector amplitudes and classical Pauli tracking must identically match the pre-optimization output.
