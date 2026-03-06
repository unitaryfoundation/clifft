# UCC Implementation Plan: State-Agnostic Global Optimization (Lite)

## Executive Summary & Constraints

In standard DAG-based quantum compilers, optimization is constrained by rigid temporal graphs, hardware topology, and "Hadamard Gadgetization." UCC's Factored State Architecture bypasses all of this. Because the Front-End mathematically rewinds all physical non-Clifford operations and measurements back to the $t=0$ vacuum, chronological time is completely erased. The Heisenberg IR (HIR) is inherently a **timeless, global phase polynomial** perfectly aligned to the topological geometry of the physical circuit.

Because we do not use a DAG, we do not need rigid "temporal barriers" like clock ticks, classical logic, or measurement gates to prevent invalid optimizations. A $T$ gate can safely commute past a measurement *if and only if* their spatial geometries commute at $t=0$.

To establish the "compiler" narrative for Paper 1, we will implement **Peephole Fusion**. The optimizer will natively scan the HIR and algebraically cancel or fuse $T$ and $T^\dagger$ gates acting on the exact same virtual Pauli axis, regardless of how far apart they appeared in the original circuit.

**Strict Constraints:**
1. **State-Agnosticism:** The optimizer must manipulate the HIR purely using symplectic geometry on the mathematical masks (`destab_mask`, `stab_mask`). It must never require knowledge of an initial state, the VM's active dimensions, or physical time.
2. **No DAG/Time Assumptions:** Do not treat classical annotations or measurements as absolute barriers. The HIR has no concept of time. Commutation is defined *solely* by the $\mathcal{O}(1)$ symplectic inner product.
3. **Paper 1 Scope:** Advanced $\mathcal{O}(n^4)$ FastTODD / TOHPE linear algebra solvers are explicitly deferred to a future phase. Stick exactly to the adjacent peephole fusions defined below.

---

## Phase 1: HIR Expansion & Pass Manager

**Goal:** Expand the HIR to support arbitrary Clifford phase residuals and implement the optimizer infrastructure.

*   **Task 1.1 (HIR Update):** Update `OpType` in `src/ucc/frontend/hir.h` to include `CLIFFORD_PHASE`. Update the `HeisenbergOp` struct with a static factory method `make_clifford_phase(stim::bitword<64> destab, stim::bitword<64> stab, bool sign, bool is_dagger)` to represent an $S$ (`is_dagger = false`) or $S^\dagger$ (`is_dagger = true`) rotation applied to a specific Pauli mask.
*   **Task 1.2 (Pass Interface):** Create `src/ucc/optimizer/pass.h` and `src/ucc/optimizer/pass_manager.h` (and their `.cc` implementations).
    * Define a pure abstract base class `class Pass { public: virtual void run(HirModule& hir) = 0; virtual ~Pass() = default; };`.
    * Define `PassManager` with `add_pass(std::unique_ptr<Pass>)` and `run(HirModule& hir)` methods.
    * Update `src/ucc/CMakeLists.txt` to compile the new `optimizer/` directory files into the `ucc_core` target.
*   **Task 1.3 (Bindings Hookup):** In `src/python/bindings.cc`, inside the `compile()` function, replace the `(void)skip_optimizer;` comment block. If `!skip_optimizer`, instantiate the `PassManager`, add the soon-to-be-built `PeepholeFusionPass`, and run it on the `hir` before passing it to `ucc::lower()`.

## Phase 2: Symplectic Peephole Fusion

**Goal:** Scan the timeless HIR left-to-right to algebraically fuse $T$ gates on the same axis using pure symplectic geometry.

*   **Task 2.1 (The Sweep):** Implement `PeepholeFusionPass` (in `src/ucc/optimizer/peephole.h/.cc`). Wrap the logic in a `bool changed = true; while (changed) { changed = false; ... }` loop so nested mirror circuits fully resolve. Inside, iterate through the HIR nodes left-to-right. For each `T_GATE` at index $i$, scan ahead (index $j > i$) looking for another `T_GATE` with the exact same `destab_mask` and `stab_mask`.
    *   *Performance Note:* Do not call `std::vector::erase` during the nested loops to avoid $\mathcal{O}(N^2)$ memory shifting. Track deletions via a `std::vector<bool> deleted` mask and compact `hir.ops` at the very end of each while-loop iteration.
*   **Task 2.2 (The Commutation Barrier):** As you scan $j$ forward, determine if node $j$ blocks the $T$ gate at $i$ from commuting by checking for anti-commutation. If they anti-commute, **halt the forward scan immediately (`break`)**.
    *   *Standard Nodes (`T_GATE`, `CLIFFORD_PHASE`, `MEASURE`, `CONDITIONAL_PAULI`):* Use the symplectic inner product over the `stim::bitword` masks: `((op_i.destab_mask() & op_j.stab_mask()) ^ (op_i.stab_mask() & op_j.destab_mask())).popcount() % 2 != 0`. If `true` (odd), they anti-commute.
    *   *Noise Nodes (`NOISE`):* To keep the struct at 32-bytes, `NOISE` nodes store their geometric masks in a side-table. Look up the channels: `hir.noise_sites[static_cast<uint32_t>(op_j.noise_site_idx())].channels`. Evaluate the inner product between node $i$ and *every* `NoiseChannel` in that site. (Note: `NoiseChannel` uses raw `uint64_t`, so you must use `std::popcount((x1 & z2) ^ (z1 & x2))`). If node $i$ anti-commutes with *any* channel, it is blocked.
    *   *Classical Nodes (`DETECTOR`, `OBSERVABLE`, `READOUT_NOISE`):* These are purely classical tracking nodes with implicitly zeroed geometric masks. They mathematically commute with everything. Ignore them and continue scanning.
*   **Task 2.3 (Algebraic Fusion Math):** If a matching, commuting `T_GATE` is found at $j$:
    *   Calculate effective angles. A $T$ has angle 1, $T^\dagger$ has -1. If the mapped Pauli has a negative phase (`sign() == true`), the rotation direction is inverted ($e^{-i \frac{\pi}{8} (-Z)} = e^{+i \frac{\pi}{8} Z}$): `int eff = (op.is_dagger() ? -1 : 1) * (op.sign() ? -1 : 1);`
    *   Sum them: `int total = eff_i + eff_j;`
    *   If `total == 0`: They perfectly cancel ($T \cdot T^\dagger = I$). Mark both node $i$ and $j$ as deleted.
    *   If `total == 2` or `-2`: They fuse into an $S$ or $S^\dagger$. Replace node $i$ with a `CLIFFORD_PHASE` node (`is_dagger = (total == -2)`, `sign = false`) and mark node $j$ deleted.
    *   Set `changed = true; break;` so the outer loop restarts the sweep.
*   **DoD:** A native Catch2 C++ test constructs a `HirModule` with `T 0; DETECTOR; MEASURE Z; T 0`. The pass cleanly slides the $T$ past the classical `DETECTOR` and the commuting diagonal `MEASURE`, fusing the two $T$s into a `CLIFFORD_PHASE`.

## Phase 3: Back-End Lowering

**Goal:** Extend the Back-End compiler to geometrically compress the new `CLIFFORD_PHASE` nodes and emit them as VM bytecode.

*   **Task 3.1 (Add `emit_s_dag` helper):** In `src/ucc/backend/backend.cc`, below the existing `emit_s` helper, create an `emit_s_dag(ctx, trans_cum, trans_local, v)` helper. It must append `SQRT_Z_DAG` (Stim's $S^\dagger$ gate) to the transposed tableaux, and emit `make_array_s_dag(v)` if active, or `make_frame_s_dag(v)` if dormant.
*   **Task 3.2 (Expansion Safety):** In `backend.cc` `lower()`, add a switch case for `OpType::CLIFFORD_PHASE`. The structural routing must perfectly mirror the logic used for `T_GATE`.
    *   Call `map_to_virtual` and `compress_pauli`.
    *   If the result is dormant and the basis is `X_BASIS`, the compiler **must** still emit `OP_FRAME_SWAP` to route it to the top, append `OP_FRAME_H`, emit `OP_EXPAND` to double the array, and call `ctx.reg_manager.activate()`. (Applying an $S$ gate to $|+\rangle$ yields $|+i\rangle$, which breaks the dormant $|0\rangle_D$ invariant, requiring array expansion).
    *   If `!is_dormant` and the basis is `X_BASIS`, emit `OP_ARRAY_H`.
*   **Task 3.3 (Opcode Emission):** After handling the routing, determine the final phase direction: `bool phase_flip = op.is_dagger() ^ result.sign;`.
    *   If `is_dormant` (which at this point guarantees it's a `Z_BASIS` since `X_BASIS` was activated): emit `OP_FRAME_S_DAG` (if `phase_flip`) else `OP_FRAME_S`.
    *   If `!is_dormant`: emit `OP_ARRAY_S_DAG` (if `phase_flip`) else `OP_ARRAY_S`.
*   **DoD:** A Catch2 test feeds `H 0; T 0; T 0` through the full pipeline. The resulting bytecode successfully emits the array/frame updates for a single $S$ gate instead of two $T$ gates, without prematurely expanding the `peak_rank`.

## Phase 4: Optimization Verification & Oracles

**Goal:** Establish strict mathematical guarantees that the optimizer does not corrupt quantum probability.

*   **Task 4.1 (Tier 1: Exact Statevector Fuzzing):** Write a Python test in `test_structural_oracles.py` (or update `test_qiskit_aer.py`) that generates random 8-qubit Clifford+T circuits (using the existing `random_clifford_t_circuit` helper). Compile and execute them twice: once with `skip_optimizer=True` and once with `skip_optimizer=False`. Assert that `ucc.get_statevector()` yields identical complex arrays (fidelity > 0.9999).
*   **Task 4.2 (Tier 2: The Identity Uncompute Test):** Update `TestBoundedTMirrorFuzzer` in `test_structural_oracles.py`. Change `skip_optimizer=True` to `skip_optimizer=False`. Because the optimizer mathematically traverses the deep Clifford routing and the outer `while` loop handles nested pairs, it should perfectly cancel all $T$ and $T^\dagger$ gates across the mirror boundary. The resulting output program must have `peak_rank == 0` (zero runtime memory expansion).

---

## [DEFERRED] Future Phases (Paper 2)

*   **Phase 5:** Virtual Diagonalization & The Parity Table Bridge
*   **Phase 6:** The TOHPE Pass (Algorithm 2)
*   **Phase 7:** The op-T-mize Benchmark Suite
*   **Phase 8:** The FastTODD Upgrade (Algorithm 3)
