# AI Development Diary

Notes on interesting AI-assisted development work on UCC.

---

## 2026-02-26: The Teleportation T-Gate Bug

**PR:** [#46](https://github.com/unitaryfoundation/ucc-next/pull/46)

### The Bug

A teleportation identity circuit — prepare `H T |0>` on Q0, teleport to Q2 via a Bell pair, apply `T_DAG H` on Q2, measure — should always measure 0. Instead it failed ~15% of the time (1466/10000 shots). The bug only appeared when `peak_rank > 0` (i.e., a non-Clifford gate created a surviving GF(2) dimension through the teleportation). Replacing T with S (pure Clifford) made it pass immediately.

### The Debugging Journey

This was a multi-day, multi-agent debugging effort. The core difficulty was that the bug lived at the intersection of three systems: the Stim tableau simulator (front-end), the GF(2) basis algebra (back-end), and the Schrodinger VM's Pauli frame tracking (runtime).

**ucctrace tool.** Early on we built `ucctrace`, a CLI diagnostic tool that traces the full UCC pipeline for any `.stim` file: parsed AST, Heisenberg IR, compiled bytecode, and step-by-step SVM execution showing the `v[]` array and Pauli frame state after each instruction. This was essential — without it we were debugging blind. Usage: `ucctrace circuit.stim [-s seed]`. It revealed that seed 0 passed and seed 42 failed, with identical `v[]` arrays but different error frames after the AG_PIVOT measurements.

**Approach 1 — Phase polynomials in the constant pool.** An external LLM proposed tracking base phases for each basis vector through AG_PIVOTs using a new `AgPivotBasisUpdate` struct in the constant pool, evaluated O(rank^2) inside `op_ag_pivot`. This was mathematically correct but turned out to be a no-op for the actual divergence path because the mapped basis vectors had `z_mask=0x0` (no Z-components to apply phases to).

**Approach 2 — OP_BASIS_PHASE opcode.** A second attempt added a new opcode emitted after AG_PIVOT to apply accumulated phases. Same mathematical insight, same result: a no-op for this circuit because the phase mechanism targeted Z-components that didn't exist.

**The real root cause** turned out to be simpler and more subtle than either approach assumed. After AG_PIVOT frame changes, Stim's tableau tracking leaves "phantom" stabilizer bits in destab observables. The back-end's `GF2Basis::transform` correctly projects these bits out of the basis vectors (clearing `ag_stab_slot`), but the incoming T-gate observables from the front-end still carry them. So `find_in_span(0x6)` on a basis containing `[0x2]` failed — the phantom bit 2 (from `ag_stab_slot=2`) wasn't in the basis. The back-end incorrectly emitted `OP_BRANCH` instead of `OP_COLLIDE`, doubling `peak_rank` and leaving the `v[]` array trapped in an uncancelled superposition.

### The Fix (and a Fix for the Fix)

**Attempt 3 — projected_mask.** A third external LLM proposed a `projected_mask` that accumulates `(1 << ag_stab_slot)` at each AG_PIVOT and strips those bits from `beta` before the `find_in_span` lookup. This fixed the teleportation test but broke a different test (`peak_rank with interleaved T and M`) — the circuit `H 0, T 0, M 0, H 0, T 0` needs the second T to create a genuine new dimension on qubit 0, but the mask permanently blocked bit 0.

**Attempt 4 — "match span" heuristic.** We fed back the failure to the LLM with the failing test source and trace. It proposed a refinement: if stripping the mask reduces `beta` entirely to zero (but the original was non-zero), the bits represent a genuinely re-prepared qubit dimension — fall back to the unstripped `beta` and clear those bits from the mask. The intuition: phantom bits always piggyback on existing active dimensions (so stripping leaves a non-zero remainder), while legitimate re-use produces a beta that is *entirely* composed of masked bits (stripping gives zero).

This passed all 221 C++ tests and 129 Python tests.

### Takeaways

- **Build diagnostic tools early.** `ucctrace` paid for itself many times over. Without per-instruction visibility into `v[]` and the Pauli frame, we would have been stuck much longer.
- **Multi-agent iteration works.** The final fix took four LLM attempts across two agents, with human-curated feedback files bridging them. Each attempt narrowed the search space even when the code itself didn't work.
- **The failing test matters as much as the passing one.** The interleaved T/M test caught the over-correction immediately. Having a good existing test suite turned a subtle regression into an obvious failure.
- **Phantom bits are a general hazard.** Any system that maintains parallel representations of quantum state (Stim's tableau vs. UCC's GF(2) basis) will have synchronization bugs at frame boundaries. The `projected_mask` is a targeted fix; a more principled approach might have the front-end strip stabilizer components before emitting HIR ops.
