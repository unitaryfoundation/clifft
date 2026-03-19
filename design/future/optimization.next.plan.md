# UCC Implementation Plan: Architecture Cleanup & Statevector Squeezing

## Executive Summary

This plan outlines the implementation of a globally state-agnostic optimization pass for the UCC compiler. To minimize the "active spacetime volume" (\int 2^{k(t)} dt) evaluated by the Virtual Machine, we will implement a Bidirectional Statevector Squeeze.
Because UCC's Front-End rewinds all operations to the t=0 vacuum, we bypass physical DAG constraints and evaluate commutations using ultra-fast \mathcal{O}(1) symplectic inner products. 

First, we will clean up the Front-End to remove a temporal dependency (use_last_outcome), making the Heisenberg IR (HIR) a pure, position-independent algebraic graph. Then, we will implement the Squeezer.


### Phase 1 (High Priority): Front-End Determinism (use_last_outcome Removal)

Context: The reset decomposition in frontend.cc currently relies on a lazy temporal flag (use_last_outcome = true) to link a conditional Pauli to its preceding hidden measurement. This breaks position-independence. Because the Front-End knows the total number of visible measurements (circuit.num_measurements), it can assign absolute classical memory indices to hidden measurements natively during tracing.
Agent Tasks:
 * Update src/ucc/frontend/hir.h:
   * Add uint32_t num_hidden_measurements = 0; to HirModule.
   * Delete FLAG_USE_LAST_OUTCOME.
   * Delete the getter use_last_outcome() and setter set_use_last_outcome() from HeisenbergOp.
   * Remove use_last_outcome from the HeisenbergOp union comments.
 * Update src/ucc/frontend/frontend.cc:
   * In trace(), right before the main loop over circuit.nodes, declare: uint32_t hidden_meas_idx = circuit.num_measurements;
   * In the GateType::R, RX, RY, MR, MRX, MRY switch case (around line 431):
     * Refactor the measurement emission logic:
       uint32_t this_meas;
if (hidden) {
    this_meas = hidden_meas_idx++;
    auto meas_op = HeisenbergOp::make_measure(destab_mask, stab_mask, sign, MeasRecordIdx{this_meas});
    meas_op.set_hidden(true);
    emit(meas_op);
} else {
    this_meas = static_cast<uint32_t>(meas_idx++);
    emit(HeisenbergOp::make_measure(destab_mask, stab_mask, sign, MeasRecordIdx{this_meas}));
}

     * For the conditional Pauli, assign the specific index:
       auto cond_op = HeisenbergOp::make_conditional(corr_destab, corr_stab, corr_sign, ControllingMeasIdx{this_meas});
emit(cond_op);
// Ensure cond_op.set_use_last_outcome(true); is deleted!

   * At the very end of trace(), just before returning, set hir.num_hidden_measurements = hidden_meas_idx - circuit.num_measurements;.
 * Update src/ucc/backend/backend.cc:
   * In lower(), update the total_meas_slots calculation: uint32_t total_meas_slots = hir.num_measurements + hir.num_hidden_measurements;.
   * Remove the hidden_meas_count pre-calculation loop entirely.
   * Remove last_meas_idx tracking and hidden_meas_emit_idx.
   * In the OpType::MEASURE case, remove the if (op.is_hidden()) branching. Just unconditionally use uint32_t classical_idx = static_cast<uint32_t>(op.meas_record_idx());.
   * In the OpType::CONDITIONAL_PAULI case, completely remove the if (op.use_last_outcome()) branch. Unconditionally read cond_idx = static_cast<uint32_t>(op.controlling_meas());.
 * Update Introspection & Bindings:
   * src/ucc/util/introspection.cc: In format_hir_op, remove the use_last_outcome ternary string formatting for CONDITIONAL_PAULI.
   * src/python/bindings.cc: Remove "use_last_outcome" from the HeisenbergOp as_dict builder.
Definition of Done (Phase 1): The C++ codebase compiles. The Python and Wasm tests pass without error. The HIR is now perfectly position-independent.


### Phase 2 (High Priority): Statevector Squeezing (Bidirectional Bubble)
Context: We will squeeze the active lifespan of virtual qubits by aggressively bubbling measurement collapses as early as physically possible (leftward), and non-Clifford operations (T_GATE, PHASE_ROTATION) as late as possible (rightward). Since Pauli NOISE is evaluated purely in the coordinate frame and doesn't expand the active array, we do not bubble it.
Agent Tasks:
 * Create Commutation Helper (src/ucc/optimizer/commutation.h and .cc):
   * Create a new header to hold shared commutation logic.
   * Move the anti_commute inline function from peephole.cc here.
   * Write a strict helper function: bool can_swap(const HeisenbergOp& left, const HeisenbergOp& right, const HirModule& hir).
   * Quantum Commutation: Two operations commute if their \mathcal{O}(1) symplectic inner product is even.
     * Check Pauli overlap: anti_commute(left.destab_mask(), left.stab_mask(), right.destab_mask(), right.stab_mask()).
     * If left or right is NOISE, check the inner product against every channel in the NoiseSite. If any anti-commute, return false.
   * Temporal/PRNG Barriers (Strict): We must not swap two operations that both consume PRNG to preserve the exact fixed-seed trajectory. If both left and right are probabilistic (e.g. MEASURE, NOISE, READOUT_NOISE), return false.
   * Classical Dataflow Barriers:
     * If left writes to a classical index (e.g., MEASURE) and right reads from it (e.g., CONDITIONAL_PAULI), they intrinsically cannot be swapped. In this compiler, we strictly avoid moving any classical reader to the left of any MEASURE just to be safe. If left is MEASURE and right is CONDITIONAL_PAULI, DETECTOR, OBSERVABLE, or READOUT_NOISE, return false.
 * Implement StatevectorSqueezePass:
   * Create src/ucc/optimizer/statevector_squeeze_pass.h and .cc inheriting from HirPass.
   * Sweep 1: Eager Compaction (Leftward Bubble):
     * Iterate i from 1 to hir.ops.size() - 1.
     * If hir.ops[i].op_type() == OpType::MEASURE:
       * Set curr = i.
       * While curr > 0 and can_swap(hir.ops[curr - 1], hir.ops[curr], hir):
         * std::swap(hir.ops[curr - 1], hir.ops[curr]);
         * If !hir.source_map.empty(): std::swap(hir.source_map[curr - 1], hir.source_map[curr]);
         * curr--;
   * Sweep 2: Lazy Expansion (Rightward Bubble):
     * Iterate i from hir.ops.size() - 2 down to 0.
     * If hir.ops[i].op_type() == OpType::T_GATE or OpType::PHASE_ROTATION:
       * Set curr = i.
       * While curr < hir.ops.size() - 1 and can_swap(hir.ops[curr], hir.ops[curr + 1], hir):
         * std::swap(hir.ops[curr], hir.ops[curr + 1]);
         * If !hir.source_map.empty(): std::swap(hir.source_map[curr], hir.source_map[curr + 1]);
         * curr++;
 * Wiring:
   * Add statevector_squeeze_pass.cc and commutation.cc to CMakeLists.txt in both src/ucc/ and src/wasm/.
   * Update peephole.cc to #include "ucc/optimizer/commutation.h" and use the shared anti_commute.
   * Expose StatevectorSqueezePass via Python bindings (bindings.cc) and add it to default_hir_pass_manager (immediately after PeepholeFusionPass).
Definition of Done (Phase 2): Write a Python unit test containing H 0; T 0; H 1; T 1; M 0; M 1;. When compiled with the Squeezer, it natively reorders the HIR to measure qubit 0 before allocating qubit 1. The peak_rank of the resulting Program drops exactly from 2 to 1.


### Phase 3 (Optional / Later Phase): Global Phase Folding
Context: The current PeepholeFusionPass is strictly O(N^2) but only looks for adjacent pairs. We can fold redundant phases globally by tunneling through intermediate routing.
Agent Tasks:
 * Implement PhaseFoldingPass.
 * Maintain a std::unordered_map<std::pair<uint64_t, uint64_t>, int> tracking active axes to their \mathbb{Z}_8 phase accumulations (T=1, S=2, Z=4, T_DAG=7). Use the first 64 bits of destab_mask and stab_mask as the key.
 * Sweep left-to-right. Absorb T_GATE and CLIFFORD_PHASE nodes into the map (do not emit them).
 * When encountering a barrier (MEASURE, NOISE, CONDITIONAL_PAULI), check all masks in the hash map. If a mask in the map anti-commutes with the barrier, pop it from the map, resolve its \mathbb{Z}_8 value into a new phase node, and emit it to the HIR just before the barrier.
 * At the end of the circuit, sequentially emit any remaining phases in the map.

### Phase 4 (Optional / Later Phase): Greedy Spider-Nest Reduction
Context: Deep multi-controlled Toffoli redundancies (like Galois Field multipliers) yield 4-term linear dependencies (P_A \oplus P_B \oplus P_C \oplus P_D = 0) in the phase polynomial.
Agent Tasks:
 * Implement SpiderNestPass.
 * Extract maximal cliques of mutually commuting T-gates (stopping at barriers). Let the clique size be m.
 * Within the clique, iterate all pairs (A, B) and store collision_map[A.mask ^ B.mask] = {A, B}.
 * If a new pair (C, D) yields a mask already in the map (and the 4 gates are disjoint), a weight-4 linear dependency is found.
 * Verify their \mathbb{Z}_8 sums align with a known PyZX Spider-Nest identity. Replace the 4 T-gates with exactly 2 T-gates and residual Clifford phases to balance the polynomial. Restart the sweep.
Phase 5 (Optional / Later Phase): The op-T-mize Benchmark Suite
Agent Tasks:
 * Create paper/optimize/benchmark_shootout.py.
 * Download canonical op-T-mize benchmark .qasm circuits (e.g., Adder16, barenco_tof_10).
 * Write a utility to auto-inject MEASURE instructions onto garbage ancillas at the end of arithmetic uncomputation blocks.
 * Implement isolated subprocess calls to compile the circuits using:
   * TKET (pytket): FullPeepholeOptimise()
   * PyZX (pyzx): full_reduce()
   * UCC: Our native compilation pipeline.
 * Extract and plot the resulting peak active rank (k_{\max}), final T-count, and compilation wall-clock latency (ms) for all three tools. Generate a publication-ready CSV and Matplotlib chart.
 
