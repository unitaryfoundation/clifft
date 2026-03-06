# UCC Implementation Plan: State-Agnostic Global Optimization (Lite)

**Status: COMPLETED**

## Executive Summary & Constraints

In standard DAG-based quantum compilers, optimization is constrained by rigid temporal graphs, hardware topology, and "Hadamard Gadgetization." UCC's Factored State Architecture bypasses all of this. Because the Front-End mathematically rewinds all physical non-Clifford operations and measurements back to the $t=0$ vacuum, chronological time is completely erased. The Heisenberg IR (HIR) is inherently a **timeless, global phase polynomial** perfectly aligned to the topological geometry of the physical circuit.

Because we do not use a DAG, we do not need rigid "temporal barriers" like clock ticks, classical logic, or measurement gates to prevent invalid optimizations. A $T$ gate can safely commute past a measurement *if and only if* their spatial geometries commute at $t=0$.

To establish the "compiler" narrative for Paper 1, we implemented **Peephole Fusion**. The optimizer natively scans the HIR and algebraically cancels or fuses $T$ and $T^\dagger$ gates acting on the exact same virtual Pauli axis, regardless of how far apart they appeared in the original circuit.

**Strict Constraints:**
1. **State-Agnosticism:** The optimizer manipulates the HIR purely using symplectic geometry on the mathematical masks (`destab_mask`, `stab_mask`). It never requires knowledge of an initial state, the VM's active dimensions, or physical time.
2. **No DAG/Time Assumptions:** Classical annotations and measurements are not treated as absolute barriers. The HIR has no concept of time. Commutation is defined *solely* by the $\mathcal{O}(1)$ symplectic inner product.
3. **Paper 1 Scope:** Advanced $\mathcal{O}(n^4)$ FastTODD / TOHPE linear algebra solvers are explicitly deferred to future work.

---

## What Was Delivered

### HIR Expansion & Pass Manager (PR #72)

- Added `CLIFFORD_PHASE` OpType to represent fused $S$/$S^\dagger$ rotations on Pauli masks.
- Created `Pass` abstract base class and `PassManager` in `src/ucc/optimizer/`.
- Wired optimizer into `compile()` pipeline.

### Symplectic Peephole Fusion (PR #73)

- `PeepholeFusionPass` scans HIR left-to-right with convergence loop.
- Commutation check via $\mathcal{O}(1)$ symplectic inner product on `stim::bitword` masks.
- Noise node commutation via side-table channel lookup.
- Classical nodes (DETECTOR, OBSERVABLE, READOUT_NOISE) treated as transparent.
- Algebraic fusion: T+T-dag cancels, T+T fuses to CLIFFORD_PHASE (S or S-dag).
- Unknown OpTypes conservatively block fusion.

### Back-End CLIFFORD_PHASE Lowering (PR #73)

- `emit_s_dag` helper for S-dagger gate emission.
- CLIFFORD_PHASE case in `lower()` mirrors T_GATE routing: map_to_virtual, compress_pauli, basis-dependent expansion/H-gate emission.
- Dormant X-basis S-gate correctly triggers array expansion (S|+> = |+i>).

### Explicit Pipeline API (PR #75)

- Exposed `ucc.trace()`, `ucc.lower()`, `ucc.PassManager`, `ucc.PeepholeFusionPass`, `ucc.HirModule` to Python.
- HirModule is opaque (no stim types leak); exposes only scalar metadata.
- `ucc.compile()` is a no-optimization convenience; users must explicitly construct and run a PassManager for optimization.

### Optimizer Verification (PR #75)

- 77 Python tests in `test_peephole_oracle.py`:
  - Statevector equivalence fuzzing (2-8 qubits, depths 20-100).
  - Algebraic identity tests (T*T-dag=I, T+T=S, T^4=Z).
  - 40-qubit mirror circuit T-gate annihilation: peak_rank collapses from up to t_count to 0.
  - Explicit pipeline API tests.
- Updated `test_structural_oracles.py` mirror fuzzer with optimizer-enabled variants.

## Key Results

- The peephole optimizer achieves **perfect T-gate cancellation** on all U*U-dag mirror circuits (peak_rank -> 0).
- Statevector fidelity between optimized and unoptimized compilation exceeds 0.9999 across all random circuit tests.
- 347 C++ tests + 271 Python tests = 618 total, all passing.

---

## Pending / Deferred Items

- **Virtual Diagonalization & Parity Table Bridge:** Connect the HIR's phase polynomial representation to the parity network formalism for deeper multi-gate optimization.
- **TOHPE Pass (Algorithm 2):** Implement the TOHPE linear algebra solver for global T-gate count reduction beyond adjacent peephole fusion.
- **op-T-mize Benchmark Suite:** Establish standardized benchmarks for comparing T-count reduction against other compilers.
- **FastTODD Upgrade (Algorithm 3):** Implement the $\mathcal{O}(n^4)$ FastTODD algorithm for near-optimal T-count minimization.
- **CLIFFORD_PHASE chaining:** The peephole pass currently only fuses T-gate pairs. It does not chain further (e.g., S+T -> T^3, or S+S -> Z). This is a natural extension.
- **Multi-pass interaction:** As more passes are added, define canonical pass ordering and investigate fixed-point convergence across heterogeneous passes.
