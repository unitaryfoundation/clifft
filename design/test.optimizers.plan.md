# UCC Implementation Plan: Optimization Invariants & Exact Trajectory Synchronization

## 1. Executive Summary

We are upgrading our testing framework to rigorously validate that AOT compiler optimizations (both HIR and Bytecode passes) are mathematically sound and perfectly preserve the PRNG trajectory of the Schrödinger Virtual Machine (SVM).

Because UCC evaluates PRNG-consuming events (`MEASURE`, `NOISE`) strictly in the offline physical geometry, an optimized program must produce **bit-for-bit identical classical arrays** compared to an unoptimized program when given the same seed.

To achieve this safely (without causing Out-Of-Memory errors in the CI machine), we will:

1. Patch a floating-point fragility in the VM's active measurement logic to guarantee PRNG synchronization across precision artifacts.
2. Build targeted procedural circuit generators that explicitly create "optimizer bait" (e.g., deep star-graphs, commutation chains) while artificially bounding the active dimension ($k_{\max}$) by injecting resets/measurements or limiting overall qubit count.
3. Add a dedicated invariant test suite.

---

## 2. Phase 1: VM Epsilon Patch (Fixing PRNG Desync)

**Goal:** Prevent floating-point dust from stealing PRNG rolls in quasi-deterministic measurements.

**File:** `src/ucc/svm/svm.cc`

1. **Target `exec_meas_active_diagonal**`:
* Locate the probability check: `if (prob_b1 <= 0.0) ... else if (prob_b0 <= 0.0)`.
* Change this to use a relative epsilon to catch floating-point accumulation errors that occur when $T$-gates conceptually interfere to zero but leave residual noise (e.g., `1e-32`):
```cpp
if (prob_b1 <= 1e-14 * total) {
    b = 0;
} else if (prob_b0 <= 1e-14 * total) {
    b = 1;
} else {
    double rand = state.random_double(); ...
}

```




2. **Target `exec_meas_active_interfere**`:
* Locate the similar check for `prob_minus` and `prob_plus`.
* Apply the exact same `1e-14 * total` relative epsilon threshold.


3. **Target `exec_swap_meas_interfere**`:
* Apply the identical relative epsilon threshold to the fused swap-measure opcode logic.



---

## 3. Phase 2: Memory-Bounded "Optimizer Bait" Generators

**Goal:** Procedurally generate circuits with specific topological patterns that optimizers look for, while strictly preventing Out-Of-Memory (OOM) crashes by bounding the $T$-gate spread.

**File:** Create a new file `tests/python/utils_fuzzing.py`.

Implement the following generator functions:

1. **`generate_commutation_gauntlet(num_qubits: int, depth: int, seed: int) -> str`**
* **Structure:** Interleaves $T$ and $T^\dagger$ gates with deep chains of *commuting* operations (e.g., $Z$, $CZ$, and $M$ gates) and *anti-commuting* operations (e.g., $X$, $H$).
* **Memory Bound:** Restrict all $T$ gates to act *only* on a small subset of qubits, e.g., indices `0` through `min(num_qubits-1, 4)`. The remaining qubits can be used for vast Clifford/measurement networks. This mathematically guarantees $k_{\max} \le 5$, making it completely RAM-safe while testing $\mathcal{O}(1)$ mask commutations across the entire register.


2. **`generate_star_graph_honeypot(num_qubits: int, depth: int, seed: int) -> str`**
* **Structure:** Forces the creation of contiguous `CX` gates sharing a target, and contiguous `CZ` gates sharing a control, to explicitly trigger the `MultiGatePass`. Sprinkle `DEPOLARIZE1` blocks in between to test `NoiseBlockPass`.
* **Memory Bound:** Limit $T$ gates to the first 4 qubits, or periodically emit `M` on all qubits to flush the active array, ensuring $k_{\max}$ stays strictly $\le 10$.


3. **`generate_uncomputation_ladder(num_qubits: int, depth: int, seed: int) -> str`**
* **Structure:** Builds a highly entangling Clifford+$T$ sequence, and then dynamically appends the *exact analytical inverse* of that sequence (reversing order and applying Daggers).
* **Memory Bound:** Limit the maximum number of unmeasured $T$ gates in the forward pass to $\le 10$. This tests the `PeepholeFusionPass`'s ability to algebraically collapse $k_{\max}$ without causing memory overflow in the unoptimized baseline.



---

## 4. Phase 3: The Differential Invariant Test Suite

**Goal:** Wire the generators into a parameterized test suite that asserts the compiler's strict physical and statistical contracts.

**File:** Create a new file `tests/python/test_optimization_invariants.py`.

1. **Test Harness Setup:**
* Create a helper `run_differential_trajectory(circuit_str: str, shots: int, seed: int)`.
* Compile the circuit twice: once with `hir_passes=None, bytecode_passes=None` (the baseline), and once with the default passes enabled (`hir_passes=ucc.default_pass_manager()`, etc.).


2. **Strict Memory & Latency Invariants:**
* **OOM Guard:** Assert `base_prog.peak_rank <= 12`. If a generator accidentally produces a circuit that requires $>4000$ complex amplitudes, the test must instantly fail rather than freezing the CI environment.
* Assert `opt_prog.peak_rank <= base_prog.peak_rank` (Memory footprint never grows).
* Assert `opt_prog.num_instructions <= base_prog.num_instructions` (Latency never grows).
* Assert `opt_prog.num_measurements == base_prog.num_measurements` (Classical interfaces are unbroken).


3. **The Synchronization Asserts:**
* Execute `ucc.sample(..., seed=seed)` for both programs.
* Assert exact `np.array_equal` for the returned `measurements`, `detectors`, and `observables` arrays.


4. **Test Cases:**
* Use `@pytest.mark.parametrize` to feed the three generators (Gauntlet, Star-Graph, Uncomputation) into the test harness across multiple seeds.
* Test configurations spanning $(N=10, depth=100)$ up to $(N=50, depth=500)$ to prove algorithmic safety at scales well beyond the standard statevector limits.
