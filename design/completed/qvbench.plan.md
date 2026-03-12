# UCC Implementation Plan: Quantum Volume (Worst-Case) Benchmarking

## Executive Summary & Architectural Context

This phase implements a Quantum Volume (QV) benchmarking suite to evaluate the execution speed of the newly merged continuous arbitrary rotations (R_X, R_Y, R_Z, U_3) in the Unitary Compiler Collection (UCC).

**The Architectural Reality:** Quantum Volume circuits consist of layers of random permutations followed by highly entangling Haar-random SU(4) gates. This represents the absolute worst-case scenario for UCC's Factored State Architecture. The rapid global scrambling guarantees that the peak active dimension (k_max) will equal the total physical qubits (N) almost immediately. This forces the VM to allocate the full 2^N statevector array, neutralizing its dimensional advantage.

**The Great Equalizer (Terminal Measurements):** Because UCC tracks the coordinate frame virtually, executing a circuit without measurements would leave the state in a permuted virtual basis, avoiding the O(n^2) cost of a final physical basis rotation. To ensure a strictly fair, apples-to-apples comparison against standard statevector simulators, **all benchmark circuits must end with terminal measurements on all qubits.** Generating physical bitstrings forces UCC to evaluate probabilities across the 2^N array and pass the outcomes through the Pauli tracking frame, matching the computational payload of the other engines.

**Strict Constraints & Rules:**

1. **Target Simulators:** The shootout will strictly compare **UCC** against **Qiskit-Aer**, **Qulacs**, and **Qsim**.
2. **Instruction Set Parity:** Do NOT allow Qiskit to simulate native 4x4 blocks. The circuit must be transpiled into a `['cx', 'u3']` basis so all simulators are executing the same raw sequence of 1-qubit and 2-qubit operations.
3. **Single CPU Core:** All simulators should be strictly pinned to a single physical CPU core to measure fundamental algorithmic baseline efficiency. As an alternate mode should allow libraries to use all cores/configurable number if they support it to see the impact.
4. **Subprocess Isolation:** To prevent Python garbage collection drift and memory fragmentation, every individual benchmark run MUST execute in a fresh, isolated Python subprocess.
5. **Configurable Sweeps:** The orchestrator script must allow the user to explicitly define the minimum and maximum qubit counts (or a specific array of N values) via CLI arguments to facilitate quick local correctness checks before launching massive OOM-bound scaling runs.
6. **Memory Cap:** Default maximum N=26 for 8GB machines. The orchestrator should accept `--max-q` to allow larger runs on bigger machines.

**Future Work:**
- Revisit qsim gate fusion fairness. Currently qsim uses its default `max_fused_gate_size=2` which fuses adjacent gate pairs. This gives qsim a potential advantage. A future comparison should disable fusion (`max_fused_gate_size=1`) to measure raw gate-by-gate performance parity.

---

## Phase 1: Unified Circuit Generation & QASM Translation

**Goal:** Generate the standardized QV circuits, transpile them, append terminal measurements, and export them as OpenQASM 2.0 to serve as the universal source of truth for all simulators.

* **Task 1.1 (Circuit Generation):** In `paper/qv_benchmark/generator.py`, use Qiskit's `qiskit.circuit.library.quantum_volume(num_qubits, seed=...)` function (not the deprecated class) to generate the base circuit.

* **Task 1.2 (Transpilation & Measurement):**
  * Transpile the circuit using `qiskit.compiler.transpile(qc, basis_gates=['cx', 'u3'], optimization_level=0)`.
  * Append terminal `measure_all()` instructions to all qubits.

* **Task 1.3 (QASM Export):** Export the transpiled circuit to a QASM 2.0 string using `qiskit.qasm2.dumps()`.

* **Task 1.4 (Unified QASM Parser & Adapter Layer):** Write a single QASM parser in `paper/qv_benchmark/qasm_adapter.py` that extracts `(gate_name, params, qubits)` tuples from the QASM string, then provides adapter functions to build circuits for each framework:
  * **Qiskit-Aer:** Load via `QuantumCircuit.from_qasm_str()`.
  * **UCC:** Translate the parsed gate list to UCC Stim-superset format (mapping `u3(t,p,l) q[i]` to `U3(t,p,l) i` and `cx q[i],q[j]` to `CX i j`).
  * **Qulacs:** Build circuit programmatically using `qulacs.gate.U3()` and `qulacs.gate.CNOT()`.
  * **Qsim:** Load via Cirq's `cirq.contrib.qasm_import.circuit_from_qasm()`.

* **DoD:** The generator produces a standardized QASM string of `u3` and `cx` gates ending in measurements, which is successfully ingested by all four frameworks.

## Phase 2: Mathematical Validation (Heavy Output Probability)

**Goal:** Mathematically prove that UCC's U3 rotation math and QASM translation are sound by verifying the Heavy Output Probability (HOP) converges to ~84.6% on small, noiseless circuits.

* **Task 2.1 (The HOP Oracle):** Write `paper/qv_benchmark/validate_hop.py`. For a given exact statevector, calculate the true probabilities of all 2^N bitstrings. Find the median probability. Sum the probabilities of the "heavy" bitstrings (those strictly greater than the median).

* **Task 2.2 (UCC Validation):** For a small sweep of N in [4, 6, 8]:
  1. Generate the unmeasured, transpiled circuit.
  2. Extract the exact dense statevector from UCC (`ucc.get_statevector()`).
  3. Compare against a reference statevector from Qiskit-Aer to assert fidelity > 0.999.
  4. Assert that the calculated HOP is > 0.70 (converging toward the theoretical 0.846).

* **DoD:** The suite mathematically guarantees UCC's execution logic is correct before proceeding to the timing phase.

## Phase 3: The Execution Timing Harness

**Goal:** Execute the measured circuits and time the simulation runtime. Report both compile time and sample time separately for UCC, and total time for all simulators.

* **Task 3.1 (Isolated Worker Script):** Write `paper/qv_benchmark/worker.py` taking arguments for `simulator_name`, `num_qubits`, and `seed`.
  * Ingest the QASM circuit (generation + transpilation happens inside the worker to include fair parsing overhead for all simulators, but is NOT timed).
  * Start a high-resolution timer.
  * For UCC: time `compile()` and `sample(shots=1)` separately, report both plus total.
  * For other simulators: time the simulation call for 1 shot.
  * Stop the timer and print timing results as JSON.

* **Task 3.2 (The Orchestrator):** Write `paper/qv_benchmark/run_benchmark.py`.
  * Implement CLI arguments: `--min-q` (default 6), `--max-q` (default 26), or `--qubits 10,14,18,22` for explicit list.
  * `--repeats N` (default 3) to run each (N, simulator) combo multiple times for reliable timing.
  * `--simulators` to select which simulators to run (default: all four).
  * For each N in the requested sweep, and each simulator, spawn `worker.py` via `subprocess.run()`.
  * Capture the timing output and append it to `paper/qv_benchmark/results.csv`.
  * If a subprocess crashes or OOMs, catch the exception, log "OOM", and continue.

* **DoD:** A robust CSV is generated containing execution times for N=6 up through N=26 (or the machine's memory limit).

## Phase 4: Data Visualization

**Goal:** Generate publication-ready plots demonstrating the scaling performance.

* **Task 4.1 (Plotting Script):** Write `paper/qv_benchmark/plot_qv.py` using `matplotlib` and `pandas`.
* **Task 4.2 (Formatting):**
  * X-axis: Number of Qubits (N).
  * Y-axis: Execution Time in seconds (Logarithmic Scale).
  * Show median across repeats, with error bars (min/max or IQR).
  * If a simulator OOMs or hits a timeout, denote the failure point with a distinct marker (e.g., a red 'X') at the ceiling of the graph.
  * Save the output to `paper/qv_benchmark/qv_scaling.pdf`.

* **DoD:** The script outputs a clean, legible graph comparing UCC's worst-case array iteration speed directly against the specialized dense simulators.
