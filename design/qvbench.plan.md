# UCC Implementation Plan: Quantum Volume (Worst-Case) Benchmarking

## Executive Summary & Architectural Context

This phase implements a Quantum Volume (QV) benchmarking suite to evaluate the execution speed of the newly merged continuous arbitrary rotations ($R_X, R_Y, R_Z, U_3$) in the Unitary Compiler Collection (UCC).

**The Architectural Reality:** Quantum Volume circuits consist of layers of random permutations followed by highly entangling Haar-random $SU(4)$ gates. This represents the absolute worst-case scenario for UCC's Factored State Architecture. The rapid global scrambling guarantees that the peak active dimension ($k_{\max}$) will equal the total physical qubits ($N$) almost immediately. This forces the VM to allocate the full $2^N$ statevector array, neutralizing its dimensional advantage.

**The Great Equalizer (Terminal Measurements):** Because UCC tracks the coordinate frame virtually, executing a circuit without measurements would leave the state in a permuted virtual basis, avoiding the $\mathcal{O}(n^2)$ cost of a final physical basis rotation. To ensure a strictly fair, apples-to-apples comparison against standard statevector simulators, **all benchmark circuits must end with terminal measurements on all qubits.** Generating physical bitstrings forces UCC to evaluate probabilities across the $2^N$ array and pass the outcomes through the Pauli tracking frame, matching the computational payload of the other engines.

**Strict Constraints & Rules:**

1. **Target Simulators:** The shootout will strictly compare **UCC** against **Qiskit-Aer**, **Qulacs**, and **Qsim**.
2. **Instruction Set Parity:** Do NOT allow Qiskit to simulate native $4 \times 4$ blocks. The circuit must be transpiled into a `['cx', 'u3']` basis so all simulators are executing the same raw sequence of 1-qubit and 2-qubit operations.
3. **Single CPU Core:** All simulators should be strictly pinned to a single physical CPU core to measure fundamental algorithmic baseline efficiency. As an alternate mode should allow libraries to use all cores/configurable number if they support it to see the impact.
4. **Subprocess Isolation:** To prevent Python garbage collection drift and memory fragmentation, every individual benchmark run MUST execute in a fresh, isolated Python subprocess.
5. **Configurable Sweeps:** The orchestrator script must allow the user to explicitly define the minimum and maximum qubit counts (or a specific array of $N$ values) via CLI arguments to facilitate quick local correctness checks before launching massive OOM-bound scaling runs.

---

## Phase 1: Unified Circuit Generation & QASM Translation

**Goal:** Generate the standardized QV circuits, transpile them, append terminal measurements, and export them as OpenQASM 2.0/3.0 to serve as the universal source of truth for all simulators.

* **Task 1.1 (Circuit Generation):** In `paper/qv_benchmark/generator.py`, use Qiskit's `qiskit.circuit.library.QuantumVolume(num_qubits)` to generate the base circuit.
* **Task 1.2 (Transpilation & Measurement):** * Transpile the circuit using `qiskit.compiler.transpile(qc, basis_gates=['cx', 'u3'], optimization_level=0)`.
* Append terminal `measure` instructions to all qubits.


* **Task 1.3 (QASM Export & Ingestion):** Export the transpiled circuit to a QASM string. Write adapter functions to ingest this QASM string into the respective simulator formats:
* **Qiskit-Aer:** Load directly via `QuantumCircuit.from_qasm_str()`.
* **UCC:** Translate the QASM to the UCC `.stim` superset format (mapping `u3` to `U3` and `cx` to `CX`).
* **Qulacs / Qsim:** *[Note: Integration touchpoints to be implemented based on specific QASM-import contrib modules available for these libraries.]*


* **DoD:** The generator produces a standardized QASM string of `u3` and `cx` gates ending in measurements, which is successfully ingested by all four frameworks.

## Phase 2: Mathematical Validation (Heavy Output Probability)

**Goal:** Mathematically prove that the transpilation, QASM translation, and new UCC $U_3$ rotation math are sound by verifying the Heavy Output Probability (HOP) converges to $\sim 84.6\%$ on small, noiseless circuits.

* **Task 2.1 (The HOP Oracle):** Write `paper/qv_benchmark/validate_hop.py`. For a given exact statevector, calculate the true probabilities of all $2^N$ bitstrings. Find the median probability. Sum the probabilities of the "heavy" bitstrings (those strictly greater than the median).
* **Task 2.2 (Validation Execution):** For a small sweep of $N \in [6, 8, 10]$:
1. Generate the unmeasured, transpiled circuit.
2. Extract the exact dense statevector from Qiskit-Aer, Qulacs, Qsim, and UCC (`ucc.get_statevector()`).
3. Assert that all statevectors match (fidelity $> 0.999$) AND that the calculated HOP for each simulator is $> 0.70$ (converging toward the theoretical $0.846$).


* **DoD:** The suite mathematically guarantees the execution logic is correct before proceeding to the timing phase.

## Phase 3: The Execution Timing Harness

**Goal:** Execute the measured circuits and time *only* the simulation runtime (strictly excluding circuit generation, transpilation, and parsing overhead). UCC timing should include the "compilation" that lowers from HIR to bytecode.

* **Task 3.1 (Isolated Worker Script):** Write `paper/qv_benchmark/worker.py` taking arguments for `simulator_name` and `num_qubits`.
* Ingest the QASM circuit.
* Start a high-resolution timer.
* Execute the circuit for exactly **10 shots** (e.g., `ucc.sample(prog, shots=10)`).
* Stop the timer and print the execution time.


* **Task 3.2 (The Orchestrator):** Write `paper/qv_benchmark/run_benchmark.py`.
* Implement CLI arguments (e.g., `--min-q`, `--max-q`, or `--qubits 10,14,18,22`) so users can easily test small scales before pushing to the memory limit.
* For each $N$ in the requested sweep, and each simulator (`qiskit`, `qulacs`, `qsim`, `ucc`), spawn `worker.py` via `subprocess.run()`.
* Capture the timing output and append it to `paper/qv_benchmark/results.csv`.
* If a subprocess crashes or OOMs, catch the exception, log "OOM", and continue.


* **DoD:** A robust CSV is generated containing execution times for $N=10$ up through the machine's memory limit (e.g., $N=28$).

## Phase 4: Data Visualization

**Goal:** Generate publication-ready plots demonstrating the scaling performance.

* **Task 4.1 (Plotting Script):** Write `paper/qv_benchmark/plot_qv.py` using `matplotlib` and `pandas`.
* **Task 4.2 (Formatting):** * X-axis: Number of Qubits ($N$).
* Y-axis: Execution Time in seconds (Logarithmic Scale).
* If a simulator OOMs or hits a timeout, denote the failure point with a distinct marker (e.g., a red 'X') at the ceiling of the graph.
* Save the output to `paper/qv_benchmark/qv_scaling.pdf`.


* **DoD:** The script outputs a clean, legible graph comparing UCC's worst-case array iteration speed directly against the specialized dense simulators.
