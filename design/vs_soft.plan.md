# UCC SOFT Validation & Benchmark Implementation Plan

## Executive Summary & Constraints
You are implementing the validation and benchmarking suite to compare UCC against the SOFT simulator for Magic State Cultivation (MSC). The goal is to prove that UCC's Ahead-Of-Time (AOT) "Symplectic Determinism" architecture allows a conventional multi-core CPU to compete with, or outperform, a 16-GPU cluster by completely eliminating dynamic $\mathcal{O}(n^2)$ tableau updates and aggressively early-aborting failed shots.

To ensure unimpeachable correctness, we will also implement an independent validation layer using `qiskit-aer` to verify our exact physics engine before targeting the massive $d=5$ MSC circuits.

**Strict Constraints for this specific plan (Overrides general guidelines if in conflict):**
1. **No External Multithreading Dependencies:** Do not introduce OpenMP, Intel TBB, or other heavy threading libraries to the C++ core. Use standard `<thread>` and `<future>` for parallelizing the Monte Carlo shots.
2. **Deterministic Concurrency:** The RNG seed for shot $i$ must strictly be `base_seed + i`, regardless of which thread executes it. Results must be bit-identical whether running on 1 core or 32 cores.
3. **The 32-Byte Invariant:** The `Instruction` struct is exactly 32 bytes. Do not alter its size when implementing `OP_POSTSELECT`. Reuse existing union payload structures (e.g., the `detector` or `meta` variant).
4. **Zero Qiskit C++ Dependencies:** Qiskit-Aer is a validation oracle only. It must be used exclusively in the Python `tests/python/` suite. The C++ core must remain completely ignorant of Qiskit.

---

## Phase 1: Independent Ground Truth (Qiskit-Aer Validation)
**Goal:** Establish a solid, third-party source of truth (Qiskit-Aer) for noisy, mid-circuit measurement Clifford+T circuits to prove UCC's correctness independently of SOFT.
*   **Task 1.1 (Dependencies):** Add `qiskit>=1.0` and `qiskit-aer>=0.14` to the `dev` dependency group in `pyproject.toml`.
*   **Task 1.2 (Qiskit Translator):** Create `tests/python/utils_qiskit.py`. Write a translator that takes a `ucc.Circuit` (or `.stim` text) and converts it into a `qiskit.QuantumCircuit`. It must support `H`, `S`, `S_DAG`, `T`, `T_DAG`, `CX`, `X_ERROR`, `Z_ERROR`, `DEPOLARIZE1`, `DEPOLARIZE2`, and mid-circuit `M`.
*   **Task 1.3 (Statevector Oracle Test):** In `tests/python/test_qiskit_aer.py`, write a test that generates random noiseless Clifford+T circuits (up to 10 qubits), runs them in `AerSimulator(method="statevector")`, and asserts exact fidelity match ($>0.9999$) with `ucc.get_statevector()`.
*   **Task 1.4 (Density Matrix Distribution Test):** Write a test that generates small (e.g., 4-qubit) *noisy* circuits with mid-circuit measurements. Run them in `AerSimulator(method="density_matrix")` to extract exact expected measurement probability distributions. Sample the same circuit 10,000 times in `ucc.sample()` and assert the sampled distribution matches the Qiskit density matrix probabilities within statistical tolerance ($5\sigma$).
*   **Definition of Done (DoD):** `uv run pytest tests/python/test_qiskit_aer.py` passes reliably, proving UCC's amplitude interference and noise accumulation are mathematically flawless.

## Phase 2: The Fast-Fail Architecture (`OP_POSTSELECT`)
**Goal:** Implement the ability to immediately abort doomed shots to bypass deep non-Clifford evaluations, capitalizing on the 85-99% discard rates of MSC circuits.
*   **Task 2.1 (Compiler Option):** In `backend.h` and `backend.cc`, update `ucc::lower()` to accept a `bool abort_on_detector = false` argument. Expose this flag in the Python `ucc.compile()` binding. *(Note: Future syntax `ASSERT_DETECTOR` can eventually drive this per-node, but a global compiler flag serves the MVP benchmark).*
*   **Task 2.2 (Back-End Lowering):** In `backend.cc`, when lowering `OpType::DETECTOR`, if `abort_on_detector` is true, emit an `Opcode::OP_POSTSELECT` instruction *immediately after* the `OP_DETECTOR` instruction. The payload must reference the same `detector_targets` index (using the `instr.detector` union field) and expected parity (typically 0).
*   **Task 2.3 (SVM State):** In `svm.h`, add `bool discarded = false;` to `SchrodingerState` and reset it to `false` in the `reset()` method. Update `SampleResult` to include `std::vector<uint8_t> passed;` (1=passed, 0=discarded).
*   **Task 2.4 (SVM Execution):** In `svm.cc`, implement `case Opcode::OP_POSTSELECT:`. It evaluates the XOR parity of the target measurement record indices. If the parity is 1 (diverges from the expected 0 for a clean detector), set `state.discarded = true;` and `return;` immediately to exit the bytecode execution loop for that shot.
*   **Task 2.5 (Python API Update):** Update nanobind bindings so `ucc.compile` accepts `abort_on_detector`. Update `sample()` to return a 4-tuple `(measurements, detectors, observables, passed)`. Fix existing Python tests to unpack 4 values instead of 3.
*   **DoD:** A Python test proves that compiling with `abort_on_detector=True` and running a circuit with a deterministic error that triggers a detector yields `passed=0` for those shots, and drastically reduces execution time by returning early.

## Phase 3: Multi-Core Saturation (Thread Pool)
**Goal:** Parallelize the SVM `shots` loop natively in C++ to leverage 16-32 core CPUs.
*   **Task 3.1 (API Update):** Update `sample(program, shots, seed, threads)` in `svm.h/.cc` and `bindings.cc` to accept `uint32_t threads = 0` (default to `std::thread::hardware_concurrency()`).
*   **Task 3.2 (GIL Release):** In `bindings.cc`, wrap the `ucc::sample` call in `nanobind::gil_scoped_release` so Python does not block the C++ threads.
*   **Task 3.3 (Thread Pool Implementation):** In `svm.cc:sample()`:
    *   Pre-allocate the `SampleResult` vectors to full size. Initialize `passed` to all 1s (true).
    *   Launch $T$ `std::thread` workers, giving each thread an equal chunk of the total shots.
    *   Inside the thread lambda, allocate a *thread-local* `SchrodingerState`. Loop over the assigned global shot indices $i$. Call `state.reset(seed + i)`. Execute the program. Write results directly to the pre-allocated offsets in the `SampleResult` vectors (completely lock-free, no mutexes needed). Write `!state.discarded` to the `passed` array.
    *   Join all threads.
*   **DoD:** Running `ucc.sample(..., threads=4)` utilizes multiple CPU cores and produces bit-identical results to `threads=1` for the same seed, completing significantly faster.

## Phase 4: Composable Rank Profiling
**Goal:** Expose the dynamic rank history to visually and empirically prove the "Dynamical Shift-Rank Bound" theorem without bloating the `CompiledModule` in memory.
*   **Task 4.1 (C++ Scanner):** In `backend.h/.cc`, add a standalone function `std::vector<uint32_t> get_rank_history(const CompiledModule& prog)`. It iterates over `prog.bytecode`, keeping a running counter. Increment on `OP_BRANCH`, decrement on `OP_MEASURE_MERGE`. Record the rank at each step.
*   **Task 4.2 (Python Binding):** Expose `ucc.get_rank_history(program)` as a standalone function returning a numpy array in `bindings.cc`.
*   **Task 4.3 (Plotting Script):** Create `tools/bench/plot_rank.py` which loads `magic_state_cultivation/circuits/circuit_d5_p0.001.stim` from the `haoliri0-soft` codebase, compiles it, calls `ucc.get_rank_history()`, and uses matplotlib to plot the rank history step function.
*   **DoD:** The `plot_rank.py` script automatically verifies the $d=5$ MSC circuit yields a `peak_rank` of exactly 10, and generates a plot showing the "breathing" state. The `CompiledModule` remains lean and zero-overhead.

## Phase 5: The Benchmarking Harness
**Goal:** Create the script that runs the head-to-head comparison against the data from the SOFT paper.
*   **Task 5.1 (Data Setup):** Pull the SOFT $d=3$ and $d=5$ `.stim` circuits (with `p=0.001` and `p=0.0005`) into the `tools/bench/circuits/` folder.
*   **Task 5.2 (The Benchmark Script):** Create `tools/bench/run_soft_benchmarks.py`. It should:
    1. Load the `.stim` file.
    2. Compile it using `ucc.compile(..., abort_on_detector=True)`.
    3. Start a high-resolution timer.
    4. Call `ucc.sample(..., shots=100_000, threads=os.cpu_count())`.
    5. Stop the timer.
*   **Task 5.3 (Metrics Calculation):** Calculate and print exactly the metrics from Table III and V of the SOFT paper:
    *   Discard Rate: `1.0 - np.mean(passed)`
    *   Logical Error Rate: Mean of the final observable bit for shots where `passed == True`.
    *   Throughput: Total shots / Total time (Shots/sec).
*   **DoD:** The script runs successfully on the $d=5$ MSC circuit. The console output proves that UCC perfectly matches the discard rates reported in the SOFT paper, and generates the CPU throughput numbers needed for the paper's comparison table.
