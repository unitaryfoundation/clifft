# UCC SOFT Validation & Benchmark Implementation Plan

## Executive Summary & Constraints
You are implementing the validation and benchmarking suite to compare UCC against the SOFT simulator for Magic State Cultivation (MSC). The goal is to prove that UCC's Ahead-Of-Time (AOT) "Symplectic Determinism" architecture allows a conventional multi-core CPU to compete with, or outperform, a 16-GPU cluster by completely eliminating dynamic $\mathcal{O}(n^2)$ tableau updates and aggressively early-aborting failed shots.

To ensure unimpeachable correctness, we will implement an independent validation layer using `qiskit-aer` to verify our exact noiseless physics engine before targeting the massive $d=5$ MSC circuits. Noisy circuit validation is already covered by existing Stim-based statistical equivalence tests.

You can look at the SOFT codebase https://github.com/haoliri0/SOFT/ as necessary to understand the best way to do comparisons. The circuits we want to test are in `magic_state_cultivation/circuits/` folder in that repository.

**Strict Constraints for this specific plan (Overrides general guidelines if in conflict):**
1. **No External Multithreading Dependencies:** Do not introduce OpenMP, Intel TBB, or other heavy threading libraries to the C++ core. Use standard `<thread>` and `<future>` for parallelizing the Monte Carlo shots. Consider pinning threads for best performance. Have the user be explicit on the number of threads they want to use.
2. **Deterministic Concurrency:** The RNG seed for shot $i$ must strictly be `base_seed + i`, regardless of which thread executes it. Results must be bit-identical whether running on 1 core or 32 cores.
3. **The 32-Byte Invariant:** The `Instruction` struct is exactly 32 bytes. Do not alter its size when implementing `OP_POSTSELECT`. Reuse existing union payload structures (e.g., the `detector` or `meta` variant).
4. **Zero Qiskit C++ Dependencies:** Qiskit-Aer is a validation oracle only. It must be used exclusively in the Python `tests/python/` suite. The C++ core must remain completely ignorant of Qiskit.

---

## Phase 1: Independent Ground Truth (Qiskit-Aer Validation)
**Goal:** Establish a solid, third-party source of truth (Qiskit-Aer) for noiseless Clifford+T circuits to prove UCC's statevector correctness independently of SOFT. (Note: SOFT itself uses Qiskit only for noiseless statevector verification. Noisy circuit validation is already covered by UCC's Stim-based statistical equivalence tests in `test_statistical_equivalence.py`.)

*   **Task 1.1 (Dependencies):** Add `qiskit>=1.0` and `qiskit-aer>=0.14` to the `[dependency-groups] dev` section in `pyproject.toml`.
*   **Task 1.2 (Shared Test Utilities):** Extract shared test utilities (`assert_statevectors_equal`, `binomial_tolerance`, `random_clifford_t_circuit`, `random_clifford_circuit`) from `tests/python/test_sample.py` into `tests/python/conftest.py` so they can be reused across test files.
*   **Task 1.3 (Qiskit Translator):** Create `tests/python/utils_qiskit.py`. Write a translator that takes `.stim` text and converts it into a `qiskit.QuantumCircuit`. It must support the noiseless gate set: `H`, `S`, `S_DAG`, `T`, `T_DAG`, `CX`, and mid-circuit `M`.
*   **Task 1.4 (Statevector Oracle Test):** In `tests/python/test_qiskit_aer.py`, write a test that generates random noiseless Clifford+T circuits (up to 10 qubits), runs them in `AerSimulator(method="statevector")`, and asserts exact fidelity match ($>0.9999$) with `ucc.get_statevector()`.
*   **Definition of Done (DoD):** `uv run pytest tests/python/test_qiskit_aer.py` passes reliably, proving UCC's amplitude interference is mathematically correct against an independent third-party simulator.

## Phase 2: The Fast-Fail Architecture (`OP_POSTSELECT`)
**Goal:** Implement the ability to immediately abort doomed shots to bypass deep non-Clifford evaluations, capitalizing on the 85-99% discard rates of MSC circuits.

*   **Task 2.1 (Compiler Option):** In `backend.h` and `backend.cc`, update `ucc::lower()` to accept a `bool abort_on_detector = false` argument. Expose this flag in the Python `ucc.compile()` binding.
*   **Task 2.2 (Back-End Lowering):** In `backend.cc`, when lowering `OpType::DETECTOR`, if `abort_on_detector` is true, emit an `Opcode::OP_POSTSELECT` instruction *immediately after* the `OP_DETECTOR` instruction. The payload must reference the same `detector_targets` index (using the `instr.detector` union field). The expected parity is always 0.
*   **Task 2.3 (SVM State):** In `svm.h`, add `bool discarded = false;` to `SchrodingerState` and reset it to `false` in the `reset()` method. Add `std::vector<uint8_t> passed;` to the C++ `SampleResult` struct (1=passed, 0=discarded).
*   **Task 2.4 (SVM Execution):** In `svm.cc`, implement `case Opcode::OP_POSTSELECT:`. It evaluates the XOR parity of the target measurement record indices (looked up via `detector_targets[target_idx]`). If the parity is 1 (diverges from expected 0), set `state.discarded = true;` and `return;` immediately to exit the bytecode execution loop for that shot.
*   **Task 2.5 (Python SampleResult Wrapper):** Create a Python `SampleResult` class in `ucc/__init__.py` that wraps the result arrays. It must:
    *   Have named attributes: `.measurements`, `.detectors`, `.observables`, `.passed` (where `.passed` is `None` when `abort_on_detector` was not used during compilation).
    *   Implement the 3-tuple protocol (`__iter__`, `__getitem__`, `__len__`) yielding `(measurements, detectors, observables)` for backward compatibility. Existing `meas, det, obs = ucc.sample(...)` unpacking continues to work unchanged.
    *   Users who need the `passed` array access it via `result = ucc.sample(...); result.passed`.
    *   The C++ `SampleResult` struct stays clean (pure data transport). The nanobind binding in `bindings.cc` extracts numpy arrays from it and constructs the Python `SampleResult`.
*   **Task 2.6 (Tests):** A Python test proves that compiling with `abort_on_detector=True` and running a circuit with a deterministic error that triggers a detector yields `passed=0` for those shots, and drastically reduces execution time by returning early. Existing Python tests must continue to pass with no changes to their unpacking patterns.
*   **DoD:** All existing tests pass unchanged. New tests verify postselection correctness and early-abort behavior.

## Phase 3: Composable Rank Profiling
**Goal:** Expose the dynamic rank history to visually and empirically prove the "Dynamical Shift-Rank Bound" theorem without bloating the `CompiledModule` in memory.
*   **Task 3.1 (C++ Scanner):** In `backend.h/.cc`, add a standalone function `std::vector<uint32_t> get_rank_history(const CompiledModule& prog)`. It iterates over `prog.bytecode`, keeping a running counter. Increment on `OP_BRANCH`, decrement on `OP_MEASURE_MERGE`. Record the rank at each step.
*   **Task 3.2 (Python Binding):** Expose `ucc.get_rank_history(program)` as a standalone function returning a numpy array in `bindings.cc`.
*   **Task 3.3 (Plotting Script):** Create `tools/bench/plot_rank.py` which loads the $d=5$ MSC circuit from `tools/bench/circuits/`, compiles it, calls `ucc.get_rank_history()`, and uses matplotlib to plot the rank history step function.
*   **DoD:** The `plot_rank.py` script generates a plot showing the rank "breathing" pattern. The `CompiledModule` remains lean and zero-overhead.

## Phase 4: The Benchmarking Harness
**Goal:** Create the script that runs the head-to-head comparison against the data from the SOFT paper.
*   **Task 4.1 (Data Setup):** Download the SOFT $d=3$ (`p=0.001`) and $d=5$ (`p=0.001`, `p=0.0005`) `.stim` circuits into `tools/bench/circuits/`. Include a `README.md` with attribution to the SOFT project (https://github.com/haoliri0/SOFT/, Apache-2.0 license).
*   **Task 4.2 (The Benchmark Script):** Create `tools/bench/run_soft_benchmarks.py`. It should:
    1. Load the `.stim` file.
    2. Compile it using `ucc.compile(..., abort_on_detector=True)`.
    3. Start a high-resolution timer.
    4. Call `ucc.sample(..., shots=100_000, threads=N)` where N is specified by the user.
    5. Stop the timer.
*   **Task 4.3 (Metrics Calculation):** Calculate and print exactly the metrics from Table III and V of the SOFT paper:
    *   Discard Rate: `1.0 - np.mean(result.passed)`
    *   Logical Error Rate: Mean of the final observable bit for shots where `result.passed == 1`.
    *   Throughput: Total shots / Total time (Shots/sec).
*   **DoD:** The script runs successfully on the $d=5$ MSC circuit. The console output proves that UCC produces physically correct discard rates and logical error rates, and generates the CPU throughput numbers needed for the paper's comparison table.

## Phase 5: Multi-Core Saturation (Thread Pool)
**Goal:** Parallelize the SVM `shots` loop natively in C++ to leverage 16-32 core CPUs.
*   **Task 5.1 (API Update):** Update `sample(program, shots, seed, threads)` in `svm.h/.cc` and `bindings.cc` to accept `uint32_t threads = 1`. Default is single-threaded (explicit, no surprises). The benchmark script and callers specify thread count explicitly.
*   **Task 5.2 (GIL Release):** In `bindings.cc`, wrap the `ucc::sample` call in `nanobind::gil_scoped_release` so Python does not block the C++ threads.
*   **Task 5.3 (Thread Pool Implementation):** In `svm.cc:sample()`:
    *   Pre-allocate the `SampleResult` vectors to full size. Initialize `passed` to all 1s (true).
    *   Launch $T$ `std::thread` workers, giving each thread an equal chunk of the total shots.
    *   Inside the thread lambda, allocate a *thread-local* `SchrodingerState`. Loop over the assigned global shot indices $i$. Call `state.reset(seed + i)`. Execute the program. Write results directly to the pre-allocated offsets in the `SampleResult` vectors (completely lock-free, no mutexes needed). Write `!state.discarded` to the `passed` array.
    *   Join all threads.
    *   Verify that `threads=1` has no meaningful performance regression compared to the current non-threaded code path.
*   **Task 5.4 (Benchmark):** In `tools/bench/`, create a benchmark script that varies the number of threads and measures scaling behavior (linear vs sub-linear speedup).
*   **DoD:** Running `ucc.sample(..., threads=4)` utilizes multiple CPU cores and produces bit-identical results to `threads=1` for the same seed, completing significantly faster.
*   **NOTE:** The development VM has only 2 cores. Multi-core scaling tests beyond 2 threads must be run on a separate machine.
