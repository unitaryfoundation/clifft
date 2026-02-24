# UCC SOFT Validation & Benchmark Implementation Plan

## Executive Summary & Constraints
You are implementing the validation and benchmarking suite to compare UCC against the SOFT simulator for Magic State Cultivation (MSC). The goal is to prove that UCC's Ahead-Of-Time (AOT) "Symplectic Determinism" architecture allows a conventional multi-core CPU to compete with, or outperform, a 16-GPU cluster by completely eliminating dynamic $\mathcal{O}(n^2)$ tableau updates and aggressively early-aborting failed shots.

To ensure unimpeachable correctness, we will implement an independent exact analytical validation layer using `qiskit-aer`'s density matrix simulator to verify our exact noisy physics engine before targeting the massive $d=5$ MSC circuits.

You can look at the SOFT codebase https://github.com/haoliri0/SOFT/ as necessary to understand the best way to do comparisons. The circuits we want to test are in `magic_state_cultivation/circuits/` folder in that repository.

**Strict Constraints for this specific plan (Overrides general guidelines if in conflict):**
1. **No External Multithreading Dependencies:** Do not introduce OpenMP, Intel TBB, or other heavy threading libraries to the C++ core. Use standard `<thread>` and `<future>` for parallelizing the Monte Carlo shots. Have the user be explicit on the number of threads they want to use.
2. **Deterministic Concurrency:** The RNG seed for shot $i$ must strictly be `base_seed + i`, regardless of which thread executes it. Results must be bit-identical whether running on 1 core or 32 cores.
3. **The 32-Byte Invariant:** The `Instruction` struct is exactly 32 bytes. Do not alter its size when implementing `OP_POSTSELECT`. Reuse existing union payload structures (e.g., the `detector` variant).
4. **Zero Qiskit C++ Dependencies:** Qiskit-Aer is a validation oracle only. It must be used exclusively in the Python `tests/python/` suite. The C++ core must remain completely ignorant of Qiskit.
5. **Inline Noise Instructions for Qiskit:** Do NOT use `qiskit_aer.noise.NoiseModel`. STIM applies noise sequentially at specific moments. You must translate STIM noise operations into Qiskit `QuantumError` objects and apply them directly to the `QuantumCircuit` using `error.to_instruction()`.

---

## Phase 1: Independent Ground Truth (Qiskit-Aer Statevector Validation)
**Status: ✅ COMPLETE**
*(Note: Phase 1 is already completed for noiseless circuits in `test_qiskit_aer.py`. We proceed to extending it for exact noise validation).*

---

## Phase 2: Qiskit Density Matrix Oracle (Analytical Noise Validation)
**Goal:** Prove that UCC's Monte Carlo execution perfectly converges to Qiskit's exact analytical probability distribution across all possible correlation orders, ruling out the engine "faking" higher-order correlations.

*   **Task 2.1 (Explicit Pauli Noise Construction):** In `tests/python/utils_qiskit.py`, update `stim_to_qiskit` to support noise gates. Import `pauli_error` and `depolarizing_error` from `qiskit_aer.noise`. When encountering `X_ERROR(p)`, `Y_ERROR(p)`, `Z_ERROR(p)`, `DEPOLARIZE1(p)`, or `DEPOLARIZE2(p)`, construct the exact Qiskit `QuantumError` and append it to the circuit via `circ.append(error.to_instruction(), [qubits])`.
*   **Task 2.2 (Readout Noise Substitution):** When encountering `M(p)` or `READOUT_NOISE(p)`, the simplest mathematically equivalent approach for Z-basis measurement is to insert an `X_ERROR(p)` immediately before the Qiskit `measure` instruction. Implement this inline substitution.
*   **Task 2.3 (The Density Matrix Runner):** In `tests/python/utils_qiskit.py`, write `get_exact_probabilities(stim_text: str) -> dict[str, float]`. At the end of the translated circuit, append `circ.save_probabilities_dict()` on the measured classical registers. Execute the circuit using `AerSimulator(method="density_matrix")`. Because the noise is inline, do not pass a `noise_model` to the simulator.
*   **Task 2.4 (Extraction & Endianness):** Extract the probabilities from the result object (`result.data()["probabilities"]`). Crucial: Qiskit formats bitstrings in little-endian order (last measured bit is MSB). UCC/Stim outputs bitstrings in chronological measurement order. You must reverse or map the Qiskit bitstring keys so they match UCC's output ordering conventions.
*   **Task 2.5 (TVD Validation Tests):** Create `tests/python/test_density_matrix_oracle.py`. Write a fuzzer to generate random 4-qubit and 5-qubit circuits containing dense Clifford+T unitary logic interleaved with noise (`X_ERROR`, `DEPOLARIZE1`, `DEPOLARIZE2`) and terminal measurements. Call the Qiskit oracle for exact probabilities $P_{exact}(x)$, run UCC for $N=100,000$ shots to get empirical probabilities $P_{ucc}(x)$, and assert the Total Variation Distance (TVD) is $< 0.03$.
*   **Definition of Done (DoD):** `uv run pytest tests/python/test_density_matrix_oracle.py` passes reliably over 10 different random topologies, mathematically guaranteeing that UCC's implementation of multi-qubit Pauli interference, gap-sampled stochastic noise, and measurement merging is flawless across all higher-order correlations.

---

## Phase 3: The Fast-Fail Architecture (`ASSERT_DETECTOR`)
**Goal:** Implement the ability to immediately abort doomed shots to bypass deep non-Clifford evaluations, capitalizing on the 85-99% discard rates of MSC circuits. We will do this by extending the Stim syntax with `ASSERT_DETECTOR`.

*   **Task 3.1 (Parser Enhancement):** In `src/ucc/circuit/gate_data.h` and `src/ucc/circuit/parser.cc`, add `ASSERT_DETECTOR` to `GateType` and `kGateNames`. Parse it identically to `DETECTOR` (accepting `rec[-k]` targets). It must increment the `num_detectors` counter just like a normal detector.
*   **Task 3.2 (HIR & Back-End Lowering):** In `src/ucc/frontend/hir.h`, add `OpType::ASSERT_DETECTOR`. Update the front-end to emit it. In `src/ucc/backend/backend.cc`, lower `ASSERT_DETECTOR` to a new `Opcode::OP_POSTSELECT` instruction. The payload must reference the `detector_targets` pool exactly as `OP_DETECTOR` does.
*   **Task 3.3 (SVM State):** In `src/ucc/svm/svm.h`, add `bool discarded = false;` to `SchrodingerState` and reset it to `false` in the `reset()` method. Add `std::vector<uint8_t> passed;` to the C++ `SampleResult` struct.
*   **Task 3.4 (SVM Execution & Padding):** In `src/ucc/svm/svm.cc`, implement `case Opcode::OP_POSTSELECT:`. It evaluates the XOR parity of the referenced measurements. If the parity is 1 (or diverges from the expected parity), the shot has failed:
    *   Set `state.discarded = true;`.
    *   **Crucial padding:** Since `reset()` intentionally does not scrub the arrays between rounds to avoid memory overhead, an early abort leaves garbage in the tail end of the arrays. You MUST use `std::fill` to zero out the remaining elements of `state.meas_record` (from `meas_idx` to end) and `state.det_record` (from `det_idx` to end) with `0`. `obs_record` is already zeroed on reset, so it does not need padding.
    *   `return;` immediately to exit the bytecode execution loop for that shot.
    *   If parity is 0 (success), just write `0` to `state.det_record[det_idx++]` and `break;`.
*   **Task 3.5 (Python SampleResult Wrapper):** Create a Python `SampleResult` class in `src/python/ucc/__init__.py` that wraps the result arrays. It must:
    *   Have named attributes: `.measurements`, `.detectors`, `.observables`, `.passed`.
    *   Implement the 3-tuple protocol (`__iter__`, `__getitem__`, `__len__`) yielding `(measurements, detectors, observables)` for backward compatibility.
*   **Task 3.6 (Tests):** A Python test proves that using `ASSERT_DETECTOR` successfully aborts execution on a deterministic error, marks `passed=0`, pads the rest of the measurement/detector record with 0s, and drastically reduces execution time.
*   **Definition of Done (DoD):** All existing tests pass unchanged. New tests verify postselection correctness, early-abort behavior, and correct 0-padding of remaining results.

---

## Phase 4: Composable Rank Profiling
**Goal:** Expose the dynamic rank history to visually and empirically prove the "Dynamical Shift-Rank Bound" theorem without bloating the `CompiledModule` in memory.

*   **Task 4.1 (C++ Scanner):** In `src/ucc/backend/backend.h` and `.cc`, add a standalone function `std::vector<uint32_t> get_rank_history(const CompiledModule& prog)`. It iterates over `prog.bytecode`, keeping a running counter. Increment on `OP_BRANCH`, decrement on `OP_MEASURE_MERGE`. Record the rank at each step.
*   **Task 4.2 (Python Binding):** Expose `ucc.get_rank_history(program)` as a standalone function returning a numpy array in `src/python/bindings.cc`.
*   **Task 4.3 (Plotting Script):** Create `tools/bench/plot_rank.py` which loads the $d=5$ MSC circuit from `tools/bench/circuits/`, compiles it, calls `ucc.get_rank_history()`, and uses matplotlib to plot the rank history step function.
*   **Definition of Done (DoD):** The `plot_rank.py` script generates a plot showing the rank "breathing" pattern. The `CompiledModule` remains lean and zero-overhead.

---

## Phase 5: Multi-Core Saturation (Thread Pool)
**Goal:** Parallelize the SVM `shots` loop natively in C++ to leverage 16-32 core CPUs.

*   **Task 5.1 (API Update):** Update `sample(program, shots, seed, threads)` in `src/ucc/svm/svm.h`, `.cc` and `src/python/bindings.cc` to accept `uint32_t threads = 1`. Default is single-threaded (explicit, no surprises). The benchmark script and callers specify thread count explicitly.
*   **Task 5.2 (GIL Release):** In `src/python/bindings.cc`, wrap the `ucc::sample` call in `nanobind::gil_scoped_release` so Python does not block the C++ threads.
*   **Task 5.3 (Thread Pool Implementation):** In `svm.cc:sample()`:
    *   Pre-allocate the `SampleResult` vectors to full size. Initialize `passed` to all 1s (true).
    *   Launch $T$ `std::thread` workers, giving each thread an equal chunk of the total shots.
    *   **Architecture Note (No Thread Pinning):** Do NOT use OS-specific thread affinity (like `pthread_setaffinity_np`). Standard `<thread>` relies on the OS scheduler, preserving cross-platform compatibility. Hardware-specific pinning is handled externally via `taskset` when running the benchmarks.
    *   Inside the thread lambda, allocate a *thread-local* `SchrodingerState`. Loop over the assigned global shot indices $i$.
    *   **Architecture Note (PRNG Seeding):** Call `state.reset(seed + i)`. Do NOT use xoshiro256++ `jump()` functions. We seed *per shot* via SplitMix64 to guarantee deterministic replayability regardless of which thread executed the shot.
    *   Execute the program. Write results directly to the pre-allocated offsets in the `SampleResult` vectors (completely lock-free, no mutexes needed). Write `!state.discarded` to the `passed` array.
    *   Join all threads.
    *   Verify that `threads=1` has no meaningful performance regression compared to the current non-threaded code path.
*   **Definition of Done (DoD):** Running `ucc.sample(..., threads=4)` utilizes multiple CPU cores and produces bit-identical results to `threads=1` for the same seed, completing significantly faster.

---

## Phase 6: O(1) Memory Aggregation & Error Replay (`sample_stats`)
**Goal:** Prevent Out-Of-Memory (OOM) crashes and memory bandwidth bottlenecks when simulating billions of shots by tracking only the aggregate logical statistics natively in C++, completely bypassing array allocations. Concurrently, capture the exact seeds of failed logical shots for later replay.

*   **Task 6.1 (Stats Struct Upgrade):** In `src/ucc/svm/svm.h`, define a new return struct for statistics:
    ```cpp
    struct SampleStats {
        uint64_t total_shots = 0;
        uint64_t passed_shots = 0;
        std::vector<uint64_t> observable_ones; // Count of 1s per observable (for passed shots only)
        std::vector<uint64_t> logical_error_seeds; // Exact seeds of shots that passed but had a logical error
    };
    ```
*   **Task 6.2 (C++ Implementation):** Implement `SampleStats sample_stats(const CompiledModule& program, uint64_t shots, uint64_t seed, uint32_t threads, uint32_t max_errors_to_capture = 1000);` in `svm.cc`.
    *   Instead of allocating giant `SampleResult` arrays, create a single `SampleStats` object.
    *   Inside the thread pool, each thread must keep its own local counters (`local_passed`, `local_obs_ones`) and a local vector `local_error_seeds`.
    *   As each shot $i$ completes, if `!state.discarded`:
        *   Increment `local_passed`.
        *   Iterate through `state.obs_record` and add the bits to `local_obs_ones`.
        *   **Logical Error Capture:** If the primary logical observable failed (e.g., `state.obs_record[0] == 1`) AND `local_error_seeds.size() < max_errors_to_capture`, append the exact seed (`seed + i`) to `local_error_seeds`.
    *   After the thread's loop finishes, use a `std::mutex` to safely add the local counters and append the local error seeds into the global `SampleStats` return object.
*   **Task 6.3 (Replay API):** Implement `SampleResult sample_with_seeds(const CompiledModule& program, const std::vector<uint64_t>& seeds, uint32_t threads)` in `svm.cc`. This bypasses the sequential `seed + i` loop and instead runs the full dense array collection (returning `measurements`, `detectors`, `observables`) specifically for the provided list of seeds.
*   **Task 6.4 (Python Binding):** In `src/python/bindings.cc`, expose `ucc.sample_stats` and `ucc.sample_with_seeds` to Python (remembering `nanobind::gil_scoped_release`).
*   **Definition of Done (DoD):** A Python test runs `ucc.sample_stats` for 10 million shots using threads, without allocating megabytes of RAM. It returns identical statistical ratios to a standard `ucc.sample` call and successfully populates the `logical_error_seeds` array when errors occur. It successfully replays those seeds via `sample_with_seeds`.

---

## Phase 7: The Checkpointing Benchmarking Harness
**Goal:** Create the script that runs the head-to-head comparison against the data from the SOFT paper, utilizing chunking, `tqdm`, and checkpointing to safely execute billions of shots.

*   **Task 7.1 (Data Setup):** The SOFT $d=3$ (`p=0.001`) and $d=5$ (`p=0.0005`, `p=0.001`, `p=0.002`, `p=0.003`, `p=0.004`, `p=0.005`) `.stim` circuits are located in `magic_state_cultivation/circuits/`. Ensure they are accessible from the benchmark script.
*   **Task 7.2 (The Benchmark Script):** Create `tools/bench/run_soft_benchmarks.py`. It should:
    1. Load the `.stim` file and replace `DETECTOR` with `ASSERT_DETECTOR`.
    2. Compile using `ucc.compile()`.
    3. Accept CLI arguments for `--shots` (e.g., 100,000,000), `--chunk-size` (e.g., 10,000,000), and `--threads`.
    4. Initialize an empty statistics accumulator.
*   **Task 7.3 (Chunking & Tqdm Loop):**
    *   Add `tqdm` to the Python dependencies.
    *   Wrap the execution in a `tqdm` loop iterating over the chunks.
    *   For each chunk $c$, call `ucc.sample_stats(..., shots=chunk_size, seed=start_seed + c * chunk_size, threads=N)`.
    *   Accumulate the returned `SampleStats` into the global running totals.
*   **Task 7.4 (JSON Checkpointing):** At the end of each chunk, immediately dump the current accumulated statistics (including the `logical_error_seeds` list) to a JSON file (e.g., `checkpoint_d5_p0.001.json`). If the script is interrupted, it should be able to load this file and resume from the last completed chunk.
*   **Task 7.5 (Metrics Calculation & Reveal):**
    *   At the end of the script (or when loading from a checkpoint), calculate and print exactly the metrics from Table III and V of the SOFT paper:
        *   Discard Rate: `1.0 - (passed_shots / total_shots)`
        *   Logical Error Rate: `observable_ones[0] / passed_shots`.
        *   Throughput: `total_shots / total_time_seconds` (Shots/sec).
    *   Add a `--reveal` flag to the script. If set, it skips the main simulation, loads the JSON checkpoint, extracts the `logical_error_seeds`, and calls `ucc.sample_with_seeds()`. It then prints the raw detection/measurement events of the failures, mirroring SOFT's `.reveal` file capabilities.
*   **Definition of Done (DoD):** The script successfully processes 100 million shots of the $d=5$ MSC circuit, displays a progress bar, safely writes a checkpoint file, calculates the correct logical failure rates, and can use `--reveal` to dump the exact failed trajectories.
