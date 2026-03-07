# UCC Magic State Cultivation: Trillion-Shot & 512-Qubit Scale

## Executive Summary & Constraints

This plan outlines the complete end-to-end trajectory for reproducing the 463-qubit trillion-shot Magic State Cultivation curve.

To accelerate validation, we will first implement the fast-fail execution and sampling architecture to test the 42-qubit baseline (which fits in the MVP's 64-qubit limit). Then, we will widen the 1D Pauli frame trackers using C++ templates to achieve extreme speed at the 512-qubit scale for the final end-to-end demonstration.

**Strict Constraints:**
1. **The 32-Byte Invariant:** The `Instruction` struct MUST remain exactly 32 bytes at all times.
2. **Single-Threaded C++ Core:** Do NOT implement C++ multithreading (e.g., `<thread>`). Multi-core saturation is handled entirely by Sinter's Python worker processes.
3. **Survivor Sampling Only:** To prevent OOMs during 10-million shot batches, C++ must only allocate and push array data to Python for shots that survived post-selection.
4. **Reproducible:** When it comes to actually sampling the circuits (so not the changes to UCC itself), write scripts to do them in the `paper/magic` directory. The scripts should also generate any plots we would plan to use in a paper.

---

## Current Status (2026-03-07)

- **Phase 1:** COMPLETE. Merged as PR #76 at commit `c8b5f29`.
- **Phase 2:** COMPLETE. On branch `feat/survivor-sampling-sinter` (PR #78).
- **Phase 3:** COMPLETE. Sinter adapter working end-to-end on PR #78.
- **Phases 4-6:** Not started.

### Implementation Notes (deviations from original plan)

- **Phase 2 `SurvivorResult`** has additional fields beyond the plan:
  `logical_errors` (count of shots with any observable flipped, required by
  Sinter's `errors` contract) and `observable_ones` (per-observable counts).
- **Phase 3 adapter** does not use `postselected_detectors_predicate` CLI flag.
  Instead, `UccCompiledSampler.__init__` unpacks Sinter's bit-packed
  `task.postselection_mask` directly via `np.unpackbits(bitorder='little')`.
- **PRNG seeding** was redesigned during Phase 2/3 review. The C++ VM now
  seeds xoshiro256++ once per batch (256-bit OS hardware entropy by default)
  and streams forward across shots. `reset()` clears only memory arrays, not
  the PRNG. Deterministic replay available via explicit `seed` parameter.
  This eliminates birthday-paradox seed collisions at trillion-shot scale.
- **Noise hazard clamp** tightened from `0.9999` to `1.0 - 1e-15` to ensure
  `p=1.0` noise channels reliably fire under the streaming PRNG.
- **Vendored circuit**: `paper/magic/circuits/circuit_d3_p0.001.stim` from
  SOFT repo (Apache-2.0). Contains T/T_DAG gates. Cannot be parsed by Stim
  (non-Clifford); UCC compiles it natively. Used for future Phase 5 work.
- **Current test circuit**: `tools/bench/target_qec.stim` is a 14-qubit
  S-gate Clifford proxy (22 detectors, 1 observable, 13 postselected).
  All current validation uses this circuit. Reproducing the SOFT paper's
  exact statistics (85.60% discard, 4.59e-9 error rate at p=0.001) requires
  billions of shots on beefier hardware (Phase 6 territory).

---

## Phase 1: Sinter-Native Fast-Fail Compilation (`OP_POSTSELECT`) [DONE]

**Goal:** Allow the C++ compiler to ingest Sinter's `postselection_mask` and natively lower targeted parity checks into early-abort instructions.

*   **Task 1.1 (Compiler API):** Update `ucc::lower()` to accept an optional `std::vector<uint8_t> postselection_mask`.
*   **Task 1.2 (Bytecode Lowering):** As the Back-End iterates the HIR, if it encounters a `DETECTOR` flagged in the mask, emit `OP_POSTSELECT` instead of `OP_DETECTOR`.
*   **Task 1.3 (SVM Execution):** Implement `OP_POSTSELECT`. It evaluates the XOR parity of the referenced measurements. If it fails, set a `discarded = true` flag on the state and `return` immediately, exiting the bytecode loop. Ensure a `0` is still recorded to `det_record` to maintain PyMatching array shapes.
*   **DoD:** Doomed shots instantly abort at the exact instruction they fail, saving deep non-Clifford FLOPs.

## Phase 2: Dense Survivor Sampling ($\mathcal{O}(1)$ Discard Memory) [DONE]

**Goal:** Return measurement/detector arrays *only* for shots that survived post-selection.

*   **Task 2.1 (Stats Struct Update):** Define `SampleStats` returning `total_shots`, `passed_shots`, and flattened 1D arrays ONLY for shots where `!discarded`.
*   **Task 2.2 (C++ Sampler):** Implement `ucc.sample_survivors()` via nanobind. Use `nanobind::gil_scoped_release` so Sinter's multiple Python worker processes can run C++ concurrently.
*   **DoD:** Sinter can request 10M shots. With a 99.7% discard rate, UCC returns a tiny ~75 MB numpy array to Python instead of a 25 GB blowout.

## Phase 3: The 42-Qubit Cultivation Baseline (vs. SOFT GPU) [DONE]

**Goal:** Prove UCC's CPU execution speed and exact numerical equivalence against the Li et al. SOFT simulator's 16-GPU cluster baseline using their exact 42-qubit circuits (which natively fit inside the `stim::bitword<64>` MVP limit).

*   **Task 3.1 (The Sinter Adapter):** Create `paper/magic/ucc_soft_sampler.py` and implement `UccSoftSampler` inheriting from `sinter.Sampler`.
    *   Override `compiled_sampler_for_task(self, task)`. Extract `task.postselection_mask`, pass it directly to `ucc.compile(..., postselection_mask=mask)`, and return a custom compiled sampler.
    *   Override `sample(self, max_shots)`. Generate a random 64-bit batch seed (`import secrets; secrets.randbits(64)`) to ensure cross-worker determinism. Call `stats = ucc.sample_survivors(..., keep_surviving_records=False)`.
    *   Return a `sinter.AnonTaskStats` object populating `shots`, `errors` (using `stats.observable_ones[0]`), and `discards` to satisfy Sinter's pipeline contract.
*   **Task 3.2 (Execution & Validation):** Write `run_vs_soft.py` to register the sampler via Sinter's `custom_decoders` argument. Target the exact distance-5 MSC circuits from the SOFT repository (e.g., `haoliri0-soft/magic_state_cultivation/circuits/circuit_d5_p0.001.stim`).
*   **Task 3.3 (Postselection Filtering):** Use Sinter's `--postselected_detectors_predicate "coords[4] == -9"` flag. This natively instructs Sinter to build the post-selection mask, because the SOFT circuits use a 5th coordinate of `-9` to flag cultivation success checks.
*   **DoD:** Sinter natively utilizes all physical CPU cores and outputs a CSV exactly matching the SOFT paper's Table III: an $85.60\%$ discard rate and a $4.59 \times 10^{-9}$ logical error rate for $p=0.001$, proving $\mathcal{O}(1)$ performance scalability on a single CPU core.

## Phase 4: 512-Qubit Template Monomorphization (AVX-512)

**Goal:** Scale the C++ Core's Pauli trackers to support the 463-qubit escape stage while preserving perfect inline memory layouts.

*   **Task 4.1 (Template Architecture):** In `svm.h`, template `SchrodingerState<W>` and the inner execution loop `execute_impl<W>`.
*   **Task 4.2 (Bitword Upgrade):** Replace the hardcoded `stim::bitword<64> p_x, p_z;` trackers with `stim::bitword<W>`. (We will instantiate this for $W=64$, $W=256$, and $W=512$).
*   **Task 4.3 (CMake Config):** Add `-DUCC_MAX_QUBITS=512` as a CMake cache variable. When active, conditionally compile the $W=512$ path and add `-mavx512f` for native wide-vector math on compatible CPUs.
*   **DoD:** The codebase compiles. The 32-byte `Instruction` struct is unmodified. Catch2 tests pass flawlessly for circuits utilizing 400+ qubits, automatically vectorizing the frame XORs over AVX registers.

## Phase 5: The 463-Qubit End-to-End Splicer & Decoder Hijack

**Goal:** Convert the generated 463-qubit S-gate proxy circuit into a true T-gate circuit, apply errata topological fixes, and trick the PyMatching decoder into evaluating it via Sinter.

*   **Task 5.1 (Circuit Upgrades):** The default pipeline analysis generates `end2end-inplace-distillation` circuits using Clifford `S` gates as a scalable proxy. Write `paper/magic/stitch_escape.py` to apply the required physical fixes to create the true non-Clifford circuit:
    1.  **Gate Swap:** Replace all $S$ and $S^\dagger$ gates with $T$ and $T^\dagger$ gates.
    2.  **Measurement Fix:** The color code does not have a transversal $X+Y$ measurement. Fix the final logical check by explicitly wrapping the `MPP Y...` observable measurement with $T$ and $T^\dagger$ gates on the data qubits to correctly rotate the basis into $H_{XY}$.
    3.  **Transversality Errata Fix:** To initialize all stabilizers to the $+1$ eigenvalue, the logical $H_{XY}^L$ must be made transversal by using $H_{NXY}$ on specific data qubits. Swap $T^\dagger$ for $T$ (and vice versa) on these highlighted data qubits during the logical check.
    4.  **Feedforward Pauli Corrections:** During the $d=3 \to d=5$ code growth step, new non-deterministic stabilizers are formed. Instead of evaluating complex lookup tables, inject fast, single-classical-control Pauli feedback instructions (e.g., `CZ rec[-1] target`) to dynamically restore these stabilizers to the $+1$ eigenvalue based on the ancilla measurements.
*   **Task 5.2 (The DEM Trick):** `PyMatching` requires a Detector Error Model (DEM) to decode syndrome data, but Stim natively refuses to generate DEMs for any circuit containing non-Clifford `T` gates. We must generate the DEM from the **unpatched S-gate proxy circuit** via `stim.Circuit.detector_error_model()`. Because the topological error graph is physically identical, this allows PyMatching to successfully route and decode the true T-gate circuit's results.
*   **Task 5.3 (The PyMatching Sinter Hijack):** Create `UccDesaturationSampler` inheriting from `sinter.Sampler`.
    *   Initialize `pymatching.Matching.from_detector_error_model(dem)` during compilation using the proxy DEM.
    *   During sampling, call `stats = ucc.sample_survivors(..., keep_surviving_records=True)`.
    *   Take the returned flattened `stats.surviving_detectors` and `stats.surviving_observables` arrays and reshape them into 2D arrays: `(passed_shots, num_detectors)` and `(passed_shots, num_observables)`.
    *   Pack the arrays using `np.packbits(..., bitorder='little')` to perfectly mimic Stim's C-contiguous binary layout required by PyMatching.
    *   **Gap Evaluation:** Evaluate complementary gaps by forcing the observable detector node to 1 to get `on_weight`, and 0 to get `off_weight`. Calculate the gap as `|on_weight - off_weight| * decibels_per_w`.
    *   Populate Sinter's `custom_counts` dictionary with keys like `C{gap}` (correct) and `E{gap}` (error) to trace the discard-vs-error Pareto frontier. Return `sinter.AnonTaskStats`.
*   **DoD:** Run the patched End-to-End circuit at $p=0.0$ (no noise). The decoder must predict exactly 0 logical errors out of 10,000 shots, mathematically proving the T-gate splicing, Pauli feedforward, and 463-qubit routing are completely coherent.

## Phase 6: Trillion-Shot Distributed Cloud Execution

**Goal:** Deploy to a cloud cluster and safely aggregate a trillion shots to map the final performance curve.

*   **Task 6.1 (Active Dimension Profiling):** Create a script `plot_active_k.py`. As the C++ Back-End compiles the massive 463-qubit circuit, record the `active_k` dimension after every instruction. Plot it to visually prove to users that the statevector physically shrinks and expands predictably within system memory limits.
*   **Task 6.2 (Cluster Deployment):** Write a runner script `run_e2e_node.py` using `sinter.collect(..., decoders=['ucc_desaturation'])`. Deploy this to a fleet of AWS CPU spot instances (e.g., `c7i` instances built with `-DUCC_MAX_QUBITS=512 -mavx512f` for AVX-512 hardware vectorization).
*   **Task 6.3 (Hyper-Threading Guard):** Configure Sinter's worker count to `max(1, os.cpu_count() // 2)` to use physical cores only. Running heavy AVX-512 math on Hyper-Threads causes severe L3 cache evictions and negative scaling.
*   **Task 6.4 (S3 Aggregation):** Run a background daemon on each node that executes `aws s3 cp` to back up the CSV stats periodically. Download the bucket contents locally and run `sinter combine *.csv > final_stats.csv` to hit the trillion-shot target.
