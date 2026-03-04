# **UCC Magic State Cultivation: Unified Implementation Plan**

## **Executive Summary & Constraints**

This plan outlines the complete end-to-end trajectory for reproducing the "ungrown" 42-qubit SOFT GPU baseline and the "grown" 463-qubit trillion-shot End-to-End curve using exact T-gate physics.

**The Trillion-Shot Data Strategy:**

Simulating $10^{12}$ shots generates terabytes of raw array data. To survive this without Out-Of-Memory (OOM) crashes or I/O bottlenecks:

1. **The Fast-Fail Engine:** We pass Sinter's automatically generated postselection_mask directly into the UCC C++ compiler. C++ natively lowers flagged detectors into OP_POSTSELECT and instantly aborts doomed shots in the VM hot-loop.
2. **Surviving-Only Memory ($\mathcal{O}(1)$ Discard Overhead):** C++ will *only* allocate and push array data to Python for shots that survived post-selection. With a 99.7% discard rate, a 10-million shot batch returns a tiny ~75 MB array instead of a 25 GB memory blowout.
3. **Universal Sinter Orchestration:** We wrap the C++ core in custom Python sinter.Sampler classes. Sinter natively handles multi-core CPU saturation, tracks discards vs. errors, and continuously appends to a flat CSV file.

**Strict Architectural Constraints:**

1. **Strictly Single-Threaded C++ Core:** Do NOT implement C++ multithreading (e.g., <thread>, mutexes). UCC acts as a pure, single-threaded execution engine. Multi-core saturation is handled entirely by Sinter's Python worker processes.
2. **Deterministic Seeding:** The RNG seed for shot $i$ must strictly be base_seed + i to ensure identical behavior regardless of how Sinter batches the workload.
3. **The 32-Byte Invariant:** The Instruction struct MUST remain exactly 32 bytes at all times.
4. **Inline Noise for Qiskit Oracle:** Do not use Qiskit's global NoiseModel for validation. Construct explicit QuantumError objects and apply them inline sequentially.

**Phase 2: Sinter-Native Fast-Fail Compilation (OP_POSTSELECT)**

**Goal:** Allow the C++ compiler to ingest Sinter's postselection_mask and natively lower targeted parity checks into early-abort instructions *without* hacking the .stim syntax.

* **Task 2.1 (Compiler API Update):** In bindings.cc and backend.h/cc, update ucc::compile() and ucc::lower() to accept an optional std::vector<uint8_t> postselection_mask (defaulting to empty).
* **Task 2.2 (Bytecode Lowering):** In backend.cc, as you iterate through the HIR and encounter an OpType::DETECTOR:
  * Check if its detector_idx is flagged in the bit-packed mask: bool postselect = mask.size() > (det_idx / 8) && (mask[det_idx / 8] & (1 << (det_idx % 8)));
  * If true, emit an Opcode::OP_POSTSELECT instruction.
  * If false (or mask is empty), emit the standard Opcode::OP_DETECTOR instruction.
* **Task 2.3 (SVM State):** In svm.h, add bool discarded = false; to SchrodingerState and reset it to false in the reset() method.
* **Task 2.4 (SVM Execution):** In svm.cc, implement case Opcode::OP_POSTSELECT:. It evaluates the XOR parity of the referenced measurements exactly like a detector.
  * If the parity is 1 (or diverges from expected): Set state.discarded = true; and return; immediately to safely exit the bytecode execution loop for that shot.
  * If the parity is 0 (success): You MUST still record the 0 to the detector record (state.det_record[det_idx++] = 0;) so the final array shape perfectly aligns with PyMatching's expectations later.

## ---

**Phase 3: Dense Survivor Sampling ($\mathcal{O}(1)$ Discard Memory)**

**Goal:** Return measurement/detector/observable arrays *only* for shots that survived post-selection, preventing memory blowouts when Sinter requests massive shot batches.

* **Task 3.1 (Stats Struct Update):** In svm.h, define a tightly-packed return structure:
  C++
  struct SampleStats {
      uint64_t total_shots = 0;
      uint64_t passed_shots = 0;
      std::vector<uint64_t> observable_ones;

      // Flattened 1D arrays ONLY for shots where !discarded
      std::vector<uint8_t> surviving_measurements; // Added for completeness, optional
      std::vector<uint8_t> surviving_detectors;
      std::vector<uint8_t> surviving_observables;
  };

* **Task 3.2 (C++ Sampler Implementation):** Implement ucc.sample_survivors(const CompiledModule& prog, uint64_t shots, uint64_t base_seed, bool keep_surviving_records = false) in svm.cc.
  * Loop i from 0 to shots. Call state.reset(base_seed + i) to guarantee deterministic reproducibility. Execute the program.
  * If !state.discarded: increment passed_shots, tally observable_ones.
  * If keep_surviving_records == true AND !state.discarded, push the contents of state.det_record and state.obs_record to the corresponding surviving_ vectors.
* **Task 3.3 (Python Bindings):** Expose ucc.sample_survivors via nanobind. Use nanobind::gil_scoped_release around the C++ call so Sinter's multiple Python worker processes can run natively in C++ simultaneously without blocking each other.

## ---

**Phase 4: Composable Rank Profiling**

**Goal:** Expose the dynamic rank history to visually and empirically prove the "Dynamical Shift-Rank Bound" theorem without bloating the CompiledModule in memory.

* **Task 4.1 (C++ Scanner):** In src/ucc/backend/backend.h and .cc, add a standalone function std::vector<uint32_t> get_rank_history(const CompiledModule& prog). It iterates over prog.bytecode, keeping a running counter. Increment on OP_BRANCH, decrement on OP_MEASURE_MERGE. Record the rank at each step.
* **Task 4.2 (Python Binding):** Expose ucc.get_rank_history(program) as a standalone function returning a numpy array in src/python/bindings.cc.
* **Task 4.3 (Plotting Script):** Create tools/bench/plot_rank.py which loads the $d=5$ MSC circuit from tools/bench/circuits/, compiles it, calls ucc.get_rank_history(), and uses matplotlib to plot the rank history step function.

## ---

**Phase 5: The 42-Qubit Cultivation Baseline (vs_soft)**

**Goal:** Prove CPU supremacy over the SOFT 16-GPU cluster using native Sinter orchestration.

* **Task 5.1 (The Sinter Adapter):** Create tools/bench/ucc_soft_sampler.py. Implement UccSoftSampler inheriting from sinter.Sampler.
  * In compiled_sampler_for_task(self, task), extract task.postselection_mask, optionally convert it to bytes/list, pass it to ucc.compile(..., postselection_mask=mask), and return a CompiledUccSoftSampler.
  * In sample(max_shots), generate a secure batch seed using import secrets; batch_seed = secrets.randbits(64).
  * Call stats = ucc.sample_survivors(..., keep_surviving_records=False).
  * Return sinter.AnonTaskStats(shots=max_shots, errors=stats.observable_ones[0] if stats.passed_shots > 0 else 0, discards=max_shots - stats.passed_shots).
* **Task 5.2 (Local Execution):** Write run_vs_soft.py that registers the sampler via Sinter's custom_decoders argument and runs sinter.collect(...) using --postselected_detectors_predicate "coords[4] == -9".
* **DoD:** Sinter natively utilizes all your CPU cores, prints a live progress bar, and writes a CSV proving a $4.59 \times 10^{-9}$ error rate and $85.6\%$ discard rate for the $d=5, p=0.001$ circuit.

## ---

**Phase 6: The 512-Qubit Core Upgrade (AVX-512)**

**Goal:** Scale the C++ Core to support the 463-qubit escape stage without breaking the 32-byte bytecode invariant.

* **Task 6.1 (CMake Config):** Add -DUCC_MAX_QUBITS=64 as a CMake cache variable. When compiled with 512, detect the architecture and add -mavx512f for native wide-vector math.
* **Task 6.2 (Instruction Indirection):** In backend.h, conditionally compile the Instruction payload. If > 64, replace the 8-byte destab_mask/stab_mask with 4-byte uint32_t destab_idx, stab_idx; pointing to a new std::vector<stim::simd_bits<kStimWidth>> mask_arena stored in the ConstantPool. Add explicit padding to ensure static_assert(sizeof(Instruction) == 32) passes.
* **Task 6.3 (AGMatrix Upgrade):** Update AGMatrix for > 64 to use an array of simd_bits (or stim::simd_bit_table). Its apply() method must use Stim's .for_each_set_bit() on the sign trackers to XOR the wide columns efficiently.
* **Task 6.4 (Routing Abstraction):** In svm.cc, abstract resolve_sign(). If <= 64, use standard popcount. If > 64, fetch from mask_arena and use Stim's SIMD-backed AND/XOR popcount methods. Verify the pure-Clifford test suite passes identically.

## ---

**Phase 7: The End-to-End Splicer & Decoder Hijack**

**Goal:** Splice T-gates into Gidney's 463-qubit circuit and trick the PyMatching decoder into evaluating them via Sinter.

* **Task 7.1 (The Splicer Script):** Write tools/bench/stitch_escape.py.
  1. Load Gidney's generated 463-qubit S-gate end-to-end circuit.
  2. Slice out the initial cultivation block and replace it with the exact T-gate and feedforward operations from the SOFT .stim file.
  3. Wrap the final MPP Y... check with T/T_DAG to fix the $H_{XY}$ transversality mapping.
     *(Note: We do not need to string-replace detectors because Sinter handles the postselection_mask automatically via the coordinate predicate).*
* **Task 7.2 (The DEM Trick):** PyMatching requires a DEM. Stim refuses to generate DEMs for T-gates. Generate the DEM using the **unpatched S-gate circuit** via stim.Circuit.detector_error_model(). The topological error graph is physically identical.
* **Task 7.3 (The Sinter Hijack):** Create UccDesaturationSampler (adapting Gidney's custom DesaturationSampler).
  * In its compiled_sampler_for_task, compile the UCC program with the postselection mask, and instantiate pymatching.Matching.from_detector_error_model(dem).
  * In sample(), call stats = ucc.sample_survivors(..., keep_surviving_records=True).
  * Take stats.surviving_detectors and stats.surviving_observables, reshape them into 2D arrays (passed_shots, num_detectors) and (passed_shots, num_observables), repack using np.packbits(..., bitorder='little') to perfectly mimic Stim's C-contiguous binary layout.
  * Feed the repacked arrays directly into PyMatching (self.gap_decoder.decode_batch()) to compute complementary gaps. Compare predictions against stats.surviving_observables to count errors, and return a sinter.AnonTaskStats.
* **Task 7.4 (Transitive Validation):** Run the patched End-to-End circuit at $p=0.0$ (no noise). Assert the decoder predicts exactly 0 logical errors, mathematically proving the T-gate splicing and 463-qubit feedforward routing is completely coherent.

## ---

**Phase 8: Trillion-Shot Distributed Cloud Execution**

**Goal:** Deploy to a spot cluster and safely aggregate a trillion shots. In this phase you prepare a workflow for running the distrubitued cloud data collection. Have the user (human) able to run this twice. Once with UCC on the original Gidney circuits WITHOUT replacing with T-gates. This will ensure we can recover and match his original results. Once confident, can then run using the T-gate specific circuits created in Phase 7.

* **Task 8.1 (The Runner Script):** Write run_e2e_node.py using sinter.collect(..., decoders=['ucc_desaturation'], save_resume_filepath=f"stats_{socket.gethostname()}.csv").
  * *Crucial AWS c7i config:* Set Sinter's num_workers = max(1, os.cpu_count() // 2) to use **physical cores only**. Running heavy AVX-512 math on Hyper-Threads causes severe L3 cache evictions and negative scaling.
* **Task 8.2 (Cluster Deployment):** Deploy to a fleet of AWS c7i (Sapphire Rapids) spot instances. Build UCC natively on the node with -DUCC_MAX_QUBITS=512 -DCMAKE_CXX_FLAGS="-march=native -O3" to unlock the hardware extensions.
* **Task 8.3 (S3 Aggregation):** Run a background bash daemon on each node that executes aws s3 cp stats_$(hostname).csv s3://my-bucket/ every 5 minutes.
* **Task 8.4 (Final Merge):** Download the bucket contents locally and run sinter combine *.csv > final_stats.csv. Feed this into Gidney's step3_plot script to generate Figure 14 with true T-gate physics!
