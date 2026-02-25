# **UCC Implementation Plan: Phase 2 (Stochastic Noise & QEC)**

**THIS PLAN IS COMPLETE AND RETAINED FOR HISTORICAL PURPOSES**

## **Executive Summary & Constraints**

This phase upgrades the UCC architecture from a noiseless pure-state simulator into a fault-tolerant stochastic execution engine capable of acting as a drop-in replacement for stim.compile\_detector\_sampler() (it won't be exact drop-in, but basically the same interface in spirit for extracting detector and/or observable samples).

**Strict Constraints for Phase 2:**

1. **The 32-Byte Invariant:** The HeisenbergOp and Instruction structs MUST remain exactly 32 bytes. Multi-channel noise masks (like DEPOLARIZE2) and variable-length detector target lists must be stored in side-tables in the HirModule and ConstantPool, referenced by a 32-bit index.
2. **Geometric Gap Sampling:** Quantum Pauli noise (X\_ERROR, DEPOLARIZE1, etc.) MUST NOT emit 32-byte Instruction bytecode. The Back-End must extract all quantum noise into a ConstantPool::noise\_schedule side-table. The VM will execute noise by sampling the distance to the next error, keeping the quantum hot-loop entirely free of if(rand() \< p) branch checks.
3. **Readout Noise is Classical:** M(p) must be decomposed by the parser into a clean M followed by a classical READOUT\_NOISE(p) operation. Readout noise flips the classical measurement record bit, not the quantum state.
4. **Coordinate Ignorance:** Coordinate annotations (e.g., DETECTOR(1, 2)) must be parsed successfully without throwing syntax errors, but the float data must be discarded immediately to preserve memory.

Also, please follow guidelines on AGENTS.md on how to do your work. If you find any conflicts between this document and other design documents, please ask for clarificaiton before doing any work.

**Phase 2.1: Parser Enhancements & QEC Syntax**

**Goal:** Parse the provided target circuit, decomposing noisy measurements and gracefully ignoring coordinate data.

* **Task 2.1.1 (Gate Definitions):** Update gate\_data.h/parser.cc to add: X\_ERROR, Y\_ERROR, Z\_ERROR, DEPOLARIZE1, DEPOLARIZE2, DETECTOR, OBSERVABLE\_INCLUDE, QUBIT\_COORDS, SHIFT\_COORDS, and a new internal synthetic gate READOUT\_NOISE.
* **Task 2.1.2 (Annotation Stripping):** Update the parser loop to gracefully consume and discard multiple comma-separated float arguments inside parentheses (e.g., DETECTOR(1.25, 0.25, 0)). For noise gates like X\_ERROR(0.001) or OBSERVABLE\_INCLUDE(0), extract the first float into AstNode::arg.
* **Task 2.1.3 (Readout Noise Decomposition):** In the parser, if a measurement gate (M, MX, MY) has an argument p \> 0.0, emit the standard clean measurement node (with arg=0.0), followed immediately by a new AstNode of type READOUT\_NOISE targeting the newly created rec\[-1\] absolute index with arg \= p. Handle multi-target noisy measurements by interleaving them: M 0, READOUT\_NOISE rec\[-1\], M 1, READOUT\_NOISE rec\[-1\]. *(For R and RX, emit: M, READOUT\_NOISE, then the conditional CX/CZ feedback).*
* **Task 2.1.4 (QEC Annotations):** Allow DETECTOR and OBSERVABLE\_INCLUDE to parse lists of rec\[-k\] targets. Maintain counters for num\_detectors and num\_observables (max observable index \+ 1\) in the Circuit struct.
* **DoD:** A Catch2 test parses the user's target circuit without throwing, correctly reporting the number of qubits, measurements, detectors, and observables. QUBIT\_COORDS and SHIFT\_COORDS are silently discarded. M(0.001) 0 expands into two nodes.

## **Phase 2.2: Front-End HIR Emission (Noise & Logic)**

**Goal:** Translate AST noise and logic nodes into the Heisenberg IR, resolving all multi-Pauli channels and relative indices.

* **Task 2.2.1 (HIR Structs):**
  * Add struct NoiseChannel { uint64\_t destab\_mask; uint64\_t stab\_mask; double prob; };
  * Add struct NoiseSite { std::vector\<NoiseChannel\> channels; };
  * Add std::vector\<NoiseSite\> noise\_sites to HirModule.
  * Add std::vector\<std::vector\<uint32\_t\>\> detector\_targets and observable\_targets to HirModule.
  * Add OpType::NOISE, OpType::READOUT\_NOISE, OpType::DETECTOR, OpType::OBSERVABLE to HeisenbergOp.
* **Task 2.2.2 (Quantum Noise Rewinding):** When encountering X/Y/Z\_ERROR or DEPOLARIZE1/2 in trace():
  * Resolve the operation into a list of mutually exclusive Paulis and probabilities (e.g., DEP1(p) \= X, Y, Z each with probability $p/3$. DEP2(p) \= 15 channels with $p/15$).
  * Rewind *each* Pauli through the current inv\_state tableau to get its $t=0$ destab\_mask and stab\_mask.
  * Push this list of resolved NoiseChannels as a NoiseSite to hir.noise\_sites, and emit a HeisenbergOp::NOISE node holding the payload\_idx.
* **Task 2.2.3 (Classical Logic Emission):**
  * For DETECTOR and OBSERVABLE\_INCLUDE, resolve all rec\[-k\] targets to absolute indices, store the list in the side-tables, and emit the HIR node holding the index. (For observables, encode the observable index into the union payload as well).
  * For READOUT\_NOISE, emit an HIR node holding the absolute target measurement index and the float probability.
* **DoD:** A pure Clifford circuit with DEPOLARIZE1 0 produces an HIR NOISE node pointing to a NoiseSite of exactly 3 rewound Pauli masks.

## **Phase 2.3: Compiler Back-End (Noise Scheduling & Constant Pool)**

**Goal:** Lower the HIR into executable bytecode and construct the AOT NoiseSchedule.

* **Task 2.3.1 (Constant Pool Upgrades):** Move NoiseChannel and NoiseSite to backend.h. Add uint32\_t pc; and double total\_probability; to NoiseSite. Add std::vector\<NoiseSite\> noise\_schedule to ConstantPool, along with the detector/observable target vectors. Update CompiledModule to track num\_detectors and num\_observables.
* **Task 2.3.2 (Bytecode Opcodes):** Add OP\_READOUT\_NOISE, OP\_DETECTOR, and OP\_OBSERVABLE to the Instruction opcode enum.
* **Task 2.3.3 (Lowering Logic):**
  * Iterate through the HIR.
  * If NOISE: **Do not emit bytecode.** Instead, copy the NoiseSite to the noise\_schedule. Set its pc to the *current* size of the bytecode array (the index of the *next* quantum instruction). Sum the channel probabilities to populate total\_probability.
  * If READOUT\_NOISE, DETECTOR, OBSERVABLE: Emit their respective opcodes and copy their target lists to the ConstantPool. (OP\_READOUT\_NOISE easily fits a double prob and uint32\_t meas\_idx in the 24-byte union payload using bare doubles).
* **DoD:** The emitted bytecode vector is strictly free of quantum NOISE opcodes. The noise\_schedule maps strictly to valid pc indices with computed total\_probability.

## **Phase 2.4: SVM Runtime & Gap Sampling**

**Goal:** Execute the classical opcodes and evaluate the NoiseSchedule natively at runtime.

* **Task 2.4.1 (State Expansion):** Update SchrodingerState to include std::vector\<uint8\_t\> det\_record and obs\_record.
* **Task 2.4.2 (Gap Sampling Algorithm):** In svm.cc execute(), implement the gap sampler:
  * Before the instruction loop, initialize uint32\_t next\_noise\_idx \= 0;
  * Inside the execute() loop, before the switch statement, add:
    C++
    while (next\_noise\_idx \< schedule.size() && pc \== schedule\[next\_noise\_idx\].pc) {
        const auto& site \= schedule\[next\_noise\_idx\];
        if (state.random\_double() \< site.total\_probability) {
            // Roulette wheel selection among mutually exclusive channels
            double r \= state.random\_double() \* site.total\_probability;
            double cum\_p \= 0.0;
            for (const auto& ch : site.channels) {
                cum\_p \+= ch.prob;
                if (r \<= cum\_p) {
                    state.destab\_signs ^= ch.destab\_mask;
                    state.stab\_signs ^= ch.stab\_mask;
                    break;
                }
            }
        }
        next\_noise\_idx++;
    }

* **Task 2.4.3 (Classical Opcodes):**
  * OP\_READOUT\_NOISE: if (state.random\_double() \< instr.prob) state.meas\_record\[idx\] ^= 1;
  * OP\_DETECTOR: XOR the listed meas\_record bits and push to det\_record.
  * OP\_OBSERVABLE: XOR the listed meas\_record bits and XOR them into obs\_record\[obs\_index\] (observables accumulate).
* **Task 2.4.4 (Python API Update):** Update bindings.cc and ucc.sample() to return a Python tuple of three numpy arrays: (measurements, detectors, observables).
* **DoD:** Python API successfully returns detector and observable boolean arrays. Gap sampling successfully injects noise without executing inline bytecode.

## **Phase 2.5: Exact Trajectory Validation (Physics Oracle)**

**Goal:** Prove that the Heisenberg rewinding of Pauli masks perfectly matches Stim's geometry.

* **Task 2.5.1:** Create a Python test test\_trajectory\_oracle.py.
* **Task 2.5.2:** Generate a specific circuit configuration (e.g., H 0, CX 0 1, M 0 1, DETECTOR rec\[-1\] rec\[-2\]).
* **Task 2.5.3:** Create an "erroneous" version by explicitly inserting an X 0 gate (mimicking a deterministic $X$ error).
* **Task 2.5.4:** Compile and run both the clean and "erroneous" circuit in UCC and Stim. Assert that UCC's detector arrays exactly match Stim's compile\_detector\_sampler() outputs for the specific trajectories.
* **DoD:** Exact match of detector parity flips between UCC and Stim when a specific physical error is explicitly injected.

## **Phase 2.6: Statistical Equivalence & Target Circuit**

**Goal:** Prove the full gap-sampled stochastic noise engine exactly matches Stim statistically.

* **Task 2.6.1:** Add the `c=inject[bell]+cultivate,p=0.001,noise=uniform,g=css,q=14,b=Y,r=5,d1=3.stim` circuit to the test suite.
* **Task 2.6.2:** Run the target circuit in Stim for $10^5$ shots using compile\_detector\_sampler(), extracting the marginal firing probability of each detector and observable.
* **Task 2.6.3:** Run the target circuit in UCC for $10^5$ shots. Before doing this, run for ~100 shots and report how much time 10^5 will take. We want to be sure we have optimized enough to make this feasible before actually doing it.
* **Task 2.6.4:** Assert that the marginal probability of every single detector and observable in UCC falls within $\\pm 0.005$ (a strict $5\\sigma$ bound for $N=10^5$) of the Stim distribution. Right this as a generic statistic test, versus hard coding .005. Similar to how we've done for existing measurement comparisions in the unit tests.
* **DoD:** The test passes reliably, mathematically proving that UCC's AOT noise scheduling, multi-Pauli depolarizing decomposition, readout noise, and classical logic pipelines identically mirror Stim's physics.

### **Phase 2.7: Performance Benchmarking Suite (Standalone Task)**

**Goal:** Establish an on-demand, trackable benchmarking suite in tools/bench/ to quantify UCC's compilation latency and per-shot execution overhead relative to Stim.

* **Task 2.7.1 (Tooling Setup):**
  * Add pytest-benchmark to the dev dependency group in the root pyproject.toml.
  * Save the target Clifford QEC circuit provided in the prompt to a new file tools/bench/target\_qec.stim.
* **Task 2.7.2 (The Benchmark Script):** Create tools/bench/test\_bench\_qec.py.
  * Load the .stim file into a string constant.
  * Write test\_compile\_stim(benchmark) and test\_compile\_ucc(benchmark) to strictly measure the time it takes to parse and compile the circuit into an executable object (stim.Circuit(text).compile\_detector\_sampler() vs ucc.compile(text)).
  * Write test\_sample\_stim(benchmark) and test\_sample\_ucc(benchmark) to strictly measure the execution time of $100,000$ shots using the pre-compiled objects. Use pytest fixtures to compile the objects *once* before the benchmark execution loop begins so compilation time doesn't bleed into runtime metrics.
* **Task 2.7.3 (Justfile Integration):** Add a bench recipe to the justfile that only targets the bench directory:
```
  bench \*args="":
      uv run pytest tools/bench/ \--benchmark-sort=name \--benchmark-columns=Mean,StdDev,ops {{args}}
```
* **DoD:** Running just bench successfully executes the suite and prints a formatted terminal table comparing the exact execution time (in milliseconds) and operations-per-second of both engines without interfering with the standard tests/python/ correctness suite.
