# UCC Implementation Plan: Native Error Syndrome Normalization

## Context & Goal

UCC currently simulates exact physical states and outputs raw measurement parities. Standard quantum decoders (like PyMatching) and evaluators (like Sinter) expect **Error Syndromes**, where `0` strictly means "matches the noiseless reference circuit" and `1` strictly means "error".

Furthermore, UCC's `OP_POSTSELECT` instruction eagerly discards shots whose parity is non-zero. If a perfectly clean circuit natively yields a `1` parity, UCC aggressively discards it, leading to a false 100% discard/error rate.

**Goal:** Push Error Syndrome Normalization down into the C++ Virtual Machine and compiler bindings. This ensures $\mathcal{O}(1)$ C++ execution speeds are maintained, memory layouts remain pristine, and the Python frontend provides a frictionless user experience via a `normalize_syndromes=True` flag.

---

## Phase 1: Virtual Machine Architecture (C++)

**1. Update `src/ucc/backend/backend.h**`
Add a flag for expected detector parity to the `Instruction` struct, and add an expected observables vector to `CompiledModule`.

```cpp
    // Flag bits for measurement instructions
    static constexpr uint8_t FLAG_SIGN = 1 << 0;
    static constexpr uint8_t FLAG_HIDDEN = 1 << 1;
    static constexpr uint8_t FLAG_IDENTITY = 1 << 2;
    // NEW: Expected noiseless parity is 1
    static constexpr uint8_t FLAG_EXPECTED_ONE = 1 << 3;

// ... inside CompiledModule struct ...
    uint32_t num_observables = 0;   // Total observables
    std::vector<uint8_t> expected_observables; // NEW

// ... update lower() signature ...
[[nodiscard]] CompiledModule lower(const HirModule& hir,
                                   std::span<const uint8_t> postselection_mask = {},
                                   std::span<const uint8_t> expected_detectors = {},
                                   std::span<const uint8_t> expected_observables = {});

```

**2. Update `src/ucc/svm/svm.cc` Execution Handlers**
Modify `exec_detector` and `exec_postselect` to accept the expected parity flag and initialize their branchless XOR accumulation.

```cpp
static inline void exec_detector(SchrodingerState& state, const ConstantPool& pool,
                                 uint32_t det_list_idx, uint32_t classical_idx,
                                 bool expected_one) {
    assert(det_list_idx < pool.detector_targets.size());
    const auto& targets = pool.detector_targets[det_list_idx];

    // Branchless XOR: Initializes with 1 if expected_one is true, 0 if false
    uint8_t parity = static_cast<uint8_t>(expected_one);
    for (uint32_t meas_idx : targets) {
        assert(meas_idx < state.meas_record.size());
        parity ^= state.meas_record[meas_idx];
    }
    state.det_record[classical_idx] = parity;
}

static inline bool exec_postselect(SchrodingerState& state, const ConstantPool& pool,
                                   uint32_t det_list_idx, uint32_t classical_idx,
                                   bool expected_one) {
    assert(det_list_idx < pool.detector_targets.size());
    const auto& targets = pool.detector_targets[det_list_idx];

    uint8_t parity = static_cast<uint8_t>(expected_one);
    for (uint32_t meas_idx : targets) {
        assert(meas_idx < state.meas_record.size());
        parity ^= state.meas_record[meas_idx];
    }
    state.det_record[classical_idx] = 0; // Maintain 0 for PyMatching shape

    if (parity != 0) { // Parity != 0 strictly means "error"
        state.discarded = true;
        return true;
    }
    return false;
}

```

*Update the dispatch macros/switch in `execute()` to pass `(pc->flags & Instruction::FLAG_EXPECTED_ONE) != 0` to these handlers.*

**3. Update `sample_survivors` and `sample` in `src/ucc/svm/svm.cc**`
*Crucial Architecture Note:* Observables cannot be normalized at the instruction level because multiple `OBSERVABLE_INCLUDE` instructions accumulate into the same index. They must be normalized exactly once at the end of the shot.

Inside the `for (uint32_t shot = 0; shot < shots; ++shot)` loop of **both** `sample` and `sample_survivors`, apply the normalizer. In `sample_survivors`:

```cpp
        // For sample_survivors: only run this if (!state.discarded)
        bool any_obs_flipped = false;
        for (uint32_t i = 0; i < num_obs; ++i) {
            uint8_t final_obs = state.obs_record[i];
            if (i < program.expected_observables.size() && program.expected_observables[i] != 0) {
                final_obs ^= 1;
            }

            // Only update observable_ones/logical_errors in sample_survivors()
            if (final_obs) {
                result.observable_ones[i]++;
                any_obs_flipped = true;
            }

            // Write the normalized observable to the output array
            if (keep_records) {
                result.observables.push_back(final_obs);
            }
        }

        if (any_obs_flipped) {
            result.logical_errors++;
        }

```

*(Ensure you apply the exact same XOR inversion logic to `state.obs_record[i]` inside the standard `ucc::sample()` loop before it writes to `result.observables`).*

---

## Phase 2: Compiler Back-End (C++)

**1. Update `src/ucc/backend/backend.cc**`
Modify `ucc::lower()` to inject the flag into detector instructions and save the expected observables.

```cpp
CompiledModule lower(const HirModule& hir,
                     std::span<const uint8_t> postselection_mask,
                     std::span<const uint8_t> expected_detectors,
                     std::span<const uint8_t> expected_observables) {
// ...
            case OpType::DETECTOR: {
                // ...
                bool is_postselected = det_emit_idx < postselection_mask.size() &&
                                       postselection_mask[det_emit_idx] != 0;

                Instruction instr = is_postselected ? make_postselect(cp_idx, det_emit_idx)
                                                    : make_detector(cp_idx, det_emit_idx);

                // Inject expected parity
                if (det_emit_idx < expected_detectors.size() && expected_detectors[det_emit_idx] != 0) {
                    instr.flags |= Instruction::FLAG_EXPECTED_ONE;
                }

                ctx.emit(instr);
                ++det_emit_idx;
                break;
            }
// ...
    CompiledModule result;
    // ...
    result.expected_observables.assign(expected_observables.begin(), expected_observables.end());
    return result;
}

```

---

## Phase 3: Python Bindings & Auto-Normalization

**1. Update `src/python/bindings.cc**`
Modify the `ucc.compile` binding to accept `normalize_syndromes`. If true, automatically execute a noiseless reference shot natively in C++ before compiling the target circuit. Rather than putting the reference run logic inside the compile method, find another suitable place to add it in the C++ code so we can also test it. Also, make a new "RemoveNoise" pass on HIR rather than adding the loop here. DO NOT ADD that pass to the default pass list.

```cpp
    m.def(
        "compile",
        [](const std::string& stim_text,
           std::vector<uint8_t> postselection_mask,
           std::vector<uint8_t> expected_detectors,
           std::vector<uint8_t> expected_observables,
           bool normalize_syndromes,
           ucc::PassManager* hir_passes,
           ucc::BytecodePassManager* bytecode_passes) {

            nb::gil_scoped_release release;
            ucc::Circuit circuit = ucc::parse(stim_text);
            ucc::HirModule hir = ucc::trace(circuit);

            if (hir_passes)
                hir_passes->run(hir);

            // Automatically extract the expected reference syndrome in C++!
            if (normalize_syndromes) {
                if (!expected_detectors.empty() || !expected_observables.empty()) {
                    throw std::invalid_argument("Cannot provide expected parities when normalize_syndromes=True");
                }

                ucc::HirModule clean_hir = hir;

                // 1. Erase all stochastic noise and readout errors from the IR
                clean_hir.ops.erase(
                    std::remove_if(clean_hir.ops.begin(), clean_hir.ops.end(),
                                   [](const ucc::HeisenbergOp& op) {
                                       return op.op_type() == ucc::OpType::NOISE ||
                                              op.op_type() == ucc::OpType::READOUT_NOISE;
                                   }),
                    clean_hir.ops.end());
                clean_hir.source_map.clear(); // Drop map to avoid line index mismatch

                // 2. Lower and execute exactly 1 shot of the clean circuit natively
                // Note: Don't pass postselection_mask here! The clean reference must run to completion.
                auto clean_prog = ucc::lower(clean_hir);

                if (clean_prog.num_measurements > 0) {
                    auto clean_res = ucc::sample(clean_prog, 1, 0); // seed=0 for determinism
                    expected_detectors = std::move(clean_res.detectors);
                    expected_observables = std::move(clean_res.observables);
                }
            }

            // Lower the actual circuit, injecting the reference masks
            auto program = ucc::lower(hir, postselection_mask, expected_detectors, expected_observables);

            if (bytecode_passes)
                bytecode_passes->run(program);

            return program;
        },
        nb::arg("stim_text"),
        nb::arg("postselection_mask") = std::vector<uint8_t>{},
        nb::arg("expected_detectors") = std::vector<uint8_t>{},
        nb::arg("expected_observables") = std::vector<uint8_t>{},
        nb::arg("normalize_syndromes") = false,
        nb::arg("hir_passes") = nb::none(),
        nb::arg("bytecode_passes") = nb::none(),
        "Compile a quantum circuit string to executable bytecode.\n"
        );

    // Note: Also update the `m.def("lower", ...)` binding to accept the two expected_* spans.

```

---

## Phase 4: Sinter Integration Example (For Users)

With this auto-normalization wrapped in C++, users do not need to juggle `stim.Circuit().without_noise()` or manual parity arrays. Your custom Sinter sampler drops down to the below.

** THERE IS NOTHING TO IMPLEMENT FOR THIS PHASE; this is just reference for the user **

```python
class UccSGateDesaturationSampler(sinter.Sampler):
    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        base_sampler = CompiledDesaturationSampler.from_task(task)

        # Convert the paper's set of postselected detectors into a UCC bitmask
        original_num_dets = base_sampler.gap_circuit.num_detectors
        mask = [1 if k in base_sampler.postselected_detectors else 0 for k in range(original_num_dets)]

        # Compile natively, asking UCC to auto-normalize the syndromes
        ucc_program = ucc.compile(
            str(base_sampler.gap_circuit),
            postselection_mask=mask,
            normalize_syndromes=True, # <--- UCC handles the clean reference run automatically!
            hir_passes=ucc.default_pass_manager(),
            bytecode_passes=ucc.default_bytecode_pass_manager(),
        )

        return UccSGateCompiledDesaturationSampler(task, base_sampler, ucc_program)

```

---

## Phase 5: Testing Strategy

Implement the following tests to ensure exactness and regression safety.

### 5.1 Python Integration Tests (`tests/python/test_sample.py`)

Add the following test case to verify `normalize_syndromes=True` handles complex circuits where multiple `OBSERVABLE_INCLUDE` statements XOR into a single observable index, and interacts safely with postselection.

```python
def test_normalize_syndromes_multiple_observables_xord() -> None:
    """Test normalize_syndromes=True on a circuit where multiple includes XOR together."""
    import numpy as np
    import ucc

    # Circuit design:
    # X 0, X 1, X 2 -> All measurements evaluate to 1
    # DET 0: M0 (evaluates to 1)
    # DET 1: M0 ^ M1 (evaluates to 1 ^ 1 = 0)
    # OBS 0: M0 ^ M1 ^ M2 (1 ^ 1 ^ 1 = 1) -> 3 includes!
    # OBS 1: M1 (1) -> 1 include
    circuit = """
        X 0 1 2
        M 0 1 2
        DETECTOR rec[-3]
        DETECTOR rec[-3] rec[-2]

        OBSERVABLE_INCLUDE(0) rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]

        OBSERVABLE_INCLUDE(1) rec[-2]
    """

    # 1. Baseline: Without normalization, physical parities match the math above
    prog_raw = ucc.compile(circuit, normalize_syndromes=False)
    _, det_raw, obs_raw = ucc.sample(prog_raw, shots=10, seed=0)

    assert np.all(det_raw[:, 0] == 1)
    assert np.all(det_raw[:, 1] == 0)
    assert np.all(obs_raw[:, 0] == 1)
    assert np.all(obs_raw[:, 1] == 1)

    # 2. Normalized: All output parities must be strictly 0.
    # We also apply a postselection mask on DET 0 (which natively evaluates to 1).
    # Since it is normalized, it becomes 0, meaning shots should SURVIVE post-selection.
    prog_norm = ucc.compile(
        circuit,
        normalize_syndromes=True,
        postselection_mask=[1, 0] # Flag DET 0 for postselection
    )

    res = ucc.sample_survivors(prog_norm, shots=10, seed=0, keep_records=True)

    assert res["passed_shots"] == 10  # Normalized 1^1=0, so shots survive!
    assert np.all(res["detectors"] == 0)
    assert np.all(res["observables"] == 0)
    assert res["logical_errors"] == 0

```

### 5.2 C++ Tests (`tests/test_svm_risc.cc` & `tests/test_backend.cc`)

* **`tests/test_backend.cc`:** Create a `HirModule` and pass `expected_detectors = {1, 0}` to `lower()`. Verify that `bytecode[i].flags & Instruction::FLAG_EXPECTED_ONE` is set correctly for the corresponding detectors.
* **`tests/test_svm_risc.cc`:**
* Construct an `OP_DETECTOR` with `FLAG_EXPECTED_ONE` and verify the XOR inversion works (expect 0).
* Construct an `OP_POSTSELECT` with `FLAG_EXPECTED_ONE`, feed it a `1` measurement record, and verify `state.discarded == false`. Feed it a `0` measurement record and verify `state.discarded == true`.
* Construct a program with `expected_observables = {1}` and test that `ucc::sample()` correctly normalizes `obs_record`.
