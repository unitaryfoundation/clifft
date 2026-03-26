# UCC: Non-Destructive Expectation Value Probe (EXP_VAL)

## Summary

Add a simulation-only, non-destructive expectation value instruction
`EXP_VAL` that evaluates the expectation value of a Pauli product at an
exact point in the circuit. The result is a `float64` in `[-1, 1]`
appended to a new per-shot expectation value record stream. `EXP_VAL`
does not collapse the state, mutate the Pauli frame, or affect any
existing measurement/detector/observable records.

Syntax:

    H 0
    EXP_VAL X0
    EXP_VAL X0*Y1*Z2
    EXP_VAL X0*Y1 Z2*Z3

Multiple whitespace-separated Pauli products per instruction are
supported (same syntax as `MPP`). Each product gets its own index in
the expectation value record. Indexing is implicit by encounter order
(like measurement records). No named record targets in v1.

---

## Semantics

`EXP_VAL` is a **read-only, shot-local, exact probe** of the current
trajectory. It evaluates the expectation `<P>` at the exact circuit
location where it appears.

**It must see:**

- Sampled Pauli noise that has already happened
- Classically controlled Pauli corrections that have already happened
- Post-collapse states from earlier measurements
- None of the operations that come after it

**It must not mutate:**

- Active array `v[]`
- Pauli frame (`p_x`, `p_z`)
- Global scalar `gamma`
- Active dimension `active_k`
- Measurement, detector, or observable records

**Return value:** `float64` in `[-1, 1]`, one per Pauli product per
shot. Stored in `exp_vals[exp_val_idx]`.

---

## Design Overview

New HIR op + new VM opcode + direct read-only expectation kernel.
No fake measurements, no compress/inspect/uncompress sequences.

Pipeline path:

1. **Parser**: Parse `EXP_VAL X0*Y1*Z2` -> AST node with
   `GateType::EXP_VAL`
2. **Front-end**: Rewind each Pauli product through the Clifford
   tableau, emit `OpType::EXP_VAL` with rewound masks
3. **Back-end**: Map to virtual frame, store full Pauli mask in constant
   pool, emit `Opcode::OP_EXP_VAL`
4. **VM**: Execute read-only expectation kernel on the active state

---

## Phase 0: Refactor Sampling Return Types (Prerequisite PR)

This phase is a standalone refactor, merged as its own PR before any
EXP_VAL work begins. It eliminates the fragile tuple/dict returns from
all sampling functions and establishes a unified `SampleResult` object
pattern that can absorb new fields (like `exp_vals`) without breaking
callers.

### Task 0.1: Define SampleResult Python wrapper

Touch: `src/python/bindings.cc`, `src/python/ucc/__init__.py`

Create a Python-visible `SampleResult` class wrapping the C++
`SampleResult` struct. Attributes:

- `measurements`: `numpy.ndarray` shape `(shots, num_measurements)`, uint8
- `detectors`: `numpy.ndarray` shape `(shots, num_detectors)`, uint8
- `observables`: `numpy.ndarray` shape `(shots, num_observables)`, uint8

The object should support attribute access (`result.measurements`) and
be iterable/unpackable for backward compatibility during transition
(`m, d, o = result` via `__iter__`).

### Task 0.2: Update sample() to return SampleResult

Touch: `src/python/bindings.cc`

Change `ucc.sample()` to return a `SampleResult` object instead of a
bare tuple. The `__iter__` support means existing `m, d, o = ucc.sample()`
code continues to work during a deprecation period.

### Task 0.4: Update sample_survivors() to return unified SampleResult

Touch: `src/python/bindings.cc`

Change `sample_survivors()` from returning a dict to returning a
`SampleResult` object. The `.measurements`, `.detectors`, and
`.observables` arrays contain only the surviving (post-selected)
shots. Survivor-only metadata is exposed on the same object via
`.total_shots`, `.passed_shots`, `.discards`, `.logical_errors`, and
`.observable_ones`.

### Task 0.5: Update sample_k() to return SampleResult

Touch: `src/python/bindings.cc`

Change `sample_k()` from returning a bare tuple to returning a
`SampleResult` object. Same attribute pattern.

### Task 0.6: Update sample_k_survivors() to return unified SampleResult

Touch: `src/python/bindings.cc`

Change `sample_k_survivors()` from returning a dict to returning a
`SampleResult` object. The `.measurements`, `.detectors`, and
`.observables` arrays contain only surviving shots, and survivor-only
metadata is exposed on the same object via `.total_shots`,
`.passed_shots`, `.discards`, `.logical_errors`, and
`.observable_ones`.

### Task 0.7: Update all internal callers

Touch: `tests/python/`, `paper/` scripts

Update existing Python code to use the result object pattern. Keep
`m, d, o = ucc.sample(...)` unpacking working via `__iter__` but
migrate tests to use attribute access where it improves clarity.
Update `sample_survivors()` and `sample_k_survivors()` callers from
dict key access to attribute access. Do not preserve dict-style access
on the new result object; callers should fail clearly and migrate.

**DoD:** All four sampling functions (`sample()`, `sample_survivors()`,
`sample_k()`, `sample_k_survivors()`) return `SampleResult` objects.
For survivor APIs, the shot arrays contain only surviving shots and the
same object also carries survivor metadata. Dict-style compatibility is
removed. All existing tests and docs are updated and pass. Merged as
its own PR.

---

## Phase 1: Parser and AST

### Task 1.1: Add GateType::EXP_VAL

Touch: `src/ucc/circuit/gate_data.h`

- Add `EXP_VAL` to the `GateType` enum
- Add `GateTraits` entry: arity `MULTI`, no measurement/reset/noise flags
- Mark it as non-Clifford, non-measurement (a new UCC extension gate)

### Task 1.2: Parser support

Touch: `src/ucc/circuit/parser.cc`, `src/ucc/circuit/circuit.h`

- Add `"EXP_VAL"` to `kGateNames[]` (keep sorted)
- Write `parse_exp_val()` following the `parse_mpp()` pattern:
  - Multiple whitespace-separated Pauli products (same as MPP)
  - Pauli-tagged targets via `Target::pauli(qubit, flag)`
  - Each product separated by `*` within, whitespace between products
  - No numeric args
  - Reject duplicate qubits within a product (same validation as MPP)
- Add `Circuit::num_exp_vals` counter, increment once per Pauli product
  (not once per instruction -- an `EXP_VAL X0 Z1` with two products
  increments by 2)
- Route `GateType::EXP_VAL` in `parse_line()` to `parse_exp_val()`

### Task 1.3: Parser tests

Touch: `tests/test_parser.cc` (new section)

- `EXP_VAL X0` parses correctly, one target with Pauli tag
- `EXP_VAL X0*Y1*Z2` parses correctly, three Pauli-tagged targets
- `EXP_VAL X0*Y1 Z2*Z3` parses correctly, two products
- `EXP_VAL X0 Y1 Z2` parses correctly, three single-qubit products
- Malformed syntax rejected (missing Pauli, bad separator, etc.)
- Duplicate qubits within a product rejected
- `num_exp_vals` increments per product (2 products = +2)
- Multiple EXP_VAL instructions in one circuit

**DoD:** Parser unit tests pass. `EXP_VAL` round-trips through parse
correctly.

---

## Phase 2: HIR and Front-End

### Task 2.1: Add OpType::EXP_VAL

Touch: `src/ucc/frontend/hir.h`

- Add `EXP_VAL` to `OpType` enum (before `NUM_OP_TYPES` sentinel)
- Add `ExpValIdx` strong typedef
  (`enum class ExpValIdx : uint32_t {}`)
- Add union payload variant:
  ```cpp
  struct {
      uint32_t exp_val_idx;
  } exp_val_;
  ```
- Add accessor `exp_val_idx()` with debug assert
- Add factory
  `make_exp_val(PauliBitMask destab, PauliBitMask stab, bool s, ExpValIdx idx)`
- Add `HirModule::num_exp_vals` counter

### Task 2.2: Front-end emission

Touch: `src/ucc/frontend/frontend.cc`

- Handle `GateType::EXP_VAL` in the trace loop
- For each Pauli product in the instruction (multi-product support):
  - Build a `stim::PauliString` from the AST targets (same pattern as
    MPP)
  - Rewind through `sim.inv_state(obs)` (same as MPP/R_PAULI)
  - Convert to BitMask via `stim_to_bitmask()`
  - Emit `HeisenbergOp::make_exp_val(destab, stab, sign,
    ExpValIdx{next_exp_val_idx++})`
- Increment `hir.num_exp_vals` once per product
- Maintain source map

The front-end rewind is critical: it projects the user-specified Pauli
onto the Heisenberg picture at t=0, accounting for all Clifford gates
that have been absorbed into the tableau.

### Task 2.3: Front-end tests

Touch: `tests/test_frontend.cc` (new section)

- Rewound Pauli for EXP_VAL matches the same rewind logic used for MPP
- Exp val indices are assigned in encounter order across products
- Multi-product `EXP_VAL X0 Z1` emits two HIR ops with consecutive
  indices
- `hir.num_exp_vals` is correct
- Source map entries are preserved

**DoD:** Front-end correctly rewinds EXP_VAL Paulis and emits
EXP_VAL ops.

---

## Phase 3: Optimizer Rules

### Task 3.1: Commutation barrier

Touch: `src/ucc/optimizer/commutation.cc`

Treat `EXP_VAL` as a **hard barrier**: `can_swap()` returns `false`
if either operand is `EXP_VAL`.

Rationale: the user wants the expectation value at a specific circuit
point. Even if neighboring ops commute algebraically, moving them
across EXP_VAL changes the semantic meaning of the probe.

### Task 3.2: Statevector squeeze pass

Touch: `src/ucc/optimizer/statevector_squeeze_pass.cc`

The squeeze pass bubbles measurements leftward and non-Cliffords
rightward using `can_swap()`. Since `can_swap()` returns false for
EXP_VAL, no explicit changes are needed -- the barrier is
automatically respected. Add a comment noting this.

### Task 3.3: Peephole fusion pass

Touch: `src/ucc/optimizer/peephole.cc`

The `is_blocked()` helper determines whether a T-gate can commute past
an operation. Add `case OpType::EXP_VAL: return true;` to block
T-gate movement across EXP_VAL probes.

### Task 3.4: Bytecode fusion passes

Touch: `src/ucc/optimizer/single_axis_fusion_pass.cc`,
       `src/ucc/optimizer/tile_axis_fusion_pass.cc`

Bytecode fusion passes (SingleAxisFusionPass, TileAxisFusionPass) scan
for runs of fusible opcodes. `OP_EXP_VAL` is not a fusible opcode,
so it naturally terminates any run. The `is_fusible()` helpers return
false for unknown opcodes by default. Verify this is the case and add
the new opcode to any explicit exhaustive switch statements if needed.

### Task 3.5: Optimizer tests

Touch: `tests/test_optimizer.cc` (new section)

- `can_swap(T_GATE, EXP_VAL)` returns false
- `can_swap(EXP_VAL, MEASURE)` returns false
- `can_swap(EXP_VAL, NOISE)` returns false
- StatevectorSqueezePass does not move ops across EXP_VAL
- PeepholeFusionPass does not fuse T-gates across EXP_VAL

**DoD:** No optimizer pass reorders operations across EXP_VAL.

---

## Phase 4: Back-End / Lowering

### Task 4.1: Add Opcode::OP_EXP_VAL

Touch: `src/ucc/backend/backend.h`

- Add `OP_EXP_VAL` to the `Opcode` enum (before `NUM_OPCODES`)
- Add a constant-pool table for exp val Pauli specs. Reuse the
  existing `PauliMask` shape (full virtual x, z, sign):
  ```cpp
  // In ConstantPool:
  std::vector<PauliMask> exp_val_masks;  // Indexed by cp_exp_val_idx
  ```
- Add instruction factory:
  ```cpp
  Instruction make_exp_val(uint32_t cp_exp_val_idx,
                           uint32_t exp_val_idx);
  ```
  Use the `pauli` payload variant: `cp_mask_idx` = constant pool index,
  `condition_idx` = exp val record index.
- Add `CompiledModule::num_exp_vals`

Design note: we use a separate `exp_val_masks` vector (not the existing
`pauli_masks` used by `OP_APPLY_PAULI`) to keep the two concerns
cleanly separated and avoid index confusion.

### Task 4.2: Lowering rule

Touch: `src/ucc/backend/backend.cc`

Add `case OpType::EXP_VAL:` in the lowering loop:

1. Take the rewound HIR Pauli (destab_mask, stab_mask, sign)
2. Map to current virtual frame via `map_to_virtual(ctx, ...)`
3. Convert the virtual PauliString to a `PauliMask` (extract x, z, sign)
4. Store in `constant_pool.exp_val_masks`
5. Emit `make_exp_val(cp_idx, exp_val_idx)`

**Do not** use `compress_pauli()`. The exp val kernel operates on the
full virtual Pauli mask directly. No routing, no temporary basis
change, no axis compression.

Also update:
- `CompiledModule::num_exp_vals` from `hir.num_exp_vals`
- Source map emission for the new opcode

### Task 4.3: Back-end tests

Touch: `tests/test_backend.cc` (new section)

- Lowering emits `OP_EXP_VAL` for EXP_VAL HIR ops
- `num_exp_vals` is propagated correctly
- Constant pool `exp_val_masks` contains the correct virtual Pauli
- No `compress_pauli()` routing is used (no SWAP/EXPAND/etc. emitted
  around the exp val instruction)
- Source maps preserved

**DoD:** Back-end correctly lowers EXP_VAL to OP_EXP_VAL with full
virtual Pauli mask in constant pool.

---

## Phase 5: VM State and Execution Kernel

### Task 5.1: VM state changes

Touch: `src/ucc/svm/svm.h`, `src/ucc/svm/svm_state.cc`

Add to `SchrodingerState`:
```cpp
std::vector<double> exp_vals;
```

- Constructor: allocate with `num_exp_vals` size, zero-initialized
- Update constructor signature: add `uint32_t num_exp_vals = 0`
  parameter
- `reset()`: zero exp_vals (do not reallocate)

### Task 5.2: Execution kernel

Touch: `src/ucc/svm/svm_kernels.inl`

Add `exec_exp_val()` handler and dispatch case.

**Kernel semantics:**

Let P be the stored virtual Pauli from the constant pool.

**Step 1: Frame conjugation.**
Compute the symplectic anti-commutation parity between P and the
current runtime Pauli frame (p_x, p_z). If P anti-commutes with the
frame, flip the expectation sign.

Specifically: the frame sign parity is
`popcount((P.x & state.p_z) ^ (P.z & state.p_x)) & 1`.
If odd, multiply overall sign by -1.

**Step 2: Dormant/active split.**
Using `state.active_k`, bits `[0, active_k)` are active, bits
`[active_k, n)` are dormant.

- If any dormant qubit has X support (P.x bit set, P.z bit clear):
  result is 0
- If any dormant qubit has Y support (P.x and P.z bits both set):
  result is 0
- Dormant Z support contributes +1 (can be ignored)

**Step 3: Active-space evaluation.**
Let `x_active` and `z_active` be the active-space X and Z masks
(bits 0..active_k-1 only).

The expectation value on the active array is:

    <P> = Re[ sum_j conj(a[j ^ x_active]) * c(j) * a[j] ]
        / sum_j |a[j]|^2

where:

    c(j) = s * i^(popcount(x_active & z_active))
             * (-1)^(popcount(j & z_active))

and `s` is the overall +/- sign after frame conjugation.

Implementation notes:
- Accumulate numerator in `std::complex<double>`, denominator in
  `double`
- Denominator is the active-array norm; `gamma` cancels in the ratio
- The `i^(...)` phase factor is constant across all j (depends only on
  the Pauli type, not the state), so compute it once outside the loop
- Final result should be real for Hermitian Paulis; discard tiny
  imaginary residue
- Clamp to `[-1, 1]` for numerical safety

**Step 4:** Write result to `state.exp_vals[exp_val_idx]`.
No other state changes.

**Performance:** One `O(2^k)` read-only pass per EXP_VAL site. No AVX
specialization in v1 -- scalar complex accumulation is sufficient.

### Task 5.3: Dispatch integration

Touch: `src/ucc/svm/svm_kernels.inl`

Add `case Opcode::OP_EXP_VAL:` to the main dispatch switch in
`execute_internal()`. Also add to AVX512-specific dispatch if present
in `svm_avx512.cc`.

### Task 5.4: SVM kernel tests

Touch: new file `tests/test_exp_value.cc`

Direct kernel tests using handcrafted states:

- `|0>`: `<Z>=+1`, `<X>=0`, `<Y>=0`
- `|1>`: `<Z>=-1`
- `|+>`: `<X>=+1`, `<Z>=0`
- `|+i>` (= S|+>): `<Y>=+1`
- Bell state `(|00>+|11>)/sqrt(2)`: `<Z0*Z1>=+1`, `<X0*X1>=+1`,
  `<Z0>=0`
- Dormant X/Y support gives 0
- Explicit frame flips via runtime p_x/p_z change sign correctly
- Multi-qubit products: `<X0*Y1*Z2>` on known states

**DoD:** All kernel tests pass. Expectation values are exact to
machine precision for small states.

---

## Phase 6: Sampling API and Python Bindings

### Task 6.1: Sample result changes (C++)

Touch: `src/ucc/svm/svm.h`, `src/ucc/svm/svm.cc`

Add `exp_vals` field to `SampleResult`:
```cpp
std::vector<double> exp_vals;  // Shape: [shots * num_exp_vals]
```

Update `sample()`:
- Allocate `exp_vals` array: `shots * num_exp_vals` doubles
- Pass `num_exp_vals` to `SchrodingerState` constructor
- After each shot's `execute()`, copy `state.exp_vals` into the result
- For circuits without EXP_VAL, `num_exp_vals == 0`, so the vector
  remains empty (zero allocation)

Update `sample_survivors()`, `sample_k()`, and
`sample_k_survivors()` similarly -- add `exp_vals` field and copy
for surviving/sampled shots when applicable.

### Task 6.2: Python bindings

Touch: `src/python/bindings.cc`, `src/python/ucc/__init__.py`

Because Phase 0 already established the `SampleResult` object pattern,
this task simply adds a new attribute:

- Add `exp_vals` attribute to `SampleResult`: `numpy.ndarray` of
  shape `(shots, num_exp_vals)` with dtype `float64`. For circuits
  without EXP_VAL, shape is `(shots, 0)`.

- All four sampling functions already return `SampleResult` (from
  Phase 0), so `exp_vals` is automatically available on all results.

**Program metadata:**

- Expose `Program.num_exp_vals` property
- Expose `HirModule.num_exp_vals` property

**OpType and Opcode enums:**

- Add `OpType.EXP_VAL` to the Python OpType enum
- Add `Opcode.OP_EXP_VAL` to the Python Opcode enum

**HeisenbergOp.as_dict():**

- Add `exp_val_idx` field for EXP_VAL ops

**State:**

- Expose `State.exp_vals` property (list of float64)
- Update constructor to accept `num_exp_vals` parameter

### Task 6.3: WASM bindings

Touch: `src/wasm/bindings.cc`

The WASM `simulate_wasm()` currently returns only a measurement
histogram. For v1, EXP_VAL results in WASM can be silently ignored
(the histogram is still correct). No WASM changes required unless we
want Explorer support.

Future: the Explorer could show per-EXP_VAL expectation values in a
sidebar.

### Task 6.4: Binding tests

The existing `tests/python/test_introspection.py` has completeness
checks that verify every OpType and Opcode enum value is bound. Adding
the new enum values to the Python bindings (Task 6.2) will satisfy
these automatically.

**DoD:** `SampleResult.exp_vals` is accessible from all four sampling
functions. `State.exp_vals` is accessible. Enum completeness tests
pass.

---

## Phase 7: Introspection and Documentation

### Task 7.1: Introspection formatting

Touch: `src/ucc/util/introspection.cc`, `src/ucc/util/introspection.h`

- `op_type_to_str(OpType::EXP_VAL)` -> `"EXP_VAL"`
- `format_hir_op()` for EXP_VAL: e.g.
  `"EXP_VAL +X0*Y1 -> exp[0]"`
- `opcode_to_str(Opcode::OP_EXP_VAL)` -> `"OP_EXP_VAL"`
- `format_instruction()` for OP_EXP_VAL: e.g.
  `"OP_EXP_VAL cp=3 -> exp[0]"`

### Task 7.2: Documentation updates

Touch: `docs/opcodes.json`, relevant docs pages

- Add `EXP_VAL` entry to `hir_ops` in `docs/opcodes.json`
- Add `OP_EXP_VAL` entry to `opcodes` in `docs/opcodes.json`
- Update docs describing `sample()` return value (now a result object)
- List `EXP_VAL` as a UCC simulation extension in gate docs
- Note that `EXP_VAL` is non-physical and non-destructive

**DoD:** All introspection and doc completeness tests pass.

---

## Phase 8: Python Integration Tests

Touch: new file `tests/python/test_exp_value.py`

### Task 8.1: Qiskit exact oracle

For small circuits (H/S/T/CX + EXP_VAL), compare UCC expectation
outputs to exact Qiskit statevector Pauli expectations. This is the
gold-standard correctness check.

- Single-qubit: H then EXP_VAL X (expect +1), EXP_VAL Z (expect 0)
- Two-qubit Bell: CX then EXP_VAL X0*X1 (expect +1),
  EXP_VAL Z0*Z1 (expect +1)
- Non-Clifford: T gate then EXP_VAL on various Paulis
- Multi-EXP_VAL circuit: multiple EXP_VAL at different points, verify
  each matches the Qiskit expectation at that circuit depth

### Task 8.2: Statistical equivalence to destructive measurement

For the same Pauli P at the same circuit point:

- Circuit A: `EXP_VAL P`
- Circuit B: `MPP P` (destructive measurement)

Over many shots:

    mean(exp_values) ~= mean(1 - 2*measurement_bits)

with tight statistical tolerance. This is a strong end-to-end check
that the EXP_VAL kernel agrees with the established measurement
pathway.

### Task 8.3: Noise and Pauli-frame trajectory tests

Deterministic per-shot-verifiable examples:

**Test A: Noise flips expectation**
```
H 0
Z_ERROR(1.0) 0
EXP_VAL X0
```
Expected: always -1 (the Z error anti-commutes with X, flipping sign).

**Test B: Measurement feedback**
```
H 0
M 0
CX rec[-1] 1
EXP_VAL Z1
```
Expected per shot: `exp[0] == 1 - 2*meas[0]`.

**Test C: CZ feedback**
```
H 1
H 0
M 0
CZ rec[-1] 1
EXP_VAL X1
```
Expected per shot: `exp[0] == 1 - 2*meas[0]`.

These specifically verify that the expectation reads the current Pauli
frame, not just the raw active amplitudes.

### Task 8.4: Regression on no-EXP_VAL circuits

- Existing measurement/detector/observable behavior unchanged
- `result.exp_vals.shape == (shots, 0)`
- No `OP_EXP_VAL` in bytecode
- `Program.num_exp_vals == 0`
- `sample()` still returns correct meas/det/obs

**DoD:** All Python tests pass. Correctness validated against Qiskit
oracle and against destructive measurement statistics.

---

## Performance Contract

For circuits **without** EXP_VAL:

- No extra bytecode emitted
- No extra VM loops
- No extra work in lowering
- `sample()` allocates an empty exp_vals vector (zero cost)
- Existing benchmark throughput is unaffected

For circuits **with** EXP_VAL:

- One `O(2^k)` read-only pass per EXP_VAL site per shot
- This is the correct cost model: same as a measurement kernel but
  without the destructive collapse
- No AVX specialization in v1

---

## Acceptance Criteria

The implementation is complete when all of the following hold:

1. `EXP_VAL` parses, traces through the front-end, lowers, and executes
2. Expectation values are exact and non-destructive
3. The current sampled Pauli frame is correctly accounted for
4. Values are evaluated at the exact circuit point (optimizer barrier)
5. No optimizer pass reorders operations across EXP_VAL
6. `SampleResult.exp_vals` is populated; `State` exposes `exp_vals`
7. Circuits without EXP_VAL behave identically (empty expectation
   output)
8. Enum/binding/doc completeness tests pass
9. C++ kernel tests pass with exact values on handcrafted states
10. Python oracle tests pass against Qiskit and destructive measurement

---

## Implementation Notes

### Why not lower as a measurement?

Measurements are destructive: they collapse the state, halve the active
array, and write to the measurement record. EXP_VAL must be read-only.
Faking it as a measurement and then "undoing" the collapse would be
fragile, expensive, and architecturally wrong.

### Why not compress to one axis?

The `compress_pauli()` path emits routing instructions (SWAPs, basis
changes) that mutate the virtual frame. EXP_VAL must not mutate
anything. The direct full-mask kernel reads the state without side
effects.

### Why a hard barrier?

The user places EXP_VAL at a specific circuit location to probe the
state at that point. Moving operations across EXP_VAL -- even commuting
ones -- changes what the probe measures. A hard barrier is the only
semantically correct choice for v1.

### Constant pool separation

We store exp val Pauli masks in a separate `exp_val_masks` vector
rather than reusing `pauli_masks` (used by `OP_APPLY_PAULI`). This
avoids index collision and keeps the two concerns (frame mutation vs.
read-only probing) cleanly separated.

---

## Status

- [ ] Phase 0: Refactor sampling return types (prerequisite PR)
- [ ] Phase 1: Parser and AST
- [ ] Phase 2: HIR and Front-End
- [ ] Phase 3: Optimizer Rules
- [ ] Phase 4: Back-End / Lowering
- [ ] Phase 5: VM State and Execution Kernel
- [ ] Phase 6: Sampling API and Python Bindings
- [ ] Phase 7: Introspection and Documentation
- [ ] Phase 8: Python Integration Tests
