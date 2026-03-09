# Instruction Reference

This page documents all VM opcodes and HIR (Heisenberg IR) operation types
used by the UCC compiler. The same data powers the hover tooltips in the
[Compiler Explorer](../explorer.md).

!!! tip "Explorer Tooltips"
    In the Compiler Explorer, hover over any opcode or HIR keyword to see
    its description inline.

## VM Opcodes

The VM executes a flat stream of RISC-style 32-byte instructions. Each opcode
falls into one of the categories below.

### Frame

Frame ops update the Heisenberg tracking frame U_C. They are pure bookkeeping -- no state vector work is performed.

#### `OP_FRAME_CNOT`

**CNOT update on the Heisenberg tracking frame.**

Applies a CNOT (controlled-X) conjugation to the virtual Pauli frame U_C. This is pure bookkeeping with no state vector work -- the frame tracks how Clifford gates transform the Pauli basis.

**Operands:** `axis_1 (control), axis_2 (target)`

#### `OP_FRAME_CZ`

**CZ update on the Heisenberg tracking frame.**

Applies a controlled-Z conjugation to the virtual Pauli frame U_C. Like all frame ops, this has zero cost on the state vector -- it only updates the algebraic tracking structure.

**Operands:** `axis_1, axis_2`

#### `OP_FRAME_H`

**Hadamard update on the Heisenberg tracking frame.**

Applies a Hadamard conjugation to a single virtual axis in the Pauli frame. Swaps X and Z components of that axis in the frame.

**Operands:** `axis_1`

#### `OP_FRAME_S`

**S-gate update on the Heisenberg tracking frame.**

Applies an S (pi/4 phase) conjugation to a single virtual axis in the Pauli frame. Maps X -> Y on that axis.

**Operands:** `axis_1`

#### `OP_FRAME_S_DAG`

**S-dagger update on the Heisenberg tracking frame.**

Applies an S-dagger (inverse pi/4 phase) conjugation to a single virtual axis in the Pauli frame. Maps X -> -Y on that axis.

**Operands:** `axis_1`

#### `OP_FRAME_SWAP`

**SWAP update on the Heisenberg tracking frame.**

Swaps two virtual axes in the Pauli frame. Used by the compiler to route qubits for measurement or gate decomposition.

**Operands:** `axis_1, axis_2`

### Array

Array ops apply unitary gates directly to the Schrodinger state vector |phi>_A.

#### `OP_ARRAY_CNOT`

**CNOT gate on the Schrodinger state vector.**

Applies a controlled-X gate to two active virtual axes in the state vector |phi>_A. Pairs of amplitudes separated by the target axis stride are XOR-permuted.

**Operands:** `axis_1 (control), axis_2 (target)`

#### `OP_ARRAY_CZ`

**CZ gate on the Schrodinger state vector.**

Applies a controlled-Z gate to two active virtual axes. Negates amplitudes where both axes are in the |1> state.

**Operands:** `axis_1, axis_2`

#### `OP_ARRAY_SWAP`

**SWAP gate on the Schrodinger state vector.**

Swaps amplitudes between two active virtual axes. Used to route the measurement target to the highest axis before collapse.

**Operands:** `axis_1, axis_2`

#### `OP_ARRAY_MULTI_CNOT`

**Fused multi-control CNOT on the Schrodinger state vector.**

Applies a star-graph CNOT with multiple control axes and one target axis in a single fused pass over the state vector. The control axes are encoded as a 64-bit bitmask. Equivalent to a sequence of pairwise OP_ARRAY_CNOT instructions but faster.

**Operands:** `axis_1 (target), multi_gate.mask (control bitmask)`

#### `OP_ARRAY_MULTI_CZ`

**Fused multi-target CZ on the Schrodinger state vector.**

Applies a star-graph CZ with one control axis and multiple target axes in a single fused pass over the state vector. The target axes are encoded as a 64-bit bitmask. Equivalent to a sequence of pairwise OP_ARRAY_CZ instructions but faster.

**Operands:** `axis_1 (control), multi_gate.mask (target bitmask)`

#### `OP_ARRAY_H`

**Hadamard gate on the Schrodinger state vector.**

Applies a Hadamard (butterfly) transform to one active virtual axis. Each pair of amplitudes is replaced by their sum and difference, divided by sqrt(2).

**Operands:** `axis_1`

#### `OP_ARRAY_S`

**S-gate (phase) on the Schrodinger state vector.**

Applies diag(1, i) to one active virtual axis. Multiplies all amplitudes where that axis is |1> by i.

**Operands:** `axis_1`

#### `OP_ARRAY_S_DAG`

**S-dagger gate on the Schrodinger state vector.**

Applies diag(1, -i) to one active virtual axis. Multiplies all amplitudes where that axis is |1> by -i.

**Operands:** `axis_1`

### Subspace

Subspace ops change the size of the active subspace or apply non-Clifford rotations.

#### `OP_EXPAND`

**Promote a dormant qubit to active, growing k by 1.**

Doubles the state vector by applying a virtual Hadamard on a dormant axis: k -> k+1 and gamma is divided by sqrt(2). The new axis is initialized to equal superposition. This is the only instruction that grows the active subspace.

**Operands:** `axis_1`

#### `OP_PHASE_T`

**T-gate (pi/8 phase rotation) on an active axis.**

Applies diag(1, exp(i*pi/4)) to one active virtual axis. This is the primary non-Clifford gate -- it cannot be absorbed into the Heisenberg frame and requires direct state vector work.

**Operands:** `axis_1`

#### `OP_PHASE_T_DAG`

**T-dagger (inverse pi/8 phase rotation) on an active axis.**

Applies diag(1, exp(-i*pi/4)) to one active virtual axis. The conjugate of OP_PHASE_T.

**Operands:** `axis_1`

#### `OP_EXPAND_T`

**Fused EXPAND + T-gate in one array pass.**

Combines OP_EXPAND and OP_PHASE_T into a single instruction. Promotes a dormant qubit to active (k -> k+1) and immediately applies the T phase rotation, saving one pass over the state vector.

**Operands:** `axis_1`

#### `OP_EXPAND_T_DAG`

**Fused EXPAND + T-dagger in one array pass.**

Combines OP_EXPAND and OP_PHASE_T_DAG into a single instruction. Promotes a dormant qubit to active (k -> k+1) and immediately applies the T-dagger phase rotation, saving one pass over the state vector.

**Operands:** `axis_1`

### Measurement

Measurement ops collapse qubits, either algebraically (dormant) or by filtering/folding the state vector (active).

#### `OP_MEAS_DORMANT_STATIC`

**Deterministic measurement of a dormant qubit.**

The measured Pauli observable has a fixed eigenvalue determined by the Pauli frame vector p_x. The outcome is known without any state vector work or randomness. k is unchanged.

**Operands:** `axis_1 -> rec[classical_idx]`

#### `OP_MEAS_DORMANT_RANDOM`

**Random measurement of a dormant qubit.**

The qubit is in an equal superposition within the dormant subspace. The outcome is chosen by the RNG, and an algebraic phase is absorbed into gamma. No state vector work; k is unchanged.

**Operands:** `axis_1 -> rec[classical_idx]`

#### `OP_MEAS_ACTIVE_DIAGONAL`

**Z-basis measurement of an active qubit (diagonal collapse).**

Collapses the state vector along one active axis by keeping only the half consistent with the measurement outcome. Halves the array: k -> k-1. The outcome probability is computed from the squared amplitudes of each half.

**Operands:** `axis_1 -> rec[classical_idx]`

#### `OP_MEAS_ACTIVE_INTERFERE`

**X-basis measurement of an active qubit (interference collapse).**

Collapses the state vector by folding (adding or subtracting) two halves together, depending on the measurement outcome. Halves the array: k -> k-1. This handles the case where the measured Pauli has off-diagonal terms.

**Operands:** `axis_1 -> rec[classical_idx]`

#### `OP_SWAP_MEAS_INTERFERE`

**Fused SWAP + X-basis measurement in one array pass.**

Combines OP_ARRAY_SWAP and OP_MEAS_ACTIVE_INTERFERE into a single instruction. Swaps the target axis to the highest active dimension and immediately performs the interference (fold) measurement, halving the array: k -> k-1.

**Operands:** `axis_1 (swap_from), axis_2 (swap_to) -> rec[classical_idx]`

### Meta

Meta ops handle classical feedback, noise channels, and error correction bookkeeping.

#### `OP_APPLY_PAULI`

**Apply a full N-bit Pauli mask to the frame (classical feedback).**

XORs a precomputed N-qubit Pauli bitmask from the constant pool into the Pauli frame vector P, conditioned on a prior measurement result. Implements classical feedback (e.g. Pauli corrections after teleportation).

**Operands:** `cp_mask (constant pool index), condition (measurement record index)`

#### `OP_NOISE`

**Stochastic Pauli noise channel.**

Rolls the RNG against a probability table from the constant pool. If triggered, applies a random Pauli (X, Y, or Z) to the frame. Models depolarizing, dephasing, and other Pauli noise channels.

**Operands:** `cp_site (constant pool noise site index)`

#### `OP_NOISE_BLOCK`

**Contiguous block of noise sites processed in a batch.**

Processes a contiguous range [start, start+count) of noise sites from the constant pool in one instruction. When noise is disabled for a shot, the VM skips the entire block in O(1) instead of dispatching individual OP_NOISE instructions.

**Operands:** `pauli.cp_mask_idx (start_site), pauli.condition_idx (count)`

#### `OP_READOUT_NOISE`

**Classical bit-flip noise on a measurement result.**

Rolls the RNG against a readout error probability. If triggered, flips the classical measurement bit in the record. Models measurement readout errors separately from quantum noise.

**Operands:** `cp_entry (constant pool readout noise entry index)`

#### `OP_DETECTOR`

**Detector: parity check over measurement records.**

Computes the XOR parity of a set of measurement record entries (specified by a target list in the constant pool) and stores the result as a detector outcome. Used for quantum error correction syndrome extraction.

**Operands:** `cp_targets (constant pool target list index) -> det[detector_idx]`

#### `OP_POSTSELECT`

**Post-selection: discard shot if parity check fails.**

Same parity computation as OP_DETECTOR, but if the parity is nonzero the entire shot is discarded. Used for post-selected circuits where certain outcomes are conditioned upon.

**Operands:** `cp_targets (constant pool target list index) -> det[detector_idx]`

#### `OP_OBSERVABLE`

**Logical observable accumulator.**

Accumulates the XOR parity of measurement record entries into a logical observable. Used in error correction to track logical qubit values across multiple rounds of syndrome measurements.

**Operands:** `cp_targets (constant pool target list index) -> obs[observable_idx]`

## HIR Operation Types

The Heisenberg IR is the intermediate representation produced by the front-end.
Clifford gates are absorbed into the tracking frame and do not appear in the HIR.
What remains are non-Clifford operations, measurements, and meta-instructions.

### Non-Clifford

#### `T`, `T_DAG`

**T or T-dagger gate: pi/8 phase rotation on a Pauli product.**

The primary non-Clifford operation. Applies exp(i*pi/8 * P) where P is the Pauli product shown (e.g. +X0*Z1). T_DAG applies the conjugate rotation. These cannot be absorbed into the Heisenberg frame and are passed through to the back-end.

#### `S`, `S_DAG`

**S or S-dagger gate: pi/4 phase rotation on a Pauli product.**

Although S is a Clifford gate, when it appears in the HIR it means the front-end could not fully absorb it into the frame (e.g. it acts on a Pauli product that spans multiple qubits). S_DAG applies the inverse.

### Measurement

#### `MEASURE`

**Destructive measurement of a Pauli observable.**

Measures the Pauli product shown (e.g. +X0, -Z0*Z1) and stores the outcome in the measurement record. The Pauli is the effective observable after Heisenberg frame transformation -- it may differ from the original circuit's measurement basis.

### Feedback

#### `IF`, `THEN`

**Classical feedback: apply Pauli correction conditioned on a measurement.**

If the referenced measurement record entry is 1, applies the shown Pauli product to the frame. Implements feed-forward operations like teleportation corrections and QEC feedback.

### Noise

#### `NOISE`

**Stochastic Pauli noise channel.**

Represents a noise process (depolarizing, dephasing, etc.) that may randomly apply a Pauli operator. References a NoiseSite side-table with the channel probabilities.

#### `READOUT_NOISE`

**Classical bit-flip noise on a measurement result.**

Models measurement readout errors as a classical bit-flip probability applied to the measurement record after the quantum measurement.

### QEC

#### `DETECTOR`

**Detector: parity check over measurement records.**

Defines a parity check (XOR) over a set of measurement record entries. In error correction, detectors flag syndrome changes between rounds.

#### `OBSERVABLE`

**Logical observable accumulator.**

Accumulates measurement record parities into a logical observable for error correction tracking.
