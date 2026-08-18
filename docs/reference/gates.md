# Supported Gates

Clifft parses [Stim circuit format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md) and supports nearly all Stim gates, plus non-Clifford extensions.

## Pauli Gates

| Gate | Notes |
|------|-------|
| `X`  | Pauli X |
| `Y`  | Pauli Y |
| `Z`  | Pauli Z |

All Pauli gates are single-qubit Cliffords absorbed at compile time, so they
do not become active-state actions unless a later dynamic dependency requires
one.

## Single-Qubit Clifford Gates

| Gate | Notes |
|------|-------|
| `H` | Hadamard (alias: `H_XZ`) |
| `S` | Phase gate (alias: `SQRT_Z`) |
| `S_DAG` | Inverse phase gate (alias: `SQRT_Z_DAG`) |
| `SQRT_X`, `SQRT_X_DAG` | Square root of X and inverse |
| `SQRT_Y`, `SQRT_Y_DAG` | Square root of Y and inverse |
| `H_XY`, `H_NXY` | Hadamard variants in X,Y plane |
| `H_YZ`, `H_NYZ` | Hadamard variants in Y,Z plane |
| `H_NXZ` | Hadamard variant swapping -X and +Z axes |
| `C_XYZ`, `C_ZYX`, `C_NXYZ`, `C_NZYX`, `C_XNYZ`, `C_XYNZ`, `C_ZNYX`, `C_ZYNX` | Period-3 Clifford rotations |

All single-qubit Cliffords are absorbed AOT — they update the Clifford frame $U_C$ at compile time and have zero cost at runtime.

## Pauli-Product Clifford Gates

| Gate | Syntax | Notes |
|------|--------|-------|
| `SPP` | `SPP X0*Y1*Z2` | Generalized `S` gate over a Pauli product |
| `SPP_DAG` | `SPP_DAG X0*Y1*Z2` | Inverse generalized `S` gate |

These gates accept the same product syntax as `MPP`, including multiple
whitespace-separated products in one instruction. Prefixing a Pauli term with
`!` negates the product and reverses the corresponding phase gate. A qubit may
appear only once in each product, which is stricter than Stim for some
syntactically valid Hermitian products.

The default optimizer absorbs `SPP` and `SPP_DAG` into the Clifford frame, so
they have no runtime cost. Their named-gate phase convention is exact, so
`SPP Z0` matches `S 0`, including global phase.

## Non-Clifford Extensions

Clifft extends Stim with discrete and arbitrary-angle non-Clifford gates.
These operations can add active stabilizer coordinates and expand the active
state vector.

### T Gates

| Gate | Notes |
|------|-------|
| `T` | $\pi/8$ gate |
| `T_DAG` | Inverse $\pi/8$ gate |

### Pauli-Product Phase Gates

| Gate | Syntax | Notes |
|------|--------|-------|
| `TPP` | `TPP X0*Y1*Z2` | Generalized `T` gate over a Pauli product |
| `TPP_DAG` | `TPP_DAG X0*Y1*Z2` | Inverse generalized `T` gate |

These gates accept the same product syntax as `MPP`, including multiple
whitespace-separated products in one instruction. Prefixing a Pauli term with
`!` negates the product and reverses the corresponding phase gate. A qubit may
appear only once in each product, which is stricter than Stim for some
syntactically valid Hermitian products.

`TPP` and `TPP_DAG` emit one generalized T operation per product. Their
named-gate phase convention is exact, so `TPP Z0` matches `T 0`, including
global phase.

### Rewrite Gates

These gate names are accepted by the parser as fixed rewrite rules into
Clifft-native gates. They are rewritten during parsing and do not appear as
distinct frontend, planner, or executor gate types.

| Gate | Syntax | Notes |
|------|--------|-------|
| `CH` | `CH c t` | Controlled-Hadamard; rewritten to `R_Y(0.25) t; CX c t; R_Y(-0.25) t` |
| `CCZ` | `CCZ a b c` | Controlled-controlled-Z; rewritten to 7 `T`/`T_DAG` gates and 6 `CX` gates |
| `CCX` | `CCX a b t` | Toffoli gate; rewritten as `H t; CCZ a b t; H t` |

### Continuous Rotations

Clifft extends the Stim gate set with arbitrary-angle rotation gates. All angle
parameters are in **half-turns** (multiply by pi to get radians).

#### Single-Qubit Rotations

| Gate | Syntax | Notes |
|------|--------|-------|
| `R_X` | `R_X(alpha) target` | Rotation about X axis by `alpha * pi` radians |
| `R_Y` | `R_Y(alpha) target` | Rotation about Y axis by `alpha * pi` radians |
| `R_Z` | `R_Z(alpha) target` | Rotation about Z axis by `alpha * pi` radians |
| `U3`  | `U3(theta,phi,lambda) target` | General SU(2) gate = `R_Z(phi) R_Y(theta) R_Z(lambda)` |
| `U`   | `U(theta,phi,lambda) target` | Alias for `U3` |

!!! note "Name conflicts with Stim"
    Clifft uses `R_X`, `R_Y`, `R_Z` (with underscores) to avoid collision with
    Stim's `RX` / `RY` reset-in-basis instructions.

#### Two-Qubit Pauli Rotations

| Gate | Syntax | Notes |
|------|--------|-------|
| `R_XX` | `R_XX(alpha) q0 q1` | `exp(-i * alpha * pi/2 * XX)` |
| `R_YY` | `R_YY(alpha) q0 q1` | `exp(-i * alpha * pi/2 * YY)` |
| `R_ZZ` | `R_ZZ(alpha) q0 q1` | `exp(-i * alpha * pi/2 * ZZ)` |

Duplicate target qubits (e.g. `R_XX(0.5) 3 3`) are rejected at parse time.

#### Multi-Qubit Pauli Rotation

| Gate | Syntax | Notes |
|------|--------|-------|
| `R_PAULI` | `R_PAULI(alpha) X0*Y1*Z2` | Arbitrary Pauli product rotation |

The target list uses Stim's Pauli product syntax (e.g. `X0*Y1*Z2`). Maximum
target count is 64 qubits per instruction.

## Two-Qubit Clifford Gates

| Gate | Notes |
|------|-------|
| `CX` / `CNOT` / `ZCX` | Controlled-X |
| `CY` / `ZCY` | Controlled-Y |
| `CZ` / `ZCZ` | Controlled-Z |
| `SWAP` | Qubit swap |
| `ISWAP`, `ISWAP_DAG` | Imaginary swap and inverse |
| `CXSWAP`, `SWAPCX` | CX+SWAP composites |
| `CZSWAP` | CZ+SWAP composite (alias: `SWAPCZ`) |
| `SQRT_XX`, `SQRT_XX_DAG` | Square root of XX and inverse |
| `SQRT_YY`, `SQRT_YY_DAG` | Square root of YY and inverse |
| `SQRT_ZZ`, `SQRT_ZZ_DAG` | Square root of ZZ and inverse |
| `XCX`, `XCY`, `XCZ` | X-controlled gates |
| `YCX`, `YCY`, `YCZ` | Y-controlled gates |

Two-qubit Cliffords are also absorbed at compile time.

## Measurements and Resets

| Instruction | Notes |
|-------------|-------|
| `M` / `MZ` | Z-basis measurement |
| `MX` | X-basis measurement |
| `MY` | Y-basis measurement |
| `MR` / `MRZ` | Measure + reset (Z-basis) |
| `MRX` | Measure + reset (X-basis) |
| `MRY` | Measure + reset (Y-basis) |
| `R` / `RZ` | Reset to $\|0\rangle$ |
| `RX` | Reset to $\|+\rangle$ |
| `RY` | Reset to $\|{+i}\rangle$ |

## Multi-Qubit Measurements

| Instruction | Notes |
|-------------|-------|
| `MPP` | Multi-Pauli product measurement |
| `MXX` | Pair XX measurement (desugared to MPP) |
| `MYY` | Pair YY measurement (desugared to MPP) |
| `MZZ` | Pair ZZ measurement (desugared to MPP) |

## Noise Channels

| Instruction | Notes |
|-------------|-------|
| `DEPOLARIZE1(p)` | Single-qubit depolarizing noise |
| `DEPOLARIZE2(p)` | Two-qubit depolarizing noise |
| `DEPOLARIZE3(p)` | Three-qubit depolarizing noise over triples of targets |
| `X_ERROR(p)` | Single-qubit X error |
| `Y_ERROR(p)` | Single-qubit Y error |
| `Z_ERROR(p)` | Single-qubit Z error |
| `PAULI_CHANNEL_1(px,py,pz)` | General single-qubit Pauli channel |
| `PAULI_CHANNEL_2(...)` | General two-qubit Pauli channel (15 params) |
| `PAULI_CHANNEL_3(...)` | General three-qubit Pauli channel (63 params) |
| `CORRELATED_ERROR(p)` / `E(p)` | Correlated Pauli product error |
| `ELSE_CORRELATED_ERROR(p)` | Else-branch in a correlated-error chain |
| `READOUT_NOISE(p01[, p10])` | Classical bit-flip on a measurement record (`rec[-k]` targets) |

`READOUT_NOISE` flips already-recorded bits rather than acting on a qubit,
so its targets are measurement-record references. With one argument the
flip is symmetric; with two, a recorded 0 flips with probability `p01` and
a recorded 1 with probability `p10`. Record targets do not take the `!`
inversion marker — swap the two probabilities instead. Noisy measurements
(e.g., `M(0.01) 0`) are parser shorthand for a clean measurement followed
by `READOUT_NOISE(0.01) rec[-1]`.

`DEPOLARIZE3(p) a b c` applies one of the 63 non-identity Pauli products on
`a,b,c` with probability `p/63` each, and identity with probability `1-p`.
`PAULI_CHANNEL_3` uses the same lexicographic Pauli order as
`PAULI_CHANNEL_2`, extended to three qubits: `IIX`, `IIY`, `IIZ`, `IXI`,
`IXX`, ..., `ZZZ`.

`CORRELATED_ERROR(p) X0 Z1` applies the listed Pauli product with probability
`p`. Pauli terms may be whitespace-separated or combined with `*`; all Pauli
targets on the instruction form one product. Repeated terms on the same qubit
multiply modulo Pauli phase, so `E(1) X0 Z0` is equivalent to a Y error and
`E(1) X0 X0` is an identity event.

`ELSE_CORRELATED_ERROR(p)` must immediately follow `CORRELATED_ERROR` or another
`ELSE_CORRELATED_ERROR`. Its `p` is conditional on no earlier link in the chain
firing. Clifft lowers each contiguous chain to one noise site with absolute
channel probabilities.

### Leakage and Loss Annotations (experimental)

| Instruction | Notes |
|-------------|-------|
| `LEAKAGE(p)` | Moves `g` to `leak_g` and `e` to `leak_e` with probability `p`; other levels are unchanged |
| `LOSS(p)` | Loses each target with probability `p`, from any occupied level |
| `LEVEL_TRANSITION[name]` | Fires the model's named transition matrix on each target |

Both are recognized only by the leakage/loss sampler — `clifft.compile()`
rejects them and points to `clifft.noncomp.sample`. See the
[Leakage and Loss guide](../guide/leakage-and-loss.md).

## Identity Gates

| Gate | Notes |
|------|-------|
| `I` | Single-qubit identity (parsed but not emitted) |
| `II` | Two-qubit identity (parsed but not emitted) |
| `I_ERROR` | Single-qubit identity error (no-op) |
| `II_ERROR` | Two-qubit identity error (no-op) |

These are accepted for compatibility with Stim circuits but have no effect.

## Annotations and Control Flow

| Instruction | Notes |
|-------------|-------|
| `REPEAT N { ... }` | Loop (unrolled at parse time) |
| `DETECTOR` | QEC detector declaration |
| `OBSERVABLE_INCLUDE` | Observable accumulator over measurement records |
| `MPAD` | Measurement-record padding with literal 0/1 bits |
| `TICK` | Timing layer marker |
| `QUBIT_COORDS` | Coordinate annotation (discarded) |
| `SHIFT_COORDS` | Coordinate shift (discarded) |

`OBSERVABLE_INCLUDE` currently supports `rec[-k]` measurement-record targets.
Stim also permits Pauli-term targets on `OBSERVABLE_INCLUDE`, which Clifft does
not currently parse.

`MPAD(p)` is accepted; the optional probability noisily flips the padded
measurement-record bits.

## Expectation Value Probes

| Instruction | Syntax | Notes |
|-------------|--------|-------|
| `EXP_VAL` | `EXP_VAL X0*Y1*Z2` | Non-destructive expectation value probe |

`EXP_VAL` evaluates the expectation value of one or more Pauli products at the
exact point in the circuit where it appears. It uses the same Pauli product
syntax as `MPP` — multiple whitespace-separated products per instruction are
supported, each producing one `float64` result in `[-1, 1]`.

```
H 0
EXP_VAL X0          # single Pauli: <X> on qubit 0
EXP_VAL X0*Y1*Z2    # multi-qubit product
EXP_VAL X0*X1 Z0*Z1 # two products in one instruction
```

Results are available via `SampleResult.exp_vals` (shape `(shots, num_exp_vals)`).


## Not Yet Supported

| Gate | Category | Reason |
|------|----------|--------|
| `HERALDED_ERASE` | Noise | Heralded erasure not modeled |
| `HERALDED_PAULI_CHANNEL_1` | Noise | Heralded channel not modeled |
