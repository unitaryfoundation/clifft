# Supported Gates

UCC parses [Stim circuit format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md) and supports nearly all Stim gates, plus non-Clifford extensions.

## Pauli Gates

| Gate | Notes |
|------|-------|
| `I`  | Identity (no-op, parsed but not emitted) |
| `X`  | Pauli X |
| `Y`  | Pauli Y |
| `Z`  | Pauli Z |

All Pauli gates are single-qubit Cliffords absorbed at compile time (zero VM cost).

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
| `H_NXZ` | Negated Hadamard |
| `C_XYZ`, `C_ZYX`, ... | Period-3 Clifford rotations (all 8 variants) |

All single-qubit Cliffords are absorbed AOT — they update the Clifford frame $U_C$ at compile time and have zero cost at runtime.

## Non-Clifford Gates

| Gate | Notes |
|------|-------|
| `T` | $\pi/8$ gate |
| `T_DAG` | Inverse $\pi/8$ gate |

These are UCC extensions beyond Stim's gate set. Non-Clifford gates activate qubits in the virtual machine, expanding the active statevector.

## Two-Qubit Clifford Gates

| Gate | Notes |
|------|-------|
| `CX` / `CNOT` / `ZCX` | Controlled-X |
| `CY` / `ZCY` | Controlled-Y |
| `CZ` / `ZCZ` | Controlled-Z |
| `SWAP` | Qubit swap |
| `ISWAP`, `ISWAP_DAG` | Imaginary swap and inverse |
| `CXSWAP`, `SWAPCX` | CX+SWAP composites |
| `CZSWAP` / `SWAPCZ` | CZ+SWAP composite |
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
| `X_ERROR(p)` | Single-qubit X error |
| `Y_ERROR(p)` | Single-qubit Y error |
| `Z_ERROR(p)` | Single-qubit Z error |
| `PAULI_CHANNEL_1(px,py,pz)` | General single-qubit Pauli channel |
| `PAULI_CHANNEL_2(...)` | General two-qubit Pauli channel (15 params) |

## Annotations and Control Flow

| Instruction | Notes |
|-------------|-------|
| `REPEAT N { ... }` | Loop (unrolled at parse time) |
| `DETECTOR` | QEC detector declaration |
| `OBSERVABLE_INCLUDE` | Observable accumulator |
| `MPAD` | Deterministic measurement padding |
| `TICK` | Timing layer marker |
| `QUBIT_COORDS` | Coordinate annotation (discarded) |
| `SHIFT_COORDS` | Coordinate shift (discarded) |

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

Key properties:

- **Non-destructive**: does not collapse the state or affect measurements
- **Read-only**: does not mutate the Pauli frame, active array, or any records
- **Shot-local**: each shot produces its own exact expectation value
- **Barrier**: the optimizer will not reorder operations across `EXP_VAL`

Results are available via `SampleResult.exp_vals` (shape `(shots, num_exp_vals)`).

## Continuous Rotation Gates

UCC extends the Stim gate set with arbitrary-angle rotation gates. All angle
parameters are in **half-turns** (multiply by pi to get radians).

### Single-Qubit Rotations

| Gate | Syntax | Notes |
|------|--------|-------|
| `R_X` | `R_X(alpha) target` | Rotation about X axis by `alpha * pi` radians |
| `R_Y` | `R_Y(alpha) target` | Rotation about Y axis by `alpha * pi` radians |
| `R_Z` | `R_Z(alpha) target` | Rotation about Z axis by `alpha * pi` radians |
| `U3`  | `U3(theta,phi,lambda) target` | General SU(2) gate = `R_Z(phi) R_Y(theta) R_Z(lambda)` |
| `U`   | `U(theta,phi,lambda) target` | Alias for `U3` |

!!! note "Name conflicts with Stim"
    UCC uses `R_X`, `R_Y`, `R_Z` (with underscores) to avoid collision with
    Stim's `RX` / `RY` reset-in-basis instructions.

### Two-Qubit Pauli Rotations

| Gate | Syntax | Notes |
|------|--------|-------|
| `R_XX` | `R_XX(alpha) q0 q1` | `exp(-i * alpha * pi/2 * XX)` |
| `R_YY` | `R_YY(alpha) q0 q1` | `exp(-i * alpha * pi/2 * YY)` |
| `R_ZZ` | `R_ZZ(alpha) q0 q1` | `exp(-i * alpha * pi/2 * ZZ)` |

Duplicate target qubits (e.g. `R_XX(0.5) 3 3`) are rejected at parse time.

### Multi-Qubit Pauli Rotation

| Gate | Syntax | Notes |
|------|--------|-------|
| `R_PAULI` | `R_PAULI(alpha) X0*Y1*Z2` | Arbitrary Pauli product rotation |

The target list uses Stim's Pauli product syntax (e.g. `X0*Y1*Z2`). Maximum
target count is 64 qubits per instruction.

### Compilation Path

All rotation gates are reduced by the front-end to a single HIR type
(`PHASE_ROTATION`) via Clifford absorption. The front-end conjugates each
rotation axis into the Heisenberg frame, factors out the global phase
`e^{-i*alpha*pi/2}` into `global_weight`, and passes the relative diagonal
phase `diag(1, e^{i*alpha*pi})` to the back-end.

The peephole optimizer fuses adjacent `PHASE_ROTATION` ops on the same Pauli
and demotes rotations at Clifford/T angles back to their exact discrete
counterparts (e.g. `R_Z(0.5)` becomes `S`, `R_Z(0.25)` becomes `T`).

## Not Yet Supported

| Gate | Category | Reason |
|------|----------|--------|
| `CORRELATED_ERROR` / `E` | Noise | Correlated multi-qubit error model |
| `ELSE_CORRELATED_ERROR` | Noise | Depends on `CORRELATED_ERROR` |
| `HERALDED_ERASE` | Noise | Heralded erasure not modeled |
| `HERALDED_PAULI_CHANNEL_1` | Noise | Heralded channel not modeled |
| `SPP`, `SPP_DAG` | Pauli product | Stochastic Pauli product gate |
