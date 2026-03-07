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

## Not Yet Supported

| Gate | Category | Reason |
|------|----------|--------|
| `CORRELATED_ERROR` / `E` | Noise | Correlated multi-qubit error model |
| `ELSE_CORRELATED_ERROR` | Noise | Depends on `CORRELATED_ERROR` |
| `HERALDED_ERASE` | Noise | Heralded erasure not modeled |
| `HERALDED_PAULI_CHANNEL_1` | Noise | Heralded channel not modeled |
| `SPP`, `SPP_DAG` | Pauli product | Stochastic Pauli product gate |
