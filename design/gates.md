# UCC Gate Support Reference

This document summarizes every Stim gate/instruction and its support status in UCC.
Gates are grouped by type, matching the categories in
[Stim's gate reference](https://github.com/quantumlib/Stim/blob/main/doc/gates.md).

## Pauli Gates

| Gate | UCC Status | Notes |
|------|-----------|-------|
| `I`  | Supported | Parsed for syntax validation; no AST node emitted (identity no-op) |
| `X`  | Supported | Single-qubit Clifford, absorbed AOT |
| `Y`  | Supported | Single-qubit Clifford, absorbed AOT |
| `Z`  | Supported | Single-qubit Clifford, absorbed AOT |

## Single-Qubit Clifford Gates

| Gate | UCC Status | Notes |
|------|-----------|-------|
| `C_NXYZ`     | Supported | Period-3 Clifford rotation |
| `C_NZYX`     | Supported | Period-3 Clifford rotation |
| `C_XNYZ`     | Supported | Period-3 Clifford rotation |
| `C_XYNZ`     | Supported | Period-3 Clifford rotation |
| `C_XYZ`      | Supported | Period-3 Clifford rotation |
| `C_ZNYX`     | Supported | Period-3 Clifford rotation |
| `C_ZYNX`     | Supported | Period-3 Clifford rotation |
| `C_ZYX`      | Supported | Period-3 Clifford rotation |
| `H`          | Supported | Hadamard (aliases: `H_XZ`) |
| `H_NXY`      | Supported | Hadamard in -X,Y plane |
| `H_NXZ`      | Supported | Negated Hadamard |
| `H_NYZ`      | Supported | Hadamard in -Y,Z plane |
| `H_XY`       | Supported | Hadamard in X,Y plane |
| `H_YZ`       | Supported | Hadamard in Y,Z plane |
| `S`          | Supported | Phase gate (aliases: `SQRT_Z`) |
| `S_DAG`      | Supported | Inverse phase gate (aliases: `SQRT_Z_DAG`) |
| `SQRT_X`     | Supported | Square root of X |
| `SQRT_X_DAG` | Supported | Inverse square root of X |
| `SQRT_Y`     | Supported | Square root of Y |
| `SQRT_Y_DAG` | Supported | Inverse square root of Y |

## Non-Clifford Gates (UCC extension)

| Gate | UCC Status | Notes |
|------|-----------|-------|
| `T`     | Supported | pi/8 gate (non-Clifford, not in Stim) |
| `T_DAG` | Supported | Inverse T gate (non-Clifford, not in Stim) |

## Two-Qubit Clifford Gates

| Gate | UCC Status | Notes |
|------|-----------|-------|
| `CX`           | Supported | Controlled-X / CNOT (aliases: `CNOT`, `ZCX`) |
| `CY`           | Supported | Controlled-Y (aliases: `ZCY`) |
| `CZ`           | Supported | Controlled-Z (aliases: `ZCZ`) |
| `CXSWAP`       | Supported | CX followed by SWAP |
| `CZSWAP`       | Supported | CZ followed by SWAP (aliases: `SWAPCZ`) |
| `II`           | Supported | Two-qubit identity (no-op, no AST node emitted) |
| `ISWAP`        | Supported | Imaginary SWAP |
| `ISWAP_DAG`    | Supported | Inverse imaginary SWAP |
| `SQRT_XX`      | Supported | Square root of XX |
| `SQRT_XX_DAG`  | Supported | Inverse square root of XX |
| `SQRT_YY`      | Supported | Square root of YY |
| `SQRT_YY_DAG`  | Supported | Inverse square root of YY |
| `SQRT_ZZ`      | Supported | Square root of ZZ |
| `SQRT_ZZ_DAG`  | Supported | Inverse square root of ZZ |
| `SWAP`         | Supported | Qubit SWAP |
| `SWAPCX`       | Supported | SWAP followed by CX |
| `XCX`          | Supported | X-controlled X |
| `XCY`          | Supported | X-controlled Y |
| `XCZ`          | Supported | X-controlled Z |
| `YCX`          | Supported | Y-controlled X |
| `YCY`          | Supported | Y-controlled Y |
| `YCZ`          | Supported | Y-controlled Z |

## Noise Channels

| Instruction | UCC Status | Notes |
|-------------|-----------|-------|
| `DEPOLARIZE1`       | Supported | Single-qubit depolarizing noise |
| `DEPOLARIZE2`       | Supported | Two-qubit depolarizing noise |
| `I_ERROR`           | Supported | Single-qubit identity error (parsed as no-op) |
| `II_ERROR`          | Supported | Two-qubit identity error (parsed as no-op) |
| `PAULI_CHANNEL_1`   | Supported | 3-parameter single-qubit Pauli channel |
| `PAULI_CHANNEL_2`   | Supported | 15-parameter two-qubit Pauli channel |
| `X_ERROR`           | Supported | Single-qubit X error |
| `Y_ERROR`           | Supported | Single-qubit Y error |
| `Z_ERROR`           | Supported | Single-qubit Z error |
| `E` / `CORRELATED_ERROR`      | **Not supported** | Correlated multi-qubit error |
| `ELSE_CORRELATED_ERROR`       | **Not supported** | Conditional correlated error |
| `HERALDED_ERASE`              | **Not supported** | Heralded erasure channel |
| `HERALDED_PAULI_CHANNEL_1`    | **Not supported** | Heralded single-qubit Pauli channel |

## Collapsing Gates (Measurements & Resets)

| Instruction | UCC Status | Notes |
|-------------|-----------|-------|
| `M`   | Supported | Z-basis measurement (aliases: `MZ`) |
| `MR`  | Supported | Measure + reset in Z-basis (aliases: `MRZ`) |
| `MRX` | Supported | Measure + reset in X-basis |
| `MRY` | Supported | Measure + reset in Y-basis |
| `MX`  | Supported | X-basis measurement |
| `MY`  | Supported | Y-basis measurement |
| `R`   | Supported | Reset to |0> (aliases: `RZ`) |
| `RX`  | Supported | Reset to |+> |
| `RY`  | Supported | Reset to |+i> |

## Pair Measurement Gates

| Instruction | UCC Status | Notes |
|-------------|-----------|-------|
| `MXX` | Supported | Desugared to MPP at parse time |
| `MYY` | Supported | Desugared to MPP at parse time |
| `MZZ` | Supported | Desugared to MPP at parse time |

## Generalized Pauli Product Gates

| Instruction | UCC Status | Notes |
|-------------|-----------|-------|
| `MPP`     | Supported | Multi-Pauli product measurement |
| `SPP`     | **Not supported** | Stochastic Pauli product |
| `SPP_DAG` | **Not supported** | Inverse stochastic Pauli product |

## Control Flow

| Instruction | UCC Status | Notes |
|-------------|-----------|-------|
| `REPEAT`  | Supported | Unrolled at parse time via text-level replay; nested supported |

## Annotations

| Instruction | UCC Status | Notes |
|-------------|-----------|-------|
| `DETECTOR`           | Supported | QEC detector declaration |
| `MPAD`               | Supported | Deterministic measurement padding (0 or 1) |
| `OBSERVABLE_INCLUDE` | Supported | Observable accumulator |
| `QUBIT_COORDS`       | Supported | Silently discarded (coordinate annotation) |
| `SHIFT_COORDS`       | Supported | Silently discarded (coordinate annotation) |
| `TICK`               | Supported | Timing layer marker |

## Unsupported Stim Gates

The following Stim gates are not currently implemented in UCC:

| Gate | Category | Reason |
|------|----------|--------|
| `CORRELATED_ERROR` / `E`        | Noise       | Requires correlated multi-qubit error model |
| `ELSE_CORRELATED_ERROR`         | Noise       | Depends on `CORRELATED_ERROR` |
| `HERALDED_ERASE`                | Noise       | Heralded erasure not modeled |
| `HERALDED_PAULI_CHANNEL_1`      | Noise       | Heralded channel not modeled |
| `SPP`                           | Pauli product | Stochastic Pauli product gate |
| `SPP_DAG`                       | Pauli product | Inverse of SPP |
