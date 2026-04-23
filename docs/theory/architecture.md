# Software Architecture

This page describes Clifft's concrete software architecture: how the codebase maps to the five-stage pipeline and the key integration contracts.

## Repository Layout

The source code mirrors the pipeline stages:

| Directory | Pipeline Stage | Role |
|-----------|---------------|------|
| `src/clifft/circuit/` | Input | Circuit AST, parser, target encoding |
| `src/clifft/frontend/` | Stage 1 | Drives stabilizer tableau, absorbs Cliffords, emits HIR |
| `src/clifft/optimizer/` | Stage 2 & 4 | Two-level optimization: HIR passes and bytecode passes |
| `src/clifft/backend/` | Stage 3 | Virtual frame tracking, Pauli localization, bytecode emission |
| `src/clifft/svm/` | Stage 5 | Runtime VM: executes bytecode over dense arrays |
| `src/python/` | Bindings | Python API via nanobind |

!!! important "Isolation Invariant"
    The VM (`svm/`) never includes stabilizer tableau code or evaluates tableau mathematics. It executes purely on basic C++ types and arrays.

## The Stim Integration Contract

Clifft uses [Stim](https://github.com/quantumlib/Stim) exclusively as an AOT mathematical tableau library, **not** as a circuit simulation engine. The runtime VM never touches Stim.

The compiler uses Stim to construct and manipulate the offline Clifford frame $U_C$ through the Heisenberg mapping, and exploits `TableauTransposedRaii` for efficient row operations when synthesizing the Pauli localization sequences emitted by the Back-End. Once compilation finishes, $U_C$ is discarded — the VM executes over the virtual basis alone.

## Optimization Passes

Clifft optimizes at two distinct IR levels, each with its own pass manager:

### HIR Passes (Pre-Lowering)

Operate on the Heisenberg IR before bytecode emission:

- **PeepholeFusionPass** — Algebraic T-gate cancellation and fusion (T+T=S, T+T_dag=identity)
- **StatevectorSqueezePass** — Reorders HIR operations to minimize peak active rank
- **RemoveNoisePass** — Strips all noise (not in default pipeline; used internally for noiseless reference shots)

### Bytecode Passes (Post-Lowering)

Operate on the finalized bytecode. These rewrite and fuse instructions to reduce array passes:

- **NoiseBlockPass** — Collapses runs of identical noise instructions into single block operations
- **MultiGatePass** — Fuses contiguous CNOT/CZ ops sharing an axis into star-graph instructions
- **ExpandTPass** / **ExpandRotPass** — Fuses expand + phase into single copy-and-rotate loops
- **SwapMeasPass** — Fuses swap + measurement into one operation
- **TileAxisFusionPass** — Fuses 2-qubit tile sequences into precomputed 4x4 unitaries
- **SingleAxisFusionPass** — Fuses single-axis operation chains into precomputed 2x2 unitaries

See the [Optimization Passes](../reference/passes.md) reference for detailed descriptions of each pass.

## Bytecode Format

The VM executes an instruction set with **32-byte cache-aligned instructions**. Each instruction encodes:

- An opcode (gate type, frame operation, measurement, etc.)
- Up to 2 virtual axis indices (`uint16_t`)
- Gate parameters (rotation angles, probabilities)

The fixed instruction size ensures L1 cache alignment and predictable memory access patterns during the hot simulation loop.

## Memory Model

The VM allocates a single contiguous complex array of size $2^{k_{\text{max}}}$ at program start. This array is never resized during execution. When measurements reduce the active set, the array is logically compacted (the compiler emits SWAP instructions to route measured qubits to the top axis before measurement).

The Pauli frame ($P$) is tracked as a pair of $n$-bit masks using the custom, auto-vectorized `clifft::BitMask<kMaxInlineQubits>`, supporting arbitrary scaling natively without heap allocations.

## Python Bindings

Clifft uses [nanobind](https://github.com/wjakob/nanobind) to expose the C++ core to Python. The Python layer provides:

- `clifft.compile()` and `clifft.sample()` as the primary interface
- `clifft.execute()` and `clifft.get_statevector()` for exact state inspection
- `clifft.trace()` for compilation pipeline debugging

See the [User Guide](../guide/compiling.md) for API details.
