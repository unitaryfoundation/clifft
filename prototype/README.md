# clifft — Python Prototype

A pure-Python prototype of the **clifft** Ahead-of-Time (AOT) compilation
architecture for Clifford+T quantum circuit simulation, based on the
[generalized stabilizer formalism](https://arxiv.org/abs/2512.23037)
(Li et al., 2025).

This prototype serves as the **correctness oracle** for the production C++
implementation. It validates the full AOT pipeline — compiler, bytecode, and
VM — against direct statevector simulation.

## Architecture

clifft decouples the heavy O(n²) stabilizer tableau geometry from lightweight
O(1) per-shot execution through a compile-then-execute model:

```
Quantum Circuit
      │
  AOT Compiler         ← walks circuit with stim.TableauSimulator
      │                   rewinding operators to the Heisenberg frame
      │                   emitting flat bytecode + GF(2) basis tracking
      ▼
  Program (bytecode)   ← immutable instruction sequence
      │
  VM × N shots         ← per-shot: sign tracker + dense v[] array
                         no tableau, no Clifford math
```

The compiler absorbs all Clifford gates into the tableau and emits bytecode
only for non-Clifford gates (T/T†, generic LCU) and measurements. The VM
executes these instructions using only integer popcount, XOR, and dense
array arithmetic.

See the [design documents](../design/) for the full algorithmic and systems
architecture.

## Key Concepts

### GF(2) Basis Tracking

The compiler maintains a linearly independent GF(2) basis **V** of
shift-vectors. Each non-Clifford gate or measurement produces a shift β
that is classified against V:

| Condition | Opcode | Effect on v[] |
|-----------|--------|---------------|
| β ∉ span(V) | `BRANCH` | New dimension — array doubles |
| β ∈ span(V), β ≠ 0 | `COLLIDE` | In-place butterfly mix |
| β = 0 | `SCALAR_PHASE` | Diagonal phase — no structural change |

### Dominant Term Factoring (The Tangent Trick)

For T-gates (I ± iZ), the compiler factors out the dominant cos(π/8) term.
The identity branch carries weight 1.0 (zero FLOPs), and the spawned branch
gets relative weight ±i·tan(π/8) ≈ ±0.414i. All relative weights stay ≤ 1.0,
preventing IEEE-754 drift.

### Sign Tracking

Instead of carrying the full n×n tableau at runtime, the VM tracks only two
bitmasks (`destab_signs`, `stab_signs`) — one bit per virtual qubit. Stochastic
events (noise, measurement randomness) flip individual sign bits via pre-compiled
XOR masks. The compiler pre-computes a `mapped_gamma` mask per instruction so
the VM resolves all topological interference via `popcount(idx & mapped_gamma) % 2`.

## Files

### Core

| File | Lines | Description |
|------|-------|-------------|
| **`aot_compiler.py`** | ~1400 | AOT compiler + VM: GF(2) basis tracking, bytecode emission, per-shot VM execution, generic LCU support, AG pivot matrices, detector logic |
| **`state.py`** | ~420 | `GeneralizedStabilizerState`: direct (v, tableau) simulation used as reference oracle |
| **`tableau.py`** | ~390 | Stabilizer/destabilizer tableau in binary symplectic form (galois GF(2)) |
| **`clifford_proxy.py`** | ~180 | Fast pre-pass: computes upper bound on |v| via GF(2) shift-vector rank analysis |

### Tests

| File | Tests | Description |
|------|-------|-------------|
| **`test_aot.py`** | 58 | AOT compiler + VM validated against numpy statevector reference |
| **`test_clifford_proxy.py`** | 18 | GF(2) rank-based |v| bound predictions |
| **`test_state.py`** | 22 | Direct generalized stabilizer ops vs statevector |
| **`test_inverse_tableau.py`** | 13 | Inverse tableau correctness (Heisenberg rewinding) |
| **`test_measurement_divergence.py`** | 2 | Measurement edge cases |

### Other

| File | Description |
|------|-------------|
| **`examples.py`** | Worked examples showing how the (v, tableau) representation evolves |
| **`walkthrough.ipynb`** | Interactive notebook exploring the formalism |

## Usage

```bash
# Run all tests
uv run pytest -v

# Run only AOT compiler tests
uv run pytest test_aot.py -v

# Run examples (uses the direct state.py oracle)
uv run python examples.py
```

### AOT Compiler + VM

```python
from aot_compiler import AOTCompiler, VM

# Define a circuit as (gate, *args) tuples
circuit = [
    ("H", 0),
    ("T", 0),
    ("CNOT", 0, 1),
    ("M", 0),
    ("M", 1),
]

# Compile once
compiler = AOTCompiler(num_qubits=2)
program = compiler.compile(circuit)

# Execute many shots
for _ in range(1000):
    vm = VM(program)
    measurements = vm.run()
```

### Direct State Simulation (Oracle)

```python
from state import GeneralizedStabilizerState
import numpy as np

s = GeneralizedStabilizerState(2)
s.apply_h(0)          # Clifford: updates tableau only, v unchanged
s.apply_cnot(0, 1)    # Bell state, still |v|=1
s.apply_t(0)          # T gate: |v| may double

# Verify against statevector
sv = s.to_statevector()
print(f"|v| = {s.num_entries}, norm = {np.linalg.norm(sv):.6f}")
```

## Relationship to Production C++

This prototype validates the algorithms that will be implemented in C++:

| Prototype (Python) | Production (C++) |
|--------------------|-------------------|
| `AOTCompiler` class | `src/clifft/compiler/` — drives `stim::TableauSimulator<W>` |
| `VM` class | `src/clifft/vm/` — 32-byte bytecode interpreter |
| `stim.TableauSimulator` | Same (unmodified upstream via CMake FetchContent) |
| `numpy` arrays | Cache-aligned `std::complex<double>` (≤64q) / SIMD batched (GPU) |
| `dict`-based v | Dense `v[]` array with peak-rank pre-allocation |

The prototype is a **development tool**, not a user-facing component.
