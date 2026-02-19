# UCC — Architecture

This document describes the concrete software architecture of UCC: repository
structure, dependency management, Stim integration strategy, build system, Python
bindings, and implementation phasing.

For the system overview and pipeline design see [`overview.md`](overview.md).
For C++ data structures and VM execution semantics see
[`data_structs.md`](data_structs.md).

---

## 1. Overview

**UCC** — the **Universal Compiler Collection** — is a multi-level compiler
infrastructure and execution engine for universal quantum circuits (Clifford + T
and beyond).

> **Note:** UCC has evolved from its origins as a Clifford tableau tracker. By
> natively embracing *non-unitary* operations (measurements, state collapse,
> stochastic noise, LCU expansions) and the T-gate, it is now a *universal*
> fault-tolerant compiler and execution engine. The bytecode execution engine is
> just *one* of the back-ends — another primary use-case is exporting the
> optimized IR to physical hardware routers (like tqec/PBC).

UCC uses a **multi-level Ahead-of-Time (AOT) compilation** model with four
distinct stages:

```
ucc::Circuit (AST)         ← parser reads .stim-superset text
        │
   1. Front-End               ← drives stim::TableauSimulator<W> for Clifford
        │                       math; emits HeisenbergOps for everything else
        ▼
   Heisenberg IR (HIR)
        │
   2. Middle-End (Optimizer)   ← O(1) commutation, fusion, cancellation
        │
        ▼
   Optimized HIR               ← [export point for tqec / PBC routing]
        │
   3. Compiler Back-End        ← GF(2) basis tracking, bytecode emission,
        │                       NoiseSchedule compilation
        ▼
   Program (bytecode + ConstantPool)
        │
   4. VM × N shots             ← lightweight per-shot execution; no Tableau,
                                 no Stim simulator, only sign bits + v[] array
```

**Key properties:**

- The Front-End absorbs all O(n²) Boolean tableau geometry. The VM sees only
  O(1) bitwise sign-tracking and dense array arithmetic.
- The Middle-End optimizes at the speed of bitwise arithmetic — no DAG
  traversal, no matrix algebra.
- Stim is an **unmodified upstream dependency** fetched via CMake FetchContent.
  UCC never patches, vendors, or forks Stim.
- UCC has its **own circuit parser and AST** (`ucc::Circuit`) that accepts
  a superset of the Stim `.stim` format (adding `T`, `T_DAG`, and future
  non-Clifford gates).
- The C++ core is exposed to Python via nanobind bindings.

---

## 2. Data Transfer, Serialization & CLI

### 2.1 Multi-Tier Serialization Model

UCC uses an **in-memory object/pointer model** for fast JIT Python execution.
All pipeline stages pass C++ objects directly — no serialization overhead during
normal compilation and sampling workflows.

### 2.2 One-Way Export Formats (MVP)

For debugging, interoperability, and physical routing, UCC supports **one-way
export** of intermediate representations:

| Export Point | Format | Purpose |
|--------------|--------|--------|
| Optimized HIR | `.hir` (human-readable text) | Debugging optimizer passes; export to tqec/PBC routing |
| Optimized HIR | `.hir.json` (JSON) | Machine-readable HIR for external tooling |
| Program | `.bin` (FlatBuffers or custom binary) | HPC cluster loading; pre-compiled circuit distribution |

**🚨 MVP Scope:** For the MVP, these are strictly *one-way exports*. Writing parsers
to load `.hir` or `.bin` back into C++ memory is deferred to Phase 4. The focus
is on correctness and debuggability, not round-trip serialization.

### 2.3 LLVM-Style CLI

UCC provides a modular command-line interface using subcommands:

```bash
# Compile circuit to optimized HIR (debugging, export to routing tools)
ucc opt circuit.stim -o opt.hir

# Lower optimized HIR to bytecode program
ucc lower opt.hir -o program.bin

# Execute program with sampling
ucc run program.bin --shots 1M --threads 8

# Full pipeline (compile + run)
ucc sample circuit.stim --shots 1M --threads 8
```

This separation enables:
- **Debugging:** Inspect the HIR after optimization to verify fusion/cancellation.
- **Caching:** Pre-compile expensive circuits once, distribute `.bin` files.
- **Integration:** Export `.hir` directly to physical routing tools without VM overhead.

---

## 3. Repository Layout

```
ucc/
├── CMakeLists.txt                # Top-level: builds libraries, tests, Python module
├── pyproject.toml                # scikit-build-core config + Python metadata
├── README.md
├── LICENSE
├── design.                       # Design documents (this file lives here)
│   ├── architecture.md           # ← you are here
│   ├── overview.md               # System overview, pipeline, performance thesis
│   └── data_structs.md           # HIR, VM bytecode, constant pool, execution loop
├── src/
│   └── ucc/
│       ├── circuit/              # Circuit AST, parser, gate metadata
│       │   ├── circuit.h/.cc     #   ucc::Circuit — flat instruction list + REPEAT nesting
│       │   ├── parser.h/.cc      #   Text → ucc::Circuit (stim-superset syntax)
│       │   └── gate_data.h       #   Gate enum, arity, Clifford/non-Clifford classification
│       ├── frontend/             # Front-End: Clifford absorption + HIR emission
│       │   ├── frontend.h/.cc    #   Main trace loop; drives stim::TableauSimulator
│       │   └── hir.h   #   HeisenbergOp, HirModule definitions
│       ├── optimizer/            # Middle-End: HIR optimization passes
│       │   └── optimizer.h/.cc   #   Commutation, fusion, cancellation, layering
│       ├── backend/              # Compiler Back-End: HIR → bytecode lowering
│       │   ├── backend.h/.cc     #   Code generation loop, GF(2) basis tracking
│       │   ├── gf2_basis.h/.cc   #   GF(2) shift-vector basis V management
│       │   └── ssa_map.h/.cc     #   Virtual → physical qubit mapping for REPEAT
│       ├── vm/                   # Runtime VM (runs per shot)
│       │   ├── program.h/.cc     #   Bytecode program + ConstantPool constant pool
│       │   ├── vm.h/.cc          #   Bytecode interpreter / execution engine
│       │   └── shot_state.h/.cc  #   Per-shot mutable state (signs, v[], meas_record)
│       └── util/
│           └── config.h          #   Compile-time configuration (W, precision, limits)
├── src/python/
│   ├── CMakeLists.txt            # nanobind module target
│   ├── bindings.cc               # nanobind module definition (_ucc_core)
│   └── ucc/
│       ├── __init__.py           # Public Python API (re-exports from _ucc_core)
│       └── py.typed              # PEP 561 marker
├── tests/
│   ├── CMakeLists.txt
│   ├── test_parser.cc            # Circuit parser round-trips
│   ├── test_frontend.cc          # HIR emission for known circuits
│   ├── test_optimizer.cc         # Commutation, fusion, cancellation correctness
│   ├── test_backend.cc           # Opcode emission, GF(2) basis, NoiseSchedule
│   ├── test_vm.cc                # VM opcode execution (unit)
│   ├── test_end_to_end.cc        # Full pipeline, compare to known distributions
│   └── python/
│       └── test_api.py           # Python binding smoke tests (pytest)
├── benchmarks/                   # C++ microbenchmarks (Catch2 BENCHMARK)
├── tools/bench/                  # Cross-simulator comparison (separate pyproject.toml)
├── prototype/                    # Python correctness oracle (dev tool, not user-facing)
├── cmake/
│   ├── FetchStim.cmake           # FetchContent for upstream Stim
│   └── FetchCatch2.cmake         # FetchContent for Catch2 v3
└── .github/workflows/            # CI: build, test, wheels
```

### Rationale for the four-directory split

The `frontend/`, `optimizer/`, `backend/`, `vm/` directories mirror the
multi-level pipeline:

| Layer | Lifetime | Stim dependency | Hot path? |
|-------|----------|-----------------|----------|
| **circuit/** | Parse time | None | No |
| **frontend/** | Trace time (once) | Heavy — TableauSimulator, Tableau, PauliString | No |
| **optimizer/** | Optimize time (once) | None — pure bitwise arithmetic on HIR | No |
| **backend/** | Code gen time (once) | Light — GF(2) basis math only | No |
| **vm/** | Per shot (×millions) | Light — `simd_bits<W>` only (>64 qubits) | **Yes** |

This separation enforces key invariants:
- The VM never touches `stim::Tableau` or `stim::TableauSimulator`.
- The optimizer never sees VM memory layout (x_mask, commutation_mask).
- The Front-End's Stim dependency is contained to a single directory.

---

## 4. Dependencies

### C++

| Dependency | Version | Mechanism | Role |
|------------|---------|-----------|------|
| **Stim** | HEAD (pinned tag) | CMake FetchContent | Tableau math in frontend; `simd_bits`/`simd_bit_table` in VM |
| **Catch2** | v3.x | CMake FetchContent | Unit tests + microbenchmarks |
| **nanobind** | latest | CMake FetchContent (via scikit-build-core) | Python bindings |

**Stim is an unmodified upstream dependency.** We do not fork, patch, or vendor
it. `cmake/FetchStim.cmake` fetches a pinned commit from
`https://github.com/quantumlib/Stim.git` and links `libstim` as a static
library. If Stim's internal API changes, we update our call sites — not Stim.

### Python (build + dev)

| Package | Role |
|---------|------|
| scikit-build-core | PEP 517 build backend (drives CMake) |
| nanobind | Binding generator |
| numpy | Array return type for `sample()` |
| pytest | Python-side tests |

### Python (prototype / dev-only)

| Package | Role |
|---------|------|
| stim | Python prototype uses `stim.TableauSimulator` |
| numpy | Statevector oracle |

Dev-only dependencies are **not** required by the built package.

---

## 5. Circuit Format & Parser

UCC accepts a **superset of the Stim `.stim` format**. Any valid Stim
circuit that uses only supported gates is also a valid UCC circuit.

### Extensions beyond Stim

| Gate | Syntax | Notes |
|------|--------|-------|
| `T` | `T 0 3 7` | π/8 phase gate |
| `T_DAG` | `T_DAG 2` | T† (inverse π/8 phase) |
| *(future)* | `CCZ 0 1 2` etc. | Additional non-Cliffords via LCU |

### ucc::Circuit AST

The parser produces a `ucc::Circuit` — the **Abstract Syntax Tree (AST)** — a
flat vector of AST nodes with nested `REPEAT` blocks represented as a
sub-circuit pointer + repeat count. The term "IR" is reserved strictly for the
**Heisenberg IR (HIR)** — the output of the Front-End:

```cpp
namespace ucc {

enum class GateType : uint16_t {
    // Clifford gates (matching Stim's gate set)
    H, S, S_DAG, X, Y, Z, SQRT_X, SQRT_X_DAG, SQRT_Y, SQRT_Y_DAG,
    CX, CY, CZ, SWAP, ISWAP, ISWAP_DAG,
    // Non-Clifford extensions
    T, T_DAG,
    // Noise channels
    DEPOLARIZE1, DEPOLARIZE2, X_ERROR, Z_ERROR,
    // Annotations / control flow
    M, MR, MX, MY, MPP, R, RX, RY,
    DETECTOR, OBSERVABLE_INCLUDE, TICK,
    REPEAT,
    // ... full Stim gate coverage added incrementally
};

struct AstNode {
    GateType gate;
    double arg;                         // gate argument (e.g. noise probability)
    std::vector<uint32_t> targets;      // qubit indices or measurement record refs
    std::vector<double> coords;         // DETECTOR coordinates (empty for most gates)
};

struct Circuit {
    std::vector<AstNode> nodes;
    // REPEAT blocks: the AstNode with gate==REPEAT stores
    // {repeat_count, sub_circuit_index} and sub-circuits are held here:
    std::vector<Circuit> blocks;
};

}  // namespace ucc
```

### Why we write our own parser

- **Stim's parser has no extension mechanism.** Adding `T` requires modifying
  Stim's internal gate dictionaries, keyword tables, and decomposition maps.
  That means maintaining a fork.
- **Moderate effort.** The `.stim` text format is simple (one instruction per
  line, `REPEAT k { ... }` blocks, `rec[-k]` references, `DETECTOR(x,y,z)`
  coordinate syntax). A clean-room parser is ~500–800 lines.
- **The Front-End already needs custom iteration.** It must intercept
  non-Clifford gates, stream REPEAT blocks, resolve `rec[-k]` indices, and
  propagate DETECTOR coordinates — all logic that sits outside Stim's
  `CircuitInstruction` iteration.
- **No validation burden.** We scope the parser to the gates we actually
  implement. Unknown gates produce a clear error.

---

## 6. Stim Integration

**Stim is used as a tableau math library, not as a circuit engine.** This is
the most important architectural decision in ucc.

### What we use from Stim (unmodified, via FetchContent)

#### Front-End (heavy use, runs once)

| Stim type / method | How the Front-End uses it |
|----|-----|
| `stim::TableauSimulator<W>` | Clifford frame proxy. For each Clifford gate, the Front-End constructs a `stim::CircuitInstruction` and calls the corresponding `do_*` method. This incrementally builds the inverse tableau that encodes the Heisenberg picture. |
| `stim::Tableau<W>` — inverse row reads | Heisenberg rewinding: `sim.inv_state.zs[q]` yields the rewound Z-observable for qubit q. Used to extract `destab_mask`/`stab_mask` for every non-Clifford gate and measurement. |
| `stim::Tableau<W>` — composition | AG pivot matrices: after a measurement collapses the tableau, the Front-End computes the GF(2) change-of-basis via `fwd_after.then(inv_before)`. These matrices are stored in a side-table (`ag_matrices`) owned by the `HirModule` and indexed by `HeisenbergOp::MEASURE` payloads. The Back-End later moves them to the `ConstantPool` constant pool. |
| `stim::Tableau<W>::inverse()` | Forward tableau extraction (needed occasionally for generator enumeration and AG pivot diffs). |
| `stim::PauliString<W>` | Pauli algebra for dominant-term factoring. `inplace_right_mul_returning_log_i_scalar` computes exact log-i phase without floating-point arithmetic. |
| `stim::CircuitInstruction` | Constructed per-gate to pass target qubit lists to `TableauSimulator::do_*` methods. |

**Multi-Pauli Measurement Support:** Multi-Pauli measurements (e.g., `MPP X1*X2*Z3`) are
natively supported with zero architectural changes. Because the Front-End rewinds any
observable into the $t=0$ frame, a multi-Pauli measurement simply becomes a single
`HeisenbergOp::MEASURE` with multiple bits set in its Pauli masks. The Middle-End and
Back-End process a weight-10 parity check exactly as fast as a weight-1 Pauli measurement.

#### VM Runtime (light use, runs per shot)

| Stim type | How the VM uses it |
|-----------|-----|
| `stim::simd_bits<W>` | Sign tracker storage when qubit count exceeds 64. For ≤64 qubits, the VM uses `uint64_t` and does not touch Stim at all. |
| `stim::simd_bit_table<W>` | AG pivot matrices in ConstantPool (>64 qubit path). |
| Measurement formatting | Stim-compatible output formats (.b8, .01, .smp) for interoperability. |

**The VM never uses `stim::Tableau` or `stim::TableauSimulator`.** This is the
fundamental invariant that makes per-shot execution lightweight.

### What we write ourselves

| Component | Why not Stim |
|-----------|-------------|
| Circuit parser & AST (§5) | Stim's parser cannot be extended without forking |
| Front-End trace logic | Heisenberg rewinding, HIR emission, dominant-term factoring |
| Middle-End optimizer | Commutation, fusion, cancellation — none of this exists in Stim |
| Back-End code generator | GF(2) basis tracking, bytecode emission, `commutation_mask` computation, NoiseSchedule compilation |
| VM execution engine | Bytecode interpreter, coefficient array math, measurement sampling |
| REPEAT block streaming | Incremental bytecode emission with SSA qubit mapping |
| `rec[-k]` resolution | Measurement index tracking during compilation |
| DETECTOR coordinate propagation | Accumulating coordinate offsets through REPEAT iterations |
| Noise schedule & gap sampling | Geometric gap sampling is a VM-side optimization orthogonal to Stim |

### Why not patch Stim

1. **No extension mechanism.** Stim's gate dictionaries, decomposition tables,
   and keyword maps are internal. Adding `T` touches >5 files.
2. **Fork maintenance burden.** Stim evolves rapidly. Rebasing a fork that
   modifies internal data structures is fragile.
3. **Architectural mismatch.** Stim is a dynamic per-shot simulator. Our
   multi-level pipeline needs fundamentally different iteration logic.
4. **Scoped parser.** Our parser implements exactly the syntax we need without
   replicating Stim's comprehensive validation machinery.

---

## 7. SIMD Strategy

SIMD width is controlled by a single compile-time parameter `W` inherited from
Stim's `MAX_BITWORD_WIDTH` convention.

### Where W flows

| Component | W-dependent types | Notes |
|-----------|-------------------|-------|
| Front-End | `Tableau<W>`, `TableauSimulator<W>`, `PauliString<W>` | Full n×n bit-matrix operations |
| VM (>64q path) | `simd_bits<W>` for sign tracker, `simd_bit_table<W>` for AG matrices | Only sign/generator mask dimensions scale with n |
| VM (≤64q path) | `uint64_t` | Fast path: no Stim types, no SIMD overhead |

### Where W does NOT matter

- **Coefficient array `v[]`** — indexed by stabilizer rank, not qubit count.
- **Bytecode instructions** — 32-byte fixed-size structs with `uint32_t` masks
  (rank-indexed, not qubit-indexed).
- **HIR** — uses `uint64_t` Pauli masks (≤64 qubits) or will use indices into
  a mask arena (>64 qubits).
- **Python bindings** — W is invisible to the user.

### Build configuration

- **Development:** `-march=native` selects the best available ISA.
- **Release wheels:** Explicit `-DSIMD_WIDTH=256` (AVX2) or per-platform
  selection, matching Stim's approach.

---

## 8. Build System

### CMake structure

```cmake
# src/ucc/CMakeLists.txt
add_library(ucc_core
    circuit/circuit.cc
    circuit/parser.cc
    frontend/frontend.cc
    optimizer/optimizer.cc
    backend/backend.cc
    backend/gf2_basis.cc
    backend/ssa_map.cc
    vm/program.cc
    vm/vm.cc
    vm/shot_state.cc
)
target_link_libraries(ucc_core PUBLIC libstim)
target_include_directories(ucc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
```

Logically, the code separates into five layers (circuit → frontend → optimizer
→ backend → vm) as described in §2. If link-time granularity matters later
(e.g., a VM-only embedding or optimizer-only export), we can split into separate
targets. For now, one target keeps the build simple.

### External dependencies

```cmake
# cmake/FetchStim.cmake
include(FetchContent)
FetchContent_Declare(
    stim
    GIT_REPOSITORY https://github.com/quantumlib/Stim.git
    GIT_TAG        v1.14.0   # pinned release
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stim)
```

Catch2 follows the same pattern in `cmake/FetchCatch2.cmake`.

### Python packaging

`pyproject.toml` uses **scikit-build-core** as the PEP 517 build backend:

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"
```

### Local development workflow

```bash
# C++ only
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build

# Python (editable install)
uv pip install -e .
pytest tests/python/
```

---

## 9. Python Bindings (nanobind)

The compiled nanobind module `_ucc_core` is wrapped by a pure Python package
`ucc` that provides the public API.

### 9.1 Seed Management & Determinism

The Python `sample()` API accepts an optional `base_seed` for deterministic
reproducibility:

```python
results, seeds = ucc.sample(program, shots=1000, base_seed=42)
```

**Hybrid Base-Seed Architecture:**

1. The C++ backend uses the `base_seed` to initialize a master PRNG.
2. The master PRNG generates a deterministic array of $N$ exact 64-bit sub-seeds
   (one per shot).
3. Each shot's `ShotState` is initialized with its assigned sub-seed.
4. The `sample()` function returns both the measurement results and the array
   of $N$ exact seeds used.

**Replay Failed Shots:** A future API overload will accept an explicit list of
seeds to enable perfect mathematical replay of specific failed shots:

```python
# Replay specific shots with their exact seeds
results = ucc.sample_with_seeds(program, seeds=[12345, 67890])
```

This enables debugging non-deterministic failures by re-running the exact RNG
sequence that produced them.

### 9.2 Python API Example

The compiled module provides:

```python
import ucc

# Compile a circuit (runs the full AOT pipeline once)
program = ucc.compile("H 0\nT 0\nM 0")

# Sample many shots (runs the VM N times)
results = ucc.sample(program, shots=10_000)
# → numpy uint8 array, shape (shots, num_measurements)

# Sample with deterministic seeding (for reproducibility)
results, seeds = ucc.sample(program, shots=10_000, base_seed=42)
# → results: numpy uint8 array, shape (shots, num_measurements)
# → seeds: numpy uint64 array, shape (shots,) — exact per-shot seeds

# Access the optimized HIR (for export to tqec / PBC routing)
hir = ucc.compile_to_hir("H 0\nT 0\nM 0")
print(hir.num_ops)           # number of HIR operations
print(hir.num_gate_ops)      # non-Clifford gate count (optimization metric)

# Export HIR to text format (one-way, for debugging)
hir.export_text("circuit.hir")

# Inspect the compiled program
print(program.num_qubits)
print(program.peak_rank)
print(program.num_instructions)
```

### Binding internals

```cpp
// src/python/bindings.cc
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/optimizer.h"
#include "ucc/backend/backend.h"
#include "ucc/vm/vm.h"

namespace nb = nanobind;

NB_MODULE(_ucc_core, m) {
    nb::class_<ucc::HirModule>(m, "HirModule")
        .def_prop_ro("num_ops",      &ucc::HirModule::num_ops)
        .def_prop_ro("num_gate_ops", &ucc::HirModule::num_gate_ops)
        .def_prop_ro("global_weight", &ucc::HirModule::global_weight);

    nb::class_<ucc::Program>(m, "Program")
        .def_prop_ro("peak_rank",        &ucc::Program::peak_rank)
        .def_prop_ro("num_instructions", &ucc::Program::num_instructions)
        .def_prop_ro("num_qubits",       &ucc::Program::num_qubits);

    m.def("compile", &ucc::compile_string, nb::arg("circuit_text"));
    m.def("compile_to_hir", &ucc::compile_to_hir, nb::arg("circuit_text"));
    m.def("sample", &ucc::sample,
          nb::arg("program"), nb::arg("shots") = 1,
          nb::arg("base_seed") = nb::none());
}
```

The `W` template parameter is resolved at build time inside the C++ layer.
Python callers never see it.

---

## 10. Testing Strategy

### Why Not Text Round-Trip Testing?

A naive testing approach would be: `.stim` text → compile → `.stim` text →
assert string equality. **This is fundamentally flawed.** The Front-End
*destructively absorbs* Clifford gates into the Pauli frames of non-Cliffords.
Decompiling the HIR back to circuit text would require brittle Clifford
synthesis algorithms, and even then the output would be semantically equivalent
but textually different from the input.

Instead, UCC uses a **5-Stage Semantic Invariant Testing Strategy** that
isolates each pipeline stage with mathematically precise oracles.

### Test 1: Lexical AST Round-Trip (Isolates Parser)

```
Generate random .stim text
    → Parse into ucc::Circuit AST
    → Format back to text
    → Assert exact string equality
```

This tests the parser and formatter in isolation. The AST is a pure syntactic
representation with no semantic transformation, so exact round-trip is expected.

**File:** `test_parser.cc`

### Test 2: The Rewinding Oracle (Isolates Front-End)

```
Generate a pure-Clifford circuit with a single `M 0` at the end
    → Run through the Front-End
    → Extract the stab_mask of the resulting HeisenbergOp::MEASURE
    → Assert exact equality with stim::TableauSimulator.inv_state.zs[0]
```

This validates that the Front-End's Heisenberg rewinding exactly matches Stim's
inverse tableau. The `stab_mask` encodes the Z-component of the rewound Pauli
string, which must be bit-identical to Stim's internal representation.

**File:** `test_frontend.cc`

### Test 3: The Annihilation Fuzz (Isolates Middle-End)

```
Construct: T 0 → [random Cliffords commuting with rewound axis] → T_DAG 0
    → Run through Front-End + Optimizer
    → Assert ops.size() == 0 and global_weight == 1.0 (perfect cancellation)

Then: Insert an anti-commuting NOISE node between T and T_DAG
    → Assert Optimizer refuses to fuse (barrier awareness)
    → Assert ops.size() == 2 (both gates preserved)
```

This fuzz-tests the optimizer's commutation, fusion, and cancellation logic.
The T/T† pair on the same axis must annihilate when the path is clear, but
barriers must block the sweep.

**File:** `test_optimizer.cc`

### Test 4: Peak Rank Bounds (Isolates Compiler Back-End)

```
Create circuit: 4 independent T-gates on qubits 0-3
    → Measure in commuting basis (triggers OP_MEASURE_FILTER)
    → Compile through Back-End
    → Assert peak_rank == 4

Then: Add anti-commuting measurement (triggers OP_MEASURE_MERGE)
    → Assert active rank properly shrinks to 3
```

This validates the Back-End's GF(2) basis tracking and memory allocation
computation. Incorrect rank tracking would cause OOM or array overflows.

**File:** `test_backend.cc`

### Test 5: Statevector Oracle (Full Pipeline Integration)

```
Generate random Clifford+T circuit (2-8 qubits, 10-50 gates)
    → Compile to bytecode
    → Execute 1 shot on C++ execution engine
    → Expand VM's final compact v[] array + GF(2) basis into dense 2^N statevector
    → Assert np.allclose(expanded_vm_state, pure_python_state * global_weight)
```

The `prototype/` directory contains a pure-Python generalized stabilizer
simulator (~800 lines) validated against numpy statevector simulation (500+
random Clifford+T circuits, zero failures). This serves as the ground-truth
oracle for full-pipeline integration testing.

**File:** `test_end_to_end.cc` (C++ side) + `prototype/` (Python oracle)

### Test File Summary

| File | Stage | Invariant |
|------|-------|----------|
| `test_parser.cc` | Parser | Lexical AST round-trip |
| `test_frontend.cc` | Front-End | Rewinding matches `stim::TableauSimulator` |
| `test_optimizer.cc` | Middle-End | Annihilation + barrier awareness |
| `test_backend.cc` | Compiler Back-End | Peak rank tracking |
| `test_vm.cc` | Execution Engine | Individual opcode semantics |
| `test_end_to_end.cc` | Full Pipeline | Statevector oracle match |

All C++ tests use Catch2 with `catch_discover_tests()` for automatic CTest
registration.

### Python Binding Tests (pytest)

`tests/python/test_api.py` exercises the public API: `compile`, `sample`,
`compile_to_hir`, result shapes, error handling for invalid circuits.

---

## 11. Packaging & CI/CD

*Deferred to Phase 4.* Brief plan:

- **cibuildwheel** for wheel builds on GitHub Actions.
- **Platforms:** Linux x86_64 (AVX2), macOS arm64 (NEON) initially.
  Windows and manylinux aarch64 added later.
- **SIMD:** Platform-specific `-DSIMD_WIDTH` flags in cibuildwheel config.
- **Publishing:** PyPI via trusted publishing (OIDC).
- **CI matrix:** Build + test on every PR; wheel build on tagged releases.

---

## 12. Implementation Phasing

Each phase builds on the previous one and is designed to maximize early
engineering velocity. The MVP focuses purely on proving the core performance
thesis of the AOT math before adding complexity.

### Phase 1 — MVP / Proof of Concept

**Goal:** Prove the core performance thesis of multi-level AOT compilation.
Strictly limited scope to minimize time-to-first-benchmark.

**Scope:**
- `ucc::Circuit` parser for the stim-superset format
- Front-End driving `stim::TableauSimulator<64>` for Clifford gates
- HIR emission for T/T† and measurements (`HeisenbergOp` with `T_GATE`, `T_DAG_GATE`, `MEASURE`)
- **Middle-End Optimizer:** $\mathcal{O}(1)$ commutation, fusion, and cancellation passes
- Back-End bytecode emission: OP_BRANCH, OP_COLLIDE, OP_SCALAR_PHASE
- VM interpreter with ShotState (`uint64_t` sign tracker, dense `v[]` array)
- Measurement opcodes: OP_MEASURE_MERGE, OP_MEASURE_FILTER, OP_AG_PIVOT
- Basic Python bindings with `compile()` and `sample()` APIs
- One-way HIR text export (`.hir` format) for debugging
- Single-threaded or basic OpenMP execution
- Catch2 test suite; fuzz-tested against Python prototype

**Strict Limits:**
- **≤64 qubits only** (inline `uint64_t` masks, no `simd_bits` in VM)
- **Single-allocation array sizing** (peak_rank computed at compile time)
- **No noise, no REPEAT blocks** (dynamic REPEAT simply unrolled by Front-End if memory allows)
- **No generic LCU** (T/T† fast-path only)

### Phase 2 — Control Flow & Advanced Logic

**Goal:** Production-ready control flow and noise modeling.

- **OP_POSTSELECT:** Mid-circuit early abort optimization for magic state cultivation
- Pauli noise channels (DEPOLARIZE1/2, X_ERROR, Z_ERROR) in HIR
- NoiseSchedule compilation in Back-End; geometric gap sampling in VM
- REPEAT block streaming in the Front-End (SSA qubit mapping across iterations)
- Dynamic REPEAT block streaming/limit-cycles to prevent unrolling memory bloat
- `rec[-k]` measurement index resolution
- DETECTOR / OBSERVABLE_INCLUDE coordinate propagation
- Full Stim gate coverage (ISWAP, SWAP, SQRT_X, etc.)
- Reset gates (R, RX, RY)
- Generic LCU support: OP_BRANCH_LCU, OP_COLLIDE_LCU, OP_SCALAR_PHASE_LCU

### Phase 3 — Scale & Performance

**Goal:** Hardware-optimized execution at scale.

- **>64 qubit support:** Template monomorphization with `simd_bits<W>` in constant pool
- **SIMD SoA Batching:** VectorBatch layout for multi-shot SIMD execution
- Cache-aware batch sizing
- C++ microbenchmarks; comparison benchmarks in `tools/bench/`

### Phase 4 — Production

**Goal:** Production-quality release with full tooling.

- **Two-way serialization:** Parsers to load `.hir` and `.bin` back into C++ memory
- Memory compaction for unbounded circuits (see `data_structs.md` Appendix B)
- `compile_to_hir` API for tqec / PBC routing export
- scikit-build-core packaging
- cibuildwheel CI/CD pipeline
- PyPI publishing
- Documentation (README, API reference, examples)

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **Front-End** | The first AOT pass: parses `ucc::Circuit`, absorbs Cliffords via `stim::TableauSimulator`, emits the HIR. Runs once per circuit. |
| **Middle-End (Optimizer)** | The second AOT pass: optimizes the HIR via O(1) commutation, fusion, and cancellation. Runs once per circuit. |
| **Compiler Back-End** | The third AOT pass (also called the "Code Generator"): lowers optimized HIR to execution engine bytecode and ConstantPool. Tracks GF(2) basis, computes x_mask/commutation_mask/AG matrices. Runs once per circuit. *Note: "Back-End" here refers to the compiler stage, not a QPU hardware backend.* |
| **VM** | The bytecode interpreter that executes a `Program` for a single Monte Carlo shot. Runs millions of times. |
| **HIR (Heisenberg IR)** | Flat list of `HeisenbergOp` structs — abstract Pauli-string operations with explicit weights and no VM memory routing. Output of Front-End, input to Optimizer. Uses explicit `T_GATE` and `T_DAG_GATE` OpTypes for T-gates and a generic `GATE` OpType for other non-Clifford gates. The Back-End uses these enum values to deterministically route to fast-path or LCU opcodes. |
| **Program** | Compiled output: a vector of 32-byte `Instruction` structs + a `ConstantPool` constant pool. Immutable, shared across shots. |
| **ShotState** | Per-shot mutable state: sign tracker (`destab_signs`, `stab_signs`), coefficient array `v[]`, measurement record, PRNG. |
| **GF(2) basis V** | The set of linearly independent shift-vectors tracked by the Back-End. Stabilizer rank = dim(V) = log₂(len(v)). |
| **commutation_mask** | A bitmask encoding commutation structure of a Pauli operator against V, enabling O(1) phase evaluation via `popcount(idx & commutation_mask)`. |
| **NoiseSchedule** | A sorted table of `NoiseSite` entries in ConstantPool, mapping bytecode PCs to pre-computed noise masks. Each site may contain multiple `NoiseChannel` entries for multi-Pauli noise (e.g., DEPOLARIZE1/2). The VM uses geometric gap sampling over this table. |
| **W** | SIMD bitword width (64, 128, 256, or 512 bits). Compile-time template parameter inherited from Stim. |
| **Dominant Term Factoring** | LCU decomposition technique that absorbs the largest-magnitude Pauli term into the tableau, making the identity path free. |
