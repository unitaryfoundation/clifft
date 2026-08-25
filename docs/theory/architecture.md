# Software Architecture

Clifft separates circuit analysis from repeated execution. Compilation does
the tableau, coordinate, and dependency work once. Sampling then executes a
fixed symbolic-coordinate program using preallocated state.

The main pipeline is:

```text
Circuit -> HIR -> optimized HIR -> SamplingPlan -> ExecutablePlan -> Executor -> results
```

These boundaries have different responsibilities:

| Stage | Decides or owns | Deliberately excludes |
|---|---|---|
| HIR | Circuit operations expressed in the Heisenberg frame | Active-coordinate and kernel choices |
| `SamplingPlan` | Symbols, affine dependencies, active-coordinate actions, width transitions, and outputs | CPU instruction-set selection, descriptor layout, and target-specific fusion |
| `ExecutablePlan` | Immutable descriptors, dependency tables, fused actions, and kernels prepared for one execution target | Per-shot state and runtime topology analysis |
| `Executor` | Mutable coefficients, symbols, expression registers, records, outputs, scratch space, and RNG state | Tableau evolution, coordinate planning, and dependency discovery |

The [Theoretical Overview](overview.md) explains the state factorization and
symbolic-coordinate method. This page focuses on how the implementation
prepares and executes that model.

The original Clifft preprint documents the earlier localized-Pauli SVM. See
[Symbolic Sampling in Clifft](../updates/symbolic-sampling.md) for how the
implementation moved from that design to the pipeline described here.

## Repository Layout

| Directory | Role |
|---|---|
| `src/clifft/circuit/` | Circuit AST, parser, and target encoding |
| `src/clifft/frontend/` | Clifford tracing and Heisenberg IR construction |
| `src/clifft/optimizer/` | HIR optimization passes |
| `src/clifft/sampling/` | Semantic planning, executable preparation, shot execution, and kernels |
| `src/clifft/noncomp/` | Leakage/loss trajectory planning and continuation handling |
| `src/python/` | Python API via nanobind |
| `src/wasm/` | Browser bindings for compilation, inspection, and sampling |

## Compilation Stages

### Clifford Trace and HIR Optimization

The front end uses Clifft's native tableau implementation to absorb Clifford
gates and express the remaining rotations, measurements, noise, feedback, and
outputs in the Heisenberg frame. HIR passes then simplify this representation
using Pauli algebra and dataflow. Neither stage fixes a sampling data layout.

### Semantic Planning

The planner chooses active stabilizer coordinates, resolves basis changes,
maps Pauli operations into those coordinates, and derives the affine Boolean
expressions needed for branch-dependent signs. The resulting `SamplingPlan`
is a target-independent description of the work common to every shot.

Keeping this boundary semantic matters. A CPU executor and a future target can
share the same coordinate choices and output contract without sharing action
layouts, kernel selectors, or fusion policy.

### Executable Preparation

The current executable-plan builder lowers a validated `SamplingPlan` for the
host CPU. It transposes affine dependencies for incremental evaluation,
combines supported adjacent rotations, and converts semantic actions into
compact fixed descriptors. On supported x86 builds, it also selects the
scalar, AVX2, or AVX-512 executor backend once for the plan.

The action's kernel tag describes a prepared operation shape, such as a
diagonal or lane-paired rotation. It does not independently select an ISA.
This keeps target selection consistent across the whole executor.

`ExecutablePlan` is immutable after construction. Its builder is temporary
and is discarded once the fixed storage is ready; it is not another persistent
IR or a runtime pass manager.

!!! important "Planning boundary"
    Runtime kernels do not evolve tableaus, localize Paulis, analyze
    commutation, choose coordinates, or discover symbolic dependencies. Those
    decisions belong to compilation and executable preparation.

## Execution and Memory

An `Executor` combines an `ExecutablePlan` with the mutable state of one shot.
The sampling drivers reset and reuse it across shots, while collecting records,
detectors, observables, expectation values, or survivor statistics.

The executor allocates its coefficient array for the plan's maximum active
width. Record storage, symbol values, expression accumulators, and scratch
buffers are also prepared before ordinary dispatch. Ordinary actions and
kernels are allocation-free and exception-free; construction validates
external and compiler-produced inputs, while Debug assertions protect internal
hot-path invariants.

The coefficient array contains $2^k$ complex amplitudes for the current active
coordinates, not one amplitude per physical-qubit basis state. Clifford
structure outside that active space remains encoded by the prepared coordinate
description.

Noncomputational transitions are the deliberate exception to fixed ordinary
execution. A leakage or loss instrument may stop at an explicit boundary,
prepare a trajectory-specific suffix, grow reusable storage if necessary, and
resume without losing the live coefficients, records, symbols, or RNG
position. See [Noncomputational States](noncomputational.md).

## Execution Targets

Architecture-specific kernels live in separate translation units compiled
with their required flags. Portable code contains no SIMD types and retains
the scalar reference path. Supported x86 builds can therefore select explicit
AVX2 or AVX-512 kernels without allowing those instructions to leak into
fallback execution on other CPUs. Portable ARM, Windows, and WebAssembly
builds use the scalar backend.

WebAssembly follows the same HIR, `SamplingPlan`, and `ExecutablePlan`
boundaries. The playground can inspect the semantic and prepared plans, while
browser sampling consumes the prepared scalar executable.

## Python Bindings

`clifft.compile()` returns a `Program` backed by an `ExecutablePlan`.
`clifft.sample()`, survivor sampling, fixed-fault sampling, and exact queries
all consume that same type. One compiled program can be reused for many calls
and shot counts.

`clifft.get_statevector(program)` expands an eligible pure-state program into
the full physical-qubit state vector for debugging and validation. The result
represents the final state ray and is defined only up to global phase. Its
final coordinate metadata is not read during ordinary sampling.

See [Compiling Circuits](../guide/compilation.md) and
[Simulation](../guide/simulation.md) for API examples.
