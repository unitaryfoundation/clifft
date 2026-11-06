# Software Architecture

Clifft's compiled modes separate circuit analysis from repeated execution. Compilation does
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
| `Program` | Original CPU executable and optional reconstructed sampling plans for retained records and survivor counts | Per-shot state and runtime compilation |
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
| `src/clifft/api/` | Program ownership, sampling-plan selection, and exact queries |
| `src/clifft/sampling/` | Semantic planning, executable preparation, shot execution, and kernels |
| `src/clifft/noncomp/` | Leakage/loss fixed, trajectory, and online execution |
| `src/python/` | Python API via nanobind |
| `src/wasm/` | Browser bindings for compilation, inspection, and sampling |

## Compilation Stages

### Clifford Trace and HIR Optimization

The front end uses Clifft's native tableau implementation to absorb Clifford
gates and express the remaining rotations, measurements, noise, feedback, and
outputs in the Heisenberg frame. HIR passes then simplify this representation
using Pauli algebra and dataflow. Neither stage fixes a sampling data layout.

The phase-polynomial and rotation-simplification passes recover stabilizer
constraints through measurement and feedback, then use only fixed-sign
constraints to justify static rewrites shared by all shots. This analysis stays
local to compilation; it adds no conditional execution or runtime analysis.
See [Optimization for Simulation](stabilizer-optimization.md) for the principles
and [Optimization Passes](../reference/passes.md) for configuration and limits.

### Semantic Planning

The planner chooses active stabilizer coordinates, resolves basis changes,
maps Pauli operations into those coordinates, and derives the affine Boolean
expressions needed for branch-dependent signs. The resulting `SamplingPlan`
is a target-independent description of the work common to every shot.

Keeping this boundary semantic matters. The CPU executor and the experimental
HIP and CUDA backends share the same coordinate choices and output definitions
without sharing action layouts, kernel selectors, or fusion policy.

### Executable Preparation

The current executable-plan builder lowers a validated `SamplingPlan` for the
host CPU. It transposes affine dependencies for incremental evaluation,
combines supported adjacent rotations, and converts semantic actions into
compact fixed descriptors. On supported x86 builds, it also selects the
scalar, AVX2, or AVX-512 executor backend once for the plan.

The action's kernel tag describes a prepared operation shape, such as a
diagonal or lane-paired rotation. It does not independently select an ISA.
This keeps target selection consistent across the whole executor.

`ExecutablePlan` is immutable after construction, and its builder is discarded
once the prepared descriptors are finalized.

The experimental HIP and CUDA backends start from the same `SamplingPlan`
boundary but prepare a backend-private GPU executable and device workspace.
Their packed actions and launch controls are not part of the stable CPU API;
see [HIP Backend](../development/hip-backend.md) and
[CUDA Backend](../development/cuda-backend.md).

!!! important "Planning boundary"
    Compiled executors and their kernels do not evolve tableaus, localize Paulis, analyze
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

Shared trajectory execution retains packed classical values and an independent
random stream for every shot, while equivalent shots use one coefficient state.
Prepared rotation signs and measurement outcomes split groups as needed.
Before an action can exceed the retained state capacity, dispatch suspends at
an explicit storage boundary. The pool grows without changing sampled noise,
branches, or live coefficients, then the same prepared action resumes.

Entrance postselection eliminates unavailable measurement and readout variables
from selected detector equations during preparation. For example, requiring
both `m XOR n0 = 0` and `m XOR n1 = 0` implies the noise-only condition
`n0 XOR n1 = 0`. These necessary checks reject individual shots before quantum
work; the original detector checks still determine final acceptance. Expected
reference parities are included in the equations.
Preparation also schedules selected detectors after the last contributing record
write, including readout noise. This preserves each detector's record snapshot
while allowing rejected trajectories to skip subsequent quantum actions. Workers
share this action order and apply the plan's selected scalar or SIMD kernels.

Noncomputational transitions use explicit boundaries to change future
execution. A leakage or loss instrument may stop at an explicit boundary,
prepare a trajectory-specific suffix, grow reusable storage if necessary, and
resume without losing the live coefficients, records, symbols, or RNG
position. See [Noncomputational States](noncomputational.md).

## Online Noncomputational Execution

Most Clifft execution modes resolve the Clifford frame ahead of time. In some
noncomputational simulations, leakage or loss changes downstream operations:
for example, a two-qubit gate is skipped when one target is lost. Trajectory
execution recompiles the remaining circuit to account for these changes. When
many sites can leak or be lost, recompilation can dominate sampling time.
Users can instead select `execution="online"`, which updates the Clifford frame
during sampling and handles these changes as they occur. See
[Online execution](../guide/leakage-and-loss.md#online-execution) for usage and
performance tradeoffs.

## Execution Targets

Architecture-specific kernels live in separate translation units compiled
with their required flags. Portable code contains no SIMD types and retains
the scalar reference path. Supported x86 builds can therefore select explicit
AVX2 or AVX-512 kernels without allowing those instructions to leak into
fallback execution on other CPUs. Apple arm64 builds use NEON kernels. Linux
arm64, Windows, and WebAssembly builds use the scalar backend.

The online noncomputational backend currently uses the portable scalar dense
kernels on all platforms.

WebAssembly follows the same HIR, `SamplingPlan`, and `ExecutablePlan`
boundaries. The playground can inspect the semantic and prepared plans, while
browser sampling consumes the prepared scalar executable.

## Python Bindings

`clifft.compile()` returns a `Program` owning the original `ExecutablePlan`.
`clifft.sample()`, survivor sampling, fixed-fault sampling, and exact queries
all consume that same type. One compiled program can be reused for many calls
and shot counts.

With `reconstruction=True`, compilation also prepares bounded reconstructed
plans for full-record sampling and for detector/observable survivor counts.
`sample()` selects the full-record plan; `sample_survivors()` selects according
to `keep_records`. Each falls back to the original plan when its candidate is
ineligible or does not reduce peak active width.

Because the reconstructed plans preserve only the classical output distribution
and postselection acceptance probability, state and exact-probability queries,
replay, and fixed-fault sampling always use the original plan, exposed through
`program.original_plan`. `Program.peak_active_width`, `num_actions`, and
inspection describe the full-output sampling plan.
`program.sampling_plan(keep_records=...)` returns a read-only `PlanView` with
metrics, inspection, and reconstruction diagnostics for the chosen output mode.
Views retain the owning program and refer to its immutable executables. Circuit
counts and noise-site probabilities continue to describe the original program.
Experimental GPU and WebAssembly paths prepare a single executable.

`clifft.get_statevector(program)` expands an eligible pure-state program into
the full physical-qubit statevector for debugging and validation. The result is
normalized and defined only up to global phase. Its
final coordinate metadata is not read during ordinary sampling.

See [Compiling Circuits](../guide/compilation.md) and
[Simulation](../guide/simulation.md) for API examples.
