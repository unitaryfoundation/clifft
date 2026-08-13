# Software Architecture

Clifft separates circuit analysis from repeated execution. Compilation does
the tableau and dependency work once; sampling executes a fixed
symbolic-coordinate plan using preallocated storage.

## Repository Layout

| Directory | Role |
|---|---|
| `src/clifft/circuit/` | Circuit AST, parser, and target encoding |
| `src/clifft/frontend/` | Clifford tracing and Heisenberg IR construction |
| `src/clifft/optimizer/` | HIR optimization passes |
| `src/clifft/sampling/` | Active-coordinate planning, executable-plan preparation, executor, and kernels |
| `src/clifft/noncomp/` | Leakage/loss trajectory planning and continuation handling |
| `src/python/` | Python API via nanobind |

The older localized SVM implementation remains internal while migration
tests compare it with the production path. It is not part of the public
Python compilation API.

## Compilation

The front end uses Stim's tableau implementation to absorb Clifford gates and
express the remaining operations in the Heisenberg frame. HIR passes simplify
that representation before the sampling planner chooses active symbolic
coordinates.

Executable-plan preparation then converts affine expressions, measurements,
rotations, active-width transitions, detectors, observables, and
post-selection checks into fixed descriptors. It also selects supported
scalar or SIMD kernels for the current host.

!!! important "Planning boundary"
    Runtime kernels do not evolve tableaus, localize Paulis, analyze
    commutation, or discover symbolic dependencies. Those decisions belong to
    compilation and plan preparation.

## Execution and Memory

The executor allocates its coefficient array for the plan's maximum active
width. Record storage, symbolic registers, expression accumulators, and
scratch buffers are likewise prepared before the hot dispatch loop. Ordinary
actions and kernels are allocation-free and exception-free.

The coefficient array contains $2^k$ complex amplitudes for the currently
active symbolic coordinates, not one amplitude per physical-qubit basis
state. Clifford structure outside that active space remains represented by
the compiler's symbolic frame.

## Python Bindings

`clifft.compile()` returns an executable symbolic-coordinate `Program`.
`clifft.sample()`, survivor sampling, fixed-fault sampling, and exact queries
all consume that same type. `clifft.get_statevector(program)` expands a small
pure-unitary program into the full physical-qubit state vector for debugging
and validation.

See [Compiling Circuits](../guide/compilation.md) and
[Simulation](../guide/simulation.md) for API examples.
