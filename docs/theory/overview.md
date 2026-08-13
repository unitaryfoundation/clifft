# Theoretical Overview

Clifft is a compiler and execution engine for universal quantum circuits. For
circuits whose non-Clifford effects remain localized, it reduces the
exponential part of exact simulation from the total qubit count to a dynamic
active dimension.

More details are in our [arXiv preprint](https://arxiv.org/abs/2604.27058).

## Symbolic Clifford Coordinates

The compiler absorbs deterministic Clifford evolution into an offline frame.
Measurements, noise outcomes, and conditional operations become affine
Boolean expressions, while the genuinely interfering degrees of freedom are
represented by a dense complex state over $k$ active symbolic coordinates.

The dense array has $2^k$ entries. Physical circuits may contain many more
than $k$ qubits because dormant Clifford structure remains in the symbolic
frame. Non-Clifford rotations and active measurements can add coordinates;
measurements can also remove them. The largest width reached by a compiled
program is exposed as `program.peak_rank`.

For magic-state preparation, distillation, and related QEC circuits, frequent
measurements can keep this active width small even when the physical circuit
contains hundreds of qubits. Clifft then allocates $2^{k_max}$ amplitudes
instead of $2^n$.

## Compilation and Execution

```text
Circuit text
    |
    v
Parse and Clifford trace
    |  Absorb Clifford gates and emit Heisenberg IR operations.
    v
Heisenberg IR
    |  Fuse/cancel operations and reduce active lifetimes.
    v
Optimized HIR
    |  Choose active coordinates and derive affine dependencies.
    v
Sampling plan
    |  Prepare fixed action storage and select scalar/SIMD kernels.
    v
Executable program
    |  Reuse the plan for every shot.
    v
Measurement, detector, observable, and expectation-value results
```

### Clifford Trace

The front end uses Stim's tableau implementation to absorb physical Clifford
operations. For an explicit operation $P$, it computes the corresponding
Pauli in the current Clifford frame:

$$P_{frame} = U_C^\dagger P U_C$$

The resulting Heisenberg IR keeps rotations, measurements, noise, classical
dependencies, detectors, and observables explicit without replaying every
Clifford gate during each shot.

### HIR Optimization

HIR passes use Pauli algebra and dataflow information to fuse or cancel
operations and, when safe, shorten active-coordinate lifetimes. This work is
performed before the runtime representation is fixed.

### Coordinate Planning

The planner chooses coordinates for consecutive operations, records
active-width transitions, and converts measurement and noise dependencies into
affine expressions. It also computes the descriptors needed by rotations,
measurements, transition instruments, and exact-output queries.

### Executable-Plan Preparation

Preparation transposes symbolic dependencies for incremental evaluation,
combines supported rotation runs, and selects scalar or architecture-specific
kernels. All coefficient, record, expression, and scratch storage is sized
before the hot dispatch loop.

### Sampling

Each shot assigns presampled fault symbols, evaluates dynamic expressions, and
applies the prepared active-state actions. Runtime execution performs no
tableau evolution, Pauli localization, commutation analysis, or dependency
discovery.

!!! note "Leakage and loss trajectories"
    A sampled transition can change which later operations remain physical.
    `clifft.noncomp.sample` therefore compiles and resumes symbolic-coordinate
    continuations at explicit transition boundaries. See
    [Noncomputational States](noncomputational.md).

## Exact Queries

For pure-unitary programs, `clifft.basis_probabilities()` computes selected
full-register computational-basis probabilities without expanding all $2^n$
amplitudes. `clifft.get_statevector()` performs that full expansion when a
dense state is useful for small-circuit debugging. Programs with measurements
can use `clifft.record_probabilities()` to evaluate selected measurement
records exactly.

See [Basis-State Probabilities](basis_probabilities.md) and
[Strong Simulation with Exact Probabilities](../guide/strong-simulation.md).
