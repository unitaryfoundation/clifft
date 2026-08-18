# Theoretical Overview

Clifft is a compiler and execution engine for exact simulation of universal
quantum circuits. For circuits where non-Clifford effects remain localized,
it confines exponential work to an active-state dimension of $2^k$, set by the
dynamic active width $k$, instead of the total physical-qubit count.

## Method Provenance

The original Clifft design established the factored active-state model,
described by Bradley A. Chase and Farrokh Labib in the
[Clifft paper](https://arxiv.org/abs/2604.27058).

[SymFT](https://arxiv.org/abs/2607.28600), by Wang Fang, Huazhe Lou, and Riling
Li, describes itself as the second-generation successor to
[SOFT](https://arxiv.org/abs/2512.23037). Its planner builds on SOFT's
generalized-stabilizer simulation and Clifft's dense active-state
representation. SymFT adds symbolic Clifford-Pauli-frame factorization,
adaptive stabilizer-coordinate planning, and direct multi-coordinate kernels.

The current Clifft sampler adopts these SymFT developments. `SamplingPlan`,
host-specific executable preparation, instruments and continuations, and the
executor organization remain Clifft-specific implementation choices.

See [Symbolic Sampling in Clifft](../updates/symbolic-sampling.md) for the
release-oriented migration history and matched performance comparison.

## Symbolic Clifford Coordinates

Sampling a circuit with noise or mid-circuit measurements produces a
trajectory. After planned step $j$, let $s_{\le j}$ denote the Boolean outcomes
sampled so far. The resulting pure state has the factorization

$$
|\psi_j(s_{\le j})\rangle
=
\gamma_j(s_{\le j})\,
C_j\,
P_j(s_{\le j})\,
\Big(
|\phi_j(s_{\le j})\rangle_{A_j}
\otimes
|0\rangle_{D_j}
\Big).
$$

The factors have distinct roles:

- **$C_j$ (Clifford coordinate map):** Maps the current stabilizer coordinates
  into the physical qubit basis. The front end and planner determine its
  evolution before sampling; a shot performs no tableau operations.

- **$P_j(s_{\le j})$ (Pauli frame):** Represents branch-dependent Pauli
  corrections from noise, measurement outcomes, and classical feedback. It is
  a mathematical component of the trajectory state, but the sampling executor
  does not materialize it as a mutable $n$-qubit runtime frame.

- **$A_j$ and $D_j$ (active and dormant coordinates):** Partition the current
  stabilizer coordinates into an ordered active prefix of width $k_j$ and
  $n-k_j$ dormant coordinates. Dormant coordinates are stabilized in the
  computational zero state in this basis.

- **$|\phi_j\rangle_{A_j}$ (active coefficient state):** A dense complex array
  of size $2^{k_j}$ containing the non-Clifford interference. Ordinary sampling
  keeps this coefficient state normalized up to floating-point drift.

- **$\gamma_j$ (global scalar):** Carries a common complex weight and phase
  outside the active coefficient array.

Active and dormant coordinates are basis elements, not subsets of physical
qubits. After Clifford gates change the basis, one coordinate may represent a
different, possibly multi-qubit, physical Pauli without changing the size of
the active array. Throughout these docs, **active width** $k_j$ means the number
of stabilizer coordinates represented in the dense coefficient array. The
corresponding **active-state dimension** is $2^{k_j}$, the number of amplitudes
in that array. Neither is the number of physical qubits touched by the circuit.

For a normalized physical circuit, the unconditional noisy state is the
ensemble over trajectories,

$$
\rho_j
=
\mathbb{E}_{s_{\le j}}
\left[
|\psi_j(s_{\le j})\rangle
\langle\psi_j(s_{\le j})|
\right].
$$

Clifft samples members of this ensemble rather than materializing the full
density matrix.

### Why the Factorization Matters

The dense active array is the only state component with exponential size. If
$k_{\max}$ is the largest active width reached by a plan, coefficient and
scratch storage scale as $O(2^{k_{\max}})$ instead of $O(2^n)$. Frequent
measurements can return coordinates to the dormant set, so fault-tolerant
circuits may retain a small $k_{\max}$ even when they contain hundreds of
physical qubits.

## Planning Pauli-Frame Effects as Symbolic Signs

The main symbolic idea is to resolve the branch-dependent Pauli frame during
planning instead of updating it after every event in every shot. Following the
symbolic-frame strategy introduced by
[Fang, Lou, and Li](https://arxiv.org/abs/2607.28600), the planner expresses
each relevant dependence as an affine formula of Boolean symbols and attaches
the resulting sign to the affected operation.

Each stochastic event needed later in a shot receives a Boolean symbol. A
symbol may represent a presampled Pauli fault, a sampled measurement branch, a
readout flip, or an [instrument outcome](noncomputational.md). These effects
are represented as affine Boolean expressions,

$$
\ell(s) = c \oplus \bigoplus_{r \in R} s_r.
$$

For any Pauli observable or operation $Q$, conjugation by the trajectory's
Pauli frame can change only its sign:

$$
P_j(s)^\dagger Q P_j(s)
=
(-1)^{\ell_Q(s)} Q.
$$

The planner computes $\ell_Q$ once and maps the unsigned body of $Q$ into the
current stabilizer coordinates. An operation that needs coefficient work is
therefore represented by an active-coordinate Pauli and an affine sign, rather
than by a physical Pauli string plus a mutable runtime frame.

As a result, a shot does not carry and update an $n$-qubit Pauli frame after
each fault or measurement. It evaluates the prepared expressions as their
symbols become available and uses the realized signs when applying rotations,
measurements, and output actions.

## Adaptive Stabilizer Coordinates

The planner maintains a stabilizer-destabilizer basis in which the first
$k_j$ coordinates are active and the remainder are dormant. In this basis,
each dormant coordinate is in $|0\rangle$ and has $Z$ as its stabilizer. A
promotion moves a dormant coordinate into the dense state when it must carry
coherent amplitudes; a planned measurement and collapse can return an active
coordinate to the dormant set.

The planner resolves these changes once for all shots. Some representative
cases are:

| Situation | Example | Planned state action | Width |
|---|---|---|---|
| Rotation already supported on active coordinates | A mapped $Z$ rotation that touches only active coordinates | Rotate the active Pauli directly | $k \to k$ |
| Rotation requiring dormant coherent support | An $X$-axis pi/4 rotation on a dormant $\lvert 0\rangle$ coordinate | Promote the coordinate, then rotate | $k \to k+1$ |
| Measurement with active support | A mapped $Z$ measurement that touches the active state | Sample and collapse, then remove the chosen coordinate | $k \to k-1$ |
| Random measurement in dormant space | An $X$ measurement of a dormant $\lvert 0\rangle$ coordinate | Replace its stabilizer and define the sampled branch | $k \to k$ |
| Classical result | A deterministic $Z$ record or a detector parity of earlier records | Update only symbols, records, or outputs | $k \to k$ |

This is Clifft's adaptation of SymFT's adaptive stabilizer-coordinate planning.
For an active measurement, the planner chooses a basis and array dimension that
can be removed after collapse. A random dormant measurement changes the
stabilizer description and symbolic correction without traversing the dense
coefficient array.

[Transition instruments](noncomputational.md) use the same state model. They
may act classically, filter or collapse the active state, promote a coordinate,
or stop execution so that a trajectory-specific continuation can be prepared.

## From a Circuit to Samples

The boxes below name the main compiler and runtime objects. The text between
them describes the work that prepares or consumes each object.

```text
[Circuit text]
      |
      | Parse and absorb Clifford gates
      v
[Heisenberg IR]
      |
      | Fuse, cancel, and simplify Pauli operations
      v
[Optimized HIR]
      |
      | Choose coordinates and derive symbolic dependencies
      v
[SamplingPlan]
      |
      | Prepare fixed storage, fusion, and scalar or SIMD kernels
      v
[ExecutablePlan]
      |
      | Allocate reusable shot state
      v
[Executor]
      |
      | Run shots and collect outputs
      v
[Records, detectors, observables, and other results]
```

### Clifford Trace

The front end turns circuit text into the operations Clifft needs to simulate.
It absorbs physical Clifford gates into an offline tableau and maps rotations,
measurements, noise, feedback, detectors, and observables into the Heisenberg
basis. These operations form the Heisenberg IR, or HIR.

### HIR Optimization

HIR passes use Pauli algebra and dataflow to fuse or cancel operations and,
when safe, shorten how long coordinates must remain active. This reduces the
work handed to the planner without fixing a runtime representation.

### Coordinate Planning

The symbolic-coordinate planner decides which coordinates are active at each
step. It also resolves basis changes, Pauli support, measurement collapse, and
the symbolic signs described above. The result is a `SamplingPlan` that gives
every shot the same semantic sequence of possible actions.

`SamplingPlan` is the executor-independent semantic boundary. It describes
symbols, affine expressions, active-coordinate actions, width transitions,
records, outputs, noise sites, instruments, and continuation boundaries. It
does not select an ISA, SIMD kernel, descriptor layout, or target-specific
fusion.

### Executable-Plan Preparation

Preparation turns the semantic plan into fixed storage for the selected
executor backend. It arranges symbolic dependencies for incremental
evaluation, combines supported rotation runs, and selects scalar or
architecture-specific kernels. On x86 builds with runtime dispatch, this
selects the scalar, AVX2, or AVX-512 backend once for the plan; portable builds
use the scalar backend.

### Sampling

The executor allocates coefficient, scratch, symbol, record, output, and RNG
storage before the hot loop, then reuses the prepared plan and that storage for
every shot. It samples fault and measurement symbols, evaluates the affected
expressions, and applies prepared active-state actions. It performs no tableau
evolution, commutation analysis, Pauli localization, coordinate selection, or
dependency discovery.

## Continuations and Noncomputational Trajectories

A noncomputational transition, such as leakage or loss, can change whether
later gates still act quantum mechanically on a site. Clifft therefore places
explicit boundaries after relevant instruments. A boundary is a prepared point
where ordinary execution may stop so that the remaining circuit can be
rewritten for that trajectory's sampled site status.

Clifft then plans a continuation and resumes. The continuation preserves the
live coefficients, coordinate meaning and order, active width, symbol and
record values, and RNG position. The same factored trajectory state therefore
spans the boundary even though the replacement suffix is planned later.

See [Noncomputational States](noncomputational.md) for the hybrid
quantum-classical model.

## Exact Final-State Queries

For eligible pure-state plans, Clifft retains the final Clifford coordinate map
needed to relate the active coefficient state back to physical qubits. This
metadata is not read by ordinary sampling dispatch. It supports exact queries
such as dense statevector expansion and sparse computational-basis
probabilities.

[`clifft.basis_probabilities()`](basis_probabilities.md) computes selected
full-register probabilities without expanding the full $2^n$ statevector. Its
exponential component scales as $2^k$: exponentially in the active width $k$,
or linearly in the active-state dimension.

For pure-state programs whose only stochastic events and outputs are visible
measurements, and which do not use postselection,
`clifft.record_probabilities()` evaluates selected records exactly without
sampling.

See [Basis-State Probabilities](basis_probabilities.md) and
[Strong Simulation with Exact Probabilities](../guide/strong-simulation.md).

## References

- Bradley A. Chase and Farrokh Labib, "Clifft: Fast Exact Simulation of
  Near-Clifford Quantum Circuits," [arXiv:2604.27058](https://arxiv.org/abs/2604.27058),
  2026.
- Wang Fang, Huazhe Lou, and Riling Li, "SymFT: Universal Fault-Tolerant
  Quantum Circuit Simulation via Symbolic Clifford-Pauli Frames and Stabilizer
  Coordinates," [arXiv:2607.28600](https://arxiv.org/abs/2607.28600)
  (quant-ph), 2026.
