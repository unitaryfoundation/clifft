# Theoretical Overview

Clifft is a compiler and execution engine for exact simulation of universal
quantum circuits. For circuits where non-Clifford effects remain localized,
it confines exponential work to a dynamic active dimension instead of the
total physical-qubit count.

## Method Provenance

The factored active-state model is part of Clifft's original simulation
design, described by Bradley A. Chase and Farrokh Labib in the
[Clifft paper](https://arxiv.org/abs/2604.27058).

The symbolic sampling strategy described below draws directly on
[SymFT](https://arxiv.org/abs/2607.28600), introduced by Wang Fang, Huazhe
Lou, and Riling Li. In particular, Clifft adapts SymFT's symbolic
Clifford-Pauli-frame factorization and adaptive stabilizer-coordinate
planning. `SamplingPlan`, target-specific executable lowering, instruments and
continuations, and the executor organization are Clifft-specific implementation
choices.

## Symbolic Clifford Coordinates

Sampling a circuit with noise or mid-circuit measurements produces a
trajectory. At action boundary $j$, condition on the Boolean outcomes
$s_{\le j}$ sampled so far. The resulting pure state has the factorization

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
  into the physical qubit basis. Its evolution is determined during
  compilation and planning, not by per-shot tableau operations.

- **$P_j(s_{\le j})$ (Pauli frame):** Represents branch-dependent Pauli
  corrections from noise, measurement outcomes, and classical feedback. It is
  a mathematical component of the trajectory state, but the symbolic executor
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

Active and dormant coordinates are not subsets of physical qubits. A planned
Clifford basis change can change the physical Pauli support of every coordinate
while preserving this factorization. "Active width" therefore means the number
of stabilizer coordinates represented in the dense coefficient array, not the
number of physical qubits touched by the circuit.

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

## Compiling the Pauli Frame into Symbolic Signs

Each stochastic event needed later in a shot receives a Boolean symbol. A
symbol may represent a presampled Pauli fault, a sampled measurement branch, a
readout flip, an instrument outcome, or a named parity of earlier symbols.
Classical effects are represented as affine Boolean expressions,

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

Following the symbolic-frame strategy introduced by
[Fang, Lou, and Li](https://arxiv.org/abs/2607.28600), the planner computes
$\ell_Q$ once and attaches it to the planned operation. It also maps the
unsigned body of $Q$ into the current stabilizer coordinates. Operations that
need coefficient work are consequently represented by an active-coordinate
Pauli and an affine sign, rather than by a physical Pauli string plus a mutable
runtime frame.

The full symbolic Pauli frame is planning workspace. Before hot execution, its
effects have been lowered into plan expressions, expression registers, and
record outcomes. A shot evaluates those prepared expressions as symbols become
available; it does not update an $n$-qubit Pauli frame after each fault or
measurement.

## Adaptive Stabilizer Coordinates

The planner maintains a stabilizer-destabilizer basis in which the first
$k_j$ coordinates are active and the remainder are dormant. It resolves basis
changes, Pauli support, measurement pivots, and active-width transitions once
for all shots. The resulting semantic actions have the following effects:

| Situation | Planned state action | Width transition |
|---|---|---|
| Pauli rotation supported on active coordinates | Rotate the active Pauli directly | $k \to k$ |
| Rotation requiring dormant coherent support | Promote one dormant coordinate, then rotate | $k \to k+1$ |
| Measurement with active support | Sample and collapse, then remove the planned pivot | $k \to k-1$ |
| Random measurement resolved in the dormant space | Replace a dormant stabilizer and define a branch symbol | $k \to k$ |
| Deterministic record or derived parity | Update only symbolic or record state | $k \to k$ |

This is Clifft's adaptation of SymFT's adaptive stabilizer-coordinate planning.
A promotion installs the required Pauli as the next active generator. An active
measurement selects a compile-time basis and pivot that permit the measured
coordinate to be removed after collapse. A dormant random measurement changes
the stabilizer description and symbolic correction without traversing the
dense coefficient array.

Instruments use the same state model. Depending on their source, they may act
classically, filter or collapse the existing active state, promote a coordinate,
or stop at an explicit continuation boundary.

## Semantic Sampling Pipeline

At the semantic level, the symbolic sampling path is

```text
Circuit
  -> Heisenberg IR (HIR)
  -> optimized HIR
  -> symbolic-coordinate planner
  -> SamplingPlan
  -> target executable lowering
  -> ExecutablePlan
  -> Executor
  -> records, detectors, observables, and other results
```

The front end absorbs physical Clifford operations into an offline tableau and
maps relevant operations into the Heisenberg basis. HIR optimization reasons
algebraically about the resulting Pauli operations. The symbolic-coordinate
planner then performs all stabilizer-coordinate changes and symbolic dependency
discovery before emitting `SamplingPlan`.

`SamplingPlan` is the executor-independent semantic boundary. It describes
symbols, affine expressions, active-coordinate actions, width transitions,
records, outputs, noise sites, instruments, and continuation boundaries. It
does not select an ISA, SIMD kernel, descriptor layout, or target-specific
fusion.

Target executable lowering prepares expressions, descriptors, fusion products,
and kernels for a particular executor. The executor owns the mutable per-shot
coefficient, scratch, symbol, record, output, and RNG state. Ordinary dispatch
uses only prepared actions: it performs no tableau evolution, commutation
analysis, Pauli localization, coordinate selection, or dependency discovery.

## Continuations and Noncomputational Trajectories

A noncomputational transition can invalidate the remaining compiled operations
for one trajectory. At an explicit instrument boundary, Clifft may stop,
rewrite the remaining circuit for the sampled site status, plan a continuation,
and resume. The continuation preserves the live coefficients, coordinate
meaning and order, active width, symbol and record values, and RNG position.
The same factored trajectory state therefore spans the boundary even though the
replacement suffix is compiled later.

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
exponential component still scales with the active width.

For eligible programs with measurements, `clifft.record_probabilities()`
evaluates selected measurement records exactly without sampling.

## References

- Bradley A. Chase and Farrokh Labib, "Clifft: Fast Exact Simulation of
  Near-Clifford Quantum Circuits," [arXiv:2604.27058](https://arxiv.org/abs/2604.27058),
  2026.
- Wang Fang, Huazhe Lou, and Riling Li, "SymFT: Universal Fault-Tolerant
  Quantum Circuit Simulation via Symbolic Clifford-Pauli Frames and Stabilizer
  Coordinates," [arXiv:2607.28600](https://arxiv.org/abs/2607.28600)
  (quant-ph), 2026. DOI:
  [10.48550/arXiv.2607.28600](https://doi.org/10.48550/arXiv.2607.28600).
