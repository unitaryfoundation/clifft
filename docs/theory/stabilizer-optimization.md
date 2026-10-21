# Optimization for Simulation

Clifft compiles a circuit once so that repeated sampling does less work. Its
default optimization passes rewrite the simulator's internal representation
while preserving joint measurement-record distributions. Extra compilation
effort can therefore pay off over many shots.

## Cost Model

Clifft stores non-Clifford interference in a dense array of $2^k$ amplitudes,
where **active width** $k$ counts active stabilizer coordinates, not physical
qubits. Clifford structure stays outside this array; measurements can shrink it.
See the [state factorization](overview.md#symbolic-clifford-coordinates) for
the underlying representation.

Peak width sets the exponential coefficient-storage cost, $O(2^{k_{\max}})$.
Sampling work also depends on how many operations process a large active array.
A useful structural estimate is

$$
W = \sum_{a \in \mathcal{A}_{\mathrm{dense}}} 2^{k(a)},
$$

where $\mathcal{A}_{\mathrm{dense}}$ contains operations that process the dense
state, and $k(a)$ is the width of the array they process. This is a cost proxy;
actual runtime also depends on kernels, memory traffic, and classical work.

Clifft reduces this cost by finding equivalent circuits that require less
simulation work. For example, consider this three-qubit circuit, starting from
$|000\rangle$. In Clifft's Stim-like syntax, rotation angles are in half-turns:

```text
R_XX(0.125) 0 1
R_XX(0.25) 1 2
R_XX(0.125) 0 1
MPP Z1*Z2 Z2
```

[Open this circuit in the explorer]({{ playground_url }}?code=EofQGmAUAMB0CMAmArASgATXfAUKCMsKG86ie4UcSam2OAsgApPoBa8AVG4u%2BUA).
To reproduce the stages below, use **Passes** to disable all passes, enable
only `PeepholeFusionPass`, then also enable `StatevectorSqueezePass`.

![Active-width curves for a three-qubit circuit. Fusion removes an operation at width two, reducing estimated work from 16 to 12. Squeezing reduces peak width from two to one and estimated work to 8.](../assets/theory/optimization-profile-light.svg#only-light)
![Active-width curves for a three-qubit circuit. Fusion removes an operation at width two, reducing estimated work from 16 to 12. Squeezing reduces peak width from two to one and estimated work to 8.](../assets/theory/optimization-profile-dark.svg#only-dark)

All three rotations use only $X$ axes, so they commute. The first and third
therefore fuse into one `R_XX(0.25) 0 1`, removing work without changing peak
width. The $Z_1Z_2$ measurement also commutes with the $X_1X_2$ rotation:
the anticommutation signs on qubits 1 and 2 cancel. Moving this measurement
before that rotation then reduces the peak.

The curves use operation order within each rewritten circuit; they are not
timing measurements. Here rotations process the width after the operation,
and active measurements process the width before collapse.

## What the Passes Do

In practice, Clifft works like a compiler: a sequence of passes rewrites the
circuit to reduce simulation cost, with each pass targeting different
opportunities.

| Pass family | How it reduces simulation cost |
|---|---|
| Peephole fusion | Combine or cancel compatible rotations and remove phases consumed by measurements. |
| Phase-polynomial and rotation simplification | Use algebra and known state constraints to remove non-Clifford work. |
| Statevector squeezing | Move measurements earlier and expanding rotations later where dependencies allow, shortening active lifetimes. |
| Active-width scheduling (opt-in) | Search legal operation orders, preferring lower peak width, then lower estimated dense work. |

These HIR rewrites run after Clifford tracing and before active-coordinate
planning. They complement the planner's coordinate choices and the executable
builder's kernel fusion; see [Software Architecture](architecture.md).
The [pass reference](../reference/passes.md) lists defaults, configuration,
search budgets, and restrictions for continuation-based sampling. Larger
search budgets trade compilation time for opportunities to reduce sampling
cost, without guaranteeing an optimal result.

## Stabilizer-Based Simplification

Starting from the circuit's known all-zero input state, Clifft tracks which
stabilizer constraints hold across all trajectories. These constraints let it
replace rotations with simpler operations that have the same action on the
reachable states. Measurement and feedback can restore useful constraints
after non-Clifford gates have removed them. `PhasePolynomialPass` and
`RotationSimplificationPass` use this information during HIR optimization,
before sampling plans are built. Their rewrites apply to every shot.

### Recovering Fixed Constraints

`SymbolicStabilizers` tracks Pauli operators whose eigenvalues are known either
as fixed signs or as expressions in measurement outcomes. Each sign is
`(-1)^s`, where `s` is a constant XOR a set of Boolean symbols. The symbols
represent physical measurement outcomes and readout-error flips; the analysis
assumes no probability distribution or independence for them.

For example, measuring `Z` with outcome `m` establishes eigenvalue `(-1)^m`.
Applying `X` when `m = 1` flips that sign again, leaving `m XOR m = 0` and a
fixed `+Z` constraint. If feedback instead uses a corrupted record `m XOR e`,
the remaining sign is `e`, so the same fixed constraint cannot be inferred.

Products of stabilizers can also have fixed signs when their generators do
not. If `Z0` and `Z1` both have eigenvalue `(-1)^m`, their product `Z0*Z1`
has eigenvalue `+1`. The analysis finds such combinations by eliminating the
Boolean symbols while preserving Pauli multiplication phases. It exposes the
result through `KnownStabilizers`, which stores only fixed-sign constraints.

### Simplifying Rotations

Fixed constraints can identify rotation axes with the same action on the
reachable states. The rotation-simplification pass uses these relations to
merge or cancel rotations, including rotations with arbitrary angles.

The phase-polynomial pass collects commuting T rotations and expresses their
combined phase using Boolean eigenvalue coordinates for independent Pauli axes.
It expands the phase into a polynomial modulo eight, where cancellations can
reveal a smaller non-Clifford core. The pass then synthesizes T rotations for
that core and retains the exact Clifford correction. Subsequent squeezing can
reduce active execution width further, using the final measurements as well
as the preparation.

Collecting a larger complete region can expose cancellations that smaller
regions miss. The optional wide search also tries coordinates derived from the
final Clifford tableau, since coordinate choice affects synthesis. It retains
a candidate only when the rewrite meets the T-count and active-width cost
guards. The number of variables needed to analyze a region can therefore be
larger than either its synthesized core or its final execution width.

Merlin's [phase-polynomial treatment of measured QEC circuits](https://arxiv.org/abs/2610.06811)
motivated the investigation of recovered constraints and complete phase regions.
Clifft uses its own static constraint analysis and
[TOHPE synthesis](../acknowledgments.md#tohpe). See the
[Merlin acknowledgment](../acknowledgments.md#merlin) for the design influence
and reused validation circuits.

### Conservative Analysis and Bounded Search

Constraints tracked by `SymbolicStabilizers` must hold for every trajectory and
the noiseless reference used for syndrome normalization. For Pauli noise, the
forward analysis retains only constraints commuting with each possible fault,
including products unaffected by a shared fault. Instruments clear the analysis
state. When compiling with detector postselection, `SymbolicStabilizers` and
`PhasePolynomialPass` also recover constraints from noiseless measurements
whose outcomes are fixed on surviving shots (directly by a single-record
detector or by unit propagation across two-record difference detectors).
Although upstream faults mix stabilizer signs across unconditional shots,
postselection discards trajectories with the rejected measurement outcome,
projecting all survivors back into a definite Pauli eigenspace. When a later
postselected measurement commutes with the intervening prefix (and no
intervening expectation-value probe depends on a non-commuting prefix),
conditioning on the accepted outcome allows the corresponding projector to be
moved to region entry via bounded lookahead.

The analysis bounds symbolic expressions, retained records, and elimination
work. Reaching a limit can lose a constraint and miss an optimization;
later measurements can still recover useful constraints. The exact constraints
required to justify a selected rewrite are retained separately from these
inference limits. Wide-search exhaustion preserves a completed valid candidate
or falls back to ordinary smaller regions.

Unresolved symbolic signs do not produce per-history specialization. Analysis
symbols stay inside the compiler, and runtime sampling uses its own planned
coordinates and symbols. For defaults, opt-in controls, and numerical limits,
see [Optimization Passes](../reference/passes.md).
