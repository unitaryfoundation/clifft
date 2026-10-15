# Stabilizer-Based Optimization

Clifft can simplify rotations using stabilizer constraints on the states that
reach them. Measurement and feedback can restore these constraints after
non-Clifford gates have removed them. `PhasePolynomialPass` and
`RotationSimplificationPass` use this information during HIR optimization,
before sampling plans are built. Their rewrites apply to every shot and preserve
joint measurement-record distributions.

## Recovering Fixed Constraints

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

## Simplifying Rotations

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

## Conservative Analysis and Bounded Search

Constraints must hold for every trajectory and the noiseless reference used
for syndrome normalization. For Pauli noise, the analysis retains only
constraints commuting with each possible fault, including products unaffected
by a shared fault.
Instruments clear the analysis state, and postselection supplies no additional
constraints. Region lookahead may discard entry constraints but cannot infer
new ones from future measurements.

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
