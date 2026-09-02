# Active-Width Scheduling

The active width $k$ introduced in the [Theoretical Overview](overview.md) is
not just an outcome the compiler reports: it is the value of one GF(2)
linear-algebra object, tracked as the Heisenberg IR is walked in some order.
This page describes that object, the structural facts that make it possible
to search over legal reorderings of an HIR exactly, and what the compiler's
scheduling passes do with that search space.

## The Structural Width Model

Fix an HIR with $n$ qubits. At every point while walking its operations, the
compiler tracks the **dormant subspace** $S$: an unsigned, isotropic (pairwise
commuting) subspace of the $2n$-dimensional GF(2) space of Pauli bodies,
initialized to the span of $Z$ on every qubit. "Isotropic" here just means
every two elements of $S$ commute as Paulis, so $S$ can always be extended to
a full stabilizer group; $S$ starts at its maximum possible dimension, $n$,
one generator per qubit.

The active width is $S$'s codimension, $k = n - \dim S$: how far $S$ currently
falls short of full rank. A rotation or measurement with unsigned Pauli body
$p$ updates $S$ by one of two rules, depending on the operation type:

```text
rotation about p:    S <- S cap p-perp          k -> k+1 iff p is not in S-perp
measurement of p:     S <- (S cap p-perp) + <p>  k -> k-1 iff p is in S-perp but not in S
noise, feedback, detector, observable, readout,
and every other non-rotation, non-measurement op: S unchanged
```

Here $S^\perp$ is the symplectic-orthogonal complement of $S$: the set of
Pauli bodies that commute with every generator of $S$. Both update rules are
uniform formulas, not case splits; the width only moves on the branch where
the formula actually changes $\dim S$:

- **Rotation.** If $p$ commutes with all of $S$, $S \cap p^\perp = S$ and
  nothing changes: either $p \in S$ (the rotation is a global phase the
  planner emits no action for) or $p \notin S$ (the rotation acts on the
  active coordinate array without growing it). If $p$ anticommutes with some
  generator, intersecting drops $\dim S$ by exactly one and $k$ grows by one:
  a dormant coordinate is promoted into the active array.
- **Measurement.** Intersecting first mirrors the rotation rule, then
  $\langle p \rangle$ is added back. If $p$ anticommutes with some generator
  of $S$, the intersection's one-dimensional loss is immediately restored by
  re-adding $p$, netting to no change: a random outcome in dormant space that
  never touches the active array. If $p$ commutes with all of $S$ but was not
  already in it, adding it is a genuine rank increase: $\dim S$ grows by one
  and $k$ shrinks by one, an active coordinate collapsing out of the dense
  state. If $p$ was already in $S$, adding it again changes nothing: a
  deterministic classical outcome.

Every other HIR operation type -- noise, feedback, detectors, observables,
readout corrections -- reads or writes classical state only and leaves $S$
untouched.

### Pivot Independence

An implementation represents $S$ by some list of generators, and rewriting
that list in place (choosing a different pivot row during elimination) is
exactly what both update rules above do internally. Two representations
describe the same $S$ exactly when each one's generators are all contained in
the other's span, not when the generator lists match element-for-element.
Every question this page's model ever asks of $S$ -- does $p$ commute with
all of it, is $p$ already in it -- is a question about the *subspace*, so it
has the same answer no matter which generators happen to represent $S$ at the
time. Consequently the width trace of a fixed operation sequence is
independent of pivot choice: a coordinate-frame implementation and a
from-scratch GF(2) elimination over the same ops necessarily agree on every
width transition, even though they may never choose the same intermediate
generators.

### Confluence and the Order-Invariant Final Width

The two update rules above read only the current $S$ and the operation's own
Pauli body; nothing else about history matters. For two operations that are
independent -- neither is constrained to precede the other, because swapping
them changes nothing observable -- applying them in either order lands on the
same $S$: a rotation or measurement that currently commutes with $S$ keeps
commuting with $S$ under any operation independent of it, because $S$ only
shrinks along directions that anticommute with something already applied,
and applying an independent operation cannot introduce a new such direction.
The subspace reached by executing a given *set* of operations is therefore
the same regardless of the order they executed in.

This is what makes reordering worth searching over at all: the final width
after executing every operation in an HIR is invariant across every legal
reordering, so a scheduler cannot make the exit state worse, only change how
$k$ moves on the way there. A search state is fully determined by its set of
already-executed operations, not by the sequence that reached it -- so two
different partial schedules that happen to have executed the same operations
are provably equivalent, which is what lets a search memoize on that set
instead of re-exploring every path to it.

## The Closure Theorem

Call an operation **ready** once every operation that must precede it has
executed, and **expanding** if executing it now would grow $k$ (the
`RotationPromote` case above, or the analogous branch for a transition
instrument). Every other ready operation is **non-expanding**: any
measurement, any rotation that currently commutes with $S$, and every fixed
operation.

Delaying a ready non-expanding operation can only ever leave the eventual
peak the same or larger, never smaller: by confluence, executing it now
versus later reaches the same $S$ either way, but executing it later leaves
$S$ larger (width higher) for longer, over whatever operations come between
now and then. So some schedule that minimizes peak active width executes
every ready non-expanding operation immediately -- a repeated step called
**closure**: sweep the lowest-index ready non-expanding operation, over and
over, until none remains. In plain terms, closure means a scheduler never has
to decide whether to fire a measurement or a width-neutral rotation; that
choice is already forced. The only real decision left, at any point in a
schedule, is *which* ready expanding rotation to fire next when more than one
is available -- everything else follows deterministically from closure.

## Noise Transparency

The dependence relation a scheduler searches over is conservative by
default: it keeps every operation in its original position relative to a
noise operation whose channel does not commute with it, because a naive
reorder would change which noise realization a later sign correction
observes. This can be relaxed. Every noise channel is presampled: its
Boolean outcome exists before the action stream runs, independent of where
in a schedule a planner happens to consume it. If the planner resolves an
operation's noise-dependent sign from that operation's *logical* (original,
pre-reorder) position rather than its position in the reordered schedule,
then for any fixed noise realization a noise-transparent reorder is just an
ordinary legal reordering of the noise-free circuit with that realization's
Pauli errors absorbed into downstream signs -- and ordinary legal
reorderings are already known to preserve the sampling distribution. Averaged
over the presampled distribution, the reordered schedule therefore samples
identically to the original, not merely on average per shot but for every
fixed realization individually.

This is the same symbolic-frame idea the sampler already uses for
measurement and feedback signs, described in
[Planning Pauli-Frame Effects as Symbolic Signs](overview.md#planning-pauli-frame-effects-as-symbolic-signs):
each stochastic event gets a Boolean symbol, and a sign is an affine formula
over the symbols relevant to it. Noise transparency reuses exactly that
mechanism, keyed to an operation's logical rather than scheduled position.

## Exact Certificates

Given the dependence relation above, an exact search can ask: does some
legal reordering of this HIR reach a strictly lower peak active width than a
given threshold? Because of closure, the search only ever branches on which
ready expanding operation to fire next, and because of confluence, the set
of already-executed operations is a sound key for memoizing infeasible
branches. A bounded, budgeted version of this search produces a witness
schedule together with a proven lower bound; when the two meet, the schedule
is certified optimal.

That certificate has a specific, narrow scope. It is a minimum over the
**trace class** of one exact HIR -- every linear extension of the dependence
relation the search ran with -- scored by the same structural width model
described above. It says nothing about:

- schedules reachable only through a *rewrite* that exposes new gate fusion
  the HIR does not already contain,
- amplitude-level stabilizers that a purely structural analysis does not
  track, or
- any other circuit that merely samples the same output distribution as
  this one.

A schedule outside the searched trace class, a rewritten circuit, or a
distributionally-equivalent but structurally different circuit may still do
better; the certificate is silent about all three.

## `ActiveWidthSchedulePass`

Exact search does not scale to every circuit worth compiling: its budget can
exhaust before proving anything on a large or highly branching HIR. The
compiler's scheduling pass trades that certificate for scalability: a beam
search over the same closure/readiness trace class, keeping a bounded number
of the best-looking partial schedules at each step and extending each one by
its most promising ready expanding operation, ranked first by the peak width
its closure sweep reaches and second by how many operations that sweep
executes. It optionally spends a further bounded budget asking the exact
search to polish its own answer.

Whatever schedule the pass settles on, it is compared against the input
HIR's own width trace, lexicographically by peak active width and then by a
dense-work estimate (the planning-time proxy $\sum 2^{w}$ over actions that
touch the active array, at the width $w$ each one runs at). The pass applies
its candidate only when that comparison is strictly better; otherwise the
HIR is left byte-for-byte untouched. This "never worse than the incumbent"
guarantee is what makes the pass safe to run unconditionally: unlike the
exact search, it carries no certificate of optimality, only a guarantee that
it cannot regress the circuit it started from.

## Measured Effect

Measured once, single host, Release build, at commit `2d99ec5c` (before
later performance work changed the pass's own wall time without changing
peaks or dense work): sampling throughput with the production pipeline
(`PeepholeFusionPass` and `StatevectorSqueezePass`) versus production plus
`ActiveWidthSchedulePass` at its default options, on the QEC circuits used
to validate the structural model above. "Plan work" is the planner's own
$\sum 2^{k}$ over dense actions -- the same quantity `estimate_dense_work`
approximates ahead of planning.

| Circuit | Production: peak / plan work / shots per s | With pass: peak / plan work / shots per s | Pass wall time |
|---|---|---|---|
| coherent d3 r3 | 5 / 1247 / 1.09M | 4 / 403 / 2.18M | 27 ms |
| coherent d5 r1 | 12 / 35875 / 83k | 0 / 0 / 3.8M | 2 ms |
| coherent d5 r5 | 13 / 1.99e6 / 2082 | 13 / 1.23e6 / 3385 | 25.4 s |
| distillation | 5 / 155 / 1.15M | 3 / 71 / 1.15M | 9 ms |
| cultivation d5 | 10 / 56618 / 61.6k | unchanged / 61.6k | 1.36 s |

Lower dense work turns into throughput wherever dense work dominates the
shot cost, as on coherent d3 r3 and coherent d5 r5. Peak width alone is not a
throughput predictor: distillation loses two units of width but gains
nothing, because its per-shot cost is dominated by frame, record, and
detector work rather than dense active-state work. Coherent d5 r1's coherent
noise is invisible in the output distribution once every rotation lands
behind a commuting measurement, so the pass finds a schedule with zero
active width at all. Cultivation d5's plan work is unchanged because its
production schedule was already intrinsic to that fixture under the
structural model above -- there is no better legal reordering to find, which
is itself a fact this page's certificates make it possible to check rather
than merely suspect.

## References

- Bradley A. Chase and Farrokh Labib, "Clifft: Fast Exact Simulation of
  Near-Clifford Quantum Circuits," [arXiv:2604.27058](https://arxiv.org/abs/2604.27058),
  2026.
- Wang Fang, Huazhe Lou, and Riling Li, "SymFT: Universal Fault-Tolerant
  Quantum Circuit Simulation via Symbolic Clifford-Pauli Frames and Stabilizer
  Coordinates," [arXiv:2607.28600](https://arxiv.org/abs/2607.28600)
  (quant-ph), 2026.
