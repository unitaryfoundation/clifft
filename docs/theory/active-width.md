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

Fix two operations with unsigned Pauli bodies $p$ and $q$ that commute with
each other, $p \in q^\perp$. Applying their updates to $S$ in either order
reaches the same subspace. For two rotations this is immediate: intersection
commutes, $(S \cap p^\perp) \cap q^\perp = (S \cap q^\perp) \cap p^\perp$.
For two measurements, apply $p$'s update first:
$S_1 = (S \cap p^\perp) + \langle p \rangle$. Because $p \in q^\perp$, every
element of $\langle p \rangle$ already lies in $q^\perp$, so intersecting
$S_1$ with $q^\perp$ only filters the $S \cap p^\perp$ part:
$S_1 \cap q^\perp = (S \cap p^\perp \cap q^\perp) + \langle p \rangle$, and
adding $\langle q \rangle$ back gives
$(S \cap p^\perp \cap q^\perp) + \langle p \rangle + \langle q \rangle$.
Running $q$'s update first and then $p$'s reaches the same set by the
identical argument with $p$ and $q$ exchanged, valid since commutation is
symmetric ($q \in p^\perp$ too). The same reduction handles one rotation and
one measurement: a rotation's update is the special case of a measurement's
with no $\langle \cdot \rangle$ term ever added.

Two rotations or measurements are independent under the dependence relation
-- free to swap in a legal schedule -- exactly when their bodies commute
(see [Noise Transparency](#noise-transparency) below for the one relaxation
to that rule, around `NOISE` operations). So the identity above applies to
every swap a search or scheduling pass can actually make: the subspace
reached by executing a given *set* of operations is therefore the same
regardless of the order they executed in.

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
executed, and **expanding** if executing it now would grow $k$: a
`RotationPromote` (a rotation whose body anticommutes with some generator of
$S$), or a transition instrument taking its `Activate` branch. Every other
ready operation is **non-expanding**: any measurement, any rotation that
currently commutes with $S$, and every other fixed operation.

Delaying a ready non-expanding operation can only ever leave the eventual
peak the same or larger, never smaller. Consider swapping a ready
non-expanding operation earlier, past some operation it is independent of
(and therefore free to swap, by the previous section). If it is a
measurement, its own update never raises $k$ -- that is what non-expanding
means -- so firing it sooner can only lower, never raise, the width at every
point it now precedes; and because independent updates commute, the final
$S$ after both operations execute is unchanged either way. If it is a
width-neutral rotation, persistence keeps it neutral across the swap: its
body already commutes with $S$, and, by independence, with the other
operation's body too, so it still commutes with $S$ once that operation has
updated it -- moving it earlier changes no width at all. Either way, moving
a ready non-expanding operation earlier never raises any intermediate width,
so repeating this exchange shows some peak-minimizing schedule executes
every ready non-expanding operation as soon as it is ready -- a repeated
step called **closure**: sweep the lowest-index ready non-expanding
operation, over and over, until none remains.

In plain terms, closure means a scheduler never has to decide whether to
fire a measurement or a width-neutral rotation; that choice is already
forced. The only operation type that is both fixed in program order and can
be expanding is a transition instrument taking its `Activate` branch: an
`INSTRUMENT` is a positional barrier under the dependence relation, so
whenever it is ready it is also the only ready operation, and firing it is
forced rather than chosen. The only real decision left, at any point in a
schedule, is *which* ready expanding rotation to fire next when more than
one is available -- everything else follows deterministically from closure.

## Noise Transparency

The dependence relation a scheduler searches over is conservative by
default: it keeps every operation in its original position relative to a
noise operation whose channel does not commute with it. The reason is
mechanical: the planner folds a noise site's symbols into an operation's
sign only when it reaches that operation after the site, in schedule order,
so moving an operation across the site would silently drop or add that
contribution. This can be relaxed. Every noise channel is presampled: its
Boolean outcome exists before the action stream runs, independent of where
in a schedule a planner happens to consume it. If the planner instead
resolves an operation's noise-dependent sign from that operation's
*logical* (original, pre-reorder) position rather than its position in the
reordered schedule, then for any fixed noise realization a
noise-transparent reorder is just an ordinary legal reordering of the
noise-free circuit with that realization's Pauli errors absorbed into
downstream signs -- and ordinary legal reorderings are already known to
preserve the sampling distribution. For every fixed noise realization the
reordered and original schedules have the same distribution over the
remaining randomness, so they agree for the presampled mixture as well.

This is the same symbolic-frame idea the sampler already uses for
measurement and feedback signs, described in
[Planning Pauli-Frame Effects as Symbolic Signs](overview.md#planning-pauli-frame-effects-as-symbolic-signs):
each stochastic event gets a Boolean symbol, and a sign is an affine formula
over the symbols relevant to it. Noise transparency reuses exactly that
mechanism, keyed to an operation's logical rather than scheduled position.

## Exact Certificates

The exact search this section describes is not part of the shipped
library: it lives as a research tool under
`research/active_width_certificates/` on the repository's research branch.
The certificates quoted on this page, including the ones in
[Measured Effect](#measured-effect), were produced with that tool.

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
search over the same closure/readiness trace class. At each step it extends
every partial schedule currently in the beam by each of its own ready
expanding operations, pools every resulting candidate from every kept
schedule into one generation, and keeps only the best-scoring `beam_width`
of them across that whole generation -- not the best child of each parent,
but the best children overall -- ranked by the peak width each candidate's
closure sweep reaches, then by the width the sweep settles at, then by how
many operations the sweep executed (more is preferred), with a final
tie-break on operation index for determinism. The pass is opt-in -- off by
default -- because its compile-time cost on a large, highly branching HIR
is not yet bounded to a small multiple of the rest of the pipeline (see
[Measured Effect](#measured-effect)).

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

Measured once, single host, Release build, at commit `d169751b`: sampling
throughput with the production pipeline (`PeepholeFusionPass` then
`StatevectorSqueezePass`) versus the production pipeline plus the opt-in
`ActiveWidthSchedulePass` (at its default options), on the `clifft-paper`
QEC corpus at commit `db7dc9f`, best of 3 timed batches after warmup. "Plan
work" is the planner's own $\sum 2^{k}$ over dense actions -- the same
quantity `estimate_dense_work` approximates ahead of planning.

| Circuit | Production: peak / plan work / shots per s | Default pipeline: peak / plan work / shots per s | Pass wall time |
|---|---|---|---|
| coherent d3 r3 | 5 / 1247 / 1.08M | 4 / 381 / 2.03M | 17 ms |
| coherent d5 r1 | 12 / 35875 / 82k | 0 / 0 / 3.8M | 2 ms |
| coherent d5 r5 | 13 / 1.99e6 / 2099 | 13 / 7.8e5 / 4938 | 600 ms |
| distillation | 5 / 155 / 1.12M | 3 / 71 / 1.18M | 6 ms |
| cultivation d5 | 10 / 56618 / 58k | 10 / 54710 / 60k | 90 ms |

Lower dense work turns into throughput wherever dense work dominates the
shot cost, as on coherent d3 r3 and coherent d5 r5. Peak width alone is not a
throughput predictor: distillation loses two units of width but gains
almost nothing, because its per-shot cost is dominated by frame, record, and
detector work rather than dense active-state work. Coherent d5 r1's coherent
noise is invisible in the output distribution once every rotation lands
behind a commuting measurement, so the pass finds a schedule with zero
active width at all. Cultivation d5 is where this page's certificates
matter most for reading the table correctly: the exact search certifies
cultivation d3's peak of 4 (a smaller, related fixture, not shown above) as
optimal for its trace class, but on d5 the same search exhausts its node
budget without settling either bound, so d5 carries no certificate. The
pass finds a slightly cheaper schedule at the same peak here -- a real
improvement, not evidence that no better schedule exists.

## References

- Bradley A. Chase and Farrokh Labib, "Clifft: Fast Exact Simulation of
  Near-Clifford Quantum Circuits," [arXiv:2604.27058](https://arxiv.org/abs/2604.27058),
  2026.
- Wang Fang, Huazhe Lou, and Riling Li, "SymFT: Universal Fault-Tolerant
  Quantum Circuit Simulation via Symbolic Clifford-Pauli Frames and Stabilizer
  Coordinates," [arXiv:2607.28600](https://arxiv.org/abs/2607.28600)
  (quant-ph), 2026.
