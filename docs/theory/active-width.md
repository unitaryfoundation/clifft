# Active-Width Scheduling

Legal reordering can reduce peak active width without changing the sampling
distribution. Clifft tracks width using a subspace of Pauli operators. This
page derives that model, explains which reorderings are allowed, and describes
the opt-in `ActiveWidthSchedulePass`.

For the active-state representation and its role in simulation cost, see the
[Theoretical Overview](overview.md).

## The Structural Width Model

For an HIR with $n$ qubits, the compiler tracks a **dormant subspace** $S$ of
unsigned Pauli bodies. These bodies form vectors in $\mathrm{GF}(2)^{2n}$.
Every pair in $S$ commutes: in symplectic terminology, $S$ is *isotropic* and
has dimension at most $n$. Initially,

$$
S = \operatorname{span}\{Z_0, \ldots, Z_{n-1}\}.
$$

The active width counts the missing dormant directions:

$$
k = n - \dim S.
$$

Let $p^\perp$ denote the Pauli bodies that commute with $p$, and $S^\perp$
those that commute with every element of $S$. A rotation about $p$ and a
measurement of $p$ update the subspace as follows:

$$
\begin{aligned}
R_p(S) &= S \cap p^\perp, \\
M_p(S) &= (S \cap p^\perp) + \langle p \rangle.
\end{aligned}
$$

Here $\langle p \rangle$ is the span of $p$.

- **Rotation.** If $p \notin S^\perp$, the intersection removes one dimension
  and $k$ increases by one: a dormant coordinate becomes active. Otherwise
  the width is unchanged. When $p \in S$, the rotation is a global phase and
  the planner emits no action; when $p \in S^\perp \setminus S$, it acts on
  the existing active array.
- **Measurement.** If $p \notin S^\perp$, the intersection removes one
  dimension and adding $p$ restores it. This gives a random dormant outcome
  without changing $k$. If $p \in S^\perp \setminus S$, adding $p$ increases
  the dimension and reduces $k$ by one: an active coordinate collapses. If
  $p \in S$, neither the subspace nor the width changes, and the outcome is
  deterministic.

A transition instrument classified as `Activate` uses the rotation update.
Its other modes leave $S$ unchanged. All remaining HIR operations, including
noise, feedback, readout corrections, detectors, and observables, also leave
$S$ unchanged.

### Pivot Independence

The update rules depend on the subspace, not on the basis used to represent
it. Changing an elimination pivot changes the generator list but not its
span. Membership and commutation tests therefore give the same answers for
any basis of $S$.

For a fixed operation sequence, a coordinate-frame implementation and a
separate GF(2) elimination must agree on every width transition, even if their
intermediate generators differ.

### Confluence and the Order-Invariant Final Width

The scheduler represents ordering constraints as a dependence graph. A legal
schedule is a topological order of that graph. Only rotations and measurements
move; all other operations retain their relative order. `EXP_VAL` and
`INSTRUMENT` are barriers that no operation may cross.

The plain relation uses `can_swap` to decide which pairs may exchange order.
Two rotations or measurements may swap when their Pauli bodies commute.
[Noise transparency](#noise-transparency) additionally permits them to cross
`NOISE` operations, with sign corrections described below.

Suppose $p$ and $q$ commute. Their width updates commute as well:

$$
\begin{aligned}
R_q(R_p(S))
  &= S \cap p^\perp \cap q^\perp
   = R_p(R_q(S)), \\
M_q(M_p(S))
  &= (S \cap p^\perp \cap q^\perp) + \langle p \rangle + \langle q \rangle
   = M_p(M_q(S)), \\
R_q(M_p(S))
  &= (S \cap p^\perp \cap q^\perp) + \langle p \rangle
   = M_p(R_q(S)).
\end{aligned}
$$

The rotation identity follows from intersecting subspaces in either order.
For the measurement identities, commutation gives
$\langle p \rangle \subseteq q^\perp$, so intersecting with $q^\perp$ leaves
the added span of $p$ intact. The same argument applies with $p$ and $q$
exchanged.

These identities cover swaps between movable operations. Other permitted
swaps involve operations that leave $S$ unchanged. Thus any two legal
prefixes that execute the same set of operations reach the same dormant
subspace and current width. This property is **confluence**. In particular,
the final width is invariant under legal reordering, though the intermediate
widths may differ.

Confluence does not make the full search state independent of its history.
The peak reached so far and the accumulated dense work depend on the path.
A search can share subspace and readiness information by executed set, but
must retain the cost information needed by its objective.

## The Closure Theorem

An operation is **ready** when all its predecessors have executed. It is
**expanding** if executing it now would increase $k$: a rotation whose body
anticommutes with some generator of $S$, or an instrument classified as
`Activate`. All other ready operations are **non-expanding**.

**Closure theorem.** Some peak-minimizing schedule executes ready
non-expanding operations before choosing a ready expanding operation.

To see why, take a legal continuation and move a ready non-expanding operation
to its front. Readiness means every operation it crosses is independent of it.

- For a measurement, confluence lets us view each reordered prefix as the
  measurement applied after that prefix's other operations. A measurement
  never increases width, so this prefix has no greater width than it had
  before the move. The subspaces coincide once both orders have executed the
  same operations.
- A width-neutral rotation stays neutral through independent updates. Its
  body commutes with $S$ and with every Pauli body added by an independent
  measurement. Intersections cannot break that commutation. Moving the
  rotation earlier therefore changes no width.
- Other non-expanding operations leave $S$ unchanged and can likewise move
  earlier when independent.

Repeating this exchange produces **closure**: execute the lowest-index ready
non-expanding operation until none remains. The index rule makes the sweep
deterministic. The theorem concerns peak width; it does not establish minimum
dense work.

After closure, the scheduler chooses among ready expanding rotations. An
expanding instrument offers no choice: as a barrier, it is the only ready
operation when it becomes ready.

## Noise Transparency

In the plain dependence relation, an operation cannot cross a noise site if
it anticommutes with one of that site's channels. The planner normally folds
noise symbols into later operations' signs in schedule order. Moving an
operation across such a site would add or remove a sign contribution.

Noise transparency preserves that contribution using the operation's
**logical position**: its position before reordering. Each `NOISE` channel
is presampled, so its outcome is available regardless of where the site
appears in the schedule. The HIR's `logical_noise_prefix` records how many
noise sites originally preceded each operation. This metadata moves with the
operation, allowing the planner to correct its sign after reordering.

For a fixed noise realization, absorb the realized Pauli errors into the
downstream signs. A noise-transparent schedule is then a legal reordering of
that noise-free circuit, with the same distribution over the remaining
randomness. This holds for each realization, so averaging over the noise
distribution preserves the full sampling distribution.

The correction uses the existing symbolic-frame machinery: signs are affine
expressions in Boolean event symbols. See
[Planning Pauli-Frame Effects as Symbolic Signs](overview.md#planning-pauli-frame-effects-as-symbolic-signs).

## Exact Certificates

The exact search is a research tool, not part of the shipped library. It lives
under `research/active_width_certificates/` on the repository's research branch
and produced the certificates reported below.

The search asks whether any legal schedule stays below a proposed peak-width
threshold. Closure limits branching to ready expanding operations. Among
prefixes that respect the threshold, confluence makes the executed set a
valid key for memoizing failed continuations. A budgeted search can return a
witness schedule and a proven lower bound; when they meet, the schedule is
certified optimal.

The certificate applies only to one HIR, the dependence relation used, and
the structural width model above. The legal schedules under that relation
form the HIR's **trace class**. The certificate does not cover:

- circuit rewrites that expose new gate fusion;
- amplitude-level stabilizers absent from the structural model; or
- other circuits with the same output distribution.

These alternatives may reach a lower peak than any schedule in the searched
trace class.

## `ActiveWidthSchedulePass`

`ActiveWidthSchedulePass` uses a beam search to limit the number of partial
schedules retained. It starts with closure, then repeats these steps:

1. Extend each retained schedule by each of its ready expanding operations.
   Apply closure to every candidate.
2. Group candidates by executed set. Discard a candidate if another in its
   group is no worse in both peak and dense work, and strictly better in at
   least one. Resolve exact cost ties by comparing the full operation
   sequences.
3. Rank all remaining candidates by peak reached, width after closure, number
   of operations executed in this step (more is preferred), and expanding
   operation index.
4. Keep at most `beam_width` candidates across the whole generation. Record
   completed schedules and continue extending the rest.

The best completed schedule is chosen by peak width, then estimated dense
work. By default, the pass also moves width-neutral rotations rightward past
independent non-expanding operations to improve executable-plan rotation
fusion.

The dense-work estimate sums $2^w$ over actions that touch the active array,
using the width $w$ at which each action runs. The pass replaces the input HIR
only if the candidate has a lower peak, or the same peak and lower estimated
dense work. Otherwise it leaves the HIR untouched. This does not guarantee an
optimal schedule or improved sampling throughput.

The pass is opt-in because scheduling can be expensive on large, highly
branching HIRs. Run it last in the HIR pipeline, after `PeepholeFusionPass` and
`StatevectorSqueezePass`. A noise-transparent reorder can prevent later
peephole fusion. See [Optimization Passes](../reference/passes.md) for pipeline
configuration and the measurements below for compile-time costs.

## Measured Effect

The following measurements compare the production pipeline
(`PeepholeFusionPass` then `StatevectorSqueezePass`) with production plus
`ActiveWidthSchedulePass` at its default options. They use a Release build at
commit `d169751b` and the `clifft-paper` QEC corpus at commit `db7dc9f`, on one
host. Throughput is the best of three timed batches after warmup. "Plan work"
is the planner's $\sum 2^k$ over dense actions, which `estimate_dense_work`
approximates before planning.

| Circuit | Production: peak / plan work / shots per s | Production + schedule pass: peak / plan work / shots per s | Pass wall time |
|---|---|---|---|
| coherent d3 r3 | 5 / 1247 / 1.08M | 4 / 381 / 2.03M | 17 ms |
| coherent d5 r1 | 12 / 35875 / 82k | 0 / 0 / 3.8M | 2 ms |
| coherent d5 r5 | 13 / 1.99e6 / 2099 | 13 / 7.8e5 / 4938 | 600 ms |
| distillation | 5 / 155 / 1.12M | 3 / 71 / 1.18M | 6 ms |
| cultivation d5 | 10 / 56618 / 58k | 10 / 54710 / 60k | 90 ms |

Reducing dense work improves throughput when it dominates shot cost, as on
coherent d3 r3 and coherent d5 r5. Distillation gains little despite a lower
peak: its frame, record, and detector work dominate. On coherent d5 r1, the
pass moves every rotation behind a commuting measurement, making the
coherent noise invisible in the output distribution and reducing active
width to zero.

Exact search certified cultivation d3 at peak 4 (a smaller fixture, not shown)
and cultivation d5 at peak 10 under both the plain and noise-transparent
relations. The dense-work gain on cultivation d5 is therefore at its minimum
peak within either trace class. Coherent d5 r5 was the only corpus circuit
whose certificate did not complete within the 200k-node budget; its reported
peak of 13 remains an upper bound, not a certified optimum.

## References

- Bradley A. Chase and Farrokh Labib, "Clifft: Fast Exact Simulation of
  Near-Clifford Quantum Circuits," [arXiv:2604.27058](https://arxiv.org/abs/2604.27058),
  2026.
- Wang Fang, Huazhe Lou, and Riling Li, "SymFT: Universal Fault-Tolerant
  Quantum Circuit Simulation via Symbolic Clifford-Pauli Frames and Stabilizer
  Coordinates," [arXiv:2607.28600](https://arxiv.org/abs/2607.28600)
  (quant-ph), 2026.
