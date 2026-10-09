# Automatic conditional reduction on complete factory benchmarks

Date: 2026-10-09. Branch: `codex/bt27-fault-specialization`.

## Assessment question and scope

Can the existing algebraic reductions discover useful structure in complete
physical circuits supplied by another research effort, without code matrices,
protocol labels, or manually selected boundaries? Does this improve practical
execution compared with ordinary Clifft and simpler fault specialization?

The reduction rules were frozen for this assessment. The new work connects
them behind one research entry point and adds input support for categorical
correlated Pauli chains. There are no changes to `src/clifft`, Stim, public APIs,
or native execution contracts. All trajectory construction and planning remain
in the Python host before entering the existing executor.

The benchmark package is from local commit `1bcbf960` on
`codex/quadcycle-factory-study`, in the sibling worktree
`/home/exedev/.codex/worktrees/e82c/clifft-codex`. Its handoff is
`research/conditional_clifford/BENCHMARK_HANDOFF.md`; its exporter is
`tools/profile/export_factory_benchmarks.py`. The recorded
[manifest](factory-benchmark-manifest.json) pins the 19 full input files,
including 50 selected exact D/E probability queries. Raw files are regenerated
by that exporter and were read directly from
`/tmp/clifft-factory-benchmarks-20261009`, without merging the sibling branch.
The reference scripts are checked against their pinned Git contents. Merlin
helpers are pinned to `097380fac1a3968ca47925146e211fe990f4c396`.

These are additional diagnostic workloads within a related CSS/tricycle/CCZ
family. Nineteen exported variants are not nineteen independent families.
D/E preparation and final data probes are ideal; their residual/extraction
noise has diagnostic weights. The larger factory includes noisy preparation,
CCZ, escape and feedback, but uses simplified recovery/scheduling and ideal
parts of its final diagnostics. This study does not reproduce operational
acceptance, yield, or logical-error rates from a published protocol.

## Unified research entry point

`ConditionalPhase(source)` receives only raw circuit text. It:

1. Builds the complete physical Pauli fault model and discovers the first
   eligible phase region, including dependency-safe measurement deferral.
2. Shares that region's stabilizer-support and phase analysis across shots.
   Each shot draws its own faults and preparation outcomes; no histories are
   cached or discarded.
3. Chains another automatically discovered region after a certified Clifford
   exit, preserving every earlier visible record. The default budget is eight
   regions.
4. If exactly one T rotation survives, uses the existing bounded Pauli carrier
   through supported Clifford measurements/resets, then returns the complete
   conditional source for ordinary HIR optimization. It does not run the
   previously unhelpful second polynomial synthesis.
5. Otherwise retains the reduced state and complete ordinary continuation.
   Initial unsupported structure uses materialized original input. A later
   fallback retains already sampled records and their conditional state.

The initial-state assumption is all zero only at circuit entry. Internal states
are reconstructed from actual conditional trajectories. Unsupported paths can
still require excessive width; safe fallback is a correctness property, not a
promise of cheap execution. This entry point chooses reduction routes, not the
fastest simulator backend. In particular, ordinary Clifft remains much faster
on several already-small inputs and the ideal factory.
The existing analysis bounds still apply: at most 128 affine support variables
and 200,000 phase terms, as well as the bounded region count.

The correlated-chain handler was brought across from the benchmark effort.
Each adjacent `CORRELATED_ERROR` / `ELSE_CORRELATED_ERROR` chain is one
categorical event. An ELSE weight is multiplied by the probability that all
earlier alternatives failed. Scheduling keeps the entire chain together and
checks the union of all its Pauli supports against deferred measurements.
There are no extra physical fault locations inside CCZ's exact decomposition.

## Correctness evidence

The new frontend, rather than the old benchmark reducer, is used to construct
the candidate programs in the following checks:

- **Complete small instruments:** 70 original/renamed circuits, 94 fixed
  histories, and 183 enumerated branches. Independent Aer density matrices
  retain all physical state entries for each visible record. Maximum entry
  error is below `2.8e-16`; optimized Clifft record-law error is below `7.5e-16`.
  These cover automatic Clifford chaining, one-rotation transport, unsupported
  initial/continuation structure, surviving multiple rotations, resets,
  feedback, annotations, and correlated channels before/inside the region.
- **Noise and scheduling:** three complete categorical mixtures check moved,
  blocked, and prefix correlated chains. In the blocked case an ELSE
  alternative touches a measured wire, so the entire chain must stay put.
  Independent Stim channel controls check nonuniform and zero/one weights,
  plus all 63 three-qubit nonidentity outcomes. The latter also retain the
  benchmark effort's small Aer continuation checks.
- **Fresh quantum outcomes:** 2,048 complete shots each for a measured chain
  and a carrier circuit pass comparisons with the complete Aer record law.
  Exhaustive validation forces every branch with its probability; the actual
  sampling path does not postselect.
- **D/E independent rational references:** all 50 exported exact queries pass.
  An additional 55 histories per protocol cover the ideal case, cancellation
  and backpropagation controls, 32 selected extraction single faults, and 18
  mixed region/extraction histories. Checks include 3,422 complete
  coarse-syndrome support probabilities, 550 selected raw-record probabilities,
  and 368 selected conditional-state probe probabilities in two logical bases.
  Known hidden faults are used only to construct validation probes, never
  operational decoder information or optimizer hints.
- **Factory conditional reference:** five exported histories, including the
  repeated identity draw, each use two actual preparation branches. Exact
  stabilizer-flow checks certify the complete prefix record/state-sign law.
  The original physical non-Clifford circuit is retained on the reference
  side after directly simulating its Clifford preparation. Across 160
  full-record probability comparisons from both candidate and reference
  samples, all checks pass. This reference still uses the existing HIR
  optimizer; it is not an independent full physical-state oracle.
- **Prior regressions:** the complete deferred-phase validator passes its
  small instruments, full noise mixtures, dependency certificates, BT27/BT81
  scored-law comparisons, distillation/cultivation cases, and noisy scoring
  variants. The large scored-law comparison uses the prior shared algorithm.

Across the D/E and factory probability checks, maximum absolute error is below
`7e-18` and maximum relative error below `3.9e-14`. Relative tolerances prevent
tiny full-record probabilities from being incorrectly accepted as zero.
Complete coarse-syndrome laws are stronger than selected raw queries, but are
still marginals. The selected large-circuit raw/state probes do not certify
equality of every possible large-circuit quantum instrument or noisy law.

Artifacts: [frontend instruments](conditional-frontend-validation.json),
[factory and D/E references](factory-frontend-validation.json), and
[deferred-phase regressions](factory-deferred-regression.json).

## Complete sampling comparison

The retained [timing artifact](factory-frontend-study.json) uses 128 complete
shots per input/backend, one native thread and CPU affinity zero. Each
input/backend runs in a fresh process. Fault histories are freshly drawn,
including duplicates; unsupported and width-failure outcomes remain in the
results. Each trajectory samples its measurement outcomes too. Timing includes
fault draws, rewriting, full HIR analysis, lowering, and one-shot sampling.
Input generation and correctness audits are outside the timing. One-time setup
is reported separately. These short runs establish bounded feasibility and
rough cost comparisons, not rare-event accuracy or precise throughput curves.

Baselines are ordinary optimized Clifft compiled once, materializing faults
then optimizing the whole circuit per shot, the same with initial Clifford
preparation sampling, and pinned Merlin when it accepts the supplied circuit.
Plans exceeding active width twelve are inspected without dense allocation.

All costs below are milliseconds per complete shot, excluding one-time setup.

| New workload | Ordinary width | Frontend width | Frontend ms | Fault-only ms | Ordinary ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Factory ideal | 9 | 9 | 37.02 | 83.88 | 0.01225 |
| Factory noisy | 33 | 8-9 | 38.65 | 85.65 | over budget |
| D: separate ancillas | 33 | 8-9 | 15.36 | 94.53 | over budget |
| E: reused ancilla | 33 | 8-9 | 11.88 | 81.19 | over budget |

Relative to fault-only recompilation, the noisy factory is 2.22x faster, D is
6.15x faster, and E is 6.83x faster. The alternative that also samples and
reconstructs the initial Clifford prefix takes 464, 101, and 86.3 ms/shot,
respectively. It provides no throughput advantage on these inputs.

| Regression/control | Ordinary width | Frontend width | Frontend ms | Ordinary ms | Merlin ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BT27 scored | 33 | 0 | 3.124 | over budget | 1.484 |
| BT27 direct X | 33 | 4-7 | 4.96 | over budget | unsupported |
| Cultivation d3 | 4 | 1 | 2.928 | 0.001377 | 0.04307 |
| Cultivation d5 | 10 | 10 | 10.68 | 0.01929 | 0.714 |
| 15-to-1 scored | 5 | 0 | 0.2342 | 0.001015 | 0.01759 |
| Noncommuting control | 4 | 4 | 0.5125 | 0.0008743 | unsupported |
| Unsupported rotation | 1 | 1 | 0.03263 | 0.0006581 | unsupported |

| Input | Frontend setup s | Fault-only setup s | Frontend peak RSS MiB | Fault-only peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| Factory ideal | 0.400 | 0.017 | 66.9 | 60.2 |
| Factory noisy | 0.721 | 0.059 | 97.8 | 63.7 |
| D: separate ancillas | 0.208 | 0.018 | 62.6 | 55.9 |
| E: reused ancilla | 0.176 | 0.017 | 61.7 | 55.7 |
| BT27 scored | 0.770 | 0.069 | 165.9 | 62.2 |
| BT27 direct X | 0.744 | 0.071 | 164.9 | 62.4 |

Peak RSS is the measured process high-water mark, including Python, imports,
loading the benchmark panel, setup and execution. The artifact also records
the peak before backend setup. These values are not estimates of coefficient
storage or isolated incremental sampler memory. Ordinary width-33 runs never
allocate their full dense state, so their RSS cannot represent execution cost.

| Frontend stage | Factory noisy ms | D ms | E ms |
| --- | ---: | ---: | ---: |
| draw | 0.420 | 0.094 | 0.073 |
| rewrite | 4.359 | 1.333 | 1.252 |
| analyze | 26.961 | 12.747 | 9.946 |
| lower | 6.823 | 1.148 | 0.577 |
| sample | 0.085 | 0.041 | 0.037 |

No frontend shots exceeded the width budget across the 11 inputs (1,408
complete shots). The three dynamic methods used identical original fault
histories; their quantum outcomes need not agree shot by shot. Available
backend comparisons passed the recorded single-record and parity moment
checks. Neither the width observations nor these finite sampling checks are
a guarantee over every fault history.


Ordinary noisy factory/D/E compilation exceeds the execution width budget.
Both simpler fault-only compilation and the conditional frontend make these
cases executable. The result therefore establishes automatic discovery and a
cost advantage, not unique executability over every available dynamic method.

Merlin rejects the supplied factory/D/E inputs at an H instruction. No Merlin
speed comparison is established for these files; this is a limitation of the
measured input/backend combination, not proof that the underlying task is
outside every variant of Merlin's method. It does accept scored BT and the
cultivation controls. The ideal factory already compiles to a small ordinary
Clifft program and strongly favors compiling once.

The factory and D/E cases use the general surviving-remainder route. They do
not need the one-rotation carrier, and do not establish further chaining across
multiple surviving non-Clifford rotations. Scored BT uses Clifford completion;
cultivation uses the carrier; unsupported initial rotation input falls back
correctly. Existing full HIR optimization can further shrink the returned
BT direct-X conditional circuit beyond the width of its earlier raw reduction.

## Assessment and next decision

The automatic entry point now works on complete larger diagnostic circuits
without manually supplied boundaries. The independent probability reference
also exercises it on physically different ancilla arrangements. There is useful
signal here for continuing an optional conditional simulation mode.

The main unresolved costs are rebuilding and optimizing the complete emitted
circuit, lowering it, and retaining multiple Python representations during
setup. Small native sampling costs do not imply small complete-shot costs.
An assessment of reusable structure should target those measured host stages
and count setup/memory, without assuming repeated fault histories. A faster
result on these inputs would not require a more general magic-state carrier.

Before expanding the mathematics to multiple surviving rotations, obtain one
structurally different intended workload where the present representation
blocks practical execution. These related factory additions are useful, but
do not close the broad-applicability question. Keep ideal and unsupported
controls, and retain ordinary Clifft as an available choice; this prototype
has no automatic cost model for choosing between backends.

Reproduction commands and dependency details are in
[the profiling guide](../../tools/profile/README.md#automatic-conditional-reduction-assessment).
