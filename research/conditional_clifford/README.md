# Conditional Clifford reduction: research review

This branch explores an optional way to sample circuits whose ordinary Clifft
active width is too large. It specializes on sampled Pauli faults and
measurement outcomes, using algebraic structure of the prepared state to
replace part of the circuit before invoking Clifft. The result may be Clifford
or may retain a much smaller non-Clifford computation.

This is a research prototype for review, not a merge-ready public mode. It is
based on [PR 552](https://github.com/unitaryfoundation/clifft/pull/552) at
`ed2a125ad7daf75deef7e21d66af2a4af4a366bf`. Relative to that base, the work is
in profiling tools, opt-in build targets, and research records; it does not
change production compiler/executor semantics. Review that incremental scope,
not the inherited PR 552 changes, when assessing this experiment.

## Reading guide

1. This summary: the current approach, useful cases, and limits.
2. [Latest capability survey](CAPABILITY_SURVEY.md): complete circuits, costs,
   validation, and the assessment checkpoint.
3. [Method and implementation](METHOD.md): algebra, interfaces, and code map.
4. [Lessons from approaches set aside](LESSONS.md): failed shortcuts, mixed
   performance results, and what those results actually rule out.
5. [Reproduction and evidence](REPRODUCING.md): run retained circuit inputs
   without this VM's temporary directories; inspect the selected evidence.

## How it works for a Clifft user

Ordinary Clifft compiles a circuit once, retaining stochastic operations for
many shots. This host instead prepares reusable analysis once and constructs
a conditional program for each shot:

1. Discover a Clifford preparation and a supported phase region automatically.
   Derive the stabilizer state's affine support and a phase polynomial modulo
   eight. Dependency-safe measurement deferral can enlarge that region.
2. Separate the fixed non-Clifford phase from Clifford corrections controlled
   by faults and preparation outcomes. Draw the complete original noise once
   per shot and evaluate those corrections. Reuse does not require repeated
   fault histories.
3. Reconstruct an equivalent conditional physical state, restore earlier
   visible records, and retain the complete continuation. A Clifford exit can
   feed another reduction. One surviving Pauli rotation has a bounded carrier
   through supported Clifford measurements/resets.
4. Optimize and sample the remaining complete program with Clifft. Where
   eligible, reuse the optimized preparation, traced Clifford continuation,
   and a guarded squeeze permutation. Planning remains fresh per shot.

The all-zero assumption applies only at circuit entry. This is an equivalence
of prepared conditional states, not a unitary identity valid for arbitrary
unknown input. Measurements, hidden reset branches, feedback, detectors and
observables remain part of the sampling contract. No postselection or
fault-weight cutoff is introduced.

## Latest findings

The frozen survey covers 33 inputs: 24 complete protocol variants and nine
small controls. All 4,224 prototype shots fit the width-12 execution budget;
nine inputs exceed that budget under ordinary Clifft. These are observed
widths under the existing bounded optimizer, not lower bounds on every
possible simulator.

| Complete input | Ordinary active width | Conditional active width |
| --- | ---: | ---: |
| Noisy factory | 33 | 8-9 |
| Steane D / E | 33 | 8-9 / 6-9 |
| Noisy BT27, scored / direct X | 33 | 0 / 5-7 |
| Noisy BT81, scored / direct X | 87 | 0 / 3-7 |
| Noise-free BT81, scored / direct X | 56 | 0 / 6 |
| Cultivation d3 | 4 | 1 |
| Cultivation d5 | 10 | 10 |

BT81 direct X retains 23 optimized T gates: becoming fully Clifford is not
necessary for a useful reduction. The noise-free BT81 result also supports
pursuing noise-free optimizations independently.

The structural fit is a large CNOT/diagonal-phase computation whose action
simplifies on a stabilizer preparation's actual support, possibly after
legal measurement movement. The tested protocols are mostly related
Clifford+T error-correction/distillation constructions. Variants of their
noise and readout do not count as independent circuit families.

Ordinary Clifft is faster on all 24 cases it can sample within the survey
budget. Small distillation circuits simplify but were already cheap. Noisy
BT81 takes about 36-53 ms per prototype shot and 20 seconds of setup. For its
scored variant Merlin takes about 35 ms per shot and 0.19 seconds of setup;
the installed Merlin rejects the direct-X readout. Full comparisons, including
setup amortization and timing limitations, are in the survey.

## Applicability and workflow limits

- The input state is all zero. Supported noise consists of categorical Pauli
  events and readout flips, including adjacent correlated/ELSE chains.
  General quantum channels and arbitrary input states are outside scope.
- Discovery is conservative. The current bounds are 128 affine support
  variables, 200,000 phase terms, and eight regions. Unsupported structure
  retains an exact ordinary continuation where the input language permits it;
  that continuation may still be too wide.
- Only one surviving rotation has specialized transport. Multiple rotations
  remain for ordinary Clifft. Two/three-region chaining is independently
  checked on small controls; practical gains from chaining several interacting
  non-Clifford regions in a large protocol remain unestablished.
- This is a host workflow, not the public compile-once/batched sampler or a
  mid-dispatch executor continuation. Its control evaluation, tableau work,
  source reconstruction and planning finish before ordinary execution. It
  does not alter allocation or topology-planning rules inside hot dispatch.
- Setup and per-shot compilation can dominate. There is no automatic choice
  between ordinary Clifft, this prototype and Merlin. Seeded bitstrings need
  not coincide across different sampling implementations.

The checks include independent small quantum instruments, exact selected
large-circuit probabilities, and full recompilation comparisons for reused
fragments. The 128-shot timing batches are not rare-error-rate validation or
an exhaustive large-circuit equivalence proof.

## Assessment and next step

Compilation reuse has reached a useful checkpoint. Stop tuning the current
panel and obtain an independently motivated hard circuit with at least two
interacting non-Clifford regions, preferably outside the existing protocol
cluster. Keep the current frontend and one-rotation bound fixed initially,
and pair it with an independent small-instance oracle or exact marginal
checks. Learn which boundary actually blocks useful width reduction before
choosing a larger carrier representation.

## Historical provenance

This review snapshot consolidates the earlier incremental reports and removes
superseded output artifacts from the current tree. The complete earlier
research record, scripts and measurements remain at
[`d126c29e`](https://github.com/unitaryfoundation/clifft/tree/d126c29ec60d9dfcecfcbe0171632fd91f4809dc/research/conditional_clifford).
The retained survey and validation artifacts keep their original source and
binary hashes; they are measurements of those revisions, not newly relabeled
runs of the review packaging. Older helper modules remain where the current
prototype or validators import them. Standalone historical drivers may require
their archived inputs; use the reproduction guide for current entry points.
