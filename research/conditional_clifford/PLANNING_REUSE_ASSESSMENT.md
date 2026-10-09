# Optimization and planning reuse assessment

This follows the [boundary-composition study](BOUNDARY_COMPOSITION.md). It
measures and inspects the existing improved construction without changing
which optimizer passes run, sharing plans, or modifying production execution.
The question is which remaining work has stable inputs or decisions across
sampled Pauli faults and prefix measurement outcomes.

## What is stable, and what changes

Every inspected trajectory of each eligible complete input uses the same
optimized operation order for that input. The peephole and rotation-simplification
passes leave their HIR unchanged; the squeeze pass applies one stable
permutation. However, the unsigned Pauli operands and sampling plans vary.
The variation survives removal of constant Boolean signs and fixing the ideal
prefix sampler seed. Equal peak active width does not imply equal plans.

The assessment records complete action kinds, widths, operands, pivots,
symbol dependencies, and record destinations. It separately compares plans
after clearing only affine/record-parity constants. Those constants represent
one possible value-specialization layer; removing them does not erase sampled
branch dependencies, change angles, or turn random measurements into deterministic
ones. This distinction prevents interpreting a width histogram as a reusable
plan template.

There are **753 exact conditional-HIR checks**, 512 matched timed/audited
samples, and 34 unchanged fallback checks. Across all 753 inspected trajectories,
both simplification passes make zero changes and each circuit has one squeeze
permutation. Ordinary noisy draws nevertheless produce the following variants:

| Input | Trajectories | Squeeze orders | Plans after clearing constants | Kind/width sequences | Common action prefix after clearing constants |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noisy factory | 128 | 1 | 44 | 43 | 43 |
| Steane D | 128 | 1 | 82 | 82 | 48 |
| Steane E | 128 | 1 | 87 | 86 | 48 |
| BT27 direct-X | 128 | 1 | 104 | 104 | 16 |

For a concrete fixed-prefix-seed factory witness, both histories reach peak
width nine. With fault `(1226, 1)`, plan action 44 writes record 301 as an
affine expression of existing symbols. With faults `(144, 6), (1985, 1)`,
the same action instead samples a new dormant-random branch and adds it to
that record expression. Both use ideal prefix seed 419. The artifact retains
analogous witnesses for D, E, and BT. These are changes in measurement
classification and dependencies, not only constant output signs.

The no-fault group has one normalized plan per input across all 16 prefix
seeds. Factory and BT still have differing full plans because sampled-prefix
record constants vary. The fixed-prefix-seed noisy group still has multiple
normalized plans in every input. This separates fault-dependent structure
from the host choice of ideal measurement outcomes.

Mean cost of the ordinary `diagonal` requests, in milliseconds per shot:

| Stage | Factory | D | E | BT |
| --- | ---: | ---: | ---: | ---: |
| Peephole simplification | 0.349 | 0.062 | 0.037 | 0.044 |
| Rotation simplification | 0.789 | 0.323 | 0.182 | 0.169 |
| Squeeze scheduling | 2.088 | 0.177 | 0.073 | 0.127 |
| Width analysis | 1.307 | 0.153 | 0.079 | 0.098 |
| Ordinary planning | 6.689 | 1.087 | 0.541 | 0.761 |
| Complete attempt | 14.658 | 2.457 | 1.492 | 2.861 |

The scheduling pass alone costs about 2.09 ms on the factory, or 14% of
complete attempt time. That is an upper bound on savings from eliminating
this pass: a reusable permutation still needs a correctness guard and
application work. D/E/BT have smaller scheduling costs. No speedup from
skipping or reusing a pass is claimed by this assessment.

The stable schedules are empirical observations across the tested trajectories.
They are not, by themselves, a proof that an arbitrary future correction can
reuse that schedule. The algebraic predicate below provides a more useful
basis for the next experiment than caching a list of observed histories.

## A candidate certificate for scheduling

The worker now checks a simple setup-time property: transform every surviving
prefix rotation axis through the prefix's fixed final Clifford frame. Is every
result Z-only at the physical boundary? All four eligible panel inputs pass.
The test uses Pauli algebra and no circuit identity or hand-supplied boundary.

If this property holds, any diagonal Clifford correction at that boundary fixes
all those rotation axes. It preserves:

1. Commutation among prefix rotations, which are unchanged.
2. Commutation among continuation axes, which undergo a common Clifford
   coordinate transformation.
3. Commutation between a prefix rotation and a continuation axis, since the
   boundary correction fixes the former in boundary coordinates.

Pauli faults and readout controls in the shared continuation only affect signs;
they do not change these unsigned commutation tests. The template keeps record
indices, annotation targets, operation types, and classical dependencies fixed.
The current squeeze pass's decisions use those dependencies and pairwise Pauli
commutation, so its ordering decisions are invariant for this raw interface.

There is an important scope condition: the squeeze pass runs *after* two other
optimizers. Their observed no-op behavior is not a general license to remove
them. A sound next experiment should retain those passes and reuse the stored
permutation only after confirming they left the relevant input unchanged;
otherwise it must run ordinary squeezing. Alternatively it could establish a
stronger setup-time certificate for their behavior, but that is separate work.

The [small diagnostic controls](planning-reuse-controls.json) include an
unmeasured non-Clifford prefix that fails the predicate. Changing its boundary
from identity to S changes whether the peephole pass can eliminate a rotation,
and changes the plan's action kinds. This demonstrates why arbitrary
non-Clifford prefixes cannot inherit the observed no-op/schedule behavior.
A readout-only control verifies the other distinction: full actions change,
while normalized actions and kind/width transitions remain identical.

## Where planning spends time

A separate 256-trajectory native profile uses `cpu-clock:u` at 499 Hz with
DWARF call stacks. It profiles the native worker only, with ordinary `diagonal`
requests and no diagnostic snapshot requests. Setup is included once; Python
host work and process-idle intervals are outside the sampled CPU time.
The [profile metadata](planning-reuse-profile.json) records source/binary/data
hashes; the [compact perf report](planning-reuse-perf.txt) retains the report.

Inclusive native CPU sample shares are approximately 55.6% in
`plan_sampling`, 51.4% in `CoordinateFrame::to_current`, 17.7% in squeeze
scheduling, and 11.1% in active-width analysis. These inclusive percentages
overlap and must not be added. No samples were reported lost. The profile is
an approximate hotspot measurement from this worker/build, not a cost model
for every circuit or a profiled-versus-unprofiled speed comparison.

The profile and source agree: resolving physical Pauli axes in the changing
planner coordinates dominates. `CoordinateFrame::to_current` normally finds
coordinates through commutation with generator rows, including sign recovery;
it only materializes an inverse after enough lookups in an unchanged frame.
Measurements can change that frame. Action text does not capture every dormant
coordinate, so matching action prefixes alone do not establish an identical
planner checkpoint. Sharing a stable action prefix or keeping the same peak
width does not remove the later coordinate work automatically.
The measured profile does not justify a change to that inverse-cache policy
without a separate experiment.

## Method and validation limits

The [full assessment artifact](planning-reuse-study.json) separates:

- 128 unconditioned fault histories per eligible input with independently
  selected ideal prefix seeds. These requests provide the cost measurements.
- 16 no-fault histories with varying prefix seeds, to distinguish ideal
  measurement variability from fault-dependent structure.
- The first 32 ordinary histories replayed with one fixed ideal prefix seed.
  Fault responses still alter actual prefix records where appropriate.
- Selected single/multiple-fault and all-sites-faulted stress histories.

The host sends each audited history through full fresh tracing for an exact
conditional-HIR comparison. It additionally checks matched native samples,
widths, T counts, detectors, observables, and expectations between the timed
ordinary construction and the diagnostic construction. Diagnostic snapshots
and provenance bookkeeping run outside timed requests.

Diagnostic HIR provenance uses synthetic one-based operation identities, not
user source lines. Passes carry these identities through permutations and
eliminations; complete integer origin lists determine the schedule counts.
The per-pass `changed` flag comes from exact semantic HIR comparison, including
the final frame, rather than a digest. Signed/unsigned HIR row and frame
fingerprints are compact FNV-1a diagnostic summaries, not collision-proof
correctness certificates. Plan action comparisons use complete untruncated
inspection text and symbol kinds; they exclude source provenance. No plan is
executed because a fingerprint matches.

The [independent continuation validation](planning-reuse-validation.json)
passes in `audit` mode, including 256 full-state Aer comparisons, 288 complete
record laws with final-state tomography, 144 exact Stim Clifford-law checks,
72 frontend conditional-trace checks, original-source instrument comparisons,
and seven rejection controls. The new diagnostic controls compare audited and
ordinary construction and their record laws with Aer. The
[prefix regression](planning-reuse-prefix-regression.json) checks the preceding
worker paths, including their non-Clifford continuations.

The large exact HIR comparisons certify this construction relative to the
previous conditional source. They do not independently re-prove every original
large-circuit reduction or certify rare accepted-error probabilities. The timing
sample is used to assess cost and observed structural variation, not rare events.
Diagnostic host memory includes retained snapshots and should not be compared
with the previous frontend's execution-memory figures.

## Decision and next bounded experiment

Proceed first with **certified reuse of the squeeze permutation**, retaining
fresh planning and the other passes. Its eligibility should follow the
boundary-axis predicate and actual optimizer behavior, with a conservative
fallback. Validate exact optimized HIR, final frame, and records against the
full pass pipeline on the same panel and the noncommuting-prefix control.
Then measure the net saving, including the eligibility/unchanged-input checks.

This targets a measured cost and has an algebraic reason to generalize. It does
not require repeated fault histories or a catalog of circuit-specific gates.
Do not interpret the observed no-op passes as authorization to disable them,
or the stable schedule as evidence for copying an ideal sampling plan.

Plan reuse remains a later, harder question because the downstream coordinate
state and measurement classifications vary. Any proposal that changes native
planning/execution responsibilities needs the repository's architectural
review. This assessment changes neither those responsibilities nor the
single-rotation carrier. Independent noise-free optimization remains recorded
as a worthwhile separate direction; it is not needed to complete this noisy
scheduling experiment.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#optimization-and-planning-reuse-assessment).
