# Guarded reuse of squeeze scheduling

This follows the [optimization and planning assessment](PLANNING_REUSE_ASSESSMENT.md).
The research worker now reuses one operation permutation when an algebraic
boundary certificate and an exact per-shot guard both hold. It retains the
preceding optimizer passes and constructs a fresh sampling plan for every
trajectory. Complete-shot time falls by about 14% on the noisy factory and
4-8% on D/E/BT in the retained paired run.

This is a host-side research option, `mode="squeeze"`. There are no changes to
`src/clifft`, the production compiler/executor, circuit eligibility, or the
single-rotation carrier. The default worker mode remains unchanged.

## What is reused and why it is sound

The reused object is an ordering of HIR operations, principally measurements
and rotations. It is not a qubit permutation, a sampling plan, or a cached
fault history. Its size depends on the circuit's operation count, not the
number of shots or distinct histories.

At worker setup:

1. Transform each prefix rotation axis through the fixed final prefix frame.
   Require every resulting axis to be Z-only at the physical boundary.
2. Compose the raw prefix and continuation with identity boundary correction
   and no fault/readout controls. Run ordinary `StatevectorSqueezePass` on this
   representative input, carrying synthetic operation identities through it.
3. Store the resulting permutation. The other optimizer passes do not run on
   this representative input: the certificate describes the raw interface.

Any allowed diagonal Clifford boundary correction fixes the prefix axes.
It conjugates continuation axes by a common Clifford transformation, preserving
their pairwise commutation. Cross-prefix/continuation commutation is preserved
because the prefix axes are fixed. Fault and readout controls change signs,
not these commutation tests. Operation kinds, classical record dependencies,
annotation target lists, and positional expectation barriers remain fixed.
These are the inputs to the current squeeze algorithm's deterministic decisions.
Consequently it produces the same permutation for every raw trajectory in this
certified interface, even when no complete fault history repeats.

Earlier optimization can invalidate that interface. Each `squeeze` request
therefore snapshots its raw operation payloads, signed masks, and record/annotation
metadata, then runs the usual peephole, optional phase-polynomial, and rotation
simplification passes. A linear exact comparison checks that their resulting
scheduling input is unchanged. It compares actual values, not hashes, operation
counts alone, or an observed list of successful histories. Only then does the
worker apply the stored permutation to the current operations and provenance.
Otherwise it runs the ordinary squeeze pass. A failed boundary predicate also
uses ordinary squeezing.

The guard intentionally excludes the final Clifford frame and source provenance:
neither affects squeeze decisions. Their current values are preserved, and
validation compares the resulting final frame exactly. The guard conservatively
checks signs and angles too, although some such changes would still permit the
same schedule. Its snapshot omits the unused final-frame copy. Snapshot creation,
comparison, destruction, and permutation application are charged to shot cost.
All of this happens before ordinary planning and native execution.

## Measured cost

The [paired study](squeeze-reuse-study.json) uses 256 ordinary noisy trajectories
per eligible input, one native thread, and CPU affinity two. It rotates the
order of fresh tracing, existing diagonal composition, and guarded squeezing.
Every mode receives the same fault history, prefix seed, and sample seed.
Fault drawing, host payload construction, IPC, compilation, planning, and
sampling are included; one-time setup and diagnostic reference requests are
reported separately. There is no postselection or fault-weight truncation.

| Input | Previous diagonal mode, ms/shot | Guarded squeeze, ms/shot | Reduction |
| --- | ---: | ---: | ---: |
| Noisy factory | 14.547 | 12.457 | 14.4% |
| Steane D | 2.505 | 2.306 | 7.9% |
| Steane E | 1.576 | 1.487 | 5.6% |
| BT27 direct-X | 3.091 | 2.966 | 4.0% |

All eight successive 32-shot blocks have positive mean paired savings for each
input. Factory block means save 1.99-2.35 ms/shot; the smaller BT saving varies
more, from 0.023 to 0.283 ms. These are local cost measurements, not precise
throughput guarantees or rare-event validation. Other validation processes ran
on distinct pinned cores; shared-machine effects can still affect timings.

| Input | Ordinary squeeze, ms | Guard, ms | Apply permutation, ms | Added setup, ms | Stored indices |
| --- | ---: | ---: | ---: | ---: | ---: |
| Factory | 2.084 | 0.0720 | 0.0044 | 2.748 | 1,959 |
| D | 0.176 | 0.0036 | 0.0149 | 0.297 | 186 |
| E | 0.073 | 0.0027 | 0.0108 | 0.162 | 132 |
| BT | 0.145 | 0.0071 | 0.0139 | 0.190 | 239 |

The setup cost includes representative composition and ordinary scheduling.
The diagnostic worker constructs the schedule during common setup even when
subsequent requests use a baseline mode; the artifact reports that incremental
cost explicitly. The permutation holds at most about 15.3 KiB of indices here,
plus vector bookkeeping. The guard's temporary storage is separate.

All 1,024 ordinary candidate shots use the stored schedule, as do the additional
241 no-fault, fixed-prefix-seed, and fault-stress requests. The permutations
are nontrivial: 1,957, 186, 132, and 229 positions move respectively. Observed
active widths agree exactly between modes: factory and D span 7-9, E spans
8-9, and BT spans 3-7. The improvement removes scheduling work; it does not
reduce the state width further.

## Correctness and fallback evidence

- **Large complete inputs:** 1,265 exact comparisons against independently
  retraced, fully optimized conditional HIR, including signed operands,
  angles, operation order, record/annotation metadata, and final Clifford
  frame. The study also performs 2,530 raw trace comparisons and matches
  all 1,024 ordinary samples' records, detectors, observables, expectations,
  widths, and T counts across the three modes. Stress cases include selected
  single/multiple faults, all sites faulted, 16 no-fault prefix seeds, and
  32 ordinary histories replayed with a fixed prefix seed per input.
- **Independent small references:** the
  [continuation validator](squeeze-reuse-validation.json) passes 256 full-state
  Aer comparisons, 288 complete record laws with final-state tomography,
  144 exact Stim Clifford-law comparisons, and 72 frontend conditional-trace
  checks. It also retains original-source instrument checks and seven input
  rejection controls. Its 616 worker requests include 320 actual reuses,
  160 boundary-predicate fallbacks, and 136 optimizer-change fallbacks.
  Maximum state/record entry error is below `3.4e-16` in these reference arrays.
- **Boundary algebra:** the [boundary validator](squeeze-reuse-boundary.json)
  exercises actual schedule reuse for all 512 three-qubit diagonal Clifford
  corrections, including canceling gate sequences, with independent Aer
  full-state comparisons. Its prefix is chosen so earlier passes retain the
  rotations; it asserts reuse on every group element. Another 96 reused cases
  cover physical widths 65, 129, and 193, with reset, feedback, signed readout,
  and Pauli faults across mask-word boundaries. Maximum small-state entry
  error is below `3.9e-16`; all optimized trace comparisons pass.
- **Guard controls:** [five small cases](squeeze-reuse-controls.json) check
  ordinary reuse, a noncommuting boundary, optimizer fusion on an otherwise
  certified prefix, an optimizer rewrite that preserves operation count,
  and a surviving continuous rotation. Each uses four correction/readout
  configurations and checks exact optimized HIR, expected reuse/fallback
  status, and an independent Aer record law. Counting operations alone would
  miss the same-count rewrite.
- **Earlier paths:** the [prefix regression](squeeze-reuse-prefix-regression.json)
  passes. Scored BT27, cultivation d3/d5, and the unsupported-rotation case
  retain their previous frontend route in 34 checks.

The exact large comparisons certify the new construction relative to the
preceding conditional source. They do not independently prove the entire noisy
law of a large physical circuit. The existing benchmark limitations still
apply: factory/D/E are related diagnostic constructions, not independent
protocol families or reproductions of published operational error rates.

## Assessment and next question

This experiment succeeds at its bounded objective. An algebraically justified
shared schedule replaces nearly all measured squeeze cost, with inexpensive
guards and tested fallbacks. No gate-pattern catalog, fault-history cache,
additional non-Clifford carrier, or production architecture change is needed.
It also provides a concrete example of useful compilation reuse even when
later sampling plans differ.

Further scheduling refinements have little remaining cost to remove. Fresh
factory planning still costs 6.62 ms/shot, versus 12.46 ms total; width analysis
adds 1.30 ms. The preceding profile identified coordinate conversion as the
main planning hotspot. The next bounded compilation study should measure how
many Pauli-coordinate queries occur between planner frame changes and whether
reusing an inverse or batching queries within those stable stretches can help.
Keep the plans fresh: equal width or matching action prefixes still do not
certify identical coordinate state. Compare any candidate with current planning
on these inputs and small structurally varied controls before adopting it.

This is also a checkpoint on scope: the present speedup is modest and does not
broaden supported circuit structure. A structurally different complete circuit
family would add more generality evidence than additional factory variants.
Broader non-Clifford carry and independent noise-free optimization remain
separate directions.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#guarded-squeeze-schedule-reuse).
