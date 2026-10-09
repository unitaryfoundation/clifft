# Broader complete-circuit capability checkpoint

Date: 2026-10-09. Branch: `codex/bt27-fault-specialization`.

## Assessment

The frozen combined prototype samples all 33 surveyed inputs within active
width 12. Nine inputs exceed that budget under ordinary Clifft: the noisy
factory, D/E, the four noisy BT variants, and both noise-free BT81 variants.
Their observed conditional widths are at most nine. This is useful evidence
for an optional mode that extends practical circuit coverage.

The survey also gives a clear stopping point for compilation tuning. Every
complete protocol input uses one discovered phase region. Actual two/three
region chaining still appears only in the constructed controls. Cultivation
d5 retains width ten despite a smaller first region. These results do not
establish general handling of several interacting non-Clifford regions.

Ordinary Clifft is faster on all 24 inputs it samples within the budget.
Smaller width alone is therefore not a reason to choose the prototype. The
new mode's strongest result remains enabling otherwise excessive width;
its Python specialization and repeated planning are substantial overhead.

## Frozen methods and input scope

The harness receives complete circuit text and no reduction boundaries,
protocol matrices, or family-dependent optimization policies. It uses:

- `ContinuationPhase`, including automatic discovery, dependency-safe
  measurement deferral, Clifford-exit chaining, and the one-rotation carrier;
- fixed-preparation and Clifford-continuation reuse where eligible;
- the same `coordinate-identity` native policy for every eligible input,
  including the existing guarded squeeze schedule;
- the preceding conditional rewrite and ordinary compilation path elsewhere.

No reduction rule, executor, production source, public API, or carrier bound
changed for this survey. There is no history lookup cache. All qubits start
in zero at circuit entry; internal conditional states and observed records
are preserved. Later unsupported structure retains the conditional state
instead of restarting the original circuit. A safe fallback may still be
expensive. The ordinary compile-once backend is a comparison here, not an
automatically selected faster route.

The existing research limits also remain: 128 affine support variables,
200,000 phase terms, and at most eight chained regions. Supported categorical
Pauli/readout noise is sampled exactly; this does not cover arbitrary quantum
channels. Multiple surviving rotations can remain in the ordinary continuation,
but only one rotation is eligible for the specialized carrier.

There are 24 complete protocol variants and nine diagnostic controls:

- Four supplied factory inputs: ideal/live-noise quadcycle and Steane D/E.
- Ten existing complete protocol inputs: 15-to-1 and BH distillation with
  scored/direct-X outputs, cultivation d3/d5, and BT27/BT81 with scored/direct-X
  outputs.
- Ten matching noise-free inputs, formed by materializing the empty fault
  history. All other operations, measurements and output records remain.
- Nine controls for measured support, feedback, three-region chaining,
  branch-dependent second regions, surviving rotations, noncommuting gates,
  Clifford-only input, and unsupported initial/later rotations.

These are not 33 independent circuit families. The protocols remain largely
related Clifford+T error-correction/distillation constructions; size, readout
and noise variants are useful comparisons but not independent evidence of
broad structural generality. The controls are deliberately small and do not
establish a new large-circuit capability.

Generators and noise conventions are unchanged from the earlier studies.
Merlin helpers are pinned to `097380fac1a3968ca47925146e211fe990f4c396`.
Distillation/cultivation use their circuit noise models at p=0.001. BT uses
synthetic Pauli noise through preparation, phase and decoder, with ideal
scoring tails. D/E retain the supplied diagnostic noise weights; these are
not uniform p=0.001 models. Factory limitations from
[the benchmark scope and evidence](REPRODUCING.md) still apply.
Raw input text and hashes are retained with each case.

## Width and complete sampling cost

Each input/backend runs in a fresh process on CPU 2 with one OpenMP thread
and one OpenBLAS thread. There are 128 fresh shots per case/backend, seed
271053. The prototype draws faults once per shot, samples quantum outcomes,
and includes specialization, compilation/planning and execution in the
reported shot cost. Ordinary Clifft compiles once and batch-samples the same
complete original source. The installed Merlin backend also receives that
complete source. Setup is separate; input generation, result bookkeeping,
annotation checks and correctness audits are outside timed shot work.

The width budget is checked before allocating the dense execution state.
"Budget" means width above 12 under the existing optimizer, not a failed
attempt to allocate that state and not a lower bound on all possible methods.
No hard history is redrawn, discarded, or substituted with an easy one.

Times are local short-run means, not stable throughput estimates or a paired
comparison against previous timing artifacts. Medians, maxima, stage times,
and setup times for every backend are in the artifact. In particular, these
numbers should not be read as a regression from the earlier paired coordinate
study. Process peaks include correctness audits and Python imports; native
worker peaks are recorded separately and are not total simultaneous memory.

### Noisy protocols and supplied factory cases

"Ordinary us" is microseconds per shot; prototype/Merlin columns are
milliseconds. Widths are ordinary -> observed prototype range. Setup is the
prototype's one-time wall time in seconds.

| Complete input | Active width | Setup s | Prototype ms | Ordinary us | Merlin ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Factory, ideal | 9 -> 9 | 0.936 | 17.49 | 12.48 | rejected |
| Factory, noisy | 33 -> 8-9 | 1.562 | 19.03 | budget | rejected |
| Steane D | 33 -> 8-9 | 0.459 | 6.13 | budget | rejected |
| Steane E | 33 -> 6-9 | 0.374 | 4.02 | budget | rejected |
| 15-to-1, direct X | 5 -> 1 | 0.103 | 1.57 | 1.13 | 0.050 |
| 15-to-1, scored | 5 -> 0 | 0.103 | 0.49 | 1.28 | 0.017 |
| BH, direct X | 5 -> 1 | 0.088 | 0.89 | 1.05 | 0.047 |
| BH, scored | 5 -> 0 | 0.079 | 0.44 | 0.81 | 0.016 |
| Cultivation d3 | 4 -> 1 | 0.052 | 6.02 | 1.06 | 0.076 |
| Cultivation d5 | 10 -> 10 | 0.248 | 21.97 | 49.23 | 1.395 |
| BT27, scored | 33 -> 0 | 1.628 | 6.30 | budget | 2.57 |
| BT27, direct X | 33 -> 5-7 | 1.438 | 6.87 | budget | rejected |
| BT81, scored | 87 -> 0 | 21.150 | 53.09 | budget | 34.76 |
| BT81, direct X | 87 -> 3-7 | 19.744 | 35.57 | budget | rejected |

The BT81 direct-X case is a useful extension of the recent narrow timing
panel: 405 physical qubits, ordinary width 87, observed conditional width
3-7, and 23 final T gates. Remaining non-Clifford work is handled by ordinary
Clifft at the reduced width. It does not need to become a Clifford circuit
for the approach to be useful. The raw phase synthesis reports 24 T gates;
that is distinct from the final optimized 23.

The installed Merlin implementation rejects these BT direct-X measurements
as incompatible non-affine zero sets. It rejects the raw factory/control
inputs with an unsupported-instruction message. These are results for this
implementation and these exact inputs, not a claim that no alternate Merlin
encoding or method could handle them.

On noisy scored BT27/BT81, prototype per-shot cost is about 2.45x/1.53x Merlin,
not 10-100x. Setup changes a short-run comparison substantially: the complete
128-shot BT81 run costs 27.94 seconds for the prototype versus 4.64 seconds
for Merlin. The prototype has an approximately 21-second setup versus Merlin's
0.19 seconds. Setup amortization must be explicit in any future backend
selection or product performance claim.

The small protocols show why this should not replace ordinary Clifft. For
example, BH direct X gets from width five to one, yet spends about 0.89 ms
per shot compared with ordinary Clifft's roughly 1 us. Cultivation d5 still
has width ten and costs 21.97 ms versus 49 us. These comparisons include the
current host/recompilation overhead and do not imply an inherent lower bound
for a future implementation.

### Matching noise-free protocols

| Complete input | Active width | Setup s | Prototype ms | Ordinary us | Merlin ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 15-to-1, direct X | 1 -> 1 | 0.016 | 0.94 | 0.55 | 0.046 |
| 15-to-1, scored | 0 -> 0 | 0.008 | 0.33 | 0.68 | 0.046 |
| BH, direct X | 1 -> 1 | 0.018 | 0.63 | 0.59 | 0.044 |
| BH, scored | 0 -> 0 | 0.007 | 0.29 | 0.50 | 0.014 |
| Cultivation d3 | 4 -> 1 | 0.008 | 5.67 | 0.75 | 0.040 |
| Cultivation d5 | 10 -> 10 | 0.026 | 19.73 | 7.26 | 1.405 |
| BT27, scored | 0 -> 0 | 0.231 | 4.11 | 1.41 | 2.24 |
| BT27, direct X | 9 -> 6 | 0.310 | 3.92 | 5.71 | rejected |
| BT81, scored | 56 -> 0 | 1.737 | 28.22 | budget | 21.93 |
| BT81, direct X | 56 -> 6 | 1.807 | 16.61 | budget | rejected |

Noise-free BT81 is an independent reason to retain this direction. The ordinary
phase optimizer reports nine capped blocks and leaves width 56; the conditional
method discovers enough structure to reduce the complete scored/direct-X
circuits to widths zero/six. This demonstrates a gain against the existing
bounded optimizer, not an intrinsic obstruction to another noise-free pass.
Noise-free simulation can therefore be a worthwhile separate optimization
track even though ordinary Clifft already handles most other ideal cases well.

## What routes actually executed

Every protocol case uses one phase-region analysis. Reusable continuation
compilation is eligible for factory/D/E and direct-X BH/BT; scored BT retains
the existing all-Clifford route. Cultivation and 15-to-1 direct X use the
one-rotation carrier. No protocol depends on selecting a different planner
cache policy by name.

Cultivation d5 is the clearest warning against using first-region success as
an end-to-end score. That first region has one residual T rotation, but the
carrier stops at the next phase region. The retained whole continuation has
39 optimized T gates and width ten. The noisy original has 91 T gates and
width ten. Reducing the T count did not reduce the peak width. Cultivation
d3 follows the same bounded route but ends at one T gate and width one.

| Control | Actual result |
| --- | --- |
| Measured coset / measured feedback | One region, Clifford completion, width zero |
| Three noisy measured regions | All three chained, Clifford completion, width zero |
| Branch-dependent entry | Two regions on every shot; 67 Clifford-completion and 61 one-rotation exits; width zero |
| Two surviving rotations | Stops research reduction after one region; retains the later non-Clifford continuation; width one, three final T gates |
| Noncommuting control | Carrier cannot continue through the entry structure; width four and 24 T gates remain |
| Unsupported initial rotation | Initial fallback preserves original materialized circuit |
| Clifford-only input | Initial fallback, ordinary Clifford sampling |
| Unsupported later rotation | Keeps the already conditioned state and the unsupported continuation |

The number of discovered regions, the region's raw residual T count, the
carrier stopping reason, and the final optimized T count are different metrics.
All are recorded. A zero T count alone does not certify that a source with
arbitrary-angle rotations is Clifford. Active width concerns the sampling
representation, not the number of physical qubits.

## Correctness evidence and its limits

There were no timeouts, prototype errors, or width-budget failures among
4,224 fresh prototype shots. All visible-record counts and detector/observable
parities are checked. A further 230 fixed-history audits include zero faults,
selected categorical single faults, mixed faults, and a deterministic dense
history with the last nonidentity alternative at every site. These also remain
within width 12; they are separate from the timing draws.

On the continuation-reuse route, 83 audited histories match fresh full raw HIR,
fresh optimized HIR, and ordinary planner inspection/final tableau. This checks
the reuse mechanism against full recompilation. It is not an independent proof
of the preceding algebraic frontend. No new reducer mathematics was introduced.

For the small controls, 40 fixed-history instruments and 104 enumerated
measurement/carrier branches match independent Aer physical-state blocks and
exact optimized Clifft record laws. Maximum state-entry error is below
2.8e-16; maximum record error is below 8.4e-16. This includes both possible
second-region outcomes, fallback, and the three-region noisy chain.

The fresh record-moment screens against ordinary Clifft and Merlin all pass;
the largest score is 3.46 against a conservative threshold of seven. With 128
shots these screens catch only gross discrepancies. They do not validate
rare logical error rates, certify all possible fault/outcome histories, or
prove equality of a large complete output distribution. The earlier independent
small Stim/Aer checks, D/E exact probability references, and large BT/factory
checks remain relevant; see [the retained validation evidence](REPRODUCING.md#evidence-retained-for-review).

## Decision and next useful feedback loop

Stop tuning compilation reuse on this panel for now. The approach already
shows the intended capability extension, including residual non-Clifford work
and a useful noise-free case. Keep it optional, retain ordinary Clifft as the
natural choice for already-small circuits, and preserve the one-rotation bound.
A future automatic selector should consider setup cost and complete-circuit
width; this survey does not implement or validate such a selector.

The next structural experiment should use an independently supplied, complete
circuit with at least two interacting non-Clifford regions, where ordinary
optimized width is genuinely excessive. Prefer a family outside the existing
CSS/phase-polynomial protocol cluster. Freeze the current frontend and require
an independent small-instance oracle or exact marginal checks before considering
any new carrier representation. First learn where it stops and whether the
remaining width is the blocker. Expand beyond one carried rotation only if
that measured failure gives a specific reason to do so.

Chaining constructed regions is working; gaining additional practical circuit
families through that chaining is the remaining unestablished claim. An arbitrary
larger random circuit is not by itself a good target: the absence of useful
structure can be the correct outcome, so its role should be an explicit
negative control.

## Reproduction and retained data

For replay directly from the checked-in circuit inputs, without the original
temporary directories, use [the current review instructions](REPRODUCING.md).
The following is the original generator-based survey command.

Build the existing opt-in `profile_prefix_trace_reuse` and
`export_optimized_prefix` targets. With the pinned inputs available:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 2 .venv/bin/python \
  tools/profile/survey_conditional_capability.py \
  --benchmark-dir /tmp/clifft-factory-benchmarks-20261009 \
  --merlin-checkout /tmp/merlin-research-2026-10-06 \
  --native-worker build-profile/profile_prefix_trace_reuse \
  --exporter build-profile/export_optimized_prefix \
  --output /tmp/clifft-capability-survey-full.json --shots 128
```

The harness emits one complete JSON file and retains error/timeout/width
outcomes instead of dropping a case. The checked-in
[index and summary](capability-survey.json) contains all setup/shot costs,
widths, routes, source/binary hashes and audit summaries. To keep each artifact
below the repository size limit, complete case data are grouped in gzip JSON
files listed and hashed in the index's `artifacts` mapping. Each decompresses
to a dictionary of full case rows, including raw input text, exact selected
histories and moment vectors. No survey data were discarded by this packaging.
