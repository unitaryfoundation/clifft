# Representation study checkpoint

Latest: [complete noisy five-check factor sequences](msc_factor_sequence.md)
close the native fault-binding, final-amplitude, logical-handoff and full-output
coverage gaps on a synthetic large-state control. The 37-data-qubit sequence
executes all five checks in median 1,234 us/attempt versus 15,199 us for ordinary
Clifft (about 12.3x), over three many-shot runs. The 61-wire sequence takes
4,401 us; its width-31 dense Clifft plan requires 32 GiB for coefficients alone
and was not allocated. Width-4/10 controls still favor ordinary Clifft.

The study covers 72 full native trajectories, 54 elementary Clifft replays,
12 full CH replays including 61 wires, six original-gate Aer trajectories,
all reported output parities checked by Stim, and 512 local fault-map matrix
comparisons. Shared factor arithmetic and the complete native sequence pass
allocation checks and ASAN/UBSAN. The combined MSC regression run passes 59
tests; the additional MPS translation test brings distinct coverage to 60.

This is not full authors' MSC7: preparation and CSS measurements are ideal
operations with explicitly placed Pauli/readout noise, without growth or
ancilla extraction. No authors' large artifact or fault-distance proof has
been obtained. A bounded direct Aer MPS probe timed out before completing its
initial attempt; it is not a throughput comparison or best-in-class evidence.

The remaining implementation step is architectural. A
[concrete production proposal](logical_block_integration.md) recommends an
opt-in fused logical-block action, preserving the existing dense logical
state at certified boundaries and using compiler-owned fixed factor plans.
Noise/RNG/record ownership remains with the ordinary executor; unsupported
feature mixes decline before execution. Compile once per circuit remains
appropriate for this tested family. No production source or public API has
changed. Seek the architectural confirmation required by AGENTS.md before
adding the new planner/action/lowering contract; do not silently repurpose
instrument continuations or introduce runtime topology planning.

Previous: [fixed factor CSS syndrome sampling](msc_factor_sampling.md) avoids
X-support enumeration using precompiled prefix contractions. Measuring checks
in the single-copy elimination order allows each measured coordinate to split
into bra/ket copies without increasing the certified contraction width. The
same representation computes norms, complete syndromes, and final logical
amplitudes with coherent cross terms and physical frames retained.

On generated 7/19/37/61-data-qubit CSS blocks, maximum joint tables contain
8/8/32/64 entries. The native 37-qubit conditional syndrome kernel takes median
506 us versus 80,394 us for the sparse/Fourier kernel, about 159x faster, with
11,936 coefficient bytes. The 61-qubit kernel takes 1,745 us with 23,968
coefficient bytes. Compiled plans/gathers are additional storage. These timings
exclude physical fault binding, injection/growth, and final amplitude recovery;
they are not full-circuit speedups against Clifft or authors' MSC7 results.

Coverage includes all 95 original terminal histories, complete d3/d5 sampling
with factor replacements, independent Aer checks, 48 large/small CH prefix
comparisons, 48 native sampled-syndrome checks, and eight independent CH checks
of native-selected syndromes. All 55 MSC tests pass. Native allocation checks
and ASAN/UBSAN pass. The larger geometry comes from Stim-generated code blocks,
not a complete cultivation artifact.

The [research selection screen](msc_factor_screen_data.json) uses actual
Clifft active widths and certified contraction costs. It retains ordinary
Clifft for the measured width-4/10 controls, requests full comparisons for
intermediate sizes, and flags dense allocations above a 1 GiB study budget for
bounded factor trials. The synthetic 37/61-qubit planning controls have active
widths 19/31; reconstructed f7 remains width 44 under its existing fold engine.
The screen is not a general speed predictor or production routing policy.

Next authorized work: lower factor sampling and final amplitudes into the
complete native worker, then test a fully specified noisy five-check sequence
on 37 data qubits against the original elementary-gate circuit. Include dynamic
fault binding, code/logical-state handoff, and every output in the timing and
validation. Preserve the original controls and the separate f7 limitations.
Label the new sequence synthetic; do not claim authors' MSC7 equivalence or
fault-distance certification. No new user approval is needed for this bounded
research unless an architectural invariant or substantive scope must change.

Previous: [complete static MSC sampling and native controls](msc_sampling.md)
closes the record-sampling and offline gadget-binding gaps. Gadget faults now
select parity-polynomial monomials; Clifford input projections plus compiled
affine equations sample full records. A sparse CSS Fourier transform samples
the final syndrome without enumerating all sectors. Python and native workers
cover original d3/d5 injection, growth, feedback, spectators, readout flips,
hidden resets, detectors, observables, and final logical amplitudes.

The native study validates 64 fresh nominal/stress trajectories; sixteen also
replay original elementary gates in Clifft. Maximum joint-probability error is
1.12e-15 against the coherent reference and 4.27e-14 against elementary Clifft;
maximum logical-density error is 1.99e-14. Native allocation instrumentation and
ASAN/UBSAN pass; all 48 MSC regression tests pass.

The performance result is negative for these controls. Three 10,000-attempt
runs give median d3 times 31.60 us versus 1.02 us and d5 times 1,088.79 us versus
17.46 us for current Clifft. Clifft's peak active widths are only 4 and 10.
Physical circuit size is not sufficient to establish the hard-state regime.
Keep ordinary Clifft for these controls; do not optimize this small-state worker
or claim a new large-state speedup. The earlier reconstructed f7 result remains
separate and its authors-artifact/final-output limitations remain unchanged.

Next authorized bounded work: use actual planned active width and block
support/term counts to test a cheap eligibility/cost screen against both these
negative controls and the existing large f7 reconstruction. The physics follow-up
is fixed factor contraction for CSS syndrome sampling, to avoid enumerating the
full X-generator support as code size grows. Reuse the f7 contraction approach,
validate against this sparse Fourier reference, and keep all topology planning
at compilation. Larger code-block tests must not be labeled full MSC7 circuits.
Then extend large-state output/eligibility coverage before production dispatch.
No new approval is needed for this research batch. Only a substantive scope or
architectural-invariant change should interrupt it for a user decision.

Previous: [direct injection and complete MSC fixed-history coefficients](msc_injection.md)
removes the final CH state-extraction bridge. The single-T prefix compiles to
four weighted 2-by-2 logical maps, sixteen record constraints, and fixed fault
parities. Each reachable prefix trajectory, including hidden resets, has weight
2^-17. Its amplitudes feed growth and the coherent final block directly.

All 95 complete d3/d5 histories match joint weights and final logical states;
maximum relative probability error is 4.22e-14. Twenty original-gate Aer prefix
comparisons cover T/T-dagger and physical faults. Another 158,034 signed-flow
checks cover 2,928 prefix fault-basis histories. Five new tests bring the MSC
suite to 42. No initial/intermediate reference state enters the coefficient
path, but gadget payload binding still uses the offline topology diagnostic.
Next: compile fixed gadget fault maps, preserving relative phases and root/reset
constraints; then validate complete record sampling before native timing. This
is complete fixed-history coefficient composition, not yet a fully static
sampler. No new throughput claim; reconstructed f7 remains the large-state target.

[Coherent original-MSC terminal contraction](msc_terminal_contraction.md)
keeps four monomial terms through final cultivation and terminal measurement,
then contracts at the actual final code projection. The nineteen-data-qubit
code uses 512 support entries per logical basis state and 1,024 decoder entries;
eighteen syndrome duals cover all sectors without a table per syndrome.

All 95 complete d3/d5 histories match the isolated terminal block. The study
also connects growth amplitudes directly into the final block for all 51 d5
histories, without extracting an intermediate logical state from the reference.
Independent dense projectors cover all 64 d3 sectors and 51 selected d5 sectors;
eight Aer statevector evolutions check two nineteen-qubit four-term cases.
Six new tests bring the MSC suite to 37. No native performance claim.
Next: compute the injection prefix's weighted logical state directly and
connect it to these contractions. Offline initial-state extraction, gadget
fault binding and fixed-history rather than sampled-record execution remain.
Keep the native topology/allocation invariants; reconstructed f7 remains the
large-state performance target.

[Sparse original-MSC growth contraction](msc_growth_contraction.md)
replaces the coherent-CH input-projector calculation for growth with fixed
code lookups, syndrome-dual Paulis and scalar ancilla overlaps. Two coherent
gadget terms each visit 16 code entries; all 64 input syndromes share a 4 KiB
decoder. The output is two weighted logical amplitudes, with the prior boundary
maps retaining the physical syndrome/frame convention.

All 51 complete d5 histories match, including fourteen nonzero input frames;
maximum relative probability error is 4.78e-15. Independent dense projectors
check all 64 input syndromes and four logical corrections, with arbitrary
complex coherent inputs. Five new tests bring the MSC suite to 31. The kernel
performs no runtime tableau/dependency work, but the harness still obtains its
input state and fault-bound gadget terms offline. No native timing claim.
Next: extend the contraction to the nineteen-data-qubit final cultivation and
terminal measurement together, retaining their coherent terms until the real
final code projection. Then finish injection and static gadget fault binding
before complete static sampling/native timing. Reconstructed f7 remains the
large-state performance target.

[Weighted original-MSC instruments](msc_instruments.md) now specify
conditional branch weights and logical amplitudes at all three certified
boundaries. Fixed parity rows select a commuting input projector, reject
inconsistent records, and sign the three logical pullbacks. Probability scales
are 2^-3 for either injection interval and 2^-18 for growth. Growth factors
into six input code checks on seven data qubits and 34 single-wire projectors.

Validation: all 146 boundary observations from 95 complete histories match
branch weights and the full logical Bloch vector; 479,800 independent signed
flows cover 4,982 fault-basis histories. Dense elementary Kraus operators check
448 small-system cases including impossible records and orthogonal inputs.
Four new tests bring the MSC suite to 26. Binding is static parity evaluation;
the coherent input-projector contraction is still an offline CH oracle.
Next: replace that contraction with small fixed code/ancilla contractions,
starting with growth, then validate a complete static evaluator before native
timing. No new throughput claim; f7 remains the large-state target.

[The static original-MSC boundary study](msc_boundaries.md) certifies
the two injection-to-check intervals and the d3-to-d5 growth/syndrome interval.
All data and spectator stabilizers leave exactly one logical degree of freedom
at these entries. Fixed record/fault parity maps produce signed syndromes and
Pauli corrections without runtime tableau work. Growth also certifies all six
input syndrome flows and both logical axes. Spectator preparation/measurement
dependencies before the interval are retained explicitly.

Validation: 177,460 signed-flow checks across 4,982 ideal/fault-basis histories,
and 146 boundary observations from the existing 95 full histories, including
55 nonzero data sectors and 9,596 exact term/stabilizer comparisons. Seven new
tests and the combined 22-test MSC suite pass. Next: derive the weighted
two-amplitude contractions, reachability constraints and normalization at these
boundaries, then compare complete static execution with the full-history
oracle. Membership/sign maps alone are not a complete sampler. No new native
throughput claim; f7 remains the large-state target.

[The full original-MSC composition check](msc_protocol.md) evaluates
complete fixed histories of the actual d3/d5 benchmark circuits, retaining
physical noise, true versus reported outcomes, reset trajectories, growth,
feedforward and every detector/observable parity. Across 95 histories, d3
matches elementary-gate Aer and d5 matches elementary-gate Clifft; 4,412
conditional probabilities agree within 4.39e-15 and 51 d5 joint log probabilities
within 4.27e-14. Stim independently checks every output parity. Legal logical-Z
tails pass all detectors and retain observable one. Eight new tests pass.

This is an offline coherent-stabilizer composition reference, with observed
peak eight terms and structural bounds eight/sixteen for d3/d5. It is not a
native logical-block sampler or a static boundary certificate. Next: derive
signed code-boundary/fault maps for the actual syndrome and growth intervals,
including nonzero sectors and feedforward, using the full evaluator as oracle.
Do not introduce runtime tableau planning. No new throughput claim; f7 remains
the large-state target. The exploratory full d5 Aer MPS run was interrupted;
full-d5 external quantum-state validation remains a limitation.

[The actual MSC gadget coverage study](msc_gadgets.md) certifies a
different physical family in the original d3/d5 benchmark inputs. Five
T-conjugated measurement/reset regions reduce to two Clifford monomials,
covering 28/29 and 90/91 physical T gates. Fixed Pauli faults, both true
outcomes, ancilla records, wire relabeling, and ideal signed code boundaries
are independently checked. Seven focused tests pass. The original inputs and
license are preserved in `../fixtures/msc/` with provenance and hashes.

The full-protocol study above extends this local result to complete fixed
histories. These are composition controls; reconstructed f7 remains the
large-state target. Both corpus inputs still decline the existing fold adapter.
The available original cultivation generator supports only d3/d5, not MSC7.

[The compiled measurement-bridge study](frame_bridge.md) closes the
fault-frame measurement gap on the supported f7 families. Cat-bit substitution
produces exactly the existing two weighted monomials. Ten new legal cross-check
data-hook cases per variant leave cat-data CZ corrections and survive with
probability 1/4; 68 new full-history independent checks agree within 5.56e-17.
All 115 research tests and native sanitizer checks pass. Each of two variants
passes 3,792 fault-history branch checks, 768 incoming-Pauli checks, and 96
full-circuit fixtures, including a mixed preplanned fallback.

Full sampling does not show a practical advantage: bridge versus original
medians are 46.15 versus 45.40 us/attempt on original f7, and 46.11 versus
45.70 us on changed growth/syndrome f7. All survivor outputs agree across
600,000 paired attempted histories, retaining 350 measurements, 348 detectors,
and five probes per survivor. Stop the bridge as a separate execution direction;
keep compiled logical blocks as the preferred complete evaluator. Retain the
frame algebra and regression cases, not a new generic adaptive MPS framework.

Next: return to actual MSC benchmark coverage. Identify and certify boundaries
in a genuinely different existing MSC schedule/gadget family, with an explicit
applicability/fallback contract, before more production plumbing. The current
paper-guided f7 reconstruction remains the large-state target and d5 a control;
another rearrangement of its syndrome schedule is not broader physics coverage.
Do not claim the existing corpus is covered or that authors' artifacts have
been obtained. Production integration remains a separate decision.

[The compiled Clifford fault-frame study](fault_frames.md) factors
arbitrary core Pauli faults into one shared Clifford correction with precomputed
quadratic slots. Across 540 native-operation f7 checkpoints, the corrected bond
upper bound is at most 34 and the affine-cover union is at most 14 directions.
All 300 paired/multiple-hook checkpoints return exactly to the ideal states.
The native descriptor is 576 bytes; its update cost and the existing branch
evaluator's cost are recorded in the report and raw JSON, not claimed as complete
sampler speedups. All 110 research tests and 1,054 native frame fixtures pass,
including sanitizer checks. The sufficient data-only exit rule accepts 307/480
single Pauli patterns and 239/256 natural core samples. Cat-data corrections
change measurements and remain the key composition gap.

The subsequent bridge study above completes the bounded measurement and
throughput comparison. No generic adaptive MPS or runtime basis discovery was
implemented.

[The large-fold frame study](fold_frames.md) screens fixed frames on
702 f7 checkpoints without a dense large-state allocation. One ideal-trained
frame bounds ordinary sampled histories at bond 96, but legal multiple-hook
histories reach upper bound 10,240 and a certified lower bound 1,024. Small
per-history affine covers also vary: their conservative union reaches 44
directions. The full and final focused suites cover 104 passing research tests.
Keep compiled logical blocks as the leading implementation. A next bounded
frame experiment should factor out known common Clifford hook corrections
using offline descriptors, then measure native update costs and any composition
benefit. Do not start generic adaptive CAMPS or runtime basis discovery.

[The parsed-region adapter study](fold_adapter.md) replaces per-schedule
C++ builds with circuit-derived data plans and one reusable native worker.
A held-out shuffled-syndrome/changed-growth f7 circuit passes 200 independent
fixed-history comparisons and sustains 45.99 us/attempt with full survivor
outputs. There are 93 distinct passing research tests and sanitizer checks.
The adapter retains canonical wire labels and the fixed fold/stage skeleton;
all eight non-one-round corpus circuits decline. The schedule-catalog gap is
closed within that scope, but broader physics coverage remains open. The
subsequent frame study tests large fold states before extending frontend
plumbing or proposing production integration.

[The growth-implementation study](fold_growth.md) adds a local CSS-dual
growth encoder, alone and combined with changed syndrome ordering. Both pass
199 independent fixed-history comparisons with unchanged native execution.
The 83-test research suite passes. This variant changes physical fault maps
but preserves the original syndrome-transfer relation; broader growth maps
remain untested. The subsequent adapter study derives eligible region plans
without generating a new catalog for every schedule.

[The syndrome-schedule study](fold_schedule.md) certifies two changed
full f7 schedules from their gates. Each passes 115 independent fixed-history
comparisons with unchanged native execution and contraction capacity. A held-out
shuffled schedule also passes. Whole-circuit catalog matching is still narrow.

[The eligibility/composition study](fold_eligibility.md) provides
a parsed-circuit recognizer and full survivor-result adapter for the three
certified reconstructions, plus measured fallback through ordinary Clifft.
No production dispatcher or live-state handoff has been added.

[The bounded alternatives study](alternatives.md) records the native
Pauli-proxy comparison and the five-round Clifford-frame screen. Logical blocks
remain the leading integration candidate. The large-fold follow-up now tests
fixed frames; generic adaptive frames and a tensor executor remain unmeasured.

The current candidate is named **compiled logical-block simulation**. Research
recognition and output composition have passed the bounded integration screen;
production integration remains pending coverage and architecture review.
The user prioritizes sustained sampling of large residual states,
simple invariants, and cheap eligibility/fallback. Few-shot startup and one-round
memory are outside this study.

## Resume compiled logical blocks

The reproducible checkpoint is commit `ce702482` on
`codex/factored-state-research`. Start with [the full f7 report](fold_f7.md),
[the f5 design](fold_blocks.md), and their adjacent raw JSON data.

- The noisy f7 reconstruction is `../fixtures/fold_cultivation_f7.stim`.
  It has 99 physical qubits, 221 non-Clifford gates, and 2,205 noise sites.
- The native sampler measured 45.41 us per attempted f7 shot over 100,000
  attempts; process maximum RSS was 4,980 KiB. Ordinary scalar Clifft plans
  width 44, but sampling was skipped. There is no measured f7 speedup ratio.
- f5 measured 17.73 us per attempt against 364.66 ms in the specific scalar
  diagnostic comparison. Neither result establishes best-in-class performance.
- f7 validation: 115 histories against the independent coherent reference,
  48 research tests, and native ASAN/UBSAN checks. Both uniform verification
  records are accepted, and nonzero growth sectors are preserved.
- The invariant is two logical amplitudes plus a physical Pauli frame at code
  boundaries, with up to four coherent monomial terms between boundaries.
  Contraction schedules and fault maps are compiled once; no hot topology work.
- This is a structured-generator prototype, not an arbitrary-circuit recognizer.
  Its terminal contract is ideal syndrome postselection and logical XYZ probes.
  It lacks final-error-correction decoding and a general residual-state handoff.
  The circuit is paper-guided, not verified equivalent to an authors' artifact.

The subsequent alternatives, eligibility, schedule, growth, adapter, and frame reports complete the
bounded steps from that checkpoint. Do not start a general representation-
switching framework or optimize SIMD first. Syndrome-region certification now
supports unseen orderings derived from parsed input. The final whole-body match
checks a newly derived certificate instead of selecting a schedule catalog.
Broader physics, wire-layout, and protocol-skeleton coverage remain open.

## Bounded independent studies

1. **Pauli-proxy / Clifford-stabilizer reduction.** Read the actual applicability
   conditions and output contract. Test the smallest exact reduction on our
   fixed-fault f3/f5/f7 inputs, including hooks and noisy growth. Distinguish
   protocol-outcome equivalence from physical-state or arbitrary-probe
   equivalence. Measure all per-history preprocessing when comparing timings.
   Stop at a correctness-backed proceed/defer/reject result; a new production
   trajectory interpreter requires a separate architectural decision.
2. **Clifford-frame improvement plus exact tensors.** First measure whether
   better coordinate orders or Clifford frames reduce exact residual ranks on
   noisy trajectories, including five-round coherent memory as an independent
   workload. Include frame-selection cost. This is a feasibility screen, not
   permission to build a full MPS backend or truncate bonds.

Previously measured product components and ordinary per-shot recompilation
did not justify proceeding for the user's target. These do not rule out a
different frame or a restricted proxy. Short coherent sums are partly explored
by the current method; direct noise-averaged probability contraction remains a
separate, unstarted aggregate-output direction.
