# Reusing optimization of the fixed preparation

Date: 2026-10-09. Branch: `codex/bt27-fault-specialization`.

## Result and scope

Optimizing the fixed preparation once reduces complete per-shot cost on the
noisy factory from 37.68 to 21.10 ms, D from 15.79 to 4.50 ms, and E from 12.05
to 2.95 ms. These are paired measurements of 128 fresh trajectories per input.
All three keep active width 8-9. The reduction rules and single-rotation
carrier are unchanged; no additional non-Clifford input representation was
introduced.

This is reuse of an optimized state-preparation circuit. Parsing, tracing,
remaining optimization, planning, and executable construction still happen
for each shot. It is neither a fault-history cache nor reuse of a complete
sampling plan. All work remains in an opt-in research host before unchanged
native execution, with no public API or `src/clifft` changes.

## Why this work can be shared

The existing regional analyzer produces the following structure:

1. A fixed preparation and fixed non-Clifford remainder.
2. Clifford corrections depending on sampled faults and preparation records.
3. A fixed physical encoder, conditional Pauli offsets, and preserved records.
4. The complete conditional continuation.

Only the first part is optimized and exported once. Its equivalence is for
the accepted all-zero circuit-entry state. The resulting physical state can
then receive arbitrary trajectory-dependent corrections and continuation.
No assumption about an all-zero internal boundary state is made.

The new opt-in native tool, `export_optimized_prefix`, runs the existing four
HIR passes on an unmeasured deterministic preparation. It exports the remaining
signed Pauli T rotations **and the complete final physical Clifford frame**.
Keeping the latter is essential: the optimized rotations alone do not specify
the physical state to which the continuation must be attached. The host uses
Stim to synthesize that frame once and renders an equivalent physical circuit.
Measurements, hidden resets, stochastic sites, and unsupported HIR operations
are rejected by the exporter.

`ReusablePhase` replaces exactly this fixed prefix in the existing frontend's
conditional output. The shot retains its original faults, sampled preparation
records, corrections, annotations, and tail. A checked prefix match prevents
silently attaching the wrong interface.

For a Clifford-only continuation, the per-shot pipeline omits the phase pass
after this shared optimization. The other passes and width check still run.
If the continuation contains further non-Clifford operations, the prefix can
still be reused, but the full phase pass is retained. Existing Clifford exits,
one-rotation carrier paths, and initial unsupported inputs use the preceding
frontend unchanged. This conservative choice is not a claim that skipping
the phase pass would change circuit semantics; it protects the useful
optimization opportunities and width reductions in those paths.

## Profiling and paired comparison

Fine stage timers were followed by a native `perf` profile of one complete
conditional noisy-factory circuit, repeatedly compiled 256 times on CPU zero.
The inclusive samples attribute 44.22% to `PhasePolynomialPass`, 21.47% to
sampling planning, and 10.57% to tracing. Rotation simplification, squeezing,
and peephole fusion account for 7.53%, 6.76%, and 5.27%, respectively. These
inclusive percentages are not additive with their nested callees. Prominent
self costs include Pauli commutation, dormant-subspace intersection, coordinate
conversion, and phase-map manipulation. The evidence supports removing
repeated work across stages before tuning one inner kernel.

The [profile summary](compiled-prefix-profile.json) retains symbol percentages,
machine/event details, command, and source/binary hashes; its exact
[conditional input](compiled-prefix-profile.stim) is also retained. The event
was `cpu-clock:u` at 499 Hz with DWARF call stacks; no lost samples were
reported. Raw profiler data remains temporary rather than checked in.

The [paired timing driver](../../tools/profile/study_compiled_prefix_reuse.py)
uses the same fresh fault histories, preparation seeds, and sample seeds in
four modes. It rotates their execution order each shot. All four use the same
existing frontend setup; the shared export is timed separately.

| Mode | Fixed preparation | Per-shot phase pass |
| --- | --- | --- |
| Fresh | Original | Yes |
| Reuse, full passes | Optimized once | Yes |
| Reuse | Optimized once | Only when required by the conservative route |
| Skip only | Original | No |

Times include fault drawing, conditional rewriting, parsing, tracing,
optimization, width inspection, lowering, and one-shot sampling. Parity audits,
input generation, exact probability checks, and setup are outside the timed
shot. Validation jobs ran concurrently on other pinned CPUs. This is a short
local comparison, not a precise throughput or scaling guarantee.

| Input | Fresh ms | Reuse, full passes ms | Reuse ms | Speedup | Reuse width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noisy factory | 37.680 | 23.907 | 21.100 | 1.79x | 8-9 |
| D | 15.792 | 5.720 | 4.502 | 3.51x | 8-9 |
| E | 12.048 | 3.749 | 2.955 | 4.08x | 8-9 |
| BT27 direct X | 5.081 | 4.624 | 3.725 | 1.36x | 4-7 |
| BT27 scored | 3.309 | 3.276 | 3.273 | unchanged route | 0 |
| Cultivation d3 | 2.886 | 2.879 | 2.889 | unchanged route | 1 |
| Cultivation d5 | 10.968 | 11.012 | 10.990 | unchanged route | 10 |
| Noncommuting control | 0.538 | 0.526 | 0.527 | unchanged route | 4 |
| Unsupported rotation | 0.0362 | 0.0360 | 0.0371 | unchanged route | 1 |

On factory/D/E, fixed preparation optimization changes 216 T gates to 33.
Its additional setup cost, including subprocess export and physical Clifford
synthesis, is 54.3, 19.0, and 13.8 ms respectively. Total frontend setup is
0.773, 0.212, and 0.195 seconds. The extra work therefore amortizes after only
a few shots in this run, although it does not eliminate the earlier setup.

The full-pass reuse ablation already gives substantial savings. Omitting the
phase pass without optimizing the preparation first leaves width 16-20 on
every one of the 384 factory/D/E attempts. These attempts are recorded as
over the width-twelve budget and never densely allocated or counted as
completed shots. Thus simply removing the phase pass does not reproduce the
useful result there.

BT27 direct X behaves differently: its fixed preparation changes only 24 T
gates to 23, and skip-only already takes 3.662 ms at the same observed widths.
Most of its gain comes from omitting a pass that adds little on these
trajectories. The new preparation reuse is not needed for scored BT. Skipping
the pass on cultivation d3 loses its width-one result and leaves width four;
the actual reuse mode correctly retains the previous pipeline.

All 1,152 actual reuse attempts complete within the width budget, as do 1,152
fresh attempts. Another 102 identity, selected single-outcome, and selected
weight-2/5/12 stress histories stay within budget when inspected. These finite
observations are not a width bound over all fault histories.

The process high-water mark is 212.5 MiB across all inputs/modes, including
Python, Aer imports used by the probability checker, setup, and retained
representations. It is not an isolated per-mode memory comparison. The
artifact's `exporter_peak_bytes` is `RUSAGE_CHILDREN`: it also includes
incidental child commands and can include inherited pre-exec parent memory.
Its identical 212.5 MiB value does not measure the exporter's native working
set. A controlled memory comparison remains open.

## Correctness checks

- **Independent complete small states:** 24 randomized three-qubit Clifford+T
  preparations compare exported output with Aer density matrices. Maximum
  entry error is below `2.9e-16`.
- **Complete small instruments:** 14 original/renamed circuits, 24 fault
  histories, and 38 enumerated branches compare every visible-record/physical
  density block with the original source. These include noncommuting tails,
  complex phases, readout and correlated noise, feedback, reset, later magic,
  and unchanged fallbacks. Maximum density error is below `1.3e-16`; native
  record-law error is below `4.5e-16`. The actual candidate compilation skips
  the phase pass only where the reuse route permits it.
- **Independent D/E references:** the preceding rational-reference validator
  runs with the reused preparation as the candidate. All 50 manifest queries,
  3,422 complete coarse-syndrome support probabilities, 550 selected raw-record
  probabilities, and 368 selected state-probe probabilities pass across the
  retained 110 histories. Of the candidate query compilations, 230 use reuse
  without a phase pass and 92 retain the prior pass route.
- **Factory conditional reference:** 160 complete-record queries match the
  original physical circuit conditioned on the same preparation records.
  The reference still uses full HIR optimization; it does not use the reused
  prefix. Maximum error across the large validator is below `7e-18` absolute
  and `3.9e-14` relative.
- **Timing-run checks:** 128 selected full-record probabilities from the four
  eligible inputs agree between fresh and reuse. On the five ineligible
  inputs, 640 trajectories have identical emitted source and sampled records
  between the two routes. Large full-record replay is restricted to the
  eligible cases: cultivation d5's 92 hidden reset records exceed the bounded
  replay oracle. Every completed sample is also checked against the original
  detector/observable parities.

The small instruments use an independent physical-state oracle. The complete
D/E coarse laws are marginals, and the large raw/state/factory checks are
selected queries. Neither these checks nor the 128-shot timing samples prove
the entire large noisy law or establish a rare accepted-error rate. Benchmark
family and physical-noise limitations from the
[preceding assessment](FACTORY_FRONTEND_ASSESSMENT.md) still apply.

Artifacts: [small validation](compiled-prefix-validation.json),
[large validation](compiled-prefix-factory-validation.json), and
[paired timings and stress histories](compiled-prefix-study.json).

## Remaining cost and assessment

| Remaining reuse stage | Factory ms | D ms | E ms |
| --- | ---: | ---: | ---: |
| Fault draw and rewrite | 4.632 | 1.542 | 1.303 |
| Parse and trace | 4.571 | 0.962 | 0.582 |
| Remaining passes and width inspection | 4.808 | 0.707 | 0.407 |
| Lower | 6.759 | 1.184 | 0.581 |
| Sampling call | 0.066 | 0.030 | 0.024 |

The fixed preparation's repeated phase optimization was a useful first target.
The experiment supports reusable compiled structure without requiring repeated
fault histories or more surviving rotations. It does not yet avoid full
physical source reconstruction and planning. Sampling itself remains a small
fraction of total cost.

A next bounded reuse study should target the already identified fixed versus
changing interface: retain more parsed/traced preparation structure or reduce
the physical work presented to the planner, while validating the complete
conditional state and records. Reusing the ideal sampling plan with sign
changes alone remains invalid: earlier BT examples have fault-dependent
measurement-law rank. No such shortcut was used here. A change to the native
execution/planning lifecycle still needs a separate architecture proposal.

Broader non-Clifford carry is deferred. Ordinary compile-once Clifft remains
the appropriate comparison on already-small and ideal inputs; the preceding
ideal-factory result is orders of magnitude faster than this per-shot host.
This experiment does not add automatic backend selection.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#fixed-preparation-compilation-reuse).
