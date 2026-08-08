# Backend comparison triage findings

## Scope

This is a measurement-only triage for the legacy SVM and experimental
symbolic-coordinate backend. No production implementation was changed.

- Clifft: `3fdafa411e8e4eca812ff17df2a8f30c584fdc03`.
- Paper corpus: `db7dc9f13a2c2854690e92390c779048a1ac1400`.
- Canonical measurements: native Release Python build, one thread, processes
  pinned to CPU 3, three balanced ABBA/BAAB blocks, six samples per backend.
- Profiling: `-O3 -DNDEBUG -g -fno-omit-frame-pointer`, native ISA and
  fast-math preserved; `/usr/lib/linux-tools-6.8.0-106/perf`, cycles in user
  space, frame-pointer call graphs, zero lost samples.
- The profiling-build surface ratio was 13.9x versus 10.7x in the canonical
  Python baseline. This preserves the large-regression classification; the
  canonical Release measurements remain the reported performance numbers.

The surface A/A paired B/A ratio had median 0.9957, range 0.9652-1.0080,
and MAD 0.0110. No cell was expanded beyond three blocks because the material
gaps were far outside this envelope.

## Triage baseline

`symbolic/legacy` above one means symbolic is slower. Times are medians for
the fixed shot count in the protocol.

| Case | Legacy | Symbolic | Ratio | Absolute gap |
|---|---:|---:|---:|---:|
| surface d7 r7 aggregate | 0.642 s | 6.860 s | 10.683x | +6.218 s |
| cultivation d3 aggregate | 0.687 s | 6.991 s | 10.118x | +6.305 s |
| cultivation d3 forced k=0 | 0.553 s | 6.899 s | 12.541x | +6.346 s |
| noncomputational d17 r5 low leak | 0.400 s | 3.951 s | 9.850x | +3.551 s |
| coherent d3 r3 aggregate | 0.832 s | 1.326 s | 1.592x | +0.495 s |
| regime k=12, L=512 | 0.924 s | 1.179 s | 1.268x | +0.255 s |
| distillation aggregate | 2.139 s | 0.968 s | 0.454x | -1.172 s |

Discard and conditional logical intervals overlap between backends for every
applicable cell. Checksums confirm that outputs were consumed but are not used
as a cross-backend equality assertion.

Symbolic compile medians were 0.451 s for surface, 0.0195 s for cultivation
d3, and 0.0746 s for k=12/L=512, versus 0.00922 s, 0.00100 s, and 0.00420 s
for legacy respectively. Surface is therefore both the largest compile ratio
(49.0x) and the largest absolute compile gap (+0.442 s).

## Hardware counters

Counters cover three executions per arm in the profiling build. The generic
`cache-misses` event returned zero on this VM and is excluded rather than
interpreted.

| Case | Cycle ratio | Instruction ratio | Branch ratio | IPC legacy/symbolic |
|---|---:|---:|---:|---:|
| surface | 13.39x | 17.74x | 14.67x | 2.77 / 3.67 |
| cultivation d3 | 12.59x | 11.99x | 12.35x | 3.95 / 3.76 |
| noncomputational | 10.04x | 13.32x | 13.21x | 3.09 / 4.11 |
| coherent d3 r3 | 1.57x | 1.92x | 2.22x | 4.51 / 5.50 |
| regime k=12, L=512 | 1.34x | 2.04x | 6.41x | 3.30 / 5.02 |
| distillation | 0.56x | 0.93x | 0.91x | 2.13 / 3.56 |

Except for cultivation, the symbolic paths sustain higher IPC. The slow paths
retire materially more instructions; the profiles below distinguish affine
expression work from active-state kernels. Generic cache events on this VM are
not used for attribution: `cache-misses` returned zero, and
`cache-references` did not behave consistently enough across the two engines.

## Attribution

### Nonzero-active-width execution

These profiles were added after the initial k=0 and noncomputational pass so
that an independent reviewer can assess the active-state priorities. Each arm
has a matching `perf stat` run over three executions and a cycles call graph
over one execution. All eight call graphs recorded zero lost samples.

| Case | Max width | Symbolic/legacy | Symbolic top self costs | Legacy top self costs |
|---|---:|---:|---|---|
| cultivation d3 | 4 | 10.118x slower | expression evaluation 89.90%; rotation 2.75% | dispatch/inlined execution 76.11%; array CNOT 9.15%; phase waterfall 8.82% |
| coherent d3 r3 | 7 | 1.592x slower | rotation 66.07%; measurement probability 9.07%; expression evaluation 6.47%; collapse 4.14% | dispatch/inlined execution 51.50%; phase waterfall 24.37%; array CNOT 16.09%; active measurement 3.28% |
| regime k=12, L=512 | 12 | 1.268x slower | rotation 53.20%; expression evaluation 15.68%; measurement probability 8.51%; collapse 7.61%; promotion 2.13% | fused array U4 71.80%; dispatch/inlined execution 19.74%; phase waterfall 3.77% |
| distillation | 5 | 2.203x faster | expression evaluation 62.75%; variant dispatch 7.24%; executor loop 5.38%; measurement probability 3.71%; rotation 2.47% | dispatch/inlined execution 94.70% |

The legacy `execute_internal()` symbol contains dispatch and inlined frame and
small-k work, so its self percentage cannot be subdivided further without an
instrumented build. It is reported as a limitation, not assigned wholly to
dispatch.

#### Cultivation d3

The symbolic plan has 112 actions, 2,492 symbols, 31,278 affine terms, maximum
width four, 39 predicted dense passes, and 499 predicted coefficient visits.
The legacy module has 344 bytecode instructions at the same peak rank.

At 500,000 shots, the symbolic executor exposes about 15.6 billion affine-term
visits. `Executor::evaluate()` accounts for 89.90% of cycles, while active
rotation is only 2.75%. The 12.59x cycle and 11.99x instruction ratios are
therefore an expression-stream result despite the nonzero active width. The
ideal Amdahl ceiling from removing expression evaluation is 9.90x.

This case does not justify a dense-kernel change by itself. It is evidence for
a hybrid execution problem: preserve per-shot active-state work while reducing
the repeated affine stream around it. A k=0-only packed path would not cover
this workload.

Exact benchmark: `cultivation_d3_aggregate`, 500,000 attempted shots, all
detectors postselected, normalized references, counts-only survivor output.

#### Coherent d3 r3

The symbolic plan has 156 actions, only 369 affine terms, maximum width seven,
93 predicted dense passes, and 5,789 predicted coefficient visits. The legacy
module has 523 bytecode instructions at peak rank seven.

This is the cleanest real active-state profile: expression evaluation is only
6.47%, while rotation, measurement probability, collapse, and promotion sum to
81.0% of self samples. `apply_rotation()` alone is 66.07%, for a 2.95x ideal
Amdahl ceiling. Source-line samples concentrate in the generic complex
arithmetic, Pauli phase/popcount, and pair traversal in `sampling/kernels.cc`.
The legacy side concentrates in its AVX-512 execution loop, phase waterfall,
and array CNOT kernels.

The measured 1.59x gap is consistent with an active kernel/SIMD or operation
specialization question, not expression packing. The profile identifies the
rotation kernel as the first place to inspect; it does not by itself choose
between arithmetic specialization, pairing/fusion, or vectorization.

Exact benchmark: `coherent_d3_r3_aggregate`, 300,000 attempted shots, all
detectors postselected, normalized references, counts-only survivor output.

#### Regime k=12, L=512

The symbolic plan has 566 actions, 18,883 affine terms, maximum width 12, 26
predicted dense passes, and 38,911 predicted coefficient visits. The legacy
module has 2,208 bytecode instructions at peak rank 12.

The symbolic profile is mixed but active-dominated: rotation is 53.20%, active
measurement probability/collapse/promotion total 18.25%, and expression
evaluation is 15.68%. On the legacy side, the fused array-U4 kernel is 71.80%.
Symbolic retires 2.04x as many instructions but only 1.34x as many cycles due
to higher IPC. Eliminating rotation entirely has a 2.14x ideal ceiling;
eliminating expression evaluation alone has only a 1.19x ceiling.

This is the strongest evidence for reviewing symbolic rotation fusion or
specialized dense kernels against legacy's U4 path. Because the fixture has a
long output-relevant Clifford stream, it should not be read as a pure dense
microbenchmark.

Exact benchmark: `regime_k12_l512`, 20,000 attempted shots, no postselection,
normalized references, counts-only survivor output.

#### Distillation

The symbolic plan has 225 actions, 12,922 affine terms, maximum width five,
20 predicted dense passes, and 315 predicted coefficient visits. The legacy
module has 2,040 bytecode instructions, including 1,603 frame CNOT/CZ
instructions, at peak rank five.

Expression evaluation is still the largest symbolic self cost at 62.75%, yet
symbolic is 2.20x faster and uses 0.56x the cycles. The legacy profile is
94.70% inside its monolithic execution symbol. The plan/action-volume
difference is consistent with the symbolic planner avoiding much of the
legacy frame stream, although the monolithic legacy symbol prevents a precise
dispatch-versus-inlined-operation split.

This winning control is important: expression evaluation can dominate a
symbolic profile without implying that the backend loses. End-to-end work
elimination can outweigh the scalar expression cost.

Exact benchmark: `distillation_aggregate`, 200,000 attempted shots, all
detectors postselected, normalized references, counts-only survivor output.

### Counts-only surface execution

The surface plan has maximum active width zero, 819 actions, 20,977 symbols,
214,390 affine-expression terms (maximum 1,176 in one expression), and no
predicted dense passes. At 200,000 shots this exposes about 42.9 billion term
visits.

- Dominant cost: `sampling::Executor::evaluate()` is 87.24% of sampled cycles.
- Likely mechanism: the scalar executor replays long affine detector, record,
  and observable expressions independently for every shot. This is an
  expression/packed-execution gap, not an active-width or dense-kernel gap.
- Amdahl ceiling: removing all expression evaluation would be at most a 7.84x
  speedup of the symbolic execution. That idealized bound still does not
  guarantee legacy parity because the remaining 12.76% is material.
- One plausible direction: a guarded packed counts-only path
  that evaluates Boolean symbols and affine outputs across multiple shots,
  starting with maximum-active-width-zero plans without instruments or
  expectation probes. The corrected SymFT attribution independently identifies
  cross-shot packed evaluation as the k=0/Clifford-stream mechanism.
- Exact acceptance benchmark: `surface_d7_r7_aggregate`, 200,000 attempted
  shots, all detectors postselected, normalized references,
  `sample_survivors(keep_records=False)`. Preserve discard/logical statistics
  and rerun cultivation d3, distillation, and k=12/L=512 as guards.

### Low-leak noncomputational execution

- Dominant costs: `compile_sampling_continuation()` is 57.19% inclusive,
  including `plan_sampling()` at 55.69%; `Executor::run_shot()` is 40.10%
  inclusive. The largest self costs are measurement probabilities (22.01%),
  Stim tableau scatter (16.67%), instrument no-fire application (11.94%),
  `memmove` (9.34%), and measurement collapse (6.70%).
- Likely mechanism: transition traps create shot-specific continuations whose
  symbolic plans are expensive to build, followed by scalar measurement and
  instrument state work. This is both component/planner and executor work.
- Amdahl ceiling: eliminating continuation compilation alone is at most 2.34x,
  insufficient to close the measured 9.85x gap; executor work must also be
  addressed.
- Smallest plausible follow-up: measure continuation-topology reuse before
  considering a continuation-plan cache. Do not implement a cache from this
  profile alone. Instrument kernels are a separate follow-up for review.
- Exact acceptance benchmark: `noncomp_d17_r5_low_leak`, 1,000 shots, the
  d17/r5 fixture and p=0.01 S-hook model from the protocol, with complete raw
  results consumed.

### Surface symbolic compilation

Twenty native compile iterations took 9.254 s (0.463 s each), consistent with
the canonical 0.451 s median.

- Dominant cost: `plan_sampling()` is 98.48% inclusive;
  `transform_future_operations()` is 69.82% inclusive. Major self costs are
  Stim tableau scatter (34.63%), conditional-Pauli propagation (18.12%), Stim
  bit access (12.16%), affine XOR (5.98%), and `memmove` (5.17%).
- Likely mechanism: repeated transformation of the remaining operation suffix
  and growth/copying of affine dependencies across measurements.
- Amdahl ceiling: eliminating all suffix transformation is at most 3.31x for
  symbolic compilation.
- Smallest plausible direction: reduce repeated suffix tableau transforms and
  affine-expression copying, starting with the maximum-active-width-zero
  planner case. Compile is 0.451 s versus 6.860 s of execution for the measured
  surface call, which bounds its importance for that one-shot workflow.
- Exact acceptance benchmark: compile the surface fixture with default HIR
  passes, all-detector postselection, reference normalization, and the
  symbolic planner; compare against the 0.451 s canonical median.

## Review questions and decision gate

No production change is recommended for implementation yet. The expanded
profiles separate four questions for independent review:

1. Cultivation d3 is an affine-expression/stream bottleneck even though its
   active width reaches four. Would a hybrid packed stream around per-shot
   active work be architecturally appropriate?
2. Coherent d3 r3 and k=12/L=512 are active-kernel dominated. Do the profiles
   justify first isolating `apply_rotation()` arithmetic, SIMD, and fusion
   against the legacy phase/U4 paths?
3. Distillation wins through lower end-to-end work despite spending most of
   its symbolic cycles in expression evaluation. Which plan-volume metric
   should gate any packed or specialized path?
4. Noncomputational sampling combines continuation-planner and active executor
   costs; continuation caching alone cannot close its gap. What additional
   reuse evidence is required before component work?

The earlier k=0 packed-counts suggestion is retained as a measured opportunity,
not the priority decision. The next action should be chosen only after review
of these hotpaths. Any approved focused change should rerun the seven triage
cases before the full validation matrix.

## Profile artifacts

Raw counter CSVs and `perf.data` files are under
`tools/bench/backend_comparison/results/20260807T202928Z/profiles/`. The
nonzero-width files use the case prefixes `cultivation_d3`,
`coherent_d3_r3`, `regime_k12_l512`, and `distillation`, with `legacy` and
`symbolic` arms. Counter runs contain three executions; call graphs contain
one. Call graphs used `perf record -F 999 -e cycles:u -g --call-graph fp` and
recorded zero lost samples.
