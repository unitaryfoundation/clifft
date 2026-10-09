# One surviving Pauli rotation across a measured bridge

Date: 2026-10-09.

## Result and assessment

A surviving non-Clifford input no longer necessarily blocks the next reduction.
The new bounded host carries one Pauli T rotation on a stabilizer reference
through a supported Clifford instrument. It reaches cultivation's next phase
region while retaining its quantum state, visible records, hidden reset
mixture, and sampled faults. The carrier remains one rotation on both the
15-qubit d3 and 42-qubit d5 sources.

The useful complete-circuit result comes from the existing HIR optimizer after
this transport. Cultivation d3 drops from active width four to one. Cultivation
d5 stays at ten. Attempting a second reduction with the research phase
synthesizer adds work without improving those results: its synthesized phase
has 39 T gates and is rejected by a conservative cost check. This is a
representation/synthesis issue, not evidence that the physical state acquired
39 independent magic resources.

This establishes a limited entry contract and a real width improvement. It
does not yet extend practical executability on an otherwise inaccessible
application: ordinary Clifft already handles these cultivation widths. The
Python host is substantially slower than Merlin, even when the unsuccessful
second synthesis is skipped. This is an assessment point before expanding the
carrier to multiple rotations or further cultivation regions.

## State representation and exact rules

The [host](../../tools/profile/one_core_phase_specialization.py) starts from the
existing first regional reduction and requires exactly one remaining T or
T-dagger gate. Its conditional state has the form

`R_P(epsilon * pi/4) |s>`, where `R_P(theta) = exp(-i * theta * P / 2)`,

with a Hermitian Pauli string P, epsilon equal to +1 or -1, and a stabilizer
state |s>. T differs from this rotation by an irrelevant global phase. The
host stores one Pauli string and a Stim tableau, not a dense physical state or
an enumerated collection of gate patterns.

- Clifford gates update the reference tableau and conjugate P. Record-based
  Pauli feedback uses the actual recorded bit, including readout inversion.
- A Pauli measurement Q can be sampled on the stabilizer reference when Q
  commutes with P. Commuting the rotation through each projector proves that
  both its probability and the conditional state are correct. A noncommuting
  measurement remains live in the ordinary continuation.
- Before a reset, the host seeks an equivalent representative P' that is
  identity on the reset wire. It may multiply P by a reference stabilizer S
  only when S commutes with P. Then `P'|s> = P|s>` and P' remains Hermitian.
  A binary linear solve finds such S or rejects this crossing. The reset is
  sampled with a valid hidden-outcome decomposition; no outcome is discarded
  from the ensemble.
- If P becomes a stabilizer of the reference state, its rotation is only a
  global phase and is removed.
- At the next T gate, another linear solve tries to give P a Z-only
  representative. A successful representative becomes a CNOT parity gadget
  with one T/T-dagger gate, adjacent to the next phase region. If that is
  impossible, the exact general Pauli rotation is emitted with its necessary
  basis changes and the ordinary continuation is retained.

The commutation restriction on S matters. Multiplying by an anticommuting
stabilizer can yield an imaginary Pauli product; treating that as a Hermitian
phase rotation would silently change the state. An explicit control rejects
this tempting but invalid shortcut.

The host reconstructs the conditional stabilizer state, applies its carried
rotation, restores sampled visible records with MPAD, and appends all remaining
operations and annotations. Unsupported operations and unsuccessful crossings
stop optimization, not execution of the remaining circuit. Hidden reset
outcomes affect the conditional state but add no visible records.

Every shot draws the full original categorical Pauli history once. Only
supported reference measurements and resets are then sampled by the host.
There is no history cache, postselection, or fault-weight truncation. Forced
branches exist only in exhaustive validation, where both outcomes and their
probabilities are retained. Explicit shot seeds control the host and native
sampling streams.

The specified all-zero circuit input remains the assumption. Internal entries
are the actual conditional states, not fresh all-zero states. All tableau
updates, representative solving, reconstruction, and compilation occur in the
Python research host before ordinary native execution. There is no production
API, executor continuation, runtime planner, or Stim modification.

## Cultivation and the second analysis

Both cultivation sources pass thirteen resets and seven visible measurements
before reaching the next T layer. A literal transported Pauli would overlap
one of the later resets. Multiplication by a commuting stabilizer removes that
overlap and allows the carrier to continue. The study retains the before/after
Pauli representatives, sampled records, and hidden random decisions.

The next regional analysis uses eleven support variables in both d3 and d5.
It has one carried input rotation plus seven incoming physical T-dagger gates.
Its current polynomial synthesis emits 39 T gates. In fresh d3 histories, the
complete unoptimized candidate reaches width seven or eight instead of four.
In d5 it remains at ten but increases the T count. These candidates are not
used for execution.

The host compares the candidate and transported sources by complete raw
active width and then T count. It retains the smaller source. Both sources
encode the same already-sampled branch. Returning the original unsampled
prefix upon rejection could resample a correlated outcome and bias the law;
the fallback deliberately retains the transported conditional state.

The final compilation uses the existing HIR optimization pipeline on that
complete source. This is necessary to separate synthesis quality from what
the transport enables. On matched fixed histories and seeds:

| Complete circuit | Ordinary noisy width | First reduction plus existing optimization | Transport plus existing optimization |
| --- | ---: | ---: | ---: |
| Cultivation d3 | 4 | 4 | 1 |
| Cultivation d5 | 10 | 10 | 10 |
| 15-to-1 scored | 5 | 0 | 0 |
| 15-to-1 direct X | 5 | 1 | 1 |

Each cultivation comparison includes eleven fixed histories and two seeds per
history. The complete optimized d3 source retains one T; d5 retains 39. The
same widths and T counts hold in 256 fresh histories per case. These are
observed finite-case results, not a new all-history width theorem.

The [full experiment](one-core-phase-specialization.json) attempts the second
research synthesis. A matched [carrier-only ablation](one-core-carrier-only.json)
skips it and directly uses existing optimization. Both yield the same
cultivation record hashes under the retained seeds and the same widths. The
second research synthesis therefore adds no benefit in these cultivation runs.
Scored 15-to-1 and two constructed controls also reach Clifford outputs; their
simplifications were already available through other paths and are not new
application wins.

One small control isolates the synthesis problem: two input T gates become
thirteen in the candidate, raising raw width two to four. It has two equally
likely earlier measurement outcomes; both rejected branches retain the correct
joint record/state law. Ordinary optimization can recover width one from
either representation. Expansion here is an algorithmic choice, not intrinsic
growth of the represented quantum resource.

## Correctness and regression checks

The [validator](../../tools/profile/validate_one_core_phase_specialization.py)
enumerates the initial preparation law and every random reference measurement
and hidden reset outcome of each bounded case. Each branch carries its exact
probability. Independent Aer calculations provide an unnormalized physical
density matrix for every visible record. Comparisons cover the transported
source, the attempted second reduction, and the selected fallback or reduction.
Clifft replay separately checks complete record probabilities both before and
after the existing optimizer.

The retained [validation artifact](one-core-phase-validation.json) records:

- 36 small circuits, including wire renamings, 44 fixed-history checks, and
  80 enumerated branches. Maximum matrix error is below 2.8e-16, and both raw
  and optimized record-probability errors are below 4.5e-16.
- All 16 histories of a four-site binary-noise witness and all 32 histories
  of a two-qubit categorical channel plus readout flip. Their 32 and 64
  conditional branches retain total noise probability one. Weighted
  instrument-error bounds are below 1.1e-16.
- Explicit noncommuting-measurement, overlapping-reset, non-diagonal-entry,
  unsupported-operation, and imaginary-product controls. Dropping the carried
  phase produces a substantial quantum-state error detected by the oracle.
- An expanded-synthesis control checks rejection on both sampled records.
- 2,048 fresh host shots agree with the complete Aer record law under a
  bounded sampling check; maximum frequency discrepancy is about 0.0113.
  Explicit seeds reproduce the first eight emitted sources.

The large cultivation runs compare record/parity moments with Merlin and audit
all original detector/observable parities. These 256-shot checks are bug
checks, not full distributional equivalence or rare-event estimates. The
large-state correctness argument rests on the exact transport identities and
their checked preconditions, with independent exhaustive instrument tests on
small witnesses; no dense 42-qubit reference state was constructed.

A one-wire edge case also exposed a host issue: Stim counts literal `MPAD 1`
as an extra wire. The shared analyzer now restores the specified physical
tableau size after executing the prefix, removing that unused |0> wire. The
new measurement-erasure control covers both bit values. The complete earlier
shared-phase validator was rerun: 28 small cases, four distillation variants,
256 padding controls, and 38 BT27 conditional laws pass. Its retained
[regression artifact](one-core-shared-regression.json) has current source hashes.
Stim itself is unchanged.

## Cost and the next decision

All timings include drawing faults, prefix/reference sampling, transport,
reconstruction, final optimization, width inspection, lowering, and complete
native sampling. Initial shared analysis is separate. Audits and stress probes
are excluded. Runs use one CPU and 256 fresh shots per mode. The simpler
ablation avoids the unused second research analysis but still recompiles each
conditional complete source.

| Circuit | With second synthesis, ms/shot | Carrier plus existing optimization, ms/shot | Merlin, ms/shot |
| --- | ---: | ---: | ---: |
| Cultivation d3 | 8.17 | 2.93 | 0.044 |
| Cultivation d5 | 22.33 | 11.06 | 0.720 |
| 15-to-1 scored | 1.86 | 0.684 | 0.016 |
| 15-to-1 direct X | 0.706 | 0.713 | 0.016 |

Merlin figures in this table are from the carrier-only run. Setup for the
carrier-only d3/d5 hosts is about 0.012/0.056 seconds. Even without the wasted
second synthesis, the host is about 66 times slower than Merlin on d3 and
15 times slower on d5. Reference transport/reconstruction dominates d3, and
it plus ordinary recompilation dominates d5. These costs are material, not
negligible noise around a speed improvement.

The representation does stay bounded through the bridge: one Pauli rotation,
a 15- or 42-qubit stabilizer tableau, and no growing dense non-Clifford state.
That does not make the host constant-space or constant-time. The setup/storage
growth found in the preceding BT81 study is also not resolved here.

The next decision should be a checkpoint, not another cultivation-specific
rewrite. The demonstrated mechanism is a useful candidate building block:
conditional transport can expose structure to an existing algebraic optimizer.
The more elaborate second synthesizer is unnecessary in the measured cases.
Before supporting multiple surviving rotations or more regions, identify a
complete workload where this entry extension changes practical executability,
and bound the cost of transporting that input. Cultivation can remain a
correctness fixture, while an already fast backend handles it in practice.
Any production integration would require a separate architectural discussion.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#one-pauli-rotation-through-a-measured-bridge).
