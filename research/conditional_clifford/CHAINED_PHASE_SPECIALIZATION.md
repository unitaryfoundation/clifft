# Chaining reductions through certified Clifford exits

Date: 2026-10-09.

## Result and scope

The focused chaining experiment succeeds. Two or three eligible phase regions
can be reduced in sequence, with noncommuting Clifford gates, measurements,
resets, feedback, and sampled Pauli faults between them. Each later analysis
uses the actual conditional state and retains the previously observed records.

The certificate is conservative: another analysis is attempted only when the
previous region's reduced state preparation contains no non-Clifford remainder.
A partial reduction is still retained, but its continuation then runs normally.
This supports a chain of Clifford exits followed by an optional non-Clifford
remainder. It does not support entering a new region while an uncertified
non-Clifford state survives.

The result addresses the accepted goal of extending practically simulable
circuits. On the constructed 48-qubit example, ordinary noisy compilation has
active width 44 and one reduction still leaves width 46. Chaining makes the
complete circuit Clifford. A variant with an additional non-Clifford tail ends
at width 1 in the tested histories. These are structural demonstrations, not
new application benchmarks.

The [host](../../tools/profile/chained_phase_specialization.py) adds no
production API, optimizer pass, or executor instruction. The all-zero
circuit-entry assumption remains in force. Every conditional state is derived
from that input; an internal boundary is never assumed to be zero.

## Chaining contract

Construction shares the first
[regional analysis](REGIONAL_PHASE_SPECIALIZATION.md) across shots. A shot draws
one complete categorical Pauli-fault history at the original circuit locations.
The first step uses the shared responses, and its emitted continuation already
contains all the realized faults. Later steps cannot draw the noise again.

Each step:

1. Samples the current Clifford preparation and its stabilizer signs, including
   newly reached measurements. Previously observed records are deterministic
   MPAD entries. An explicit check requires their values and slots to remain
   unchanged.
2. Emits the reduced quantum-state preparation on the original physical wires
   and appends the complete remaining circuit.
3. Checks the constructive exit certificate: the algebraic synthesis produced
   no non-Clifford gates. Clifford corrections, affine encodings, offsets, and
   conditional preparation records then form a stabilizer preparation for the
   next analysis. The intervening supported Clifford operations preserve that
   property.
4. If certified, rebuilds the next regional analysis from this actual
   conditional source. A certified step must strictly remove source T gates,
   so it cannot loop over the same unchanged region.

A surviving non-Clifford remainder stops chaining while preserving that
reduction and the untouched continuation. Analysis limits and unsupported next
operations also retain the complete current source. A region budget, default
eight, bounds total specialization. A circuit with no initial eligible region
uses the original circuit with its sampled faults materialized.

Absence of T syntax alone does not certify that the entire continuation is
Clifford: arbitrary Pauli rotations may remain. Completion is separately
checked against the Clifford circuit language. An unsupported rotation remains
in the ordinary continuation.

The later preparation samplers receive fresh explicit seeds from the shot's
RNG. Reusing the default seed of a newly constructed sampler each shot would
repeat new outcomes and bias the law. The fault and quantum RNG streams are
separate in the timing study. No history cache, postselection, or fault-weight
truncation is used.

All of this occurs in the Python host before lowering the final specialized
circuit. The native executor receives one complete program. There is no native
mid-shot suspension or runtime topology planning inside its dispatch loop.

## Measurement-dependent second regions

One witness makes the entry dependence explicit. An initial measured bit
controls a diagonal Clifford correction produced by the first reduction.
After a Hadamard and further record-controlled operations, the same second
region has two different affine-support descriptions:

- With probability one half it becomes Clifford.
- With probability one half it retains one T gate.

Both probabilities are obtained by exact branch enumeration. Both branches are
kept and simulated. This tests that the second analysis uses its actual input
state rather than a fixed reference preparation or the first observed branch.

Other controls chain two and three parity-constrained phase regions. Each
later Pauli-product measurement establishes a new constraint, and its outcome
and subsequent feedback are retained. The algebra discovers the reductions
from the prepared state; the transformer contains no names or patterns for
these controls, code families, or encoders.

## Correctness evidence

The [validator](../../tools/profile/validate_chained_phase_specialization.py)
enumerates the conditional preparation law at each reached step. Branch
probabilities multiply down the chain and sum to one. For every visible record
string, an independent Aer calculation supplies an unnormalized physical
density matrix. Comparing these matrices checks both quantum coherences and
record/state correlations, for the specified all-zero input. Exhaustive
Clifft replay separately checks the complete visible-record probabilities.

The retained [validation artifact](chained-phase-validation.json) records:

- 24 small circuits, including 12 wire renamings, 41 selected fixed histories,
  and 100 fully enumerated chain leaves. These cover two/three reductions,
  branch-dependent support, noise on both sides of boundaries, resets,
  feedback, partial reductions, and fallbacks. Maximum joint record/state
  matrix error is below 2.8e-16 and record-probability error below 1.3e-15.
- All 16 positive-probability histories of a separate noisy measured-chain
  witness, with 64 conditional leaves. The weighted joint record/state error
  bound is below 2.4e-17. The original noise probabilities sum to one within
  floating-point tolerance.
- Exact checks with budgets of one and two regions on a three-region circuit.
  The remaining gates and measurements retain the same complete output law.
- A deliberate alteration of an old measurement record is rejected. A later
  129-variable preparation exceeds the existing 128-variable analysis budget
  and preserves exactly the one-step source. A non-T rotation is retained and
  correctly classified as an unsupported continuation for further analysis.
- Scored BT27 and scored 15-to-1 retain a non-Clifford first exit. On 22 and 32
  selected conditional sources respectively, the chain returns exactly the
  previously validated single-region source and never attempts a second entry.
  These are fallback checks, not new independent large-circuit equivalence
  proofs.
- 2,048 fresh shots of the three-region control agree with its complete Aer
  record law under a bounded sampling check. The largest frequency discrepancy
  is about 0.0083. Explicit seeds reproduce the initial eight specialized
  sources. This targets repeated sampler seeds and accidental resampling of
  old records; it is not a rare-error-rate study.

Preparation-branch enumeration reuses the existing normalized Stim flow oracle,
including its deterministic-MPAD correction. Aer supplies the independent final
joint state/record reference. None of these finite checks establishes the law
for every possible large-circuit noise history or arbitrary unknown input.

## Width and runtime

The [study](../../tools/profile/study_chained_phase_specialization.py) compares
ordinary noisy compilation, one regional reduction, and chaining on the same
source and stress histories. One-step and chained paths use matching initial
quantum seeds. Widths are inspected before lowering; the execution budget is
12. Larger ordinary and one-step programs are not allocated.

The new echo family has two groups of parity phases that cancel modulo eight,
separated by Hadamards, a measurement, and feedback. Pauli noise is distributed
through preparation, both phase groups, the intervening operations, and
readout. This is a constructed family used to isolate composability and width
growth. The protocol fallback sources use the same pinned Merlin revision,
`097380fac1a3968ca47925146e211fe990f4c396`, as the previous studies.

The table lists maximum observed widths; the measurement-dependent witness
also shows its range across fresh branches:

| Circuit | Ordinary noisy width | One reduction | Chained | Applied regions |
| --- | ---: | ---: | ---: | --- |
| Noisy measured parity chain | 1 | 1 | 0 | 2 Clifford |
| Measurement-dependent entry witness | 1 | 1 | 0 or 1 | 2, with branch-dependent remainder |
| Two echo regions, 6 qubits | 5 | 4 | 0 | 2 Clifford |
| Two echo regions, 24 qubits | 22 | 23 | 0 | 2 Clifford |
| Two echo regions, 48 qubits | 44 | 46 | 0 | 2 Clifford |
| Two echo regions and a non-Clifford tail, 48 qubits | 44 | 46 | 1 | 2 Clifford, then 1 partial |
| Scored BT27 | 33 | 9 | 9 | 1, then non-Clifford-entry fallback |
| Scored 15-to-1 | 5 | 1 | 1 | 1, then non-Clifford-entry fallback |

Each executable case also receives 32 fresh histories, with all visible outputs
and declared parities retained. The low-width results persist in those shots.
The noiseless two/three-region controls are already Clifford under ordinary
optimization, so they are correctness witnesses rather than speed targets.
The reset-after-magic and unsupported-rotation controls can also be handled
better by ordinary optimization; this host deliberately prioritizes a simple
supported-entry certificate over completeness or optimality.

Only the initial analysis is shared. Later support, phase, and fault-response
construction is repeated per shot, although the original faults are already
fixed. This is the dominant current cost:

| Constructed circuit | Initial setup, ms | Full fresh shot, ms | Later analysis within shot, ms |
| --- | ---: | ---: | ---: |
| Two regions, 6 qubits | 7.5 | 3.2 | 2.8 |
| Two regions, 24 qubits | 30.7 | 13.8 | 12.5 |
| Two regions, 48 qubits | 68.5 | 29.0 | 26.8 |
| Two regions plus partial tail, 48 qubits | 64.8 | 36.8 | 34.4 |

The 48-qubit pure-Clifford example encounters seven distinct later-entry
stabilizer bases in its 32 fresh shots. The tail variant encounters eighteen
across its later stages. These observations show why reusing one unconditional
entry description is insufficient; the current implementation simply rebuilds
each one, without assuming that caching would be useful.

Timing includes drawing faults, prefix sampling, all later analyses, rendering,
parsing, tracing, width inspection, lowering, and final sampling. Initial setup,
structural audits, output parity audits, and report generation are separate.
These measurements do not compare throughput against Merlin or establish
per-backend memory use. In the large positive examples, the useful comparison
is that the complete circuit now executes within the chosen width budget while
both previous paths exceed it.

## Remaining boundary and next assessment

Chaining through Clifford exits is now implemented and checked. The main
representational gap is a surviving non-Clifford input at the next candidate
region. The host does not continue through that state to search for a later
return to stabilizer form, even if a full reset would make this easy to certify. It
stops at the first unsupported exit and preserves the ordinary continuation.

Positive chaining examples in this step are constructed controls. The existing
BT and distillation protocol examples exercise fallback, not an additional
practical win from chaining. This is an appropriate assessment point before
building more synthetic chains or tuning their runtime in isolation.

The next useful decision should be driven by a real multi-region workload:
determine whether the obstacle is an overly conservative measurement boundary
or an actual small non-Clifford input core that must be carried into later
algebra. Sharing later analysis is a separate performance question, now measured
explicitly. This study does not claim either extension is already solved.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#chaining-regions-through-clifford-exits).
