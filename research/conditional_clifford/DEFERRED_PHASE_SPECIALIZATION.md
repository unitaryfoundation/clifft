# Diagnosing realistic measurement and non-Clifford boundaries

Date: 2026-10-09.

## Assessment

The realistic-circuit checkpoint separates two obstacles. The first BT boundary
is unnecessarily conservative: syndrome measurements are on different wires
from the later scoring computation. Commuting that computation before the
measurements exposes the cancellation to the state-preserving reducer. Full
scored BT27 becomes Clifford, and the same rule also succeeds on full scored
BT81. Direct-X BT27 retains its non-Clifford computation, as it should.

Cultivation presents the other obstacle. Its first reduced preparation contains
one T gate, and that encoded quantum input survives the intervening Clifford
computation, resets, and measurements in the checked cases. Moving disjoint
measurements does not solve this example. A later reduction would need to accept
the surviving small non-Clifford input or use a different representation.

These results support feasibility for a useful restricted class. They do not
establish a general non-Clifford optimizer, a throughput advantage over Merlin,
or satisfactory scaling of the current analysis storage.

## Small extension justified by the diagnosis

The new [research scheduler](../../tools/profile/deferred_phase_specialization.py)
buffers measurements after the first phase gate. Later supported phase gates
and Pauli channels may move before that buffer only if they touch none of the
measured wires and use no newly deferred measurement record. Earlier-prefix
feedback is allowed, with relative record references rewritten from their
absolute identities. A shared wire, new-record dependency, reset, Hadamard, or
other unsupported phase operation ends the attempt.

Measurement order, inversion, readout flips, detector/observable references,
and annotation order remain intact. Measurements still execute as quantum
instruments; they are not replaced with sampled record values. This makes the
output suitable for an arbitrary ordinary continuation within the supported
circuit language, under the existing all-zero initial-state assumption.

The permutation is computed once. Every original categorical noise site has a
bijection to a scheduled site with identical probabilities and physical Pauli
actions. Each shot draws the full original history once and maps it through
that bijection. There is no repeated-history cache, postselection, or fault
weight cutoff. Two-qubit channels retain their joint categorical outcomes.

The existing state-preserving regional reducer then analyzes the enlarged
phase region. Its algebra and the native compiler/executor are unchanged. The
new scheduler recognizes dependencies, not circuit names, encoder patterns, or
scoring annotations. This is a host experiment, not production integration.

For scored BT27, 102 scoring operations, including 42 T/T-dagger gates, commute
before 72 syndrome records. The earlier region had six cubic phase monomials
on nine support coordinates, synthesized as 24 T gates. The enlarged nominal
phase polynomial is zero. Fault- and preparation-outcome-dependent corrections
remain Clifford. Thus the construction emits a Clifford circuit for every
valid history and preparation branch of this supported model; the observed
width zero is not merely a lucky result in a few low-weight shots.

This recovers the earlier terminal-sampling result using a state-preserving
interface. It does not newly demonstrate that scored BT27 was simulable.
The distinction matters: the direct-X source lacks the inverse scoring action
and retains the six cubic terms and width nine. The rule also exposes the
single inverse T in scored 15-to-1 without any protocol-specific handling.

## Complete workloads and noise

The [study](../../tools/profile/study_deferred_phase_specialization.py) uses
the unchanged Merlin generators at revision
`097380fac1a3968ca47925146e211fe990f4c396`. Existing BT preparation, phase,
decoder, and readout noise is retained for comparisons with prior studies.
Separate diagnostic variants put heterogeneous one-qubit Pauli channels after
every supported one-qubit gate/reset, depolarizing channels after two-qubit
gates, and readout flips on measurements, including the scoring tail. These
variants use probabilities 0.0001, 0.001, and 0.01. They have no idle-noise or
hardware-timing model.

Maximum complete widths in the retained stress histories and 64 fresh shots:

| Circuit | Ordinary noisy compilation | Previous regional method | With deferral |
| --- | ---: | ---: | ---: |
| BT27 scored | 33 | 9 | 0 |
| BT27 direct X | 33 | 9 | 9 |
| 15-to-1 scored | 5 | 1 | 0 |
| Cultivation d3 | 4 | 4 | 4 |
| Cultivation d5 | 10 | 10 | 10 |
| BT81 scored | 87 | 9 | 0 |
| BT27 scored, all-gate noise at each of three probabilities | 33 | 9 | 0 |
| BT27 direct X, all-gate noise at 0.001 | 33 | 9 | 9 |

All original records and declared parities are retained. Widths are inspected
before lowering, with an execution budget of twelve; the wide ordinary
programs are never allocated. The budget is a study guard, not a claim that
width thirteen is intrinsically infeasible. No difficult histories are removed
from a rate estimate; this experiment makes no logical-error-rate estimate.

## Cultivation and discarded reset information

The first cultivation boundary is RX, before any measurement deferral can
help. The reduced input to its continuation has a single T on a known
independent |+> coordinate, followed by Clifford encoding. Until the next T,
the continuation contains only Clifford operations and instruments. A single
logical non-Clifford input is therefore a natural bounded extension to study.

The diagnostic replaces that |+>-to-T injection with half of a Bell pair and
propagates the resulting stabilizer state. Every reset swaps the old physical
wire into a fresh environment wire, then prepares the requested state. The
environment is retained in the diagnostic and excluded from the accessible
output. This handles discarded reset information without selecting a hidden
reset outcome.

Gaussian elimination on the final stabilizers finds both an X-reference and
a Z-reference correlation supported on the physical output, with identity on
every environment wire. Together these witnesses certify that the logical
qubit remains recoverable in the checked conditional Clifford instrument.
Merely checking that the reference is mixed would be insufficient: it could
be entangled only with discarded environment wires. Controls explicitly test
resetting the logical qubit, dephasing it by discarding an entangled ancilla,
and retaining a measurement plus feedback instead.

For both d3 and d5, 11 fixed histories and two explicit probe seeds per history
retain both logical witnesses. The bridge includes thirteen resets, 32 CNOTs,
and seven visible measurements. The retained probe artifact records each
witness, outcome, and measurement predictability. All seven visible measurements
are determinate on the purified Bell probe in these checks, so the certificate
does not rely on selecting a favorable measurement outcome. These probes
diagnose the channel around an arbitrary logical input; they do not sample the
full noisy magic-state outcome law or enumerate every possible fault history.

This is evidence of an actual representation boundary in the examined
cultivation bridge, rather than a reset that trivially removes the need to
carry the non-Clifford input. No solver for the subsequent non-Clifford region
has been added, and complete widths remain four and ten.

## Correctness evidence and its limits

The [validator](../../tools/profile/validate_deferred_phase_specialization.py)
checks the scheduler independently by projecting original and scheduled
instructions onto every quantum wire and onto the ordered record/annotation
stream. Identical projections, unchanged absolute record identities, and the
fault-site bijection certify a permutation of disjoint operations. This
argument applies to arbitrary inputs for the scheduling step; the algebraic
state replacement still assumes the specified all-zero circuit input.

The retained [validation artifact](deferred-phase-validation.json) includes:

- 32 small circuits including wire renamings, 50 fixed-history checks, and
  54 preparation branches. Independent Aer density matrices check the complete
  unnormalized physical state for every visible record, both after scheduling
  alone and after the full replacement/continuation. Maximum complete matrix
  error is below 2.3e-16. Clifft exhaustive record replay has maximum error
  below 3.4e-16.
- All 16 histories of a four-site binary-noise witness and all 32 histories
  of a two-qubit categorical channel plus readout flip. Both retain probability
  mass one; weighted instrument-error bounds are below 7.8e-17.
- Rejection of an intentionally noncommuting permutation and invalid original
  histories. Shared-wire, new-record, noise, and reset barriers remain intact.
- Dependency certificates for every realistic workload, including its noisy
  source. For scored cases, 156 complete conditional record laws agree exactly
  with the earlier terminal shared-phase reduction. This comparison reuses
  that phase algebra and the normalized Stim flow oracle; it is not a new
  independent physical-state proof for the large BT circuits.
- Four Bell-probe controls distinguish retained quantum information from
  discarded or dephased information, including a case where the reference
  stays mixed after the physical output has lost coherence.

Fresh samples additionally audit output parities and compare record/parity
moments with Merlin where it accepts the source. These 64-shot comparisons are
bug checks, not statistical validation of the full distribution or rare-error
rates. Merlin rejects the direct-X BT27 cases with a non-affine-zero-set
measurement error; the complete reduced circuits still execute in Clifft.

## Costs and decision

Timing and storage figures are in the
[study artifact](deferred-phase-specialization.json). Per-shot time includes
the full original fault draw, prefix sampling, history mapping, rendering,
parsing, tracing, width inspection, lowering, and final sampling. Shared
analysis and explicitly seeded prefix-sampler initialization are reported as
setup. Validation, baseline comparisons, and output audits are separate.

| Circuit | Host setup, s | Host full shot, ms | Merlin setup, s | Merlin shot, ms |
| --- | ---: | ---: | ---: | ---: |
| BT27 scored | 0.695 | 2.55 | 0.014 | 1.52 |
| BT27 direct X | 0.662 | 2.73 | rejected | rejected |
| BT81 scored | 10.16 | 23.93 | 0.106 | 19.48 |
| 15-to-1 scored | 0.038 | 0.156 | 0.001 | 0.016 |
| Cultivation d3 | 0.019 | 0.341 | 0.001 | 0.043 |
| Cultivation d5 | 0.088 | 1.83 | 0.003 | 0.741 |

These are small single-CPU runs, not a throughput tuning study. Scored BT27
and BT81 have per-shot host costs about 1.7 and 1.2 times Merlin, but including
setup makes the complete 64-shot runs about 7.7 and 8.6 times more expensive.
The already-small 15-to-1 and cultivation d3 runs are about 24 and 12 times
more expensive including setup. Existing fast backends remain preferable on
such inputs. This experiment establishes a state-preserving reduction and
Clifft executability, not a reason to replace those backends.

Across all-gate noise probabilities 0.0001, 0.001, and 0.01, scored BT27 takes
2.36, 2.60, and 3.25 ms per shot after approximately 0.7 seconds setup. At the
highest probability, fresh histories contain 27 to 57 faults, all retained.
Direct-X with all-gate noise remains executable at width nine. No repeated
history is needed for either result.

BT81 is a useful growth warning even though its complete width is zero. Its
dependency masks alone occupy about 112 MB, versus 3.7 MB for BT27. This metric
excludes Python object overhead, duplicate representations, and other analysis
storage; it is not total resident memory. Physical size grows threefold and
noise sites grow from 4,170 to 24,492, but this stored payload grows thirtyfold.
Small execution width does not ensure cheap analysis.

The checkpoint supports continuing the research with two explicit limits:

1. Capability: disjoint measurement boundaries need not block state-preserving
   phase reduction. This is now demonstrated on complete protocol circuits,
   including noisy scoring. General non-Clifford entry remains unsolved.
2. Cost: the shared fault-response representation needs a separate storage and
   setup assessment before larger-scale integration. These measurements do not
   justify extrapolating inexpensive setup to much larger circuits.

The next capability experiment should be bounded to cultivation's one logical
qubit input: carry its Clifford encoding and small non-Clifford state through
the measured bridge, then test whether the next phase region can use that
input algebraically. Preserve the complete conditional instrument and inspect
the final width. Stop to reassess if the representation expands to the physical
width or fails to expose useful structure. This would test the missing entry
contract on a real workload; it is not a commitment to a general simulator or
an expectation of beating ordinary Clifft on these already-small examples.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#measurement-deferral-and-realistic-boundaries).
