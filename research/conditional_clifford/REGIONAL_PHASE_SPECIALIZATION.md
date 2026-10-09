# Continuing after a conditional phase reduction

Date: 2026-10-09.

## Question and result

The accepted goal is to extend the range of circuits that Clifft can practically
simulate. Ordinary Clifft can remain preferable on an arbitrary subset of
circuits; this research mode need not replace it. Success on a circuit whose
ordinary active width is prohibitive is useful even if specialization adds
overhead on small circuits.

This focused experiment asks whether an eligible phase region can end before
the final readout, preserving its quantum state and classical records for a
continuation. The answer is positive within the existing input assumptions.
The new exit restores physical wire labels, the affine encoding, and conditional
bit offsets. Independent state and instrument checks pass. A continuation with
Hadamards, further T gates, noise, measurements, resets, and feedback can then
run through ordinary Clifft.

The width benefit can survive that exit. In a constructed 48-qubit circuit,
ordinary noisy compilation reaches width 44, while the reduced region and its
complete continuation reach width 2. Full scored BT27 reaches width 9 instead
of 33. A negative control shows that a sufficiently demanding continuation can
erase the width benefit even when the selected region becomes entirely Clifford.

This removes the requirement that reduction end at terminal readout. It still
reduces only the first eligible region after an initial Clifford preparation;
it does not recognize a second arbitrary non-Clifford input region.

## State-preserving construction

The existing [shared phase analysis](SHARED_PHASE_SPECIALIZATION.md) represents
the conditional state using free support coordinates, a fixed non-Clifford
phase, and fault/outcome-dependent Clifford corrections. Its sampling renderer
can turn some quantum wires into final classical outputs. That is insufficient
as an interface to later gates.

The new `preserve_state=True` research mode in
[shared_phase_specialization.py](../../tools/profile/shared_phase_specialization.py)
instead performs the following:

1. Keep preparation measurements as physical state updates. Disable absorption
   of terminal Clifford basis changes into readout axes, because that rewrite
   can change the post-measurement state while preserving its record law.
2. End the phase region before any new measurement or reset. Retain the full
   affine encoding and every output bit-offset expression.
3. Prepare the reduced phase computation, apply its Clifford correction and
   encoding, map the support coordinates back to the original physical labels,
   and apply the conditional X offsets. Constant-support wires remain quantum
   wires, initialized to their actual zero or one state. Wires first used by
   the continuation are retained at zero.
4. Restore every preparation record using MPAD, with its original detector and
   observable declarations. Append the continuation with its original quantum
   operations, record references, and realized faults.

The renderer's two interfaces are checked explicitly: a sampling-only analysis
cannot expose a quantum-state exit, and a state-preserving analysis cannot use
the sampling-only renderer. Measurements inside the selected phase region are
rejected by the state-preserving analysis.

[regional_phase_specialization.py](../../tools/profile/regional_phase_specialization.py)
finds the boundary automatically: after the initial Clifford preparation, the
first T begins the region; the first operation outside its CNOT/diagonal-phase
representation ends it. This includes a Hadamard, reset, or measurement. No
user boundary, code name, encoder/decoder annotation, or gate-pattern catalog
is supplied. This is conservative boundary discovery, not an optimal partition.

The noise model is unchanged: independent locations with heterogeneous Pauli
channels and readout flips, including categorical two-qubit channels. Faults
in the prefix, selected region, and continuation retain their locations and
probabilities. Sampling remains free of caches, postselection, and fault-weight
truncation. Only preparation outcomes are sampled by the host; later outcomes
and feedback remain live in the ordinary executor.

This is source-level composition before execution. Each shot constructs and
lowers a complete specialized circuit; it does not resume a native executor
mid-shot, transfer an unknown state into a new program, or change production
planning/allocation contracts. The all-zero circuit-entry assumption remains
essential. The replacement preserves the prepared state, not the original
region's unitary action on arbitrary unknown inputs.

## Correctness checks

The new [validator](../../tools/profile/validate_regional_phase_specialization.py)
checks the joint classical/quantum interface, not just final bit probabilities.
For every visible record string, an independent Aer density-matrix calculation
provides an unnormalized physical density matrix. Its trace is the probability
of that record, and its off-diagonal entries retain quantum coherences. Reset
branches are summed, and the record wires are treated classically.

At the region exit, Clifft statevectors of the reconstructed conditional states
are combined with their preparation-branch probabilities. After the complete
continuation, Aer compares the same full classical/quantum interface, and
exhaustive Clifft replay independently checks the complete visible-record law.
All-zero input and fixed Pauli histories suffice for these conditional checks;
there is no assertion of arbitrary-input channel equivalence.

The retained [validation artifact](regional-phase-validation.json) contains:

- 28 small circuits, including 14 wire renamings, 56 fixed histories, and 124
  enumerated preparation branches. They include general complex Clifford
  preparations, inverted Pauli-product measurements, hidden reset outcomes,
  feedback, a constant-state offset, and wires first used by the continuation.
  Maximum entry-matrix error is below 3.4e-16, final-instrument error below
  1.4e-16, and Clifft record-probability error below 2.8e-16.
- All 16 positive-probability histories of a separate noise-mixture witness,
  spanning preparation, region, continuation, and readout faults. The weighted
  final-instrument error bound is below 4.1e-17; the original probabilities sum
  to one within floating-point tolerance.
- Complete exit-state vectors for 15-to-1 and Bravyi-Haah direct-X circuits,
  each with 16 histories, plus a six-qubit parity-echo example with 12 histories.
  Maximum amplitude error, after aligning global phase, is below 5.3e-16.
  Aer evaluates both physical circuits for the 14/15-qubit cases because
  Clifft's dense state-export API intentionally stops at ten qubits. Clifft
  supplies the reduced candidate state for the six-qubit case. No API limit
  was changed.
- Scored BT27 with 11 stress histories and two sampled preparations per
  history: 22 exact complete conditional output-law comparisons. Both the
  regional composition and original conditional source are exported through
  the existing whole-circuit optimizer before comparing Stim flow laws. Exact
  prefix-record and stabilizer-sign laws also check the fault responses.
  These checks are relative to the existing optimizer; they are not an
  independent physical-state proof for the 135-qubit circuit or an enumeration
  of every fault/outcome history. The previously documented deterministic-MPAD
  oracle normalization is reused.

Two deliberate corruptions check that the stronger oracle is meaningful.
Dropping a relative phase leaves boundary Z-readout probabilities unchanged but
produces a matrix error of about 0.707. Reversing a record/state association
leaves the unconditional quantum state unchanged but produces a conditional
matrix error of 0.5. Both are detected.

The preceding shared-phase validator was also rerun after extracting the common
quantum renderer: its 28 small circuits, four distillation variants, and 38 BT27
conditional-law checks continue to pass. Historical JSON artifacts retain
their original source hashes and scopes.

## Complete-circuit width and cost

The [study driver](../../tools/profile/study_regional_phase_specialization.py)
uses the same pinned Merlin checkout,
`097380fac1a3968ca47925146e211fe990f4c396`, for existing protocol examples.
The new parity-echo family is a constructed structural control, not a new
application benchmark: parity phases accumulated eight times cancel modulo
eight, while interleaved Pauli faults leave Clifford corrections. The
continuation adds Hadamards, fresh T gates, more noise, a measurement, and
record-controlled feedback. Its number of additional T-bearing wires is varied
to test whether the savings survive.

The table gives maximum observed widths over the retained stress histories.
Every executable case additionally receives 64 fresh noise histories; their
complete widths agree with the listed values in this run.

| Complete circuit | Ordinary noisy width | Reduced region width | Complete specialized width |
| --- | ---: | ---: | ---: |
| BT27, scored | 33 | 9 | 9 |
| BT27, direct X | 33 | 9 | 9 |
| Cultivation d3 | 4 | 1 | 4 |
| Noncommuting control | 4 | 1 | 4 |
| Three-qubit preparation-state witness | 1 | 2 | 2 |
| Measurement/noise-mixture witness | 1 | 0 | 1 |
| Parity echo, 6 wires, 2 T-bearing continuation wires | 6 | 0 | 2 |
| Parity echo, 24 wires, 2 T-bearing continuation wires | 21 | 0 | 2 |
| Parity echo, 48 wires, 2 T-bearing continuation wires | 44 | 0 | 2 |
| Parity echo, 24 wires, 24 T-bearing continuation wires | 22 | 0 | 24 |

The large positive example is the direct signal for the user's goal: restoring
physical labels and applying a noncommuting continuation need not reintroduce
the original width. The final example demonstrates a real limit: new work
after the reduction can dominate the complete simulation. Even a successful
state-preserving replacement can be worse than ordinary compilation. The
simple regional synthesis also does not minimize width on small controls.

The execution width budget is 12. Wider ordinary programs and the final
specialized negative control are inspected before lowering and are not
allocated or sampled. This budget is a study guard, not a claim that every
width above 12 is infeasible. The constructed width-44 baseline is far beyond
the dense execution attempted here. No difficult histories are omitted from
a logical-error estimate: this study does not make such an estimate at all.

Scored BT27 retains a 24-T reduced region and 66 T gates in the complete
composed source before any per-shot phase optimization. Its fresh sampling
cost is 2.69 ms/shot after 0.57 seconds of shared setup. Direct-X BT27 takes
2.39 ms/shot after 0.54 seconds of setup. These are greater than the previous
terminal-only scored result of 1.40 ms/shot and width zero; the new construction
exposes an earlier quantum exit and retains the subsequent scoring computation.
It is a capability extension, not a replacement for that terminal path.

The 48-qubit positive control takes about 0.49 ms/shot after 0.09 seconds of
setup. Timing includes noise drawing, control evaluation, physical-state
rendering, parsing, tracing, width checks, lowering, and sampling. It excludes
output parity audits and report generation. No phase optimizer is called per
shot, and no conditional program is cached. Per-mode memory and comparative
throughput against another simulator were not measured in this step.

The full [study artifact](regional-phase-specialization.json) retains source
hashes, seeds, fault histories for structural checks, fresh-history hashes,
widths, and stage costs. The fresh shots audit execution and record numbering;
they do not statistically establish full noisy equivalence or rare-error rates.

## What changed about applicability

An initial supported region may now feed an ordinary noncommuting computation,
including intermediate measurements and resets. Cultivation's reset boundary
therefore becomes a valid exit instead of a reason to reject the entire
circuit. This does not make cultivation cheaper in the measured example.

The remaining restriction concerns entry, not exit. The selected region still
needs the stabilizer preparation derived from the all-zero input. If it leaves
a non-Clifford state, a later candidate region cannot simply be treated as a
fresh stabilizer preparation. Likewise, later measurement outcomes are executed
correctly but do not yet trigger another algebraic analysis.

The useful next checkpoint is repeated application at a later boundary with an
explicit certificate of a supported entry state. A bounded first case could
use a first reduction that becomes Clifford, then a measurement-dependent
second region; an unreduced non-Clifford input should remain a deliberate
fallback. Supporting small non-Clifford input cores is a broader extension.
Neither is implemented or implied by this study's successful exit checks.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#quantum-continuations-after-regional-reduction).
