# Reusing conditional phase analysis across shots

Date: 2026-10-09.

## Result and checkpoint

The next experiment succeeds at its bounded goal: one algebraic construction
finds complete and partial reductions in distillation, BT27, and measured-state
controls, then reuses its analysis across fresh faults and preparation outcomes.
It consumes raw circuits without encoder, decoder, code, or logical-operation
annotations. There is no fault-history cache or enumeration of fault patterns.

The implementation is a research host in
[shared_phase_specialization.py](../../tools/profile/shared_phase_specialization.py).
No production compiler, executor, or public API changes are made. The all-zero
circuit-entry assumption and permission to specialize on observed outcomes
continue to apply. The prepared state at an internal boundary is derived from
the circuit; it is not assumed to be zero.

This is a useful checkpoint for the proposed automatic mode. Shared analysis
removes most of the previous whole-circuit recompilation cost on BT27. Small
circuits still favor ordinary Clifft by a large margin. The supported circuit
shape is explicit and remains narrower than arbitrary non-Clifford circuits.

## Algebra and what is reused

The host locates an initial Clifford preparation followed by a region of CNOT,
Pauli, S, T, and CZ operations and terminal Pauli readouts. It absorbs eligible
terminal single-qubit Clifford basis changes into readouts while respecting
noise locations. It derives an affine support and a quadratic stabilizer phase
from the preparation, including general complex Clifford states and conditional
preparations with resets, measurements, and Pauli feedback.

Write the computational strings in that support as `q = A*x XOR b`. The matrix
`A` describes free coordinates. The offset `b` and some phase coefficients
depend on sampled faults and preparation outcomes. CNOT changes these affine
expressions. A T gate adds the corresponding bit to a phase polynomial modulo
eight. Substituting parity expressions gives a weighted polynomial of degree
at most three; stabilizer phases and CZ gates are included in the same algebra.

An X component of a fault changes a bit offset. Replacing a T phase on `l(x)`
by one on `l(x) XOR 1` changes its nonconstant phase by `-2*l(x)`. That is a
Clifford correction. Z components also add Clifford phases. Consequently the
non-Clifford part of this representation is fixed, while faults and preparation
outcomes select diagonal Clifford corrections and readout inversions. This is
an algebraic property of the supported region, including multiple faults;
it does not require identifying a transversal gate or a particular code.

Construction performs the following work once:

1. Row-reduce the preparation stabilizers and derive support coordinates.
   Compile an ideal Clifford-prefix sampler with nondisturbing stabilizer
   probes. Its records and probe signs specify each conditional preparation,
   including hidden branches introduced by resets.
2. Propagate primitive Pauli frames through that prefix and precompute their
   effect on both reported records and final stabilizer signs. Actual fault
   histories shift the ideal sampled vector by these responses.
3. Accumulate the nominal phase polynomial and conditional Clifford responses.
   Propagate all affine coordinates and choose independent output coordinates.
   This can eliminate preparation and encoding/decoding operations without
   recognizing them as named blocks.
4. Synthesize a fixed non-Clifford remainder and transpose Boolean dependencies
   into packed control-to-response masks. Conditional S phases are accumulated
   with their carries into Z phases, preserving multifault interference.

Each shot draws the original noise, samples the shared prefix law, evaluates
the controls, and emits the reduced core with its Clifford correction. Only
this smaller circuit is traced and lowered. The existing phase optimizer is
not run per shot. Prefix records and deterministic final readouts retain their
original slots through MPAD; terminal random readouts of independent fixed
bits are sampled explicitly. Detector and observable declarations retain raw
record parity semantics.

The noise model is the existing generic research host: independent locations
with heterogeneous Pauli-error, depolarizing, one- or two-qubit Pauli-channel,
and symmetric readout-flip probabilities. A two-qubit channel is categorical,
not independent draws of its constituent Paulis. There is no postselection,
fault-count truncation, or low-weight approximation. Correlated events across
locations and non-Pauli noise are outside this host.

## Structural results

The panel uses the same pinned Merlin generator revision
`097380fac1a3968ca47925146e211fe990f4c396` and circuit-wide synthetic noise as
the [preceding study](AUTOMATIC_SPECIALIZATION.md). Protocol noise probabilities
are 0.001. Scored circuits include an ideal inverse target operation before
readout; direct-X variants instead measure the logical outputs directly.

| Circuit | Physical qubits -> reduced core | Ordinary noisy active width | Reduced T count | Reduced active width |
| --- | --- | ---: | ---: | ---: |
| 15-to-1, scored | 15 -> 5 | 5 | 0 | 0 |
| 15-to-1, direct X | 15 -> 5 | 5 | 1 | 1 |
| Bravyi-Haah k=2, scored | 14 -> 5 | 5 | 0 | 0 |
| Bravyi-Haah k=2, direct X | 14 -> 5 | 5 | 2 | 2 |
| BT27, scored | 135 -> 33 | 33 | 0 | 0 |
| BT27, direct X | 135 -> 33 | 33 | 24 | 9 |
| Measured parity, with or without feedback | 2 -> 2 | 1 | 0 | 0 |

Core qubit count is distinct from non-Clifford active width: the 33-qubit
scored core is entirely Clifford and needs no dense 33-qubit amplitude array.
BT27 retains all 54 preparation records and all final records. The generic
output-coordinate construction leaves no encoder CNOTs on either distillation
or BT27, but retains one on the measured-state controls.

The direct-X examples demonstrate partial simplification. Synthesis is not
T-optimal and does not minimize active width: the earlier full optimizer gives
width one on Bravyi-Haah direct-X, versus two here. No claim about optimality
follows from the smaller BT27 T count either.

Cultivation d3/d5 explicitly fall back because a reset occurs inside the
non-Clifford region. The noncommuting control falls back at an internal H.
Their ordinary active widths are 4, 10, and 4. The current representation
supports one such region after a stabilizer preparation. It cannot continue
through arbitrary non-Clifford intermediate states, reuse a measured qubit
inside the region, or condition on a new suffix measurement for later gates.
Initial-prefix feedback into the region is supported.

Default construction limits are 128 support variables and 200,000 phase terms
or response expressions. Python integer masks avoid the earlier 64-variable
encoding ceiling, but polynomial and response storage can still grow quickly.
BT81 was not rerun in this step; the previous host's capped-region result remains
a separate representation limitation, not evidence of algebraic inapplicability.

## Timing and selection

The retained run uses one CPU, 256 fresh shots per eligible case, and ordinary
Clifft batches of 1, 256, and 1,024 shots. Fresh timing includes fault drawing,
prefix sampling and control evaluation, correction synthesis, rendering,
parsing, tracing, width inspection, lowering, and sampling. Output parity
audits and report construction are excluded. Shared construction is separate.
No previous fault history or compiled conditional program is reused.

The [JSON artifact](shared-phase-specialization.json) records the precise
timings, stage costs, hashes, seeds, and selections. Scored BT27 takes 1.395 ms
per fresh shot, versus 52.4 ms for the previous whole-circuit host and 1.476 ms
for Merlin on the scored circuit. This is about a 38-fold
improvement over that earlier host, not a claim of superiority over Merlin.
Merlin rejects the direct-X circuit with a non-affine-zero-set measurement error.
The earlier dedicated BT implementation is also a distinct, faster reference.

Shared construction costs roughly 0.4-0.5 seconds for BT27. This matters:
including setup at 256 shots, the new host remains slower than Merlin. Most
remaining per-shot time is host work; drawing noise alone accounts for about
0.6 ms, while sampling the already-lowered core takes about 0.02 ms. Of 256
BT27 histories, 254 are unique, with weights from zero through eleven. Cache
reuse is neither required by the algorithm nor responsible for these results.

On distillation, shared sampling costs approximately 0.12-0.15 ms per shot versus
about 0.0003 ms for ordinary Clifft in a 1,024-shot batch. Reduced T count does
not outweigh the host overhead there. A comparison of construction plus
1,024-shot sampling cost selects ordinary execution for every small supported
case. Unsupported controls also retain ordinary execution. BT27 selects shared
execution because ordinary width 33 exceeds the diagnostic width budget 12;
ordinary BT27 dense execution was not timed or allocated.

This selection is a research comparison after profiling both paths. It is not
a production cost model, and its totals do not include the cost of discovering
which path wins. Whole-process peak RSS is retained, but is not isolated by
backend. Scored BT27 dependency masks alone occupy about 3.74 MB of payload;
that excludes transposed masks, Python object overhead, and other storage.

## Correctness evidence and oracle correction

The [validation artifact](shared-phase-validation.json) records:

- 28 small circuits, including 14 wire-renamed versions, 80 fixed noise
  histories, and 188 enumerated preparation branches. General Clifford
  preparations include Y/Z product measurements, record feedback, and resets.
  Complete joint output distributions, after summing preparation branches,
  agree with independent Aer density-matrix calculations to below 2.5e-16.
  Both measured-state controls enumerate all positive-probability histories.
- Four distillation variants, each with 16 histories including weight-12
  faults. Complete joint record laws match independent Aer statevectors:
  every positive-probability record is checked and their total mass is one.
  Maximum probability error is below 1.8e-15.
- Scored BT27 with 19 histories: exact joint prefix-record and stabilizer-sign
  laws certify the precomputed fault response for those histories. Two sampled
  conditional preparations per history give 38 exact complete output-law
  comparisons against the existing whole-circuit optimizer and native exporter.
  These are conditional on that optimizer's correctness, rather than an
  independent proof of the original 135-qubit non-Clifford circuit. They do
  not enumerate every BT27 preparation branch or noise history.
- Fresh-noise moment comparisons with Merlin where it accepts the circuit.
  These 256-shot checks detect gross mistakes; they do not establish noisy
  distribution equivalence or rare logical-error rates. BT27 direct-X retains
  the earlier limitation of lacking a new independent full-circuit oracle.

The installed Stim 1.16.0 flow oracle reports reversed sign positions for
batched MPAD literals. A minimal local witness is `MPAD 0 1`: sampling returns
`[0, 1]`, while flow generators report `1 -> rec[1]` and `1 -> -rec[0]`.
This initially produced an apparent BT27 single-fault mismatch even though the
samplers agreed.

The new validator replaces deterministic padding with equivalent reset and
measurement instructions before asking for flow constraints. It checks this
normalization on all 256 eight-bit padding patterns against literal expected
constraints and actual Stim samples. Neither Stim nor Clifft is patched, and
the sampling implementation still uses MPAD normally. Older flow-based
artifacts have not been retroactively revalidated; mixed batched padding in
their oracle inputs requires the same care. The independent Aer checks above
do not rely on that flow-oracle behavior.

## Next decision

This experiment meets the previous checkpoint: reuse analysis, retain a
non-Clifford remainder, handle observed preparation outcomes, improve a case
that needs reduction, and prefer ordinary Clifft where it is already cheap.
Further BT-only tuning would provide less information about the general goal.

The next useful bounded step is to assess regional composition: can this
representation simplify an eligible region inside a larger circuit while
preserving the quantum state and record interface at its exit? Use a small
independent circuit with a noncommuting continuation and a measurement boundary,
with complete Aer state or instrument checks. Failure to certify that interface
should keep the original region. This would test the main current limitation
before adding more circuit families or deciding production integration.

Noise-free opportunities remain worthwhile separately. This step includes
ideal histories in correctness checks, but does not establish a new noise-free
performance result or change the priority of the ongoing noisy study.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#reusing-conditional-phase-analysis).
