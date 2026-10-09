# Current method and implementation map

## Shared algebra

The generic analyzer derives computational-basis support `q = A*x XOR b`
from the Clifford preparation's stabilizers. Free coordinates x describe the
superposition. Offset b and stabilizer-phase signs can depend on sampled
preparation records, reset branches and Pauli faults. CNOT gates transform the
affine expressions; T and diagonal Clifford gates accumulate a phase
polynomial modulo eight. Substituting parities gives terms of degree at most
three. No code matrices, logical-gate names or user boundaries are supplied.

For a Boolean parity l, flipping its offset changes the T phase from l to
`l XOR 1`. Up to a constant, the difference is `-2*l`, a Clifford phase.
Z faults also contribute Clifford phases. Consequently the non-Clifford part
is fixed across the supported fault/outcome controls; S/Z/CZ corrections vary.
This argument handles multiple simultaneous faults. Carry terms from sums of
S phases must be retained as Z corrections.

Setup prepares an ideal Clifford-prefix sampler, stabilizer probes for hidden
state information, fault-to-record/sign responses, the affine-support phase
analysis, and packed Boolean control dependencies. A shot draws the complete
categorical fault history once, samples the ideal preparation law, and shifts
its signs/records using the fault responses. Controls then select the phase
correction and physical offsets. It shares structure rather than enumerating
fault histories or measurement laws.

The noise frontend supports heterogeneous Pauli-error, depolarizing,
one/two-qubit Pauli channels, symmetric readout flips and adjacent correlated
error/ELSE chains. Each categorical event is drawn jointly. Unsupported quantum
channels are not approximated by this host.

## Safe boundaries and state reconstruction

A selected phase region begins after a Clifford preparation and ends when an
operation cannot be represented by the supported affine/diagonal algebra.
The scheduler can buffer measurements and move later phase operations or
Pauli channels ahead of them when their wires and record dependencies permit
it. For a correlated chain, every possible alternative participates in the
dependency test; the alternatives are not moved independently. Original noise
sites map bijectively to scheduled sites.

The state-preserving renderer keeps the full affine encoder, conditional X
offsets and physical qubit labels. It reconstructs a reduced preparation and
then attaches the unchanged logical continuation with realized faults.
Earlier visible records occupy their original slots using MPAD; detector and
observable declarations keep their record references. Hidden reset outcomes
influence the conditional state without adding visible records.

Preserving only a terminal output distribution is insufficient at a quantum
boundary. Terminal basis-change/readout rewrites that alter the conditional
state cannot substitute for this interface. Similarly, an exported optimized
preparation must include its complete final Clifford frame, not just the
remaining Pauli rotations.

## Chaining and the bounded carrier

`ConditionalPhase.rewrite` attempts another region when the previous exit is
Clifford. It retains the already observed record values and their conditional
state. Original physical faults have already been drawn and are not redrawn
for later regions. The default cap is eight regions.

With exactly one surviving T rotation, the carrier represents the state as
`R_P(+/- pi/4)|s>`, with P a Hermitian Pauli and |s> a stabilizer state.
Clifford operations conjugate P and update the reference. A measurement
commuting with P can be sampled on the reference. A reset can be crossed when
a linear solve finds a representative of P acting trivially on its wire.
Multiplication by a reference stabilizer is allowed only when it preserves
Hermiticity and the action on |s>.

Unsupported crossings retain an exact conditional source for ordinary Clifft.
At a later phase boundary, the current route emits the carried rotation and
remaining circuit for ordinary optimization. It does not automatically apply
another generic phase synthesis to that non-Clifford input. Multiple surviving
rotations similarly remain ordinary work. This is why the carrier's existence
does not establish unrestricted multi-region chaining.

## Compilation reuse and remaining per-shot work

The fixed preparation is optimized once and exported with its physical frame.
Where the rest is a Clifford continuation, setup traces it with synthetic
Pauli probes to derive fault-response masks. Those probes are removed before
execution. Their responses include operation signs, readout inversions and
the final frame, so feedback consumes the correct records during execution.

A native research worker receives each shot's diagonal boundary correction,
selected Pauli responses and record flips. It applies the diagonal action
algebraically, composes the fragments, performs the remaining optimization,
checks width, plans, prepares and samples one shot.

An axis/dependency certificate permits caching a squeeze permutation. Each
shot still checks the relevant input after preceding optimizer passes; a failed
check invokes ordinary squeezing. The selected coordinate policy shortcuts
identity Paulis and otherwise uses native coordinate conversion. It does not
reuse a complete sampling plan. Other coordinate policies are retained as
research controls, without a circuit-family dispatcher.

All planning and tableau evolution occurs before the ordinary executor's hot
dispatch. A future public host mode or native continuation architecture is an
open design decision, not an architectural change implemented by this branch.

## Source map

All paths below are relative to `tools/profile/`.

| Responsibility | Main files |
| --- | --- |
| Complete noise model and ordinary comparison helpers | `automatic_specialization.py` |
| Affine support, phase polynomial, control responses, state renderer | `shared_phase_specialization.py` |
| Boundary discovery and safe measurement movement | `regional_phase_specialization.py`, `deferred_phase_specialization.py` |
| Chaining/fallback routing | `conditional_phase_frontend.py` |
| One-rotation carrier | `one_core_phase_specialization.py` |
| Fixed preparation and direct emission | `compiled_prefix_reuse.py`, `prefix_trace_reuse.py`, `export_optimized_prefix.cpp` |
| Continuation responses and worker interface | `continuation_trace_reuse.py` |
| Native fragment composition and execution | `profile_prefix_trace_reuse.cpp` |
| Certified scheduling and coordinate policy | `squeeze_schedule_reuse.h`, `coordinate_reuse.cpp`, `coordinate_reuse.h` |
| Current whole-circuit assessment | `survey_conditional_capability.py` |

The native coordinate experiment compiles the existing planner implementation
under a research entry point and wraps its coordinate conversion. Production
`src/clifft/` is not patched. The Python modules still reflect research layering;
consolidating their shared helpers is cleanup for a later implementation, not
part of the scientific claim.
