# Proposed production boundary for compiled logical blocks

Historical proposal: the user authorized productionization on a fresh branch.
See [the implementation report](compiled_css_blocks.md) for the implemented
subset, validation, and public-sampler benchmarks. The proposal below records
the architectural decision before that authorization.

This is a design for review, not a change to the production architecture. The
[complete five-check study](msc_factor_sequence.md) supports a narrow exact
block action on large certified CSS states. It does not justify replacing the
ordinary state representation, selecting from physical size alone, or adding
per-shot topology planning.

## Recommended change

Add one explicitly planned logical-block action that consumes and returns the
existing dense logical coefficients at a certified code boundary. Initially
require one encoded qubit and no entangled spectators. The ordinary executor
continues to own noise symbols, RNG, records and output parities; a prepared
block sidecar owns immutable factor tables and preallocated scratch. Between
ordinary actions, the state remains the existing dense active-coordinate state.

The block acts as a fused quantum instrument: sample its true internal records,
contract the selected branch, return two normalized logical amplitudes, and
publish the records and their precompiled frame dependencies. Keep a forced
record path that returns the same conditional log probability. Readout errors
change reported records, not the physical state. Do not postselect, discard
syndromes, or insert an extra code projection that is absent from the circuit.

This is an architectural extension. Currently:

- `src/clifft/sampling/plan.h` defines `SamplingAction` and requires a continuous
  dense active-width chain for every `PlannedAction`. There is no logical-block
  action or certificate payload.
- `src/clifft/sampling/executable_plan.h` lowers that closed set of actions to
  prepared direct-Pauli operations, expressions, records and instruments.
- `src/clifft/sampling/executor.h` owns one dense `State`; `state.h` allocates
  all dense coefficients and measurement scratch for the planned peak width.
- Existing instrument continuations preserve live coefficients and their
  active-coordinate meaning. They are not an implicit representation-switch
  facility and must not be repurposed as one.

Adding an action and its planner/lowering contract needs explicit architectural
approval under `AGENTS.md`. None of these production files has been modified.
A separate whole-circuit backend would need less immediate executor integration,
but it would duplicate output/noise policies and give a weaker composition
story. Prefer the fused action if production integration is authorized.

## Compilation certificate and rejection conditions

The planner must establish, before lowering the block:

1. The actual incoming state lies in a signed CSS code encoding one logical
   qubit, with any other degrees of freedom independently certified. The
   compiler knows the change from Clifft's active coordinates to the block's
   logical X/Z convention, including relative phase. Small physical support
   or a matching gate-name pattern alone is insufficient.
2. The full interval, including quantum faults, hidden/reset measurements,
   feedback and readout errors, has a bounded coherent monomial description.
   Its exit contains the actual commuting code projection needed to return to
   the certified boundary. The synthetic sequence only certifies its explicit
   template; its generator is not a general source-circuit recognizer.
3. Every record and noise symbol has exactly one owner. The sign dependencies
   of the outgoing syndrome frame are compiled into the ordinary symbolic
   representation, so subsequent gates need no runtime tableau work.
4. Factor-table budgets and the maximum coherent term count are checked before
   allocating exponential tables. The current CSS factor kernel accepts at
   most four diagonal/flip terms. Quadratic coupled phases and entangled
   spectators outside that certificate decline. The reconstructed f7 fold
   kernel has its own, different certificate.
5. The compiler precomputes entry/exit coordinate transforms, all syndrome
   duals, factor elimination/gathers, and record/noise maps. Runtime binding
   performs only table lookups, parity evaluation and coefficient arithmetic.

First implement rejection with ordinary compilation of the *whole circuit*,
before execution or dense-state allocation. Avoid a runtime fallback that
tries to materialize an enormous physical state. Mixed block/ordinary execution
should be admitted only where the entry/exit coordinate maps have been proved;
unsupported suffixes can be rejected during compilation without sampling a shot.

## Feature and cost policy

Keep this opt-in initially. A feature gate should reject unsupported modes
before sampling and report the reason. The first production acceptance suite
must cover ordinary sampling, fixed-history replay, existing presampled noise,
readout errors and every declared output. Fixed-fault-count conditioning,
postselection/early exit, final-state queries, expectation outputs, batch lanes,
and non-Pauli instrument continuations need explicit support or a compile-time
rejection; the standalone research worker does not establish compatibility.

Select using actual Clifft active width plus certified contraction cost, not
physical qubit count. Preserve ordinary Clifft for the measured width-4/10
controls. The width-19 result supports a trial for this particular five-check
family, not a universal crossover rule. A memory-budget failure can select an
eligible block plan without ever allocating the rejected dense plan.

Compile once per circuit for this family. Physical faults and sampled outcomes
only bind fixed phase/parity data; they do not change graph topology. The study
reuses one plan for all attempts, including stress histories. Per-shot
compilation would redo work with no measured benefit here. This says nothing
about arbitrary adaptive circuits with genuinely different future topology.

## First implementation acceptance bar

- A narrow source interval recognizer produces a checked certificate; nearby
  unsupported circuits decline and use ordinary compilation.
- A prepared action passes allocation instrumentation and sanitizers, and does
  no runtime topology analysis.
- Forced trajectories and all three logical readout axes agree with original
  elementary Clifft/Aer where feasible. Large blocks retain the independent
  coherent-stabilizer reference. Test mixed prefixes and suffixes explicitly.
- Run complete five-check many-shot benchmarks with all outputs, plus actual
  original d3/d5 controls that must retain ordinary execution. Preserve circuit
  provenance and distinguish synthetic 37/61-qubit sequences from full MSC7.
- Do not advertise general acceleration or best-in-class performance. The full
  authors' MSC7/high-bond artifact and a competitive tuned external comparison
  remain missing.
