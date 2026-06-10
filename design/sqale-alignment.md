# sqale-sim alignment: spike findings and checkpoint plan

Status: design doc on the `codex/noncomputational-structural-mvp` feature
branch. Part one records a one-time validation spike: how the clifft
noncomputational MVP compares to Infleqtion's sqale-sim leakage simulator,
where they agree, where they differ, and how fast clifft is by comparison.
Part two is the plan of record for the next phase: a checkpoint that covers
sqale-sim's noise model as far as possible within clifft's ahead-of-time
architecture, with the residual approximations characterized and measured
rather than discovered.

## Why

The noncomputational MVP was built to recreate sqale-sim's loss/leakage
simulation capability in clifft's near-Clifford style, with the hope of being
substantially faster than their cirq-stabilizer prototype. The spike answered
three questions:

1. Correctness: does clifft produce the right physics where the two simulators
   model the same thing?
2. Performance: is clifft fast enough to justify continuing?
3. Gaps: what does sqale-sim do that clifft does not, ranked, so feature work
   can be prioritized by evidence.

Reference: Infleqtion `client-superstaq` PR #1353 (`cirq_superstaq/sim/`), read
at head `f007480`. The leakage sim is not on PyPI; it lives only in that PR.

## What sqale-sim is

Architecturally it is the same idea as clifft's noncomputational path: a
stabilizer/Clifford quantum core plus a classical leakage tracker. Specifics:

- `LeakageState` extends cirq's `StabilizerSimulationState` and adds a per-qubit
  `ClassicalDistribution` over a 5-level qudit. The five levels correspond 1:1 to
  clifft's default set: `0=g, 1=e, 2=leak_g, 3=leak_e, 4=lost`.
- Leakage is expressed as `JumpChannel` transition matrices attached to CZ and RZ
  gates, where `T[i, j]` is the probability of jumping from level `j` to level
  `i`, and a column summing to less than one leaves the remainder as "no jump."
  This is the same `T[to][from]` convention and the same no-jump-is-the-deficit
  semantics clifft uses.
- The readout classifier is itself a transition channel and is ternary: levels
  `g, leak_g` read 0; `e, leak_e` read 1; `lost` reads a third symbol "2" (a
  heralded-loss outcome). clifft's classifier is binary.
- It also models coherent errors (gate over-rotations, CZ/movement phase errors)
  and Pauli-twirls non-Clifford gates to stay on the stabilizer simulator. The
  clifft MVP models none of these.
- Important internal behaviors, read from `leakage_sim.py`:
  - It propagates the classical leakage distribution and samples late; an
    `oversample` option amortizes one stabilizer run over many leakage draws.
    That amortization requires terminal measurements, so it does not apply to
    multi-round QEC circuits.
  - At a jump site (`_apply_jump`) it first queries the stabilizer tableau
    (`_promote_classical`): if the qubit is deterministic in Z it takes an
    exact per-column path. Only for a genuinely indeterminate qubit does it
    approximate (`_approximate_jumps`): it pads the lower-rate computational
    column's diagonal so both columns sum to the same total (the padding is a
    pseudo self-jump, i.e. pure dephasing), draws "jump fired" at that
    equalized rate, and on fire Born-measures the qubit on the stabilizer
    state and draws the destination from the measured bit's column. A code
    comment notes this is a conservative approximation; only its dense
    reference is exact.
  - Downstream gates touching a leaked or lost qubit are skipped, lazily.
    Diagonal gates on classically-tracked qubits are no-ops; any other gate
    triggers a subspace measurement (`_collapse_qubit_subspace`): if the level
    is computational the qubit re-enters the stabilizer simulator and the gate
    proceeds, otherwise the entire operation is skipped, identity on the
    partner included. Jump channels still fire on leaked qubits (that branch
    precedes the skip), so re-entry through from-leak transition columns works.
- It ships a dense density-matrix reference, `simulate_true_distribution`, which
  its own unit tests assert against. We use that as the ground truth here.

## Methodology

Differential comparison: build a clifft `noncomp.Model` equivalent to a given
sqale scenario, run `sample_noncomputational`, and compare the output
distribution to sqale's `simulate_true_distribution` within shot noise.

Reconciliation needed to line the two up:

- Levels map 1:1 (above).
- sqale prepends a pi pulse that flips the computational subspace (`g <-> e`) and
  no-ops on leak/lost. We absorb it by swapping the `g` and `e` entries of
  clifft's `initial_state`; clifft's normal `|1>` X-prep plus a binary classifier
  then reproduces sqale's readout with no extra gate (and avoids clifft rejecting
  an X applied to an initially-leaked qubit).
- The leakage transition is hooked on a Z-type gate (`S`), which sqale treats as
  an RZ (`ZPowGate`) carrying its `rz_transition_matrix`.

Performance: a ladder on repetition-code memory circuits (H'd data qubits, CX
syndrome rounds with ancilla measure-reset, an `S` layer on the data carrying the
leakage hook, terminal data measurement), comparing clifft plain, clifft noncomp
with no events, with certain leakage, and with low-probability leakage, against
sqale's per-shot sampler. The clifft Python extension was a Release build.

Reproduction: install cirq-superstaq from the PR ref into a local venv and run
the local (uncommitted) spike scripts:

```
uv pip install "cirq-superstaq @ \
  git+https://github.com/Infleqtion/client-superstaq.git@f007480#subdirectory=cirq-superstaq"
.venv/bin/python sqale_align_local/differential.py
.venv/bin/python sqale_align_local/perf.py
```

The sqale dependency and the comparison scripts are deliberately kept local and
out of the committed dependencies and CI.

## Correctness results

Where clifft and sqale model the same thing, clifft reproduces sqale's dense
ground truth within shot noise:

| Scenario | Agrees |
| --- | --- |
| Initial leakage population + classifier | yes |
| Pure-leakage transition with a known computational source | yes |
| Source-independent leakage on a superposed qubit | yes |

## Performance results

Repetition-code memory circuits, milliseconds per shot (clifft = noncomp with
low-probability leakage; sqale = per-shot sampler):

| Qubits | clifft noncomp | sqale | speedup |
| --- | --- | --- | --- |
| 5 | 0.024 | 2.2 | ~90x |
| 17 | 0.14 | 9.2 | ~67x |
| 33 | 0.30 | 21 | ~71x |
| 49 | 0.99 | 57 | ~58x |

clifft is roughly 60-90x faster across the range, and the gap holds as the qubit
count grows. Per-shot recompilation is clifft's dominant per-shot cost in the
noncomputational path (the no-event noncomp time is close to the compile-once
cost, far above plain sampling), but clifft is far enough ahead that this is not
a competitive bottleneck. Caching the rewrite/compile would buy roughly another
order of magnitude; it is an optimization, not a requirement for parity.

## Gap analysis

Matches today (clifft equals sqale's dense truth): initial population, binary
readout classifier, pure-leakage transitions that are either known-source or
source-independent, and the performance shown above.

The spike originally identified two gaps. Their status:

- Gap A, computational-destination transitions (for example relaxation mass
  such as `T[g][e]`): **resolved**. clifft now materializes the carrier at the
  sampled destination level (a hidden reset, plus an X for a `basis_bit == One`
  destination, inserted after the base op), implementing the channel
  `rho -> |d><d| (x) Tr_q(rho)`. Relaxation and recapture entries are modeled
  exactly; the spike-era behavior of silently tracking the level label without
  updating the simulator state is gone.
- Gap B, source-dependent transitions on a qubit whose computational state is
  not known: **open**, and the gating gap for realistic circuits. clifft
  rejects these (it cannot pick a source column). sqale's fast path cannot do
  them exactly either; it takes the equalize-and-collapse approximation
  described above. Parity here means implementing the same approximation, not
  matching an exact behavior.
- A third gap became visible once Gap B's fix was scoped: **structural policy
  on leaked/lost operands**. sqale skips downstream operations touching a
  leaked or lost qubit (identity on the survivors); clifft's defaults reject
  most of these cases (two-qubit gates on a lost operand, gates on leaked
  operands, multi-qubit noise and classical feedback on lost operands). Without
  a skip-equivalent policy, any multi-round circuit rejects shortly after its
  first leak event, regardless of Gap B.

Consequence: sqale's real device matrices mix computational transfer with
leakage and apply it on superposed qubits over many rounds. Covering them needs
Gap B plus the drop policy; Gap A is already done.

## Checkpoint plan

Goal: run sqale-sim's full default noise model (the 2LQ-paper neutral-atom
parameter set) end to end on clifft, matching its fast path within shot noise
on all marginal statistics, with every residual approximation named, bounded,
and measured.

### Constraint envelope

Two architectural principles bound this checkpoint:

1. **Ahead-of-time only.** Every noncomputational event is resolved before
   compilation: the trajectory sampler decides what happens, the rewriter
   materializes it, and the compiled circuit is straight-line. No runtime
   branching, no segmented execution, no JIT.
2. **No coherent noise simulation.** Coherent control errors are converted to
   twirled Pauli channels ahead of time. This matches the reference
   simulator's own fast path, which Pauli-twirls non-Clifford gates; its dense
   simulator is the only place coherent noise is simulated coherently.

### Coverage target

| Mechanism (sqale `SqaleNoiseParams`) | clifft status under the envelope |
| --- | --- |
| `initial_state_probs` | exact today |
| `classifier_errors` (g/e confusion) | exact today |
| Transition matrices: pure leak, known-source or source-independent | exact today |
| Transition matrices: computational destinations (relaxation/recapture) | exact today (carrier materialization) |
| Transition matrices: source-dependent on an indeterminate qubit | build: equalized-rates approximation; marginals exact on Clifford circuits, two documented divergences (below) |
| Downstream ops on leaked/lost qubits | build: drop policy profile (identity on survivors, equals sqale's lazy skip) |
| Lost level at readout (ternary symbol "2") | build: three-symbol classifier; herald delivered in the sidecar, visible record stays binary |
| `cz_phase_error`, `movement_phase_error` (stochastic Z) | expressible as `Z_ERROR` instructions; verify passthrough on the noncomputational path |
| `gr_*` / `rz_relative_overrotation` (coherent) | build: twirl-to-Pauli conversion helper, channels inserted ahead of time |
| Exact state-dependent jumps; movement/atom-site semantics; correlated two-qubit leakage; `oversample` | out of scope (see successor architecture; `oversample` is inapplicable to multi-round QEC circuits anyway) |

### Work items

1. **Equalized rates plus drop policy (one unit; they are only useful
   together).** Sampler policy `unknown_source_policy = equalize_rates`: pad
   the lower-rate computational column's diagonal so both columns sum to
   `p_max` (the padding is a pseudo self-jump, i.e. dephasing), draw firing at
   `p_max` ahead of time, on fire draw the source bit uniformly and the
   destination from that renormalized column. The collapse reuses existing
   machinery: trace-out reset for leaked/lost destinations, carrier
   materialization for computational destinations (for the pseudo self-jump
   the inserted reset is the dephasing itself). Alongside it, a policy profile
   that drops operations on leaked/lost operands (two-qubit gates, noise,
   classical feedback) instead of rejecting, matching sqale's skip. The drop
   is statistically equivalent to their lazy skip because both are samplings
   of the same Markov branch process, early versus late, with downstream
   treatment depending only on the branch. Also folds in the sampler
   initial-draw floating-point-tail fix.
2. **Ternary loss herald.** Three-symbol classifier columns where the third
   symbol heralds loss. The visible record stays binary (record-layout
   invariance is load-bearing); the herald symbol is returned per measurement
   in the sidecar. An in-record herald bit is deferred until decoder work
   needs it. Includes the base-3 adapter for comparing against sqale records.
3. **Coherent-error compatibility.** Verify Pauli/`Z_ERROR` channels pass
   through the noncomputational path on alive operands (and drop on
   leaked/lost ones), then add a helper converting over-rotation and phase
   parameters to Pauli channel probabilities via the standard twirl formula
   `p_P = |tr(U P)|^2 / 4`. Kept out of the noncomputational model surface:
   these are explicit noise instructions in the circuit.
4. **Validation campaign.** See below.
5. **Demonstration notebook.** A plain `examples/*.ipynb` walking from the
   physics (extended Hilbert space, loss versus leakage, transition matrices
   as quantum instruments, state-dependent versus state-independent
   transitions, why equalized rates is the honest classical-sampling
   approximation) through the API by physical scenario (Bell-pair loss to a
   50/50 survivor, relaxation with an analytic check, heralded loss), to a
   realistic neutral-atom model on a repetition-code memory circuit, accuracy
   comparisons against the dense oracle (including exact-versus-twirled
   over-rotation, where clifft's near-Clifford exactness is an accuracy edge),
   and a closing capability/gap summary. No cirq-superstaq dependency:
   external comparisons appear as cited result tables from the local
   differential.

### Known divergences from sqale's fast path

The equalized-rates feature matches sqale's fast path exactly in all marginal
statistics on Clifford circuits. This is structural, not luck: in a stabilizer
state a qubit at a jump site is either deterministic in Z (both simulators
take an exact path) or its measurement is exactly 50/50, so drawing the source
bit uniformly ahead of time reproduces the Born marginal identically. Two
joint-statistics divergences remain:

1. **Destination-partner correlations.** sqale Born-measures the qubit at the
   jump and conditions the destination on that bit, so the destination level
   (and hence the leaked qubit's readout symbol) is correlated with entangled
   partners. clifft's ahead-of-time source draw is independent of the
   simulator's internal collapse, so the joint distribution of the leaked
   qubit's symbol and its partners' records decorrelates while every marginal
   matches. Closing this requires the destination to depend on a runtime
   outcome, which is exactly the runtime-branching boundary of the constraint
   envelope. Second order in the leak rate; measured, not fixed, in this
   checkpoint.
2. **Gate-deterministic but untracked states.** sqale's exact-path test
   queries the stabilizer tableau; clifft's status tracker only knows
   instruction-determined states (after reset or by prior status). A qubit
   made deterministic by gate algebra (for example two consecutive H gates, or
   an X echo) takes sqale's exact path but clifft's approximate path, where
   the uniform source draw can leak or flip a qubit that deterministically
   could not. The error is bounded by the equalized rate per site (the same
   scale as the noise being modeled: misattributed noise, not a new O(1)
   error) and only applies to the deterministic-but-untracked subset of sites,
   which is small in typical QEC circuits (mid-round data qubits are genuinely
   entangled; ancillas are instruction-known after reset). The mitigation,
   maintaining a Clifford tableau in the trajectory sampler to promote a
   status to known whenever the tableau is deterministic, is compatible with
   the constraint envelope but is built only if the validation probes show it
   matters. It could not promote through a measurement in any case:
   measurement outcomes live in the simulator and are not pre-sampled.

### Validation

Confidence comes from a campaign, not just unit tests:

- **Differential on the full default model.** Run the local differential on
  the complete 2LQ-paper parameter set on repetition-code circuits: clifft
  versus sqale's fast sampler (expect shot-noise agreement on marginals), and
  both versus `simulate_true_distribution` on tiny circuits (quantifies the
  approximation error the two fast paths share). Results land in this
  document.
- **Divergence probes.** Two targeted differentials: a Bell pair with a
  source-dependent leak comparing the joint distribution of the leaked
  qubit's symbol and the partner's record (measures divergence 1), and a
  deterministic-but-untracked micro-case with consecutive H gates before the
  jump site (bounds divergence 2).
- **Committed oracle extensions.** Extend the in-tree density-matrix oracle
  with the equalized channel as an explicit CPTP map and with the ternary
  classifier, so the new features are cross-checked in CI without any
  external dependency.
- **Analytic micro-cases.** A plus state under an equalized leak
  (hand-computable dephasing and jump statistics) and a single twirled
  over-rotation versus its exact simulation.

### Success criterion

Full default-parameter repetition-code circuits run end to end on clifft with
no rejects, matching sqale's fast path within shot noise on all marginal
statistics, with the residual joint-statistics divergence quantified in this
document.

## Successor architecture (out of scope, named)

The three residual approximations of this checkpoint, the destination-partner
correlation loss, the deterministic-but-untracked class, and the equalization
dephasing itself, share one root cause: the destination of a fired jump should
depend on a measurement outcome that exists only inside the simulator at
runtime. Everything else stays ahead-of-time. The successor is therefore a
segmented-continuation ladder:

1. **Per-shot two-branch precompilation.** Trajectory sampling already fixes
   where jumps fire, ahead of time and state-independently; the only runtime
   choice is one bit per fired jump. A shot needs `2^f` precompiled
   continuations where `f` is the number of fired jumps in that shot, which at
   realistic rates is almost always 0 and occasionally 1. A segmented runtime
   selects the continuation on the collapse outcome; nothing is compiled at
   runtime.
2. **JIT or cached continuations**, only if branch counts or compile costs
   ever bite.

Any rung of this ladder does more than fix the divergences: once the runtime
can branch on the collapse outcome, the equalization is unnecessary and the
exact state-dependent no-jump filter can be applied as a localized spectral
Kraus branch (`a I + b Z_v`) on the active state, at which point clifft is no
longer matching the reference fast path but its dense ground truth, while
staying near-Clifford fast. That is the designated next tier after this
checkpoint, deliberately not part of it.
