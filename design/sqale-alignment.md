# sqale-sim alignment spike

Status: prototype spike on the `codex/noncomputational-structural-mvp` feature
branch. This document is the standalone record of a one-time validation: how the
clifft noncomputational MVP compares to Infleqtion's sqale-sim leakage simulator,
where they agree, where they differ, and how fast clifft is by comparison. It is
a decision artifact, not a spec.

## Why

The noncomputational MVP was built to recreate sqale-sim's loss/leakage
simulation capability in clifft's near-Clifford style, with the hope of being
substantially faster than their cirq-stabilizer prototype. Before investing in
more features, this spike answers three questions:

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
- Two important internal behaviors:
  - It propagates the classical leakage distribution and samples late; an
    `oversample` option amortizes one stabilizer run over many leakage draws.
    That amortization requires terminal measurements, so it does not apply to
    multi-round QEC circuits.
  - For a leakage jump on a qubit that is not in a definite computational state,
    it does not handle the source-dependence exactly: it equalizes the
    computational columns and collapses the qubit, with a code comment noting
    this is a conservative approximation. Only its dense reference is exact.
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

## Gaps

Matches today (clifft equals sqale's dense truth): initial population, binary
readout classifier, pure-leakage transitions that are either known-source or
source-independent, and the performance shown above.

Two gaps define the boundary of the exact overlap:

- Gap A, computational-to-computational transfer. sqale's matrices may move
  population between computational levels (for example `g -> e`); sqale moves the
  amplitude and the readout changes, whereas clifft tracks the level label but
  does not flip the simulator state, so it silently ignores the readout effect.
  clifft should at least reject or warn on a computational-destination
  transition; modeling it faithfully means treating it as a Pauli error.
- Gap B, source-dependent leakage on a superposed qubit. clifft rejects this
  (it cannot pick a source column for an unknown computational state). Note that
  sqale's fast path does not do this exactly either; it equalizes the columns and
  collapses the qubit. So parity here is implementing the same equalized-rates
  approximation, not matching an exact behavior.

Consequence: sqale's real device matrices mix computational transfer with
leakage and apply it on superposed qubits, so they hit both gaps. The exact
overlap is the clean subset (known-source or source-independent, pure leakage).

## Ranked next work, if continuing

1. Equalized-rates / unknown-source approximation (Gap B). This is what widens
   the overlap from the clean subset to sqale's real circuits, and sqale itself
   only approximates here, so it is achievable parity. The MVP design already
   anticipates an `unknown_source_policy = equalize_rates` knob.
2. Ternary loss herald (a readout "loss" symbol). Leakage detection is the point
   of QEC leakage studies, and it is needed to compare measurement records once a
   lost level is populated.
3. Gap A guard: reject or warn on a computational-destination transition. A cheap
   correctness and trust fix; full computational-transfer modeling as a Pauli
   error is separate and lower priority.
4. Coherent errors (over-rotations, phase). Large in sqale's real model, but not
   the validation bottleneck; clifft could model these exactly rather than
   twirling them, which would be an accuracy advantage.
5. Rewrite/compile caching. A performance optimization; not needed for parity
   given the current ~60x margin.
6. Movement / atom-site semantics, correlated two-qubit leakage, exact diagonal
   filters. Future or out of scope.

## Conclusion

clifft is correct where it overlaps sqale-sim and is decisively faster (~60-90x)
on QEC-style circuits, so the approach is worth continuing. The single
highest-value next feature is the equalized-rates approximation (Gap B), which
extends the exact overlap to the kind of circuits sqale actually runs.
