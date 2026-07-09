# Noncomputational States

!!! warning "Experimental"
    The leakage/loss API is experimental and may change between minor
    releases.

Pauli noise moves a qubit around inside its two-dimensional subspace. Real
devices also leave that subspace: an atom is excited to a level outside the
qubit encoding (*leakage*), or leaves the trap entirely (*loss*). No Pauli
channel can represent either: the state is no longer a qubit state at all.

Clifft models both with a five-level structure per qubit, a per-gate jump
process between levels, and a classifier that defines what a measurement of
a non-qubit level records. This page describes the model and its
simulation semantics; the [Leakage and Loss guide](../guide/leakage-and-loss.md)
shows the API.

## The five-level model

Every qubit carries the same fixed level set:

| index | name | category | meaning |
|---|---|---|---|
| 0 | `g` | computational | the qubit $\lvert 0 \rangle$ |
| 1 | `e` | computational | the qubit $\lvert 1 \rangle$ |
| 2 | `leak_g` | leaked | carrier present, outside the qubit subspace |
| 3 | `leak_e` | leaked | a second leaked level |
| 4 | `lost` | lost | carrier gone |

Two ingredients drive the dynamics:

- **Transition matrices.** A $5 \times 5$ matrix $T[\text{to}][\text{from}]$
  attached to a circuit position gives the probability of jumping between
  levels when that position executes. Every entry is a discrete jump event
  (diagonal entries project onto the source level), and a column's deficit
  below 1 is the no-jump probability. `LOSS(p)` is the special case of a
  uniform jump to `lost` from every occupied level.
- **A classifier.** A stochastic matrix $P[\text{symbol}][\text{level}]$
  mapping the level at readout to a recorded symbol: two record symbols,
  plus an optional third that heralds the measurement (typically loss).

## Statuses: classical occupation, per trajectory

Each sampled shot is one trajectory. Along a trajectory the sampler keeps a
classical *status* per qubit: computational, `leak_g`, `leak_e`, or `lost`.
A noncomputational status is definite: the trajectory knows which level the
qubit occupies. A computational qubit carries no level claim; whether it is
$\lvert 0 \rangle$, $\lvert 1 \rangle$, or a superposition is tracked by
the simulator, not the ledger.

Classical tracking is exact for this trajectory model because jumps are
treated as incoherent: a noncomputational population carries no coherence
with the computational subspace. Under that assumption, recording
occupation per trajectory discards nothing; coherent leakage is outside
the model's scope.

## The vacated carrier

When a qubit jumps out of the computational subspace, its amplitudes still
matter: for an entangled qubit they define the partner's reduced state.
The jump therefore traces the qubit out: a hidden collapse resolves its
computational amplitude, the partner keeps the correct partial-trace
statistics, and the simulated cell (the *carrier*) is left parked while
the status ledger records the occupied level.

With the carrier vacated:

- **Gates drop.** An operation with no representable effect on a leaked or
  lost operand (a single- or two-qubit gate, a noise channel, classical
  feedback onto the site) is excised whole, acting as the identity on the
  surviving operands.
- **Measurements classify.** A measurement of a noncomputational level is
  not a Born measurement of a qubit; the classifier defines its record
  symbol. The record slot is preserved, so `rec[-k]` references, detectors,
  and observables are laid out exactly as in the noiseless circuit. The
  readout basis is incidental on a vacated carrier: `M`, `MX`, and `MY`
  all classify identically. A multi-qubit parity measurement (`MPP`) has no
  faithful single-bit substitution and is rejected up front.
- **Restoration begins with a reset.** A qubit returns to the computational
  subspace only through an operation that re-prepares the carrier: a reset
  (a leaked qubit always; a lost qubit only when the model opts in) or a
  transition whose destination is a computational level.

Leakage becomes visible to error correction through the classifier:
leaked and lost levels are classified into the measurement record before
detectors are evaluated, so they surface as detector events.

## State-dependent rates

Whether a jump fires can depend on the state. If a transition leaks out of
`g` and `e` at different rates, the fire probability on a superposition is
not a number known ahead of time — it depends on the amplitudes at that
point in that trajectory, and the no-fire branch back-acts on the state
(the general-measurement update from Kraus operators
$\smash{\sqrt{1-p_g}}\,\lvert g\rangle\langle g\rvert +
\smash{\sqrt{1-p_e}}\,\lvert e\rangle\langle e\rvert$).

Clifft resolves each potential jump *at its circuit position against the
live state*: the fire decision is drawn from the simulator's own
amplitudes, and when a jump lands, the remainder of the shot is rewritten
under the recorded event and execution resumes. Sampling is exact for
state-dependent rates, including the correlations ahead-of-time sampling
cannot produce: when the jump's destination depends on the source level,
the leaked qubit's readout stays correlated with its entangled partner.

The one approximation on this path is opt-in. The exact no-fire
back-action acts on the qubit's amplitudes, so a source-dependent site on
a coherent qubit that is still *dormant* (held in the Clifford frame,
outside the active array) expands that qubit into the array, one unit of
active dimension at that site. A qubit already active costs nothing more,
and later sites on a qubit that stays active do not stack.
`damping="neglect"` skips the expansion and the back-action, a
survivorship tilt of order $\lvert p_g - p_e \rvert$ per site and exactly
zero at source-independent rates. The default is exact.

## Validation

The implementation is checked at four levels:

1. **Closed forms.** Micro-circuits with hand-derived outcomes: partial
   trace of a Bell pair under loss, marginals under state-dependent leak
   rates, classifier confusion arithmetic.
2. **Sharp probes.** Fixed-seed tests for behavior that sampled
   distributions alone cannot check: the Bell-pair correlation that
   distinguishes live-state draws from ahead-of-time draws, and the
   damping boundary/null pair that separates `exact` from `neglect`
   exactly where the $\lvert p_g - p_e \rvert$ bound says they must
   differ and agree.
3. **A brute-force enumerator.** A dense density-matrix reference computes
   full record distributions for small circuits; sampled frequencies are
   checked against it. The enumerator shares the channel definitions with
   the sampler, so it is one independent implementation of the *dynamics*,
   not of the model; the closed forms above anchor the model itself.
4. **Statistical distance at scale.** Total-variation-distance checks on
   repetition-code rounds at realistic rates, bounded by shot noise.
