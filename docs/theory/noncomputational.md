# Noncomputational States

!!! warning "Experimental"
    The leakage/loss API is experimental and may change between minor
    releases.

Pauli noise acts within a qubit's two-dimensional computational subspace. Real
hardware can instead drive the state out of that subspace through *leakage*, or
lose the physical carrier from its site entirely through *loss*. Neither
process is a Pauli channel, so the site no longer holds an ordinary
qubit state.

Clifft models these processes with a hybrid quantum-classical trajectory
model. Within the computational subspace, the state keeps its full coherent
dynamics and entanglement. A leaked or lost site instead has a definite,
classically tracked occupation on each trajectory. Transitions between the two
are stochastic quantum jumps, including their back-action on the computational
state.

This page explains the model and how it composes with Clifft's
[factored-state simulation](overview.md). The
[Leakage and Loss guide](../guide/leakage-and-loss.md) shows the Python API.

## The effective five-level site

Each circuit wire denotes a fixed physical *site*. For modeling leakage and
loss, Clifft uses the effective per-site space

$$
\mathcal H_{\mathrm{site}}
=
\mathcal H_C \oplus \mathcal H_N,
\qquad
\mathcal H_C = \operatorname{span}\{\lvert g\rangle,\lvert e\rangle\}
\cong \operatorname{span}\{\lvert 0\rangle,\lvert 1\rangle\},
$$

with

$$
\mathcal H_N = \operatorname{span}\{
\lvert \mathrm{leak\_g}\rangle,
\lvert \mathrm{leak\_e}\rangle,
\lvert \mathrm{lost}\rangle
\}.
$$

The table below lists these levels and their categories:

| index | name | category | meaning |
|---|---|---|---|
| 0 | `g` | computational | $\lvert g\rangle$, identified with logical $\lvert 0\rangle$ |
| 1 | `e` | computational | $\lvert e\rangle$, identified with logical $\lvert 1\rangle$ |
| 2 | `leak_g` | leaked | carrier present, outside the qubit subspace |
| 3 | `leak_e` | leaked | a second leaked level |
| 4 | `lost` | lost | carrier absent from the site |

Under the model's incoherent-jump assumption, there is no coherence between
$\mathcal H_C$ and $\mathcal H_N$, or between different noncomputational
levels. The labels `g` and `e` name the computational basis levels used as
matrix indices. They do not imply that a computational site occupies a
definite level: its state may be any superposition or entangled state in
$\mathcal H_C$. The status ledger therefore records one computational status
rather than separate `g` and `e` occupations. We intentionally do not use
$\lvert 0 \rangle, \lvert 1 \rangle$ for these level labels to underscore the
distinction between the status ledger and the quantum state for a site with
computational status.

This gives a hybrid state representation. Each trajectory has two components:
Clifft's ordinary factored quantum state over the computational sites, and a
classical status ledger over all sites. The ledger entries are computational,
`leak_g`, `leak_e`, or `lost`. A computational entry leaves the corresponding
site in the quantum state. A noncomputational entry names one definite level.
The sections below describe the dynamics and interactions of these states.

## Model inputs

The user defines a model that determines how sites jump between levels and how
measurements are classified.

A transition matrix $T[\mathrm{to}][\mathrm{from}]$ attaches stochastic jumps
to circuit positions. An entry gives the probability of a jump from one source
level to one destination level. Diagonal entries are still jump events because
they project onto their source level. The unused probability in a column is the
no-jump probability. For computational sources, these probabilities define the
jump and no-jump Kraus branches described below.

For a computational source $s \in \{g,e\}$, define its total fire
probability and its computational-destination probabilities by

$$
p_{\mathrm{fire}}(s) = \sum_{\ell} T[\ell][s],
\qquad
p_{\mathrm{comp}}(s \mathbin{\to} d) = T[d][s],
\quad d \in \{g,e\}.
$$

The unconditional probability of firing into the noncomputational subspace is
the remainder

$$
p_N(s)
= p_{\mathrm{fire}}(s) - T[g][s] - T[e][s]
= T[\mathrm{leak\_g}][s]
  + T[\mathrm{leak\_e}][s]
  + T[\mathrm{lost}][s].
$$

The no-jump probability from $s$ is $1-p_{\mathrm{fire}}(s)$. A diagonal
entry $T[s][s]$ is still a jump event that lands back on its source; it is not
part of the no-jump branch. The matrix entries are branch weights. A jump from
a computational source acts on the live quantum state, while a jump from a
noncomputational source can be sampled from its already definite level.

A measurement classifier $P[\mathrm{symbol}][\mathrm{level}]$ defines the
recorded result for each level. It has two binary record symbols and may have
a third herald symbol, typically for loss. For example, the `leak_g` column
can assign independent probabilities to records 0 and 1.

## Jump back-action

When a jump leaves $\mathcal H_C$, its Kraus branch selects a computational
source level and a destination. The source is resolved against the live
quantum state, not drawn ahead of execution. Clifft realizes this back-action
as a hidden collapse and removes the site from the coherent state. The hidden
collapse writes no visible measurement record.

For an entangled site, the same collapse updates its partners. Each trajectory
keeps the partner state conditioned on the selected jump branch; averaging
over trajectories recovers the correct reduced-state statistics. A Bell-pair
partner, for example, is maximally mixed in the ensemble after
source-independent loss of the other half.

When no jump occurs, the state is also updated. If the total jump rates from
`g` and `e` are $p_g$ and $p_e$, the no-jump branch applies

$$
K_{\mathrm{stay}}
=
\sqrt{1-p_g}\,\lvert g\rangle\langle g\rvert
+
\sqrt{1-p_e}\,\lvert e\rangle\langle e\rvert.
$$

Unequal rates therefore change the surviving coherent state.

A transition
from one noncomputational level to another is purely classical, as
is the status update. A transition back to `g` or `e` restores the site to the
computational state at that basis level.

## Operations after a jump

Once a site has noncomputational status, later circuit operations follow the
policy below.

- A unitary gate, ordinary noise channel, or classical correction touching the
  site is skipped as a whole and has no effect on any operand. Correlated-error
  instructions (`E`/`CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR`) are
  retained to preserve chain conditioning; their Paulis still act on
  computational operands but are inert on the noncomputational site.
- Measurements use the classifier to determine the measured result and store it in the same record slot reserved for that measurement. Once the site is outside $\mathcal H_C$, the classifier rather than the measurement basis determines the recorded result. For a measure-reset form, the reset half
  follows the restoration policy below. The current policy rejects multi-qubit
  parity measurements (`MPP`) because they have no faithful single-bit
  substitution; a future policy could define other semantics.
- A reset restores a leaked site. It restores a lost site only when the model
  enables `reset_restores_lost`. A transition to `g` or `e` can also model
  relaxation or recapture.

This policy describes the current implementation. Future work may add additional configurable
semantics, for example by adding partner depolarization
conditioned on a noncomputational operand or transporting leakage between
sites.

The visible binary result occupies the same record slot as the original
measurement. Later `rec` references, detectors, observables, and classical
feedback all consume that substituted bit. When the classifier emits its third
symbol, a separate herald marks the slot and the binary record receives a
uniformly drawn placeholder. The herald identifies the readout, not the time or
location of the underlying jump, so it is not an exact spacetime erasure flag.

## How this composes with Clifft

Ordinary Clifft compiles a circuit once and reuses the program for many shots.
The compiler absorbs deterministic Clifford evolution into an offline frame,
localizes the remaining Pauli operations, and emits bytecode for the factored
Schrodinger virtual machine.

For a transition on a computational site, the compiled program carries the
total jump probability for each computational source and the separate weights
for `g` and `e` destinations. Their remainder is the combined weight for
`leak_g`, `leak_e`, and `lost`. The virtual machine handles a computational
destination directly. For a noncomputational destination, it returns control
to the trajectory driver, which chooses the specific level from the original
five-level matrix. Transitions whose source is already noncomputational are
handled while constructing the trajectory-specific continuation.

A noncomputational jump can change which later gates are skipped, which measurements use the classifier, and whether a site returns to the computational state. Those choices depend on the live state and differ between shots, so one model-independent program cannot describe every trajectory. `noncomp.sample` therefore interleaves execution with compilation of trajectory-specific continuations. A continuation is the circuit rewritten and compiled for the noncomputational events observed so far and the resulting status ledger. Its prefix matches the program already executed, while its remaining operations reflect the updated trajectory.

Execution continues through events that can be handled inline. When a fire requires different downstream semantics, the driver traps, records the event, compiles the matching continuation, and resumes after the trapped site. The common all-computational starting program is reused across shots. Other continuations live only for the current shot, so retained programs do not grow with the number of observed event histories.

Each continuation still uses Clifft's normal compiler and SVM architecture.
Clifford operations are absorbed ahead of time, Pauli products are localized
before execution, and the VM acts only on local array axes and frame data.

## Active-dimension cost

With the default exact damping policy, the simulation is exact for this hybrid
quantum-classical model. Most transition positions do not increase the
[active dimension](overview.md#the-factored-state-representation). A source
already in the active state uses its existing array axis. A definite dormant
source is determined entirely from the Clifford and Pauli frames.

The exceptional case is a coherent dormant site with source-dependent total
jump rates, $p_g \neq p_e$. The $K_{\mathrm{stay}}$ operator above is then not
proportional to the identity, and is non-Clifford. Exact simulation must promote the site from dormant to active.

If this occurs frequently, it can increase the peak active dimension $k$, the
dominant scaling quantity for Clifft's runtime. Users can instead choose
`damping="neglect"`, which omits the no-jump back-action. This is exact when
$p_g=p_e$ (since it is proportional to the identity in this case), and otherwise introduces a survivorship tilt of order
$\lvert p_g-p_e\rvert$ per position. Under this policy, these transition
positions do not increase $k$.
