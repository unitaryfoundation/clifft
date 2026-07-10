# Noncomputational States

!!! warning "Experimental"
    The leakage/loss API is experimental and may change between minor
    releases.

Pauli noise acts within a qubit's two-dimensional computational subspace. Real
hardware can instead drive the state out of that subspace through *leakage*, or
lose the physical carrier from its site entirely through *loss*. Neither
process is a Pauli channel. Afterward, the site no longer holds an ordinary
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

Each circuit wire denotes a fixed physical *site*. When that site is occupied
within the computational subspace, it holds a qubit. For modeling leakage and
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

The five level labels are the row and column indices of the model matrices:

| index | name | category | meaning |
|---|---|---|---|
| 0 | `g` | computational | $\lvert g\rangle$, identified with logical $\lvert 0\rangle$ |
| 1 | `e` | computational | $\lvert e\rangle$, identified with logical $\lvert 1\rangle$ |
| 2 | `leak_g` | leaked | carrier present, outside the qubit subspace |
| 3 | `leak_e` | leaked | a second leaked level |
| 4 | `lost` | lost | carrier absent from the site |

The labels `g` and `e` name basis levels against which transition rates are
defined. A computational site need not occupy either one definitely: it may be
in any superposition or entangled state within $\mathcal H_C$.

Each sampled shot is one trajectory. Its state has two parts. Sites with
computational status remain in Clifft's ordinary factored quantum state. The
sampler also keeps a classical status ledger per site. Its entries are
computational, `leak_g`, `leak_e`, or `lost`. The computational status
deliberately does not choose between `g` and `e`; that information remains
quantum. Each noncomputational status names one definite level.

This representation is exact under the model's incoherent-jump assumption.
There is no coherence between $\mathcal H_C$ and $\mathcal H_N$, or between
different noncomputational levels. Coherent leakage is outside the model's
scope.

## Model inputs

A transition matrix $T[\mathrm{to}][\mathrm{from}]$ attaches stochastic jumps
to circuit positions. An entry gives the probability of a jump from one source
level to one destination level. Diagonal entries are still jump events because
they project onto their source level. The unused probability in a column is the
no-jump probability. `LOSS(p)` is the special case of a uniform jump to `lost`
from every occupied level.

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

Unequal rates therefore change the surviving coherent state. A transition
from one definite noncomputational level to another is purely classical, as
is the status update. A transition back to `g` or `e` restores the site to the
computational state at that basis level.

## Operations after a jump

Once a site has noncomputational status, later circuit operations follow the
current structural policy.

- A gate, noise channel, or classical correction touching the site is skipped
  whole and acts as the identity on every operand.
- `M`, `MX`, `MY`, `MR`, `MRX`, and `MRY` keep their measurement slot and use
  the classifier. The basis is incidental once the site is outside
  $\mathcal H_C$. For a measure-reset form, the reset half follows the
  restoration policy below. A multi-qubit parity measurement has no faithful
  single-bit substitution and is rejected.
- A reset restores a leaked site. It restores a lost site only when the model
  enables `reset_restores_lost`. A transition to `g` or `e` can also model
  relaxation or recapture.

Gate dropping is the current interaction design, not a general law of
leakage. The model does not yet add partner depolarization conditioned on a
noncomputational operand or transport leakage between sites. Future policies
may define those interactions differently.

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

A noncomputational jump can change which later gates are skipped, which
measurements use the classifier, and whether a site returns to the
computational state. Those choices depend on the live state and differ between
shots, so one model-independent program cannot describe every trajectory.
`noncomp.sample` therefore interleaves execution with cached continuation
compilation. The workflow is shown below.

```text
Circuit + model
      |
      v
Annotate transition positions
      |
      v
Rewrite for the current event history
      |
      v
Trace -> optimize -> lower -> cache continuation
      |
      v
Execute against the live factored state
      |
      +-- event handled inline ------------------------> continue
      |
      `-- fire requiring a new continuation -> resumable trap
                                                   |
                                                   v
                                      Record jump and update status
                                                   |
                                                   v
                                      Rewrite, compile, cache, resume
```

Each continuation still uses Clifft's normal compiler and SVM architecture.
Clifford operations are absorbed ahead of time, Pauli products are localized
before execution, and the VM acts only on local array axes and frame data. The
event history determines which continuation is compiled; it does not move
global circuit topology into the runtime.

## Active-dimension cost

Most transition positions do not increase the
[active dimension](overview.md#the-factored-state-representation). A source
already in the active state uses its existing array axis. A definite dormant
source is resolved from the Clifford and Pauli frames. When $p_g=p_e$, the
no-jump operator is a scalar, so a coherent dormant source can also remain
outside the active array without approximation.

Only an exact, source-dependent transition on a coherent dormant site must
expand that site into the active state. This adds one active dimension at the
transition position. While the site remains active, later transitions on it
add no further dimension.

`damping="neglect"` avoids that expansion by omitting the no-jump back-action.
It is exact when $p_g=p_e$ and otherwise introduces a survivorship tilt of
order $\lvert p_g-p_e\rvert$ per position. The default policy keeps the exact
back-action.

## Current scope

This model deliberately covers incoherent leakage and loss trajectories. It
does not currently represent coherent superpositions involving
$\mathcal H_N$ or construct adaptive detector error models. The sampler
returns measurement records, detectors, observables, per-site final statuses,
and heralds for downstream analysis or decoding.
