# Leakage and Loss

!!! warning "Experimental"
    `clifft.noncomp` is experimental and may change between minor releases.

Real devices can leave the computational subspace (*leakage*) or lose the
physical carrier from its site (*loss*). These faults can persist until the
qubit relaxes, the carrier is recaptured, or the site is reset, so one event
can affect a sequence of later operations and detector outcomes. The
resulting correlations are not faithfully represented by independent Pauli
faults.

`clifft.noncomp` layers a classical status ledger over Clifft's ordinary
coherent simulation. A site with computational status remains in the quantum
state and can be in a superposition or entangled with other sites. A leaked or
lost site instead occupies one definite noncomputational level on each
trajectory.

Through `noncomp.Model`, the user defines the initial level distribution,
stochastic transitions between levels, how measurements are classified, and
policies such as whether reset restores a lost site. Because a jump can change
which later operations are skipped, which measurements use the classifier,
and whether a site is restored, one program compiled before sampling cannot
describe every trajectory. `noncomp.sample` therefore rewrites and compiles
trajectory-specific continuations when needed. The all-computational starting
continuation is compiled lazily and reused across shots; other continuations
are kept only while the shot that needs them executes, then discarded.

This page is the API walkthrough. The model and its simulation semantics
are described in [Noncomputational States](../theory/noncomputational.md).

## A model and its outputs

A `noncomp.Model` bundles an initial level distribution, per-gate
transition matrices, a classifier, and policy knobs. `noncomp.sample`
returns the usual records plus two sidecars: each circuit site's final status
and a per-measurement herald.

`Level` names the five rows and columns used by the model inputs.
`QubitStatus` describes the runtime status ledger: `g` and `e` both correspond
to `COMPUTATIONAL`, because a computational site remains in the coherent
quantum state, while `leak_g`, `leak_e`, and `lost` are tracked as distinct
classical statuses. An initial weight on `g` or `e` initializes the site in
the corresponding computational basis state; later evolution may put it in a
superposition or entangle it with other sites.

This minimal model has no transitions. It samples an initial mixture of
computational, leaked, and lost levels and shows how the classifier, final
status ledger, and herald output fit together.

```python
import numpy as np

from clifft import noncomp

Level = noncomp.Level

# Classifier P[symbol][level]: g and leak_g read 0, e and leak_e read 1,
# lost heralds (third symbol).
classifier = noncomp.Classifier(
    [
        [1, 0, 1, 0, 0],  # P(record 0 | level)
        [0, 1, 0, 1, 0],  # P(record 1 | level)
        [0, 0, 0, 0, 1],  # P(herald   | level)
    ]
)

model = noncomp.Model(
    initial_state=[0.95, 0.00, 0.02, 0.01, 0.02],  # P(g, e, leak_g, leak_e, lost)
    classifier=classifier,
)

r = noncomp.sample("M 0", model, shots=20_000, seed=1)

# final_status: (shots, num_qubits) of QubitStatus values.
status = r.final_status[:, 0]
lost = (status == noncomp.QubitStatus.LOST).mean()
assert abs(lost - 0.02) < 0.005

# heralds: (shots, num_measurements); 1 where the classifier heralded.
assert (r.heralds[:, 0] == (status == noncomp.QubitStatus.LOST)).all()

# symbols() reports the herald as value 2 instead of the placeholder bit.
symbols = r.symbols()
assert np.array_equal(symbols[:, 0] == 2, r.heralds[:, 0])
```

`measurements`, `detectors`, and `observables` are laid out exactly as in
ordinary sampling: a measurement of a leaked or lost site still occupies its
record slot, with the classifier supplying the bit. On a computational site,
Clifft performs the requested quantum measurement; the `g` and `e` classifier
columns can model computational-basis readout confusion for `M` and `MR`. On
a noncomputational site, the classifier replaces the quantum measurement and
applies to `M`, `MX`, and `MY` alike. When the classifier samples its herald
symbol, `heralds` marks the slot and the binary `measurements` entry holds a
uniformly drawn placeholder; `symbols()` folds the herald back in as a third
value per slot (0, 1, or 2), for comparing against tools that report loss
in-band.

`heralds[shot, slot]` means that the classifier emitted its third symbol at
that measurement slot. It does not identify when or where the underlying
transition occurred, so it is not an exact spacetime erasure flag. The
herald is side information for downstream analysis or adaptive decoding;
it is not folded into the binary detector record.

Omitting `initial_state` starts every site in `g`, which matches the standard
Clifft convention that all qubits start in $\lvert 0 \rangle$. A model capable
of leakage or loss requires a classifier whenever the circuit contains any
physical-qubit measurement, even if the measured site cannot itself become
noncomputational. `MPAD` is exempt because it appends a classical literal
rather than measuring a site.

## Transitions: hooks and inline annotations

A transition matrix `T[to][from]` is evaluated at an attached circuit
position. Each entry gives the probability of a transition event to a
destination level, which may be the source level; a column's deficit below 1
is the no-jump probability. There are
three ways to attach one to a circuit:

- **Gate hooks.** A `transitions` key that names a gate (`"CZ"`, `"S"`, …)
  evaluates its matrix after every occurrence of that gate. Keys naming
  instructions that never produce a circuit node (`MXX`/`MYY`/`MZZ`,
  `CH`/`CCX`/`CCZ`, identity no-ops) are rejected at model construction.
- **Inline references.** `LEVEL_TRANSITION[name] 0` evaluates the named matrix
  from the `transitions` mapping on site 0 at that circuit position. Any
  transition name can be referenced explicitly, whether or not it also names
  a gate hook.
- **Inline loss.** `LOSS(p) 0` loses the carrier at site 0 with probability
  `p` from any occupied level.

Before sampling, Clifft expands each gate hook into a
`LEVEL_TRANSITION[name]` annotation for every physical site operand of the
hooked gate. Classical feedback (`CX rec[-1] 0`) receives no annotation: a
record-conditioned Pauli is a frame update, not a physical execution of the
hooked gate. Use a hook when the same transition should follow
every occurrence of a gate; use an inline reference for selected circuit
positions or transitions with arbitrary names. `LOSS(p)` provides a
self-contained inline loss probability without requiring a transition matrix
in the model.

A transition back to `g` or `e` can represent relaxation or recapture into
the computational subspace; a reset represents active re-preparation.

```python
from clifft import noncomp

Level = noncomp.Level


def T(entries):
    """Build a 5x5 transition matrix T[to][from] from {(to, from): p}."""
    m = [[0.0] * 5 for _ in range(5)]
    for (to, frm), p in entries.items():
        m[to][frm] = p
    return m


classifier = noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])
leak = T({(Level.LEAK_E, Level.G): 0.01, (Level.LEAK_E, Level.E): 0.01})

model = noncomp.Model(
    transitions={
        "CZ": leak,  # hook: evaluated on each site after every CZ
        "manual_leak": leak,  # evaluated only where explicitly referenced
    },
    classifier=classifier,
)

circuit = """
    H 0
    CZ 0 1
    LOSS(0.005) 1
    LEVEL_TRANSITION[manual_leak] 0
    M 0 1
"""
r = noncomp.sample(circuit, model, shots=1000, seed=7)
assert r.measurements.shape == (1000, 2)
```

## What happens on a leaked or lost site

A unitary gate, ordinary noise channel, or classical correction touching a
leaked or lost site is skipped as a whole and has no effect on any operand.
Correlated-error instructions (`E`/`CORRELATED_ERROR` and
`ELSE_CORRELATED_ERROR`) are retained to preserve chain conditioning; their
Paulis still act on computational operands but are inert on the
noncomputational site. A single-qubit measurement keeps its record slot, but
the classifier rather than the measurement basis determines its result (`M`,
`MX`, and `MY` alike).

Losing one half of an entangled pair leaves the partner in the reduced
state; for a Bell pair, maximally mixed:

```python
import numpy as np

from clifft import noncomp

Level = noncomp.Level

lose_all = [[0.0] * 5 for _ in range(5)]
lose_all[Level.LOST][Level.G] = 1.0
lose_all[Level.LOST][Level.E] = 1.0

model = noncomp.Model(
    transitions={"S": lose_all},  # the hooked S loses its qubit with certainty
    classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]),
)

r = noncomp.sample("H 0\nCX 0 1\nS 0\nM 0\nM 1", model, shots=20_000, seed=2)

survivor = r.measurements[:, 1]
assert abs(survivor.mean() - 0.5) < 0.01  # partial trace: maximally mixed
assert (r.final_status[:, 0] == noncomp.QubitStatus.LOST).all()
assert (r.final_status[:, 1] == noncomp.QubitStatus.COMPUTATIONAL).all()
```

The classifier supplies the binary record bit before detectors and
observables are evaluated, so a noncomputational readout can produce
detector events:

```python
import numpy as np

from clifft import noncomp

Level = noncomp.Level

leak_up = [[0.0] * 5 for _ in range(5)]
leak_up[Level.LEAK_E][Level.G] = 1.0
leak_up[Level.LEAK_E][Level.E] = 1.0

model = noncomp.Model(
    transitions={"S": leak_up},
    classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]),  # leak_e reads 1
)

# A reference measurement, then the leaked one; the detector XORs them.
circuit = "M 1\nH 0\nS 0\nM 0\nDETECTOR rec[-1] rec[-2]"
r = noncomp.sample(circuit, model, shots=1000, seed=4)
assert (r.detectors[:, 0] == 1).all()  # the leaked qubit always reads 1
```

## State-dependent rates resolve at runtime

If a transition's rate differs between `g` and `e`, the jump probability for
a computational site in a superposition depends on the amplitudes at that
point in the shot. The draw happens at sample time against the live state, so
the statistics are exact — on $\lvert + \rangle$ with leak probability $p$
out of `g` only (and the leaked level reading 1), the marginal is
$\tfrac12 + \tfrac{p}{2}$:

```python
from clifft import noncomp

Level = noncomp.Level

p = 0.4
leak_from_g = [[0.0] * 5 for _ in range(5)]
leak_from_g[Level.LEAK_E][Level.G] = p

model = noncomp.Model(
    transitions={"S": leak_from_g},
    classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]),
)

r = noncomp.sample("H 0\nS 0\nM 0", model, shots=20_000, seed=5)
assert abs(r.measurements[:, 0].mean() - (0.5 + p / 2)) < 0.01
```

Resolving draws against the live state also preserves correlations that
ahead-of-time sampling cannot produce: when the jump's destination depends
on the source level, the leaked site's classified readout stays correlated
with its entangled partner. The semantics and the rank cost are in
[Noncomputational States](../theory/noncomputational.md).

## Policy knobs

- **`reset_restores_lost`** (default `False`): whether a reset on a lost site
  re-prepares it or is dropped. A reset always restores a *leaked* site; a
  measure-and-reset keeps its record either way, with
  its reset half following the same rule.
- **`damping`** (default `"exact"`): source-dependent total jump rates on a
  coherent dormant site require no-jump back-action. Exact simulation promotes
  such a site into the active state array. `"neglect"` omits that promotion
  and the no-jump back-action: an error of order
  $\lvert p_g - p_e \rvert$ per site. There is no error when $p_g = p_e$, so `LOSS(p)` is always exact.
- **`seed`**: same contract as ordinary sampling — a fixed seed is fully
  reproducible, `None` uses hardware entropy.
- **`max_rank`**: caps the compiled peak rank before allocating or growing the
  state for that module.
  The cap applies to each compiled module, including branches a given shot
  never takes, so it is conservative.

## Limits
- **Partner-error channels and leakage transport are not modeled.** An
  operation touching a leaked or lost site is dropped whole. The model
  does not add partner depolarization conditioned on that status or move
  leakage between sites.
- **Coherent leakage is outside the trajectory model.** Jumps into a
  noncomputational level are treated as incoherent, definite occupations.
- **`MPP` is not supported** under a model that can leak or lose qubits —
  a parity of levels outside the qubit subspace has no faithful single-bit
  record. Expand the parity readout into an explicit ancilla circuit; the
  ancilla's ladder gates then drop per the rules above.
- **`EXP_VAL` is not supported.** `NonComputationalSample` has no
  expectation-value output, so circuits containing these probes are rejected
  before sampling.
- **A classifier is required** whenever a model capable of leakage or loss
  meets a circuit containing any physical-qubit measurement. This is a
  model-level capability check, not a per-site reachability analysis.
- Under `damping="exact"`, a transition with source-dependent rates on a
  coherent dormant site promotes that site into the active state array,
  adding one unit of peak rank. An already-active site adds nothing, and later
  transition positions on a site that stays active do not stack, so the worst
  case is one unit per coherent *site*, not one per transition position.
  The [performance model](performance.md) otherwise applies unchanged.

Note that many of these can be revisited in future versions.

## Why there is no compile step

Ordinary Clifft separates `compile()` from `sample()` because a compiled
program is model-independent. Here it is not: the executable depends on
the model and on each shot's jump outcomes. When a jump changes downstream
semantics, `noncomp.sample` switches to a trajectory-specific continuation: a
version of the circuit rewritten and compiled for the event history observed
so far. The shot then resumes after the event. `noncomp.sample` therefore takes
the circuit and model together and compiles internally.

Each continuation uses the default optimization passes that preserve
measurement-record order. At the HIR stage, this means
`PeepholeFusionPass`; `StatevectorSqueezePass` is omitted because it can move
measurements. All of the default bytecode passes currently preserve record
order and are applied. See [Optimization Passes](../reference/passes.md) for
the full list.

This restriction matters because resolving a trapped transition may add a
forced, hidden trace-out measurement. Moving another measurement across that
collapse can change correlations. Record-order preservation is necessary but
not sufficient for resuming a shot: recompiling a continuation must also
reproduce the bytecode prefix already executed, because `resume()` reuses the
existing VM state directly. `noncomp.sample` therefore uses a fixed internal
pipeline and does not currently accept custom pass managers.

The all-computational starting continuation is compiled lazily and reused
across shots. A continuation for a noncomputational initial state or a trapped
transition is kept only while that shot executes it, then discarded. This
bounds retained compiled programs independently of the number of observed
event histories. A repeated trapped history is therefore recompiled; a
bounded cache can be added later if real workloads show that this cost matters.
The theory page explains
[how continuations compose with Clifft](../theory/noncomputational.md#how-this-composes-with-clifft).
