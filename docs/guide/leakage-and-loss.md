# Leakage and Loss

!!! warning "Experimental"
    `clifft.noncomp` is new and actively evolving. Try it and share feedback,
    but expect its API and supported models to change as use cases develop.

Pauli noise acts within a qubit's two-dimensional computational subspace. Real
hardware can instead drive the state out of that subspace through *leakage*, or
lose the physical carrier from its site entirely through *loss*. Neither
process is a Pauli channel, so the site no longer holds an ordinary qubit
state.

Leakage and loss can persist until the qubit relaxes, the carrier is
recaptured, or the site is reset. One event can therefore affect a sequence of
later gates, measurements, and detector outcomes. Clifft tracks each site's
status alongside its ordinary coherent simulation. A computational site
remains in the quantum state and can be in a superposition or entangled; a
leaked or lost site occupies one definite noncomputational level in each shot.

Ordinary Clifft compiles a circuit once and then samples the same program
repeatedly. A noncomputational jump can change which later operations are
skipped, which measurements use the classifier, and whether a site is
restored. One program compiled before sampling cannot describe every shot, so
`noncomp.sample` rewrites and compiles the remaining circuit as jumps occur.

This page is the API walkthrough. The model and its simulation semantics
are described in [Noncomputational States](../theory/noncomputational.md).

## A model and its outputs

A `noncomp.Model` defines how qubit sites enter, leave, and are observed
outside the computational subspace. It contains an initial level distribution,
transition matrices, a measurement classifier, and policies such as whether a
reset restores a lost site. The classifier maps a known level to probabilities
for a binary measurement result and, optionally, a herald.

Clifft models five `Level` values: the computational levels `g` and `e`,
identified with $\lvert 0\rangle$ and $\lvert 1\rangle$; the leaked levels
`leak_g` and `leak_e`; and `lost`. These levels index transition-matrix rows and
columns and classifier probability columns.

`QubitStatus` describes a site's status in a sampled shot. Both `g` and `e`
correspond to `COMPUTATIONAL` because the site remains in the coherent quantum
state rather than occupying a known basis level. The statuses `LEAK_G`,
`LEAK_E`, and `LOST` name definite noncomputational levels. An initial weight
on `g` or `e` prepares the corresponding computational basis state; later
evolution may put the site in a superposition or entangle it with other sites.

`noncomp.sample` returns the usual measurement, detector, and observable
arrays, plus `final_status` for each circuit site and `heralds` for each
measurement.

### A minimal model

This example randomizes the initial state across all five levels and shows how
the classifier maps those levels to measurement outputs. It has no transitions
between levels after sampling begins.

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
    initial_state=[0.92, 0.03, 0.02, 0.01, 0.02],  # P(g, e, leak_g, leak_e, lost)
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

The output arrays have the same layout as ordinary sampling: every measurement
still occupies its record slot. On a computational site, Clifft performs the
requested quantum measurement. For `M` and `MR`, the classifier's `g` and `e`
columns can then add computational-basis readout confusion. On a leaked or lost
site, the classifier supplies the result in place of a quantum measurement and
applies to `M`, `MX`, and `MY` alike.

!!! note "Computational readout confusion is Z-basis only"
    The classifier's `g` and `e` columns apply to `M` and `MR`. On a
    computational site, `MX`, `MY`, `MRX`, and `MRY` perform their ordinary
    quantum measurements without consulting those columns. Use measurement
    noise such as `MX(p)` or `MY(p)` for readout errors in another basis. If
    the site is leaked or lost, the classifier supplies the result regardless
    of measurement basis.

When the classifier emits its herald symbol, `heralds` marks the slot and the
binary `measurements` entry holds a uniformly drawn placeholder. Detectors and
observables use that binary bit as usual; the herald remains separate.
`symbols()` folds the herald back in as a third value per slot (0, 1, or 2) for
comparison with tools that report loss in-band.

`heralds[shot, slot]` means that the classifier emitted its third symbol at
that measurement slot. It does not identify when or where the underlying
transition occurred, so it is not an exact spacetime erasure flag. The herald
is side information for downstream analysis or adaptive decoding.

Omitting `initial_state` starts every site in `g`, which matches the standard
Clifft convention that all qubits start in $\lvert 0 \rangle$. A model capable
of leakage or loss requires a classifier whenever the circuit contains any
physical-qubit measurement, even if the measured site cannot itself become
noncomputational. `MPAD` is exempt because it appends a classical literal
rather than measuring a site.

## Transitions: hooks and inline annotations

A transition matrix `T[to][from]` gives the probability of a transition event
from one source level to one destination level. The destination may equal the
source; a column's deficit below 1 is the no-jump probability.

Transitions act as noise associated with specific circuit positions. There are
three ways to place them:

- **Gate hooks.** A `transitions` key that names a gate, such as `"CZ"` or `"S"`,
  evaluates its matrix after every occurrence of that gate. Keys naming
  instructions that never produce a circuit node (`MXX`/`MYY`/`MZZ`,
  `CH`/`CCX`/`CCZ`, identity no-ops) are rejected at model construction. Use a
  hook to apply the same transition after every occurrence of a gate.
- **Inline references.** `LEVEL_TRANSITION[name] 0` evaluates the named matrix
  from the `transitions` mapping on site 0 at that circuit position. Any
  transition name can be referenced explicitly, whether or not it also names
  a gate hook. Use an inline reference at selected circuit positions or with a
  transition whose name is not a gate.
- **Inline loss.** `LOSS(p) 0` loses the carrier at site 0 with probability
  `p` from any occupied level. Use it for a self-contained loss probability
  that needs no matrix in the model.

Gate hooks are shorthand for inline references. Before sampling, Clifft
expands each hook into a `LEVEL_TRANSITION[name]` annotation for every physical
site operand of the hooked gate. Classical feedback (`CX rec[-1] 0`) receives
no annotation: a record-conditioned Pauli is a frame update, not a physical
execution of the hooked gate.

A transition back to `g` or `e` can represent relaxation or recapture into
the computational subspace; a reset represents active re-preparation.

### Combining hooks and inline annotations

This example applies one transition after every `CZ`, reuses the same matrix
at one explicitly selected position, and adds an independent inline loss.

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

Most unitary gates, ordinary noise channels, and classical corrections that
touch a leaked or lost site are skipped as a whole. For a multi-qubit
operation, this leaves every computational partner unchanged. A single-qubit
measurement is different: it keeps its record position, but the classifier
rather than the measurement basis supplies its result (`M`, `MX`, and `MY`
alike).

Correlated-error instructions (`E`/`CORRELATED_ERROR` and
`ELSE_CORRELATED_ERROR`) are the exception. Clifft retains them to preserve
chain conditioning; their Paulis still act on computational operands but are
inert on a noncomputational site.

### Loss from an entangled pair

Losing one half of an entangled pair leaves the partner in the reduced
state. For a Bell pair, that reduced state is maximally mixed, so its
computational-basis measurement is unbiased:

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
assert abs(survivor.mean() - 0.5) < 0.01  # the survivor's Z readout is unbiased
assert (r.final_status[:, 0] == noncomp.QubitStatus.LOST).all()
assert (r.final_status[:, 1] == noncomp.QubitStatus.COMPUTATIONAL).all()
```

### Classifier results feed detectors

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
the statistics are exact. In the example below, the `e` component of
$\lvert + \rangle$ contributes probability $\tfrac12$ to reading 1. The `g`
component has weight $\tfrac12$, and a fraction $p$ of it leaks to a level
that also reads 1. The total is therefore $\tfrac12 + \tfrac{p}{2}$.

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

Evaluating a transition against the quantum state at that point also preserves
correlations. When the destination depends on the source level, the leaked
site's classified readout stays correlated with its entangled partner. The
semantics and rank cost are described in
[Noncomputational States](../theory/noncomputational.md).

## Policy knobs

- **`reset_restores_lost`** (default `False`): whether a reset on a lost site
  re-prepares it or is dropped. A reset always restores a *leaked* site; a
  measure-and-reset always produces its classifier record, while this option
  controls only whether the reset half restores a lost site.
- **`damping`** (default `"exact"`): source-dependent total jump rates on a
  coherent site require no-jump back-action. Exact simulation may add one unit
  of peak rank for each affected coherent site; each additional unit doubles
  the state-array size. Repeated transition positions on a site that remains
  active do not add further rank. `"neglect"` omits this cost and the no-jump
  back-action, introducing an error of order
  $\lvert p_g - p_e \rvert$ per transition position. There is no error when
  $p_g = p_e$, so `LOSS(p)` is always exact. See the
  [performance model](performance.md) for how rank affects simulation cost.
- **`seed`**: same contract as ordinary sampling: a fixed seed is fully
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
- **`MPP` is not supported** under a model that can leak or lose qubits:
  a parity of levels outside the qubit subspace has no faithful single-bit
  record. Expand the parity readout into an explicit ancilla circuit; the
  ancilla's ladder gates then drop per the rules above.
- **`EXP_VAL` is not supported.** `NonComputationalSample` has no
  expectation-value output, so circuits containing these probes are rejected
  before sampling.
- **A classifier is required** whenever a model capable of leakage or loss
  meets a circuit containing any physical-qubit measurement. This is a
  model-level capability check, not a per-site reachability analysis.

## Why there is no compile step

Ordinary Clifft separates `compile()` from `sample()` because the same compiled
program can execute every shot. Here the executable depends on the model and
on each shot's jump outcomes. When a jump changes the remaining operations,
`noncomp.sample` builds a continuation for the events observed so far. The
continuation reproduces the executed prefix, changes the remaining operations,
and resumes the shot after the jump. `noncomp.sample` therefore takes the
circuit and model together and compiles internally.

Each continuation uses the default optimization passes that preserve
measurement-record order. At the HIR stage, this means
`PeepholeFusionPass`; `StatevectorSqueezePass` is omitted because it can move
measurements. All of the default bytecode passes currently preserve record
order and are applied. See [Optimization Passes](../reference/passes.md) for
the full list.

This restriction matters because resolving a trapped transition may add a
forced, hidden trace-out measurement. Moving another measurement across that
collapse can change correlations. Record-order preservation is necessary but
not sufficient for resuming a shot: recompiling the remainder must also
reproduce the bytecode prefix already executed, because `resume()` reuses the
existing VM state directly. `noncomp.sample` therefore uses a fixed internal
pipeline and does not currently accept custom pass managers. The theory page
explains
[how this composes with Clifft](../theory/noncomputational.md#how-this-composes-with-clifft).
