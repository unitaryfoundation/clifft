# Leakage and Loss

!!! warning "Experimental"
    `clifft.noncomp` is experimental and may change between minor releases.

Real devices leak (an atom is excited out of the qubit encoding) and lose
atoms (the atom leaves the trap). Ordinary Pauli noise cannot represent
either. `clifft.noncomp` samples circuits under a five-level leakage/loss
model: transition matrices describe when qubits jump between levels, and a
classifier describes what a measurement of a leaked or lost qubit records.

This page is the API walkthrough. The model and its simulation semantics
are described in [Noncomputational States](../theory/noncomputational.md).

## A model and its outputs

A `noncomp.Model` bundles an initial level distribution, per-gate
transition matrices, a classifier, and policy knobs. `noncomp.sample`
returns the usual records plus two sidecars: each qubit's final status and
a per-measurement herald.

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
```

`measurements`, `detectors`, and `observables` are laid out exactly as in
ordinary sampling: a measurement of a leaked or lost qubit still occupies
its record slot, with the classifier supplying the bit. When the
classifier samples its herald symbol, `heralds` marks the slot and the
binary `measurements` entry holds a uniformly drawn placeholder;
`symbols()` folds the herald back in as a third value per slot (0, 1,
or 2), for comparing against tools that report loss in-band.

Omitting `initial_state` starts every qubit in `g`. A model that can leak
or lose qubits requires a classifier if the circuit measures.

## Transitions: hooks and inline annotations

A transition matrix `T[to][from]` gives the probability of jumping between
levels when it fires; a column's deficit below 1 is "no jump". There are
three ways to attach one to a circuit:

- **Gate hooks.** A `transitions` key that names a gate (`"CZ"`, `"S"`, …)
  fires after every occurrence of that gate. Keys naming instructions that
  never produce a circuit node (`MXX`/`MYY`/`MZZ`, `CH`/`CCX`/`CCZ`,
  identity no-ops) are rejected at model construction.
- **Inline references.** `LEVEL_TRANSITION[name] 0` fires the named matrix
  on qubit 0 at that circuit position. Any key can be referenced this way,
  gate-named or not.
- **Inline loss.** `LOSS(p) 0` loses qubit 0 with probability `p` from any
  occupied level.

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
    transitions={"CZ": leak},  # hook: fires after every CZ
    classifier=classifier,
)

circuit = """
    H 0
    CZ 0 1
    LOSS(0.005) 1
    LEVEL_TRANSITION[CZ] 0
    M 0 1
"""
r = noncomp.sample(circuit, model, shots=1000, seed=7)
assert r.measurements.shape == (1000, 2)
```

## What happens on a leaked or lost qubit

Gates addressing a leaked or lost qubit are dropped, acting as the
identity on the surviving operands. Measurements keep their record slot
and read the classifier
(`M`, `MX`, and `MY` alike: the readout basis is incidental once the
qubit has left the computational subspace).

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

Classification happens before detectors are evaluated, so leakage
surfaces as detector events:

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

If a transition's rate differs between `g` and `e`, the jump probability on
a superposition depends on the amplitudes at that point in the shot. The
draw happens at sample time against the live state, so the statistics are
exact — on $\lvert + \rangle$ with leak probability $p$ out of `g` only
(and the leaked level reading 1), the marginal is $\tfrac12 + \tfrac{p}{2}$:

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
on the source level, the leaked qubit's classified readout stays
correlated with its entangled partner. The semantics and the rank cost
are in [Noncomputational States](../theory/noncomputational.md).

## Policy knobs

- **`reset_restores_lost`** (default `False`): whether a reset on a lost
  qubit re-prepares the site or is dropped. A reset always restores a
  *leaked* qubit; a measure-and-reset keeps its record either way, with
  its reset half following the same rule.
- **`damping`** (default `"exact"`): a site whose rates differ between
  `g` and `e` needs its qubit in the active array for the exact no-fire
  back-action; a coherent qubit still outside it is expanded at the site.
  `"neglect"` skips the expansion and the back-action: an error of order
  $\lvert p_g - p_e \rvert$ per site, and exactly zero for
  source-independent rates (`LOSS` always qualifies).
- **`seed`**: same contract as ordinary sampling — a fixed seed is fully
  reproducible, `None` uses hardware entropy.
- **`max_rank`**: caps the compiled peak rank before any state allocation.
  The cap applies to each compiled module, including branches a given shot
  never takes, so it is conservative.

## Limits

- **`MPP` is not supported** under a model that can leak or lose qubits —
  a parity of levels outside the qubit subspace has no faithful single-bit
  record. Expand the parity readout into an explicit ancilla circuit; the
  ancilla's ladder gates then drop per the rules above.
- **A classifier is required** whenever a capable model meets a measuring
  circuit; the error names the missing piece before sampling begins.
- Under `damping="exact"`, a state-dependent site on a coherent qubit
  outside the active array expands that qubit into it, adding one unit of
  peak rank at that site. An already-active qubit adds nothing, and later
  sites on a qubit that stays active do not stack, so the worst case is
  one unit per *qubit* held coherent across its sites, not one per site.
  The [performance model](performance.md) otherwise applies unchanged.

## Why there is no compile step

Ordinary Clifft separates `compile()` from `sample()` because a compiled
program is model-independent. Here it is not: the executable depends on
the model and on each shot's jump outcomes. `noncomp.sample` takes the
circuit and model together and compiles internally, caching one module
per distinct event history within the call. Pass a parsed
`clifft.Circuit` instead of text to share parsing across calls.
