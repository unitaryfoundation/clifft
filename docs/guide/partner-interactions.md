# Partner Effects from Leakage and Loss

When a physical gate encounters a leaked or lost operand, Clifft drops the
gate. By default, its computational partner is unchanged. Some effective noise
models instead assign a residual disturbance to that partner: control pulses
can still affect it even though the intended two-qubit operation cannot act.

Configure those residual effects with `noncomp.PartnerEffect`. It combines a
single-qubit Pauli channel with an optional leakage attempt on the partner.
The original noncomputational operand retains its status. Leakage spreading
can affect later gates and measurements in the same shot.

Start with the [Leakage and Loss guide](leakage-and-loss.md) for the five-level
model, classifiers, and sampling results.

## Choose where effects apply

| Need | Configuration |
|------|---------------|
| The same effect across physical two-qubit gates | `Model(gate_partner_effects={...})` |
| A different effect for one gate type and direction | `Model(interactions=[InteractionRule(...)])` |
| An effect at an individual circuit position | `LEAKAGE_INTERACTION(...)` or `LOSS_INTERACTION(...)` |

Model settings expand into conditional circuit annotations. Explicit
annotations execute at their own positions and compose with generated ones.
They do not override or suppress another instruction.

## Model-wide defaults

Defaults are keyed by `"leaked"` or `"lost"`. `"leaked"` matches either
`LEAK_G` or `LEAK_E`. Each configured default applies in both operand
directions, provided the partner is computational. An omitted status has no
partner effect.

This model fully depolarizes the computational partner of a leaked operand.
It leaves loss interactions at the default drop behavior:

```python
from clifft import noncomp

model = noncomp.Model(
    classifier=noncomp.Classifier(
        [[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]]
    ),
    gate_partner_effects={
        "leaked": noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25)),
    },
)
result = noncomp.sample("LEAKAGE(1) 0\nCX 0 1\nM 1", model, shots=4096, seed=1)
assert abs(result.measurements[:, 0].mean() - 0.5) < 0.04
assert (result.final_status[:, 0] == noncomp.QubitStatus.LEAK_G).all()
assert (result.final_status[:, 1] == noncomp.QubitStatus.COMPUTATIONAL).all()
```

The classifier is required because this circuit can leak and measures a
physical qubit. Interaction annotations themselves add no measurement slots
and need no classifier. Existing status probes can observe spreading without
a classifier.

### Probability conventions

`PartnerEffect` accepts keyword arguments:

| Argument | Meaning | Default |
|----------|---------|---------|
| `pauli=(px, py, pz)` | Mutually exclusive X, Y, and Z probabilities on the partner | `(0, 0, 0)` |
| `spread_probability=p` | Probability of leaking the partner after the Pauli channel | `0` |

The identity probability is `1 - px - py - pz`. All probabilities must be
finite and in `[0, 1]`, and the three Pauli probabilities must sum to at most
one. The spreading probability is separate from this sum: a shot can undergo
both a Pauli error and a leakage event.

Full depolarization has equal I, X, Y, and Z probabilities, so use
`pauli=(0.25, 0.25, 0.25)`. In Stim's `DEPOLARIZE1(p)` convention this is
`p=0.75`; `p=1` always chooses a nonidentity Pauli and is a different channel.
More generally, `DEPOLARIZE1(p)` corresponds to `pauli=(p/3, p/3, p/3)`.

Spreading uses the existing source-preserving `LEAKAGE(p)` channel on the
partner: `g -> leak_g` and `e -> leak_e`. These are the partner's levels
after its Pauli channel, not a copy of the original leaked operand's level.
The original leaked operand stays leaked. The leakage event applies the
quantum jump's collapse to any entangled partners as usual.

Spreading is supported only for a leaked source. A lost-source effect with a
nonzero `spread_probability` is rejected. `PartnerEffect()` is a zero effect;
omitting configuration or specifying zero effects preserves default seeded
results.

## Specific rules replace defaults

`InteractionRule` selects a gate type, the source's operand position, and its
status. Its `effect` replaces the **entire** matching default. It does not
inherit omitted components from that default. A zero effect disables the
default for that particular gate, direction, and status.

```python
from clifft import noncomp

model = noncomp.Model(
    gate_partner_effects={
        "leaked": noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25)),
    },
    interactions=[
        noncomp.InteractionRule(
            gate="CX",
            source_operand=0,
            source_status="leaked",
            effect=noncomp.PartnerEffect(),
        ),
        noncomp.InteractionRule(
            gate="CX",
            source_operand=1,
            source_status="leaked",
            effect=noncomp.PartnerEffect(
                pauli=(0.01, 0, 0.1), spread_probability=0.02
            ),
        ),
    ],
)
```

For `CX 2 5`, the first rule applies when qubit 2 is leaked and qubit 5 is
computational. The second applies when qubit 5 is leaked and qubit 2 is
computational. `source_operand` is a position within each ordered pair,
either 0 or 1; it is not a physical qubit ID. The same convention applies to
symmetric gates such as `CZ`.

Gate aliases are canonicalized. Two rules for `CX` and `CNOT` with the same
direction and status conflict and are rejected. Different directions or
statuses are independent. `PartnerEffect` and `InteractionRule` are immutable,
and `Model` copies the supplied configuration when constructed.

### Supported gates

Defaults and rules apply to native two-qubit unitary gates, including
`CX`, `CY`, `CZ`, swap gates, square-root Pauli-pair gates, and the
parameterized `R_XX`, `R_YY`, and `R_ZZ` rotations. A rule for a rotation
applies to every angle. Both operands must be physical qubit targets.

Noise instructions, measurements, identity no-ops, and record-controlled
virtual feedback do not trigger effects. Neither do single-qubit gates or
multi-target Pauli-product gates such as `SPP` and `R_PAULI`, even when a
particular product contains two qubits.

`CH`, `CCX`, and `CCZ` are decomposed during parsing. Their resulting
constituent gates follow the configured rules; the original names cannot key
an interaction rule. Consequently, different physical decompositions can
produce different leakage models. Specify the circuit at the gate level
appropriate for the modeled hardware.

A swap gate touching a noncomputational operand is also dropped, followed by
any configured partner effect. It does not transport the leaked or lost
carrier to the other site.

## Explicit circuit annotations

The circuit instructions expose the same effects at individual positions:

```text
LEAKAGE_INTERACTION(px, py, pz, spread_probability) source partner
LOSS_INTERACTION(px, py, pz) source partner
```

The first executes only when `source` is leaked and `partner` is
computational; the second requires a lost source and a computational partner.
All arguments are required. Targets must be distinct plain qubit indices.
Multiple pairs are processed in written order, so spreading from one pair
can change a later pair's condition. An annotation need not follow a gate.

```python
from clifft import noncomp

model = noncomp.Model(
    classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])
)
result = noncomp.sample(
    """
    LEAKAGE(1) 0
    LOSS(1) 2
    LEAKAGE_INTERACTION(1, 0, 0, 1) 0 1
    LOSS_INTERACTION(1, 0, 0) 2 3
    HERALD_LEAKAGE_EVENT 0 1
    M 1 3
    """,
    model,
    shots=16,
    seed=2,
)
assert result.measurements.all()
assert (result.final_status[:, 1] == noncomp.QubitStatus.LEAK_E).all()
assert (result.final_status[:, 3] == noncomp.QubitStatus.COMPUTATIONAL).all()
```

The X error takes qubit 1 from `g` to `e` before the certain leakage event,
so it ends in `LEAK_E`. Qubit 3 receives an X error but remains computational.
The two interaction instructions produce no records: the four visible bits
come from the two status probes and two measurements.

These annotations require `noncomp.sample`. Ordinary `clifft.compile` rejects
them, just as it rejects `LEAKAGE` and `LOSS`.

## Ordering with level-transition hooks

Each physical gate expands in this order:

1. The physical gate itself.
2. Model-generated interaction annotations.
3. Model-generated `LEVEL_TRANSITION` hooks, in operand order.
4. The next handwritten circuit instruction.

If both operands enter computational, the physical gate executes and its
generated interaction annotations have no effect. If exactly one operand is
noncomputational, the gate drops and the matching partner effect applies:
Pauli noise first, then spreading. If both operands are noncomputational,
there is no computational partner to affect.

Post-gate transition hooks execute even when the physical gate was dropped.
They see any status changes caused by spreading. Leakage first introduced by
those hooks affects subsequent interactions.

!!! important "Handwritten annotations see their own circuit position"
    An explicit annotation immediately below a gate executes **after that
    gate's automatic level-transition hooks**. It can see leakage that was
    absent when the physical gate acted. Clifft does not move the annotation
    ahead of those hooks or attach it implicitly to the gate.

For example, with `transitions={"CX": T}`, this source:

```text
CX 2 5
LEAKAGE_INTERACTION(0.25, 0.25, 0.25, 0) 2 5
```

has the following order after hook expansion, assuming no model partner
effects were configured:

```text
CX 2 5
LEVEL_TRANSITION[CX] 2
LEVEL_TRANSITION[CX] 5
LEAKAGE_INTERACTION(0.25, 0.25, 0.25, 0) 2 5
```

If a hook leaks qubit 2 and leaves qubit 5 computational, the explicit
interaction depolarizes qubit 5. A model-generated interaction at that same
gate would have run before the hook and seen both operands computational.

### Control transition placement explicitly

For individual positions, give the transition a non-gate name, such as
`"after_cx"`, and write both annotations in the desired order. The following
example makes the difference deterministic. Its transition leaks `g` but
leaves `e` computational:

```python
import numpy as np
from clifft import noncomp

transition = np.zeros((5, 5))
transition[noncomp.Level.LEAK_G, noncomp.Level.G] = 1
model = noncomp.Model(
    transitions={"after_cx": transition.tolist()},
    classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]),
)

before = """
X 1
CX 0 1
LEAKAGE_INTERACTION(1, 0, 0, 0) 0 1
LEVEL_TRANSITION[after_cx] 0 1
M 1
"""
after = """
X 1
CX 0 1
LEVEL_TRANSITION[after_cx] 0 1
LEAKAGE_INTERACTION(1, 0, 0, 0) 0 1
M 1
"""
assert noncomp.sample(before, model, shots=16, seed=3).measurements.all()
assert not noncomp.sample(after, model, shots=16, seed=3).measurements.any()
```

In `before`, both operands are computational when the interaction annotation
runs, so it does nothing. In `after`, qubit 0 has leaked; the annotation flips
qubit 1 from `e` to `g`. A specific model interaction rule is the convenient
way to request the before-hook placement for every occurrence of a gate type.

### Mixing explicit and generated effects

Explicit instructions always execute as written. A model default or specific
rule does not disable an explicit annotation, and an explicit annotation does
not replace a model-generated effect. Users who mix both paths must account
for their sequential composition.

For example, a generated X channel with probability 0.2 followed by an
explicit X channel with probability 0.3 gives a net X probability of 0.38,
provided the partner remains computational. If the first annotation leaks
the partner, the second has no effect on that shot. Full depolarization can
mask accidental duplication, so avoid configuring the same intended effect
through both paths.

## Ordinary noise keeps its existing policy

`gate_partner_effects` describes residual effects of physical two-qubit
unitary gates. Noise instructions do not trigger replacement partner effects.
Their existing noncomputational behavior is unchanged:

| Noise instruction | Behavior on a leaked or lost target |
|-------------------|--------------------------------------|
| `DEPOLARIZE2/3`, `PAULI_CHANNEL_2/3` | Drop the whole affected pair or triple, including noise on computational partners |
| `E` / `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR` | Retain chain conditioning; Pauli factors act on computational operands and are inert on noncomputational sites |
| Single-qubit noise | Drop on noncomputational sites; keep on computational sites |

This is an effective-model convention: ordinary gate-noise parameters
characterized with two computational operands need not describe an interaction
with a leaked or absent carrier. Configure the residual physical disturbance
through partner effects. The simulator does not infer an association between
a noise instruction and the gate written before it.

The correlated-error exception matters. `E(p) X0 X1` can still flip
computational qubit 1 when qubit 0 is leaked. The same XX probability encoded
in `PAULI_CHANNEL_2` is dropped as a whole in that situation. Noise
representations equivalent in the computational subspace can therefore differ
after leakage. Choose and retain the representation that expresses the
intended noncomputational model.

## Comparison with deltakit-stim

[Deltakit-stim](https://github.com/Deltakit/deltakit-stim) automatically fully
depolarizes the partner of a leaked operand on its supported interaction
gates. Optional gate arguments configure directional spreading and mobility.
Clifft makes partner effects opt-in, separates leaked and lost sources, and
uses model settings or explicit annotations instead of changing unitary gate
arguments.

Using `pauli=(0.25, 0.25, 0.25)` and the same spreading probabilities matches
that local partner-effect prescription. It does not establish whole-circuit
compatibility. In particular, deltakit-stim's
[ordinary two-qubit noise implementation](https://github.com/Deltakit/deltakit-stim/blob/d29c1be6b9077d0c44e0f2499eea0a61fef5079c/src/stim/simulators/frame_simulator.inl#L811)
does not check leakage status, whereas Clifft retains the noise policy above.
Clifft also keeps separate `LEAK_G` and `LEAK_E` occupations. This feature
does not add deltakit's adaptive detector-error-model machinery.

## Exactness and limits

Partner effects use ordinary localized Pauli noise and exact source-preserving
leakage transitions. Spreading can require compiling additional continuations;
it does not add topology or status checks to ordinary executor dispatch.
Measurement slots, detector definitions, result arrays, and the `sample` call
are unchanged. Fixed seeds remain reproducible across worker counts.

The configured channel is an effective physical model. It does not describe
coherent superpositions between computational and leaked levels. Mobility,
which transfers leakage while restoring the original source, requires a joint
transition and is not implemented. Independently sampling spreading and
recovery would define a different process. See the
[noncomputational theory](../theory/noncomputational.md) for jump back-action
and the distinction between exact simulation and physical modeling assumptions.
