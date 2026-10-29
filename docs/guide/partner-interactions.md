# Partner Effects from Leakage and Loss

When a two-qubit gate touches a leaked or lost qubit, Clifft skips the gate. By
default, a computational partner qubit is left untouched. On real hardware,
however, the drive pulses for that gate can still disturb the computational
partner or cause it to leak as well.

`noncomp.PartnerEffect` models this disturbance as single-qubit Pauli noise and
optional leakage spreading on the computational partner. The originally leaked
or lost qubit keeps its status, while any spread leakage persists and can affect
later gates and measurements in the same shot.

Start with [Leakage and Loss](leakage-and-loss.md) for the five-level model,
classifiers, and sampling API.

## Choose where effects apply

| Need | Configuration |
|------|---------------|
| The same effect across physical two-qubit gates | `Model(gate_partner_effects={...})` |
| A different effect for a specific gate or operand direction | `Model(interactions=[InteractionRule(...)])` |
| An effect at an individual circuit line | `LEAKAGE_INTERACTION(...)` or `LOSS_INTERACTION(...)` |

`gate_partner_effects` and `interactions` attach automatically to physical
two-qubit gates in the circuit. `LEAKAGE_INTERACTION` and `LOSS_INTERACTION` are
explicit circuit instructions that run where you place them, in addition to any
model-configured gate effects.

## Model-wide defaults

Pass `gate_partner_effects` to `noncomp.Model` with keys `"leaked"` (matching
either `LEAK_G` or `LEAK_E`) and/or `"lost"`. Each default applies regardless of
whether the leaked or lost qubit is the first or second operand of a two-qubit
gate, whenever the other operand is computational. Omitted keys leave the
partner unchanged.

This example fully depolarizes the computational partner whenever a two-qubit
gate touches a leaked qubit, while leaving lost-qubit encounters as no-ops:

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
result = noncomp.sample("LEAKAGE(1) 0\nCX 0 1\nM 1", model, shots=4096, seed=1, final_status=True)
assert abs(result.measurements[:, 0].mean() - 0.5) < 0.04
assert (result.final_status[:, 0] == noncomp.QubitStatus.LEAK_G).all()
assert (result.final_status[:, 1] == noncomp.QubitStatus.COMPUTATIONAL).all()
```

To use Clifft's compile-once
[fixed execution](leakage-and-loss.md#fixed-execution-vs-trajectories), two-qubit
gates that can pair a leaked or lost qubit with a computational partner must
configure full partner depolarization (`pauli=(0.25, 0.25, 0.25)` with
`spread_probability=0`).

### Probability conventions

`PartnerEffect` takes two optional keyword arguments:

| Argument | Meaning | Default |
|----------|---------|---------|
| `pauli=(px, py, pz)` | Mutually exclusive X, Y, and Z error probabilities on the partner | `(0, 0, 0)` |
| `spread_probability=p` | Probability that the partner leaks after the Pauli step | `0` |

`px + py + pz` must be at most `1`, leaving `1 - px - py - pz` as the no-error
probability. `spread_probability` is an independent draw in `[0, 1]` applied
after the Pauli channel, so a partner can receive a Pauli error and then leak on
the same gate.

For full single-qubit depolarization (equal I, X, Y, and Z probabilities), pass
`pauli=(0.25, 0.25, 0.25)`. In Stim's `DEPOLARIZE1(p)` parameterization that
corresponds to `p=0.75` (more generally, `DEPOLARIZE1(p)` corresponds to
`pauli=(p/3, p/3, p/3)`).

Leakage spreading applies the same `g -> leak_g`, `e -> leak_e` jump as
`LEAKAGE(p)`, evaluated on the partner's state *after* the Pauli channel rather
than copying the source qubit's leaked level. The source qubit remains leaked,
and only `"leaked"` effects support `spread_probability > 0` (`"lost"` effects
with nonzero spreading raise `ValueError`).

## Specific rules replace defaults

Use `InteractionRule` when a particular gate or operand direction behaves
differently from `gate_partner_effects`. A matching rule **replaces** the
model-wide default for that `(gate, source_operand, source_status)` combination
rather than merging fields, so `effect=PartnerEffect()` disables the default for
that case:

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

`source_operand` is the operand index within each two-qubit pair (`0` for the
first qubit, `1` for the second), not a qubit ID. On `CX 2 5`,
`source_operand=0` matches when control qubit `2` is leaked and target qubit `5`
is computational; `source_operand=1` matches the reverse. Symmetric gates such
as `CZ` use the same written operand order (`0` and `1`).

Gate aliases share rules: `"CX"` and `"CNOT"` refer to the same gate, so
defining rules for both with the same `source_operand` and `source_status`
raises `ValueError`.

### Supported gates

Defaults and rules apply to physical two-qubit unitary gates, including `CX`,
`CY`, `CZ`, swap gates, square-root Pauli-pair gates, and parameterized
`R_XX`, `R_YY`, and `R_ZZ` rotations (across all angles).

They do **not** apply to:

- single-qubit gates, measurements, noise instructions, or identity no-ops;
- classical feedback such as `CX rec[-1] 0` (both operands must be physical
  qubits);
- multi-target Pauli-product gates such as `SPP` and `R_PAULI`, even when a
  product contains two qubits; or
- composite gates that decompose during parsing (`CH`, `CCX`, `CCZ`). Rules
  cannot name `CH`, `CCX`, or `CCZ` directly, and instead apply to the
  elementary two-qubit gates produced by their decomposition. If decomposition
  choice affects your hardware noise model, write the circuit using the target
  hardware's native two-qubit gates.

A swap gate touching a leaked or lost qubit is skipped like any other two-qubit
gate and then applies any configured partner effect; it does not move the leaked
or lost state to the other site.

## Explicit circuit annotations

Use `LEAKAGE_INTERACTION` and `LOSS_INTERACTION` directly in a circuit to apply
a partner effect at an individual position (with or without a preceding gate):

```text
LEAKAGE_INTERACTION(px, py, pz, spread_probability) source partner
LOSS_INTERACTION(px, py, pz) source partner
```

`LEAKAGE_INTERACTION` fires when `source` is leaked and `partner` is
computational; `LOSS_INTERACTION` fires when `source` is lost and `partner` is
computational. All probability arguments are required, and each
`(source, partner)` pair must name two distinct plain qubit indices. Multiple
pairs on one line run left to right, so leakage spreading in one pair can affect
a later pair on the same line.

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
    final_status=True,
)
assert result.measurements.all()
assert (result.final_status[:, 1] == noncomp.QubitStatus.LEAK_E).all()
assert (result.final_status[:, 3] == noncomp.QubitStatus.COMPUTATIONAL).all()
```

Here the certain X error flips qubit 1 from `g` to `e` before the certain
leakage jump, leaving it in `LEAK_E`. Qubit 3 receives an X error and stays
computational. Interaction instructions append no measurement records (the four
bits in `result.measurements` come from `HERALD_LEAKAGE_EVENT 0 1` and `M 1 3`).

Standalone `LEAKAGE_INTERACTION` (with `spread_probability=0`) and
`LOSS_INTERACTION` instructions support arbitrary `(px, py, pz)` probabilities
in compile-once
[fixed execution](leakage-and-loss.md#fixed-execution-vs-trajectories). Like
`LEAKAGE` and `LOSS`, these instructions require `clifft.noncomp`; ordinary
`clifft.compile` rejects them.

## Ordering with level-transition hooks

Model partner effects check whether an operand was **already** leaked or lost
entering a two-qubit gate, whereas gate transition hooks
(`Model(transitions={"CX": ...})`) model leakage or loss **caused by** that
gate. Around each physical two-qubit gate, Clifft therefore evaluates:

1. **The gate or its model partner effect:** if both operands are computational,
   the gate runs normally; if one operand is leaked or lost and the other is
   computational, the gate is skipped and the matching partner effect runs
   (Pauli noise first, then leakage spreading); if both operands are
   noncomputational, the gate is skipped with no computational partner to
   disturb.
2. **Automatic gate transition hooks** (`transitions={"CX": ...}`), in operand
   order. Hooks still run when the gate was skipped, and see any new leakage
   caused by step 1.
3. **Subsequent instructions** written on later lines of the circuit.

!!! important "Explicit interaction instructions run after a gate's transition hooks"
    Writing `LEAKAGE_INTERACTION` on the line after `CX 2 5` places it at
    step 3, **after** any `transitions={"CX": T}` hooks on qubits 2 and 5:

    ```text
    CX 2 5
    # automatic LEVEL_TRANSITION[CX] hooks on 2 and 5 run here
    LEAKAGE_INTERACTION(0.25, 0.25, 0.25, 0) 2 5
    ```

    That handwritten instruction can react to leakage just created by the `CX`
    hook, whereas a model-configured `gate_partner_effects` or `InteractionRule`
    on `CX` runs at step 1 and reacts only to leakage that existed *before*
    `CX 2 5`.

### Control transition placement explicitly

When you need explicit control over ordering at a single circuit site, give the
transition a custom non-gate name (such as `"after_cx"`) and write
`LEVEL_TRANSITION[after_cx]` and `LEAKAGE_INTERACTION` in the order you want:

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

In `before`, both qubits are still computational when `LEAKAGE_INTERACTION`
runs, so qubit 1 stays in `1`. In `after`, `LEVEL_TRANSITION[after_cx]` leaks
qubit 0 first, so `LEAKAGE_INTERACTION` flips qubit 1 to `0`. To get the
before-transition order automatically across every `CX`, configure
`gate_partner_effects` or an `InteractionRule` on the model instead.

### Mixing explicit and generated effects

Explicit `LEAKAGE_INTERACTION` and `LOSS_INTERACTION` instructions do not
override `gate_partner_effects` or `InteractionRule` entries. If both are
present at the same gate, their effects apply in sequence: for example, a
model-configured X error of `0.2` followed by an explicit X error of `0.3` gives
a net X flip probability of `0.38` while the partner stays computational (or no
second effect if the first step leaked the partner). Avoid configuring the same
physical disturbance through both mechanisms.

## Multi-qubit noise vs. partner effects

`gate_partner_effects` and `InteractionRule` attach only to physical two-qubit
unitary gates, not to noise instructions. Ordinary noise instructions keep their
standard behavior when a target is leaked or lost:

| Noise instruction | Behavior on a leaked or lost target |
|-------------------|--------------------------------------|
| `DEPOLARIZE2/3`, `PAULI_CHANNEL_2/3` | Skipped for the entire pair or triple, including computational partners |
| `E` / `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR` | Chain probabilities stay intact; Pauli terms act on computational targets and do nothing on leaked or lost targets |
| Single-qubit noise (`DEPOLARIZE1`, `X_ERROR`, ...) | Skipped on leaked or lost targets; applied as usual on computational targets |

Two-qubit gate noise such as `DEPOLARIZE2` placed after `CX 0 1` is therefore
skipped whenever qubit 0 or 1 is leaked or lost; Clifft does not automatically
convert it into single-qubit noise on the surviving partner. Use
`gate_partner_effects` or `InteractionRule` to model how a skipped two-qubit
gate disturbs its computational partner.

Note the difference between `PAULI_CHANNEL_2` and `CORRELATED_ERROR`:
`E(p) X0 X1` still flips computational qubit 1 when qubit 0 is leaked, whereas
an `XX` term inside `PAULI_CHANNEL_2` on `0 1` is skipped altogether. Two noise
encodings that are equivalent on ordinary qubits can therefore behave
differently once leakage or loss occurs.

## Acknowledgments

Clifft's leakage partner effects were inspired by
[deltakit-stim](https://github.com/Deltakit/deltakit-stim).
See [Acknowledgments and references](../acknowledgments.md#deltakit-stim) for
this and other influences on Clifft.
