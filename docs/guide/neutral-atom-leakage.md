<!--pytest-codeblocks:skipfile-->

# Tutorial: Modeling Neutral-Atom Leakage and Loss

This advanced tutorial models a neutral-atom logical experiment with Clifft's
five-level trajectory API. It recreates the `alpha=1` simulation point from
Figure 9 of Rines *et al.*, ["Demonstration of a Logical Architecture Uniting
Motion and In-Place Entanglement"](https://arxiv.org/abs/2509.13247), using the
authors' [public supplementary artifact](https://zenodo.org/records/17137995).

The example compares two treatments of state-dependent transition rates. The
first matches the stabilizer-compatible approximation used by the public
artifact. The second keeps the published transition matrices unchanged and
uses Clifft's exact conditional no-jump back-action.

This page assumes familiarity with Clifford circuits, postselection, and the
[Leakage and Loss](leakage-and-loss.md) API. For a smaller first example, start
with [Delayed Loss in a Surface Code](delayed-loss.md).

## The logical experiment

The experiment implements a precompiled order-finding instance for `N=15` and
`a=11`. The remaining quantum workload is a three-qubit Clifford circuit with
ideal output distribution

$$
P(000) = P(010) = P(101) = P(111) = \frac{1}{4}.
$$

The paper realizes that logical circuit with four physical schedules:

| Schedule | Physical realization | CZ gates |
|---|---|---:|
| Unencoded | Three physical atoms | 2 |
| Two row | Two `[[4, 2, 2]]` code patches | 11 |
| Three row | Three code patches producing two samples per shot | 22 |
| Two row with LDU | Two patches plus eight leakage-detection flags | 27 |

The `[[4, 2, 2]]` code detects a single physical-qubit error. Its stabilizers
are `XXXX` and `ZZZZ`, so invalid codewords can be rejected but an arbitrary
single error cannot be corrected. The leakage-detection unit, or LDU, adds two
controlled operations and a flag measurement for each encoded data atom. A
computational atom flips its flag once; a leaked atom does not participate and
leaves the flag set.

## Checked-in Clifft circuits

The four translated schedules are included with the documentation:

- [`unencoded_alpha1.stim`](circuits/neutral_atom/unencoded_alpha1.stim)
- [`two_row_alpha1.stim`](circuits/neutral_atom/two_row_alpha1.stim)
- [`three_row_alpha1.stim`](circuits/neutral_atom/three_row_alpha1.stim)
- [`two_row_ldu_alpha1.stim`](circuits/neutral_atom/two_row_ldu_alpha1.stim)

They use Clifft's Stim-superset text format and do not require Cirq. The export
preserves terminal measurement order and resolves physical atom motion into the
wire labels used by later operations. A move is therefore not represented as a
quantum `SWAP`. Its phase error appears as `Z_ERROR` on the atoms that arrived
at their new sites.

The circuits fix the public artifact's noise multiplier at `alpha=1`. The
[`manifest.json`](circuits/neutral_atom/manifest.json) records the public source,
archive checksum, licenses, and circuit hashes.

## Separate circuit noise from level changes

The checked-in circuits contain the Clifford operations and ordinary Pauli
noise. For example, a physical phase rotation becomes a named Clifford
rotation followed by its twirled `Z_ERROR` and an explicit level-transition
site:

```stim
SQRT_Z_DAG 10
Z_ERROR(8.8823809595495095e-05) 10
LEVEL_TRANSITION[RZ_TRANSITION] 10
```

The noncomputational model contains the effects that change or observe the
atom's level:

```python
model = noncomp.Model(
    initial_state=INITIAL_LEVELS,
    transitions={
        "CZ": cz_transition,
        "RZ_TRANSITION": rz_transition,
    },
    classifier=noncomp.Classifier(classifier_matrix),
    reset_restores_lost=False,
    damping="exact",
)
```

`initial_state` prepares a distribution over `g`, `e`, `leak_g`, `leak_e`, and
`lost`. The transition matrices use `T[to][from]`. The `CZ` key is a gate hook,
while `RZ_TRANSITION` is referenced explicitly because `RZ` means reset in the
Stim instruction set. The three-symbol classifier produces zero, one, or a
heralded loss at measurement.

The full numeric matrices and classifier are kept in the runnable
[`neutral_atom_leakage_tutorial.py`](scripts/neutral_atom_leakage_tutorial.py).
Keeping them visible there makes it possible to inspect or modify the physical
model without regenerating the circuits.

## Matched and exact no-jump behavior

A transition event collapses its computational source against the live quantum
state. The absence of an event also carries information when the total event
rates differ between `g` and `e`. For rates $p_g$ and $p_e$, the surviving
component receives the filter

$$
K_{\mathrm{stay}} = \sqrt{1-p_g}\lvert g\rangle\!\langle g\rvert
                   + \sqrt{1-p_e}\lvert e\rangle\!\langle e\rvert.
$$

The tutorial exposes two model choices:

- `matched` adds a diagonal self-jump to the lower-rate computational source
  until the two total rates agree. With equal rates, the no-jump filter is
  proportional to identity, so `damping="neglect"` is exact for the transformed
  matrices. This reproduces the public artifact's approximation.
- `exact` keeps the published unequal rates and uses `damping="exact"`. Clifft
  then applies the conditional no-jump filter and promotes coherent sites when
  required.

This changes one modeling assumption while preserving the circuit, initial
population, readout model, decoder, and Pauli twirls.

## Sample and decode the four schedules

Run the matched model on every circuit from the repository root:

```bash
uv run python docs/guide/scripts/neutral_atom_leakage_tutorial.py
```

Fixed-seed output with 5,000 trajectories per schedule:

```text
circuit       model      acceptance   heralded       TVD
unencoded     matched        99.2%       0.8%    0.0905
two_row       matched        45.9%       3.7%    0.0296
two_row_ldu   matched        17.8%      10.3%    0.0718
three_row     matched        23.9%       8.2%    0.0218
```

The script converts Clifft's measurement and herald arrays to the public
artifact's three-symbol records. It then applies the same classical workflow:

1. Reject preparation or LDU flags with invalid values.
2. Reject unheralded loss where the selected model does not correct it.
3. Decode each valid `[[4, 2, 2]]` block into two logical bits.
4. Extract the three Shor output bits and compare their distribution with the
   ideal distribution using total variation distance, or TVD.

The output reports two complementary quantities. Acceptance is the fraction of
physical trajectories that survive postselection. TVD measures the decoded
distribution among those survivors, where lower is better. The three-row
schedule emits two correlated logical samples per accepted physical trajectory;
the script accounts for that when reporting acceptance.

To compare the matched and exact models on the LDU circuit:

```bash
uv run python docs/guide/scripts/neutral_atom_leakage_tutorial.py \
  --circuits two_row_ldu --models matched exact --shots 5000
```

```text
circuit       model      acceptance   heralded       TVD
two_row_ldu   matched        17.4%      10.3%    0.0978
two_row_ldu   exact          17.9%      10.2%    0.0872
```

Five thousand trajectories make this comparison quick enough for a tutorial,
but they do not precisely resolve the difference between models. Increase
`--shots` before drawing a quantitative conclusion from the acceptance or TVD
shift.

Exact simulation is slower here because the LDU schedule has many
state-dependent transition sites. The important scientific comparison is not
runtime alone: unequal-rate no-jump back-action can change which trajectories
survive postselection even when the decoded distribution of those survivors
changes only modestly.

## Scope and provenance

This tutorial reproduces one published noise setting with checked-in Clifft
circuits. It does not regenerate the circuits from Cirq or reproduce the full
ten-value noise sweep. The source schedules, noise parameters, and decoder come
from the Apache-2.0 supplementary artifact recorded in the manifest.

The model retains the artifact's Pauli twirls for coherent pulse errors, so the
`exact` comparison changes only the unequal-rate no-jump treatment. Neither
configuration should be read as a new hardware fit.

See [Noncomputational States](../theory/noncomputational.md) for the trajectory
semantics and active-width cost of exact damping.
