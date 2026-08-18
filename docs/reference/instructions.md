# Instruction Reference

This page documents the operation types used by the Clifft compiler at both
levels of the pipeline: the **Heisenberg IR** (HIR) produced by the front-end,
and the **Sampling Plan actions** produced by the planner and executed by the
sampling backend.

The same data powers the hover tooltips in the
[Playground]({{ playground_url }}).

!!! tip "Playground Tooltips"
    In the Playground, hover over any HIR keyword or plan-action name to see
    its description inline.

---

## HIR Operation Types

The Heisenberg IR is the intermediate representation produced by the front-end.
Clifford gates are absorbed into the offline Clifford frame $U_C$ and do not
appear in the HIR. What remains are non-Clifford operations, measurements, and
meta-instructions.

{% for cat in hir_categories %}
### {{ cat }}

{% for op in hir_by_category[cat] %}
{% for display_name in op.get('display', [op['name']]) %}
#### `{{ display_name }}`
{% endfor %}

**{{ op['summary'] }}**

{{ op['detail'] }}

{% endfor %}
{% endfor %}

---

## Sampling Plan Actions

The planner compiles optimized HIR into a sampling plan: a fixed sequence of
actions in which every Clifford has been absorbed and every stochastic event
appears as a Boolean **symbol**. The plan is what repeated sampling executes,
and it is what the Playground's Sampling Plan panel displays.

### Reading a plan line

```
w1->0 MEASURE_ACTIVE Z0 pivot=0 branch=s3 outcome=s0^s1^s3 record=r0 passes=2
```

* **`w<k>` / `w<k>-><k'>`** — the active width before (and, when it changes,
  after) the action. The dense coefficient state holds $2^k$ amplitudes for
  the currently active symbolic coordinates.
* **Pauli products** such as `Z0` or `X0*Z1` are written over **active
  symbolic coordinates, not physical qubits**. The Clifford frame maps
  between the two.
* **Affine expressions** such as `1^s0^s3` are XORs of Boolean symbols, with
  a leading `1` for the affine constant. Symbols are sampled noise outcomes,
  measurement branches, or derived parities. The Playground's compact view
  truncates long expressions and reports the omitted count as `...(+N)`.
* **Typed ids** name output slots: `r` measurement records, `d` detectors,
  `o` observables, `v` expectation values, and `s` symbols.
* **`passes=<n>`** estimates full traversals of the dense coefficient state
  for a direct lowering of the action; actions without it touch no dense
  state.

### Action types

{% for cat in plan_categories %}
#### {{ cat }}

{% for op in plan_by_category[cat] %}
##### `{{ op['name'] }}`

**{{ op['summary'] }}**

{{ op['detail'] }}

**Operands:** `{{ op['operands'] }}`

{% endfor %}
{% endfor %}
