# Optimization Passes

Clifft optimizes at two distinct IR levels, each with its own pass manager.
**HIR passes** operate on the Heisenberg IR before bytecode emission.
**Bytecode passes** operate on the finalized bytecode after the
back-end has lowered the HIR.

## Default Pipeline

The default HIR pipeline:

{% for p in default_hir_passes %}
1. **{{ p['name'] }}** -- {{ p['summary'] }}
{% endfor %}

The default bytecode pipeline:

{% for p in default_bytecode_passes %}
1. **{{ p['name'] }}** -- {{ p['summary'] }}
{% endfor %}

Use `clifft.default_hir_pass_manager()` and `clifft.default_bytecode_pass_manager()`
to get these defaults, or build a custom pipeline:

```python
import clifft

# Custom HIR pipeline
pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.add(clifft.StatevectorSqueezePass())

# Custom bytecode pipeline
bpm = clifft.BytecodePassManager()
bpm.add(clifft.NoiseBlockPass())
bpm.add(clifft.MultiGatePass())
```

## Trajectory Safety Metadata

Some workflows require measurements to remain in their original order,
including hidden measurements introduced by the compiler. In particular,
`clifft.noncomp.sample` may force the result of a hidden trace-out measurement
when it resumes a trapped transition. Moving another measurement across that
collapse can change quantum correlations.

`clifft.noncomp.sample` therefore applies only passes that are enabled by
default, preserve measurement-record order, and preserve instrument prefixes.
Its HIR pipeline uses `PeepholeFusionPass` but omits
`StatevectorSqueezePass`; all default bytecode passes currently preserve both
properties and are applied.

Record-order preservation is necessary but does not by itself make a
continuation compatible with an already-running VM state. Trajectory passes
must also opt in to instrument-prefix stability: changing the circuit after an
instrument may not change optimized output through that instrument. The
runtime checks each recompiled prefix, including referenced constant-pool data,
before resuming. For this reason, `clifft.noncomp.sample` uses a fixed internal
pipeline and does not currently accept custom pass managers. See
[Leakage and Loss](../guide/leakage-and-loss.md#why-there-is-no-compile-step)
for how continuations are compiled and resumed.

---

## HIR Passes

{% for p in hir_passes %}
### {{ p['name'] }}

| | |
|---|---|
| **Kind** | HIR (pre-lowering) |
| **Default** | {{ '✅ Enabled' if p['default_enabled'] else '❌ Disabled' }} |
| **Preserves measurement-record order** | {{ 'Yes' if p['preserves_record_order'] else 'No' }} |
| **Preserves instrument prefix** | {{ 'Yes' if p['preserves_instrument_prefix'] else 'No' }} |
| **Python** | `clifft.{{ p['python_name'] }}()` |

{{ p['detail'] }}

{% endfor %}

---

## Bytecode Passes

{% for p in bytecode_passes %}
### {{ p['name'] }}

| | |
|---|---|
| **Kind** | Bytecode (post-lowering) |
| **Default** | {{ '✅ Enabled' if p['default_enabled'] else '❌ Disabled' }} |
| **Preserves measurement-record order** | {{ 'Yes' if p['preserves_record_order'] else 'No' }} |
| **Preserves instrument prefix** | {{ 'Yes' if p['preserves_instrument_prefix'] else 'No' }} |
| **Python** | `clifft.{{ p['python_name'] }}()` |

{{ p['detail'] }}

{% endfor %}
