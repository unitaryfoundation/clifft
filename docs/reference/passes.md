# Optimization Passes

Clifft optimizes at two distinct IR levels, each with its own pass manager.
**HIR passes** operate on the Heisenberg IR before bytecode emission.
**Bytecode passes** operate on the finalized RISC bytecode after the
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

---

## HIR Passes

{% for p in hir_passes %}
### {{ p['name'] }}

| | |
|---|---|
| **Kind** | HIR (pre-lowering) |
| **Default** | {{ '✅ Enabled' if p['default_enabled'] else '❌ Disabled' }} |
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
| **Python** | `clifft.{{ p['python_name'] }}()` |

{{ p['detail'] }}

{% endfor %}
