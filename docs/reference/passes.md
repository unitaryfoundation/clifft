# Optimization Passes

UCC optimizes at two distinct IR levels, each with its own pass manager.
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

Use `ucc.default_hir_pass_manager()` and `ucc.default_bytecode_pass_manager()`
to get these defaults, or build a custom pipeline:

```python
import ucc

# Custom HIR pipeline
pm = ucc.HirPassManager()
pm.add(ucc.PeepholeFusionPass())
pm.add(ucc.StatevectorSqueezePass())

# Custom bytecode pipeline
bpm = ucc.BytecodePassManager()
bpm.add(ucc.NoiseBlockPass())
bpm.add(ucc.MultiGatePass())
```

---

## HIR Passes

{% for p in hir_passes %}
### {{ p['name'] }}

| | |
|---|---|
| **Kind** | HIR (pre-lowering) |
| **Default** | {{ '✅ Enabled' if p['default_enabled'] else '❌ Disabled' }} |
| **Python** | `ucc.{{ p['python_name'] }}()` |

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
| **Python** | `ucc.{{ p['python_name'] }}()` |

{{ p['detail'] }}

{% endfor %}
