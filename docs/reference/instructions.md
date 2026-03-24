# Instruction Reference

This page documents all instruction types used by the UCC compiler at both
levels of the pipeline: the **Heisenberg IR** (HIR) produced by the front-end,
and the **VM Opcodes** (RISC bytecode) executed by the Schrodinger Virtual Machine.

The same data powers the hover tooltips in the
[Compiler Explorer](../explorer.md).

!!! tip "Explorer Tooltips"
    In the Compiler Explorer, hover over any opcode or HIR keyword to see
    its description inline.

---

## HIR Operation Types

The Heisenberg IR is the intermediate representation produced by the front-end.
Clifford gates are absorbed into the tracking frame and do not appear in the HIR.
What remains are non-Clifford operations, measurements, and meta-instructions.

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

## VM Opcodes

The VM executes a flat stream of RISC-style 32-byte instructions. Each opcode
falls into one of the categories below.

{% for cat in opcode_categories %}
### {{ cat }}

{% if cat == 'Frame' %}Frame ops update the Heisenberg tracking frame U_C. They are pure bookkeeping -- no state vector work is performed.
{% elif cat == 'Array' %}Array ops apply unitary gates directly to the Schrodinger state vector |phi>_A.
{% elif cat == 'Subspace' %}Subspace ops change the size of the active subspace or apply non-Clifford rotations.
{% elif cat == 'Measurement' %}Measurement ops collapse qubits, either algebraically (dormant) or by filtering/folding the state vector (active).
{% elif cat == 'Meta' %}Meta ops handle classical feedback, noise channels, and error correction bookkeeping.
{% endif %}

{% for op in opcodes_by_category[cat] %}
#### `{{ op['name'] }}`

**{{ op['summary'] }}**

{{ op['detail'] }}

**Operands:** `{{ op['operands'] }}`

{% endfor %}
{% endfor %}
