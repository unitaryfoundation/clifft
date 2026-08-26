# Native OpenQASM 2 Input

Clifft can natively parse and compile a unitary subset of OpenQASM 2.0. This
path has no Qiskit runtime dependency and is intended for unitary simulation
and exact-query workflows such as the
[ABSTRACTS benchmark](https://github.com/mjsutcliffe99/ABSTRACTS).

Select the format explicitly when compiling:

```python
import clifft

source = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
t q[1];
cx q[0], q[1];
"""

program = clifft.compile(source, input_format="qasm2")
probabilities = clifft.basis_probabilities(program, ["00", "11"])
```

Clifft does not guess the format from the input text. The default
`input_format="stim"` continues to select Clifft's Stim-compatible syntax.

## Supported Syntax

The initial importer supports:

- the required `OPENQASM 2.0;` version statement;
- the built-in `U` and `CX` gates;
- the built-in `qelib1.inc` include, without reading an external file;
- quantum-register declarations, indexed operands, and register broadcasting;
- `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `rx`, `ry`, `rz`, `u1`,
  `u2`, `u3`, `cx`, `cy`, `cz`, and `swap`;
- finite constant angle expressions using `pi`, arithmetic, powers, `sin`,
  `cos`, `tan`, `exp`, `ln`, and `sqrt`;
- `//` and `/* ... */` comments; and
- `barrier` statements, which are discarded as no-ops.

Declared register width is preserved even when some qubits are unused.
Register-valued gate operands are expanded using OpenQASM broadcasting rules.

The initial contract rejects classical registers, measurements, resets,
classical conditions, custom gate declarations, opaque declarations, and
nonstandard include files. These statements fail during parsing rather than
being dropped or assigned approximate semantics.

OpenQASM 3 is not part of this contract. Its standard library, gate modifiers,
global-phase operation, declarations, and dynamic-circuit features need a
separately specified importer extension.

## Parsing and Source Phase

Use `parse_qasm2()` when the lowered circuit needs to be inspected:

```python
import clifft

source = "OPENQASM 2.0; qreg q[1]; U(0, 0, 0) q[0];"
imported = clifft.parse_qasm2(source)
print(imported.circuit)
print(imported.global_phase_turns)
```

`global_phase_turns` is a value `t` representing the correction
`exp(1j * pi * t)`. OpenQASM 2's Euler gates and Clifft's internal `U3`
representation can differ by this phase. Keeping it beside the ordinary AST
lets future phase-sensitive compilation consume the exact source convention
without adding scalar bookkeeping to sampling HIR, plans, or executor state.

The current sampling, probability, and state-vector APIs are phase-insensitive,
so ordinary `compile(..., input_format="qasm2")` does not propagate this value
beyond the import boundary.

`parse_qasm2_file()` provides the corresponding file entry point.
