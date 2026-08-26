# OpenQASM 2 Input

Clifft can natively parse and compile a unitary subset of OpenQASM 2.0 without a
Qiskit runtime dependency.

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

The importer supports:

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

The importer rejects classical registers, measurements, resets, classical
conditions, custom gate declarations, opaque declarations, and nonstandard
include files. These statements fail during parsing rather than being dropped
or assigned approximate semantics. Support may expand in the future.

## Parsing and Source Phase

Use `parse_qasm2()` when the lowered circuit needs to be inspected:

```python
import clifft

source = "OPENQASM 2.0; qreg q[1]; U(0, 0, 0) q[0];"
imported = clifft.parse_qasm2(source)
print(imported.circuit)
print(imported.global_phase_half_turns)
```

Clifft is generally insensitive to global phase, but the importer retains the
source correction for potential future phase-sensitive use cases.
`global_phase_half_turns` is a value `t` representing
`exp(1j * pi * t)`. Current sampling, probability, and state-vector APIs do not
propagate it beyond the import boundary.

For gate phase conventions, Clifft follows Qiskit's de-facto convention:
`U`/`u1`/`u2`/`u3` use Qiskit's cosine-top-left Euler matrix, while `rz` is the
symmetric `RZGate` and therefore contributes no source-phase correction.

`parse_qasm2_file()` provides the corresponding file entry point.
