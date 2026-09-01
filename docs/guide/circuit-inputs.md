# Circuit Inputs

Clifft's primary circuit input format follows Stim but adds gates for
non-Clifford operations and additional noise models. Clifft supports nearly all
native Stim operations, including Clifford gates, noise, measurements,
detectors, observables, and repeat blocks. This is the most capable input path
and works with every Clifft workflow.

OpenQASM 2 support is built into `clifft`. Qiskit and Cirq support comes from
separately released companion packages. The input format is independent of the
simulation workflow and CPU execution settings.

| Starting point | Package and entry point | Important limits |
|---|---|---|
| Stim with Clifft extensions | `clifft.compile(text)` | Broadest feature support; see [Supported Gates](../reference/gates.md). |
| OpenQASM 2 text | `clifft.compile(text, input_format="qasm2")` | Unitary circuits only; no measurement, reset, classical control, or custom gate declarations. |
| Qiskit `QuantumCircuit` | [`clifft-qiskit`](https://github.com/unitaryfoundation/clifft-qiskit) | Terminal-measurement sampling; rejects mid-circuit measurement, feedforward, reset, and other non-unitary operations. |
| Cirq `cirq.Circuit` | [`clifft-cirq`](https://github.com/unitaryfoundation/clifft-cirq) | Parameter-resolved qubit circuits; no device, timing, calibration, qudit, arbitrary classical-control, or noise-channel semantics. |

## Stim with Clifft extensions

Existing Stim circuits can be passed directly to `clifft.compile()`. Add
Clifft operations such as `T` when the circuit needs non-Clifford behavior:

```python
import clifft

program = clifft.compile("""
    H 0
    CX 0 1
    T 1
    M 0 1
""")

result = clifft.sample(program, shots=1_000, seed=42)
print(result.measurements[:5])
```

Use this path for detectors, observables, post-selection, importance sampling,
leakage, loss, or other Clifft-specific workflows. Call `clifft.parse()` or
`clifft.parse_file()` only when the parsed circuit needs to be inspected before
compilation.

## OpenQASM 2

Select OpenQASM explicitly; Clifft does not guess the format from the text:

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

The importer supports the standard `qelib1.inc` unitary vocabulary, register
broadcasting, barriers as no-ops, comments, and finite constant angle
expressions. It rejects classical registers, measurements, resets, conditions,
custom or opaque gates, and nonstandard includes rather than assigning them
approximate semantics.

Use `clifft.parse_qasm2()` or `clifft.parse_qasm2_file()` when the imported
circuit needs to be inspected. OpenQASM support does not require Qiskit.

## Qiskit

Install `clifft-qiskit` and run a supported circuit through its Qiskit
`BackendV2` provider:

<!--pytest.mark.skip-->

```bash
pip install clifft-qiskit
```

<!--pytest.mark.skip-->

```python
from qiskit import QuantumCircuit
from clifft_qiskit import ClifftProvider

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

backend = ClifftProvider().get_backend("clifft")
counts = backend.run(qc, shots=1_000).result().get_counts()
print(counts)
```

The adapter returns Qiskit-style results for supported terminal-measurement
circuits. See the
[`clifft-qiskit` repository](https://github.com/unitaryfoundation/clifft-qiskit)
for its current basis, decomposition behavior, and package-specific limits.

## Cirq

Install `clifft-cirq` to convert or sample a supported Cirq circuit:

<!--pytest.mark.skip-->

```bash
pip install clifft-cirq
```

<!--pytest.mark.skip-->

```python
import cirq
import clifft_cirq

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key="m"),
)

sampler = clifft_cirq.ClifftSampler(seed=123)
result = sampler.run(circuit, repetitions=1_000)
print(result)
```

The converter supports common one-, two-, and three-qubit gates that map to
Clifft. See the
[`clifft-cirq` repository](https://github.com/unitaryfoundation/clifft-cirq)
for its current operations, conversion metadata, and package-specific limits.
