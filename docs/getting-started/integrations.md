<!--pytest-codeblocks:skipfile-->

# Front-End Integrations

Clifft natively accepts Stim-compatible circuit text with Clifft extensions for
non-Clifford operations and a documented unitary OpenQASM 2 subset. If your
circuits already live as objects in another quantum software framework,
companion packages can translate or route supported circuits to Clifft without
hand-writing either text format.

These packages are maintained separately from the core `clifft` package and are
released on their own schedule. Use their READMEs as the source of truth for the
full supported operation set and current limitations.

## Integration Options

| Starting point | Package | What it provides |
|---|---|---|
| Stim-compatible text | [`clifft`](https://pypi.org/project/clifft/) | Direct parsing, compilation, sampling, state-vector access, detectors, observables, and QEC-oriented workflows. |
| Unitary-only OpenQASM 2 text | [`clifft`](https://pypi.org/project/clifft/) | Native parsing and compilation of the [supported subset](openqasm2.md), including the ABSTRACTS gate vocabulary. |
| Qiskit `QuantumCircuit` | [`clifft-qiskit`](https://github.com/unitaryfoundation/clifft-qiskit) | A Qiskit `BackendV2` provider that runs supported circuits on Clifft and returns Qiskit-style results. |
| Cirq `cirq.Circuit` | [`clifft-cirq`](https://github.com/unitaryfoundation/clifft-cirq) | A converter to Clifft circuit text plus a Cirq-style sampler facade backed by Clifft. |

Use the native `clifft` API when you are working with supported circuit text,
detector annotations, observables, post-selection, or importance sampling. Use
an adapter when circuit construction, decomposition, or the surrounding
workflow already lives in Qiskit or Cirq.

## Qiskit

Install the Qiskit adapter:

```bash
pip install clifft-qiskit
```

Then run a supported `QuantumCircuit` through the Clifft backend:

```python
from qiskit import QuantumCircuit
from clifft_qiskit import ClifftProvider

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

backend = ClifftProvider().get_backend("clifft")
counts = backend.run(qc, shots=1000).result().get_counts()
print(counts)
```

The adapter targets terminal-measurement sampling and counts. Unsupported
semantics, such as mid-circuit measurement, feedforward, `reset`, and other
non-unitary operations, are rejected explicitly.

See the [`clifft-qiskit` repository](https://github.com/unitaryfoundation/clifft-qiskit)
for the current supported basis, decomposition behavior, and package-specific
limitations.

## Cirq

Install the Cirq adapter:

```bash
pip install clifft-cirq
```

Convert a parameter-resolved qubit circuit to Clifft text or sample it through
the Cirq-style facade:

```python
import cirq
import clifft_cirq

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key="m"),
)

converted = clifft_cirq.to_clifft_text(circuit)
print(converted.clifft_text)

sampler = clifft_cirq.ClifftSampler(seed=123)
result = sampler.run(circuit, repetitions=1000)
print(result)
```

The converter supports parameter-resolved qubit circuits and common one-, two-,
and three-qubit gates that map to Clifft. It does not model Cirq device,
timing, calibration, qudit, arbitrary classical-control, or noise-channel
semantics.

See the [`clifft-cirq` repository](https://github.com/unitaryfoundation/clifft-cirq)
for the current supported operations, conversion metadata, and package-specific
limitations.
