<!--pytest-codeblocks:skipfile-->

# Qiskit

The separately released
[`clifft-qiskit`](https://github.com/unitaryfoundation/clifft-qiskit) package
provides a Qiskit `BackendV2` provider for supported `QuantumCircuit` objects.
Use it when circuit construction, decomposition, or result handling already
lives in Qiskit.

Install the adapter:

```bash
pip install clifft-qiskit
```

Then run a supported circuit through the Clifft backend:

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
semantics, including mid-circuit measurement, feedforward, `reset`, and other
non-unitary operations, are rejected explicitly.

The companion package is maintained and released independently. See the
[`clifft-qiskit` repository](https://github.com/unitaryfoundation/clifft-qiskit)
for its current supported basis, decomposition behavior, and limitations.

Use the native [Stim-compatible input](stim-input.md) for detector annotations,
observables, post-selection, importance sampling, or other Clifft-specific
workflows.
