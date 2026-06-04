# Qiskit provider

Run Qiskit circuits on Clifft through a minimal `BackendV2` provider. Install
the optional dependency:

<!--pytest.mark.skip-->
```bash
pip install clifft[qiskit]
```

## Quickstart

```python
from qiskit import QuantumCircuit
from clifft.qiskit import ClifftProvider

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

backend = ClifftProvider().get_backend("clifft")
counts = backend.run(qc, shots=1000).result().get_counts()
print(counts)  # ~ {'00': 500, '11': 500}
```

`backend.run` also accepts a list of circuits. For each circuit the backend
transpiles into Clifft's supported basis, converts it to Stim text, samples
with Clifft, and returns Qiskit-style counts.

## Supported basis

The native basis is **Clifford+T**:

`h, s, sdg, x, y, z, cx, cy, cz, t, tdg` plus `measure`.

Higher-level gates are decomposed into this basis by the Qiskit transpiler
before execution. For example `ccx`/`ccz` decompose exactly, and arbitrary
rotations (`rx`, `ry`, `rz`, `u`, ...) are *approximated* into Clifford+T via
the transpiler's Solovay-Kitaev synthesis.

## Limitations

This is an initial prototype. Not supported in this version:

- Non-unitary / structural operations such as `reset`, mid-circuit measurement
  beyond terminal measurement, and classical control. These raise a
  `QiskitError`.
- Noise models, asynchronous jobs, Estimator/Sampler primitives, Qobj/legacy
  run configuration, and coupling maps / layout constraints.

Importing `clifft` does not import Qiskit; Qiskit is only loaded when you import
`clifft.qiskit`.
