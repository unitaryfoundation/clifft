<!--pytest-codeblocks:skipfile-->

# Cirq

The separately released
[`clifft-cirq`](https://github.com/unitaryfoundation/clifft-cirq) package
converts supported `cirq.Circuit` objects to Clifft text and provides a
Cirq-style sampler. Use it when circuit construction and result handling
already live in Cirq.

Install the adapter:

```bash
pip install clifft-cirq
```

Convert or sample a parameter-resolved qubit circuit:

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
and three-qubit gates that map to Clifft. It does not model Cirq device, timing,
calibration, qudit, arbitrary classical-control, or noise-channel semantics.

The companion package is maintained and released independently. See the
[`clifft-cirq` repository](https://github.com/unitaryfoundation/clifft-cirq)
for its current supported operations, conversion metadata, and limitations.

Use [Stim with Clifft extensions](stim-input.md) for detector annotations,
observables, post-selection, importance sampling, or other Clifft-specific
workflows.
