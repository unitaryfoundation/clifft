# Stim Circuits with Clifft Extensions

Existing Stim circuits compile directly with Clifft. Add Clifft's non-Clifford
gates when needed, while keeping Stim syntax for noise, detectors, observables,
repeat blocks, and measurements. This is the native and most capable input
path, including for post-selection, importance sampling, leakage, and loss.

Pass circuit text directly to `clifft.compile()`:

```python
import clifft

program = clifft.compile("""
    H 0
    CX 0 1
    T 1
    M 0 1
""")

result = clifft.sample(program, shots=1000, seed=42)
print(result.measurements[:5])
```

The default `input_format="stim"` is optional. Clifft accepts Stim syntax for
Clifford gates, noise channels, measurements, repeat blocks, detectors, and
observables, and adds supported non-Clifford rotations and annotations.

Use `clifft.parse()` or `clifft.parse_file()` when a circuit needs to be
inspected before compilation. Most applications should call `clifft.compile()`
directly.

See [Supported Gates](../reference/gates.md) for the complete accepted syntax,
[Quick Start](quickstart.md) for the basic workflow, and
[Choose a Workflow](choosing-a-workflow.md) for the available result types.
