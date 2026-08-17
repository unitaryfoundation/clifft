# Compiling Circuits

Clifft compiles Stim-format circuit text into an executable symbolic-coordinate
sampling plan. For most users, `clifft.compile()` is the only compilation API
needed. Lower-level APIs remain available for inspecting the circuit and
Heisenberg IR or for supplying a custom HIR optimization pipeline.

!!! note "Leakage and loss"
    Circuits containing `LEAKAGE`, `LOSS`, or `LEVEL_TRANSITION` annotations
    use `clifft.noncomp.sample()` instead. See
    [Leakage and Loss](leakage-and-loss.md).

!!! tip "Using Qiskit or Cirq?"
    Use the companion
    [`clifft-qiskit`](https://github.com/unitaryfoundation/clifft-qiskit) or
    [`clifft-cirq`](https://github.com/unitaryfoundation/clifft-cirq) package
    to translate supported circuits into Clifft's Stim-compatible text format.

## One-Step Compilation

`clifft.compile()` parses the circuit, traces Clifford operations into the
Heisenberg IR, applies the default HIR passes, plans active symbolic
coordinates, and prepares an executable plan:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")
```

The returned `Program` can be passed directly to `clifft.sample()`,
`clifft.sample_survivors()`, the fixed-fault samplers, or the exact-query APIs.

To skip HIR optimization, pass `hir_passes=None`:

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0", hir_passes=None)
```

You can also supply a custom `HirPassManager`. Compilation options such as
`postselection_mask`, `normalize_syndromes`, `expected_detectors`, and
`expected_observables` define the output contract used during sampling.

## Step-by-Step Compilation

### 1. Parse

`clifft.parse()` converts circuit text into a `Circuit`:

```python
import clifft

circuit = clifft.parse("H 0\nCNOT 0 1\nM 0 1")
```

You can also parse from a file:

<!--pytest.mark.skip-->

```python
circuit = clifft.parse_file("my_circuit.stim")
```

### 2. Trace Clifford Operations

`clifft.trace()` absorbs Clifford operations into an offline tableau and emits
the explicit non-Clifford operations, measurements, noise, and classical
outputs in the Heisenberg frame as a `HirModule`:

```python
import clifft

hir = clifft.trace(clifft.parse("H 0\nCNOT 0 1\nT 0\nM 0 1"))
print(hir)
```

### 3. Optimize the HIR

```python
import clifft

hir = clifft.trace(clifft.parse("H 0\nCNOT 0 1\nT 0\nM 0 1"))

# Use the default pipeline.
pm = clifft.default_hir_pass_manager()
pm.run(hir)

# Or construct a focused pipeline.
pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.run(hir)
```

### 4. Plan and Prepare Execution

`clifft.lower()` converts an optimized `HirModule` into the same `Program`
type returned by `clifft.compile()`:

<!--pytest-codeblocks:cont-->

```python
program = clifft.lower(hir)
```

Internally, lowering has two boundaries. Coordinate planning first produces a
semantic `SamplingPlan`: it selects active coordinates and records expressions,
measurements, rotations, width transitions, and outputs without choosing a CPU
instruction set or target-specific data layout. Executable preparation then
converts that plan into the immutable descriptors and dependency tables used
by the host executor. It also chooses any supported fusion and scalar or SIMD
kernels.

The private `SamplingPlan` is useful for separating compiler decisions from
target preparation; the public result of both steps remains one reusable
`Program`. Runtime kernels therefore do not perform tableau evolution, choose
coordinates, localize Paulis, or discover dependencies.

See [Software Architecture](../theory/architecture.md) for the contracts
between HIR, semantic planning, executable preparation, and shot execution.

## Full Custom Pipeline

```python
import clifft

circuit = clifft.parse("H 0\nT 0\nCNOT 0 1\nM 0 1")
hir = clifft.trace(circuit)
clifft.default_hir_pass_manager().run(hir)
program = clifft.lower(hir)
```

## Reference Syndrome Computation

For QEC workflows, `compute_reference_syndrome()` computes noiseless detector
and observable parities for a `HirModule`. This is the same normalization used
by `clifft.compile(..., normalize_syndromes=True)`.

<!--pytest.mark.skip-->

```python
circuit = clifft.parse(circuit_text)
reference = clifft.compute_reference_syndrome(clifft.trace(circuit))
print(reference["detectors"])
print(reference["observables"])
```
