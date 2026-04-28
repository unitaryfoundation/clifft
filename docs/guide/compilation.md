# Compiling Circuits

Clifft compiles Stim-format circuit text into an executable SVM program. For most users, `clifft.compile()` is the only compilation API needed. Lower-level APIs are available when you want to inspect intermediate representations, customize optimization passes, or build your own compilation flow.

## One-Step Compilation

For most use cases, `clifft.compile()` parses the circuit, traces Clifford operations into the Heisenberg IR, applies the default HIR and bytecode optimization passes, and returns a simulatable program:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")
```

The returned `Program` can be passed directly to `clifft.sample()`, `clifft.sample_survivors()`, or other simulation APIs.

To skip optimization, pass `None` for the corresponding stage:

```python
import clifft

program = clifft.compile(
    """
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
    """,
    hir_passes=None,
    bytecode_passes=None,
)
```

You can also supply a custom `HirPassManager` or `BytecodePassManager` to override the defaults. See [Optimization Passes](../reference/passes.md) for details.

Some simulation workflows also pass options such as `postselection_mask`, `normalize_syndromes`, `expected_detectors`, or `expected_observables` to `compile()`. These options affect how detector and observable outputs are interpreted during sampling; see [Simulation](simulation.md#detectors-observables-and-post-selection).

## Step-by-Step Compilation

The lower-level compilation APIs expose the same stages used by `clifft.compile()`. They are useful for debugging, inspecting intermediate representations, or experimenting with custom optimization passes.

### 1. Parsing

`clifft.parse()` converts Stim-format circuit text into an AST:

```python
import clifft

circuit = clifft.parse("""
    H 0
    CNOT 0 1
    M 0 1
""")
```

You can also parse from a file:

<!--pytest.mark.skip-->

```python
circuit = clifft.parse_file("my_circuit.stim")
```

### 2. Front-End: Clifford Tracing

`clifft.trace()` absorbs Clifford operations into Clifft's frame representation and produces the Heisenberg IR (`HirModule`). Non-Clifford operations, measurements, detectors, observables, and noise are represented explicitly in this IR:

```python
import clifft

circuit = clifft.parse("H 0\nCNOT 0 1\nT 0\nM 0 1")
hir = clifft.trace(circuit)

print(hir)
```

### 3. HIR Optimization

HIR passes transform the traced circuit before it is lowered to executable bytecode. The default pass manager applies Clifft's standard optimizations; you can also build a custom pass manager when experimenting with individual passes.

<!--pytest-codeblocks:cont-->

```python
# Use the default HIR pass manager
pm = clifft.default_hir_pass_manager()
pm.run(hir)

# Or build a custom one
pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.run(hir)
```

### 4. Back-End: Lowering to Bytecode

`clifft.lower()` converts optimized HIR into an executable SVM program:

<!--pytest-codeblocks:cont-->

```python
program = clifft.lower(hir)
```

Most users should call `clifft.compile()` instead. Use `lower()` directly when you are building a custom compilation pipeline or inspecting the output of individual optimization stages.

### 5. Bytecode Optimization

After lowering, bytecode passes optimize the executable program. These passes do not change the circuit semantics; they reduce runtime overhead, for example by fusing compatible operations.

<!--pytest-codeblocks:cont-->

```python
bpm = clifft.default_bytecode_pass_manager()
bpm.run(program)
```

See [Optimization Passes](../reference/passes.md) for the full list of default and optional passes available at both IR levels.

## Full Custom Pipeline

Putting it all together:

```python
import clifft

# Parse
circuit = clifft.parse("H 0\nT 0\nCNOT 0 1\nM 0 1")

# Front-end: Clifford tracing
hir = clifft.trace(circuit)

# HIR optimization
pm = clifft.default_hir_pass_manager()
pm.run(hir)

# Back-end: lower to bytecode
program = clifft.lower(hir)

# Bytecode optimization
bpm = clifft.default_bytecode_pass_manager()
bpm.run(program)
```

This is equivalent to `clifft.compile()` with default passes, but exposes each intermediate representation for inspection or customization.

## Advanced: Reference Syndrome Computation

For QEC workflows, `compute_reference_syndrome()` computes the noiseless detector and observable parities for an `HirModule`. This is the same reference used internally when compiling with `normalize_syndromes=True`.

<!--pytest.mark.skip-->

```python
import clifft

circuit = clifft.parse(circuit_text)
hir = clifft.trace(circuit)
ref = clifft.compute_reference_syndrome(hir)

print(ref["detectors"])
print(ref["observables"])
```

Most users do not need to call this directly; use `normalize_syndromes=True` when compiling instead. See [Simulation](simulation.md#syndrome-normalization) for details.
