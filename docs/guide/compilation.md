# Compiling Circuits

Clifft compiles quantum circuits through a five-stage multi-level pipeline. This page explains each stage and the Python API for interacting with them.

## The Compilation Pipeline

```text
Circuit Text --> Parse --> Front-End --> Middle-End Optimizer --> Back-End --> Bytecode Optimizer --> Program
                 |           |                 |                  |               |                  |
              Circuit     HirModule        HirModule          Program         Program            Bytecode
               (AST)       (HIR)       (Optimized HIR)    (raw bytecode) (fused instructions)   ready to
                                                                                                execute
```

## One-Step Compilation

For most use cases, `clifft.compile()` runs the full pipeline with the
default HIR and bytecode optimization passes:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")
```

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

You can also supply a custom `HirPassManager` or `BytecodePassManager` to
override the defaults (see [Optimization Passes](../reference/passes.md)).

### Syndrome Normalization

For circuits with detectors and observables (e.g., QEC circuits), detector and observable outputs represent raw measurement parities by default. A `DETECTOR` that natively evaluates to `1` in a noiseless circuit will always output `1`, even without errors. This can confuse decoders like PyMatching that expect `0` = "no error" and `1` = "error".

Set `normalize_syndromes=True` to automatically compute a noiseless reference and XOR-normalize all outputs:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile(
    circuit_text,
    normalize_syndromes=True,
)
```

This internally:

1. Strips all noise from the HIR
2. Runs a single noiseless reference shot
3. XOR-normalizes each detector and observable against the reference

After normalization, `0` strictly means "matches noiseless reference" and `1` strictly means "error". Post-selection also benefits: a detector that natively fires in the clean circuit won't cause spurious discards.

!!! note
    `normalize_syndromes=True` is mutually exclusive with manually passing
    `expected_detectors` or `expected_observables`.

You can also supply explicit reference parities if you've computed them yourself:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile(
    circuit_text,
    expected_detectors=[1, 0, 0, 1],
    expected_observables=[1],
)
```

### Combining Post-Selection and Normalization

Post-selection and syndrome normalization compose naturally:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile(
    circuit_text,
    postselection_mask=[1, 0, 0],  # Post-select on detector 0
    normalize_syndromes=True,       # Auto-normalize all syndromes
)

result = clifft.sample_survivors(program, shots=1_000_000, seed=42)
```

## Step-by-Step Compilation

You can also run each stage individually for inspection or custom pipelines:

### 1. Parsing

`clifft.parse()` converts Stim circuit text into an AST:

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

### 2. Front-End (Clifford Tracing)

`clifft.trace()` runs the Clifford front-end, absorbing Clifford gates into the offline Clifford frame $U_C$ and producing the Heisenberg IR:

```python
import clifft

circuit = clifft.parse("H 0\nCNOT 0 1\nT 0\nM 0 1")
hir = clifft.trace(circuit)

print(hir)  # HirModule(4 ops, 2 T-gates, 2 qubits)
```

### 3. Middle-End Optimization

The optimizer applies transformation passes to the HIR before lowering:

<!--pytest-codeblocks:cont-->

```python
import clifft

# Get the default HIR pass manager (includes peephole fusion)
pm = clifft.default_hir_pass_manager()

# Or build a custom one
pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())

# Run passes on the HIR module
pm.run(hir)
```

### 4. Back-End (Lowering)

`clifft.lower()` compiles the HIR down to executable VM bytecode:

<!--pytest-codeblocks:cont-->

```python
program = clifft.lower(hir)
```

`lower()` also accepts optional `postselection_mask`, `expected_detectors`, and `expected_observables` arguments for syndrome normalization and post-selection at the bytecode level (see [Syndrome Normalization](#syndrome-normalization)).

### 5. Bytecode Optimization

After lowering, a second pass manager optimizes the bytecode. This fuses instructions to reduce redundant array passes:

<!--pytest-codeblocks:cont-->

```python
# Get the default bytecode pass manager
bpm = clifft.default_bytecode_pass_manager()

# Run bytecode passes on the compiled program
bpm.run(program)
```

See [Optimization Passes](../reference/passes.md) for the full list of default
and optional passes available at both IR levels.

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

This is equivalent to `clifft.compile()` but gives you access to intermediate representations for debugging or custom optimization passes.

## Reference Syndrome Computation

If you need the noiseless reference parities without compiling, use `compute_reference_syndrome()` on an `HirModule`:

<!--pytest.mark.skip-->

```python
import clifft

circuit = clifft.parse(circuit_text)
hir = clifft.trace(circuit)
ref = clifft.compute_reference_syndrome(hir)

print(ref["detectors"])     # list of expected detector parities
print(ref["observables"])   # list of expected observable parities
```

This strips noise from the HIR, lowers and executes a single shot, and returns the noiseless parities. It is the same logic used internally by `normalize_syndromes=True`.

## Noise Removal Pass

`RemoveNoisePass` strips all stochastic noise and readout noise ops from the HIR. It is **not** included in the default pass list — it is used internally by `compute_reference_syndrome()` for noiseless reference shots, but you can use it directly if needed:

<!--pytest.mark.skip-->

```python
import clifft

pm = clifft.HirPassManager()
pm.add(clifft.RemoveNoisePass())
pm.run(hir)  # hir now has no noise ops
```

## Post-Selection

For circuits with detectors (e.g., QEC), you can mark specific detectors for post-selection. Shots where a marked detector fires are discarded:

<!--pytest.mark.skip-->

```python
import clifft

# Mark detector 0 for post-selection
program = clifft.compile(circuit_text, postselection_mask=[1])
```

See [Simulation](simulation.md) for how to use `sample_survivors()` with post-selection.
