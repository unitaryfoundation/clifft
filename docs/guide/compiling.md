# Compiling Circuits

UCC compiles quantum circuits through a four-stage Ahead-of-Time (AOT) pipeline. This page explains each stage and the Python API for interacting with them.

## The Compilation Pipeline

```text
Circuit Text --> Parse --> Front-End --> Optimizer --> Back-End --> Program
                 |           |              |            |            |
              Circuit     HirModule     HirModule    Program     Bytecode
               (AST)       (HIR)    (Optimized HIR)  (RISC)     ready to
                                                                 execute
```

## One-Step Compilation

For most use cases, `ucc.compile()` runs the full pipeline:

```python
import ucc

program = ucc.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")
```

## Step-by-Step Compilation

You can also run each stage individually for inspection or custom pipelines:

### 1. Parsing

`ucc.parse()` converts Stim circuit text into an AST:

```python
import ucc

circuit = ucc.parse("""
    H 0
    CNOT 0 1
    M 0 1
""")
```

You can also parse from a file:

<!--pytest.mark.skip-->

```python
circuit = ucc.parse_file("my_circuit.stim")
```

### 2. Front-End (Clifford Tracing)

`ucc.trace()` runs the Clifford front-end, absorbing Clifford gates into the Heisenberg frame and producing the Heisenberg IR:

```python
import ucc

circuit = ucc.parse("H 0\nCNOT 0 1\nT 0\nM 0 1")
hir = ucc.trace(circuit)

print(hir)  # HirModule(4 ops, 2 T-gates, 2 qubits)
```

### 3. Optimization

The optimizer applies transformation passes to the HIR:

<!--pytest-codeblocks:cont-->

```python
import ucc

# Get the default pass manager (includes peephole fusion)
pm = ucc.default_pass_manager()

# Or build a custom one
pm = ucc.PassManager()
pm.add(ucc.PeepholeFusionPass())

# Run passes on the HIR module
pm.run(hir)
```

### 4. Back-End (Lowering)

`ucc.lower()` compiles the HIR down to executable VM bytecode:

<!--pytest-codeblocks:cont-->

```python
program = ucc.lower(hir)
```

## Full Custom Pipeline

Putting it all together:

```python
import ucc

# Parse
circuit = ucc.parse("H 0\nT 0\nCNOT 0 1\nM 0 1")

# Front-end: Clifford tracing
hir = ucc.trace(circuit)

# Optimize
pm = ucc.default_pass_manager()
pm.run(hir)

# Back-end: lower to bytecode
program = ucc.lower(hir)
```

This is equivalent to `ucc.compile()` but gives you access to intermediate representations for debugging or custom optimization passes.

## Post-Selection

For circuits with detectors (e.g., QEC), you can mark specific detectors for post-selection. Shots where a marked detector fires are discarded:

<!--pytest.mark.skip-->

```python
import ucc

# Mark detector 0 for post-selection
program = ucc.compile(circuit_text, postselection_mask=[1])
```

See [Simulation](simulation.md) for how to use `sample_survivors()` with post-selection.
