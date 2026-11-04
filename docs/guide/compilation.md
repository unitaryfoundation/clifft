# Compiling Circuits

Clifft compiles a supported [circuit input](circuit-inputs.md) into a reusable
`Program`. Compilation resolves Clifford coordinates, symbolic dependencies,
active-state transitions, and executable actions before sampling starts.

```text
circuit input -> parse -> trace to HIR -> optimize -> plan and prepare -> Program
```

Most users only need `clifft.compile()`. The lower-level functions expose the
same pipeline for inspecting intermediate representations or supplying a custom
optimization sequence.

## Default path

Pass circuit text directly to `clifft.compile()`:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    T 2
    M 0 1 2
""")
```

The returned program can be reused by ordinary sampling, survivor sampling,
fixed-fault importance sampling, or an exact-query API when the circuit meets
that API's requirements:

<!--pytest-codeblocks:cont-->

```python
result = clifft.sample(program, shots=1_000, seed=42)
```

Compilation options such as `postselection_mask`, `normalize_syndromes`,
`expected_detectors`, and `expected_observables` configure how detectors,
observables, and post-selection are evaluated during sampling.

For detector-based post-selection, use `normalize_syndromes=True` when the
circuit has fixed detector and observable parities after stochastic and readout
noise are removed. Clifft samples the reference once during compilation without
checking determinism; coherent error rotations remain. For those errors, supply
`expected_detectors` and `expected_observables` from the intended ideal circuit,
as described in [reference selection](simulation.md#detectors-and-observables).
`sample_survivors()` applies the post-selection mask to the normalized detector
values; it does not recompute the reference per call.

Circuits containing `LEAKAGE`, `LOSS`, `LEVEL_TRANSITION`,
`LEAKAGE_INTERACTION`, or `LOSS_INTERACTION` annotations, or the status checks
`HERALD_LEAKAGE_EVENT` and `HERALD_LOSS_EVENT`, use the `clifft.noncomp` API.
Pass the circuit and model to `clifft.noncomp.sample()`, or compile once for
repeated sampling with `clifft.noncomp.compile()`. Clifft automatically selects
fixed execution or trajectories; see [Leakage and Loss](leakage-and-loss.md).

## Active-width scheduling

For repeated sampling, try the opt-in `ActiveWidthSchedulePass`. It searches
for an operation order with lower peak active width, or less estimated dense
work at the same peak. A smaller active state can save sampling time and memory,
but the search adds compilation time and may find no improvement.

Add it after the default fusion, phase reduction and squeezing passes:

```python
import clifft

pm = clifft.default_hir_pass_manager()
pm.add(clifft.ActiveWidthSchedulePass())
program = clifft.compile("H 0 1 2\nT 0 1 2\nM 0 1 2", hir_passes=pm)
```

In the playground, open **Passes** and enable **ActiveWidthSchedulePass**.
Ordinary compilation keeps this extra search disabled.

Compare compilation plus sampling time at the shot counts your application
uses. Compilation is paid once per program, so savings across many shots can
repay the search cost. A lower width or work estimate does not guarantee faster
execution; omitting the pass can be faster for short runs. Reordering preserves
sampling distributions but can change samples for a given random seed.

See [Optimization Passes](../reference/passes.md#activewidthschedulepass) for
options, result statistics, and search-budget limits, and the
[theoretical overview](../theory/overview.md#hir-optimization) for why ordering
affects active width.

## Global reconstruction for sampling

When a circuit has many non-Clifford gates whose intermediate quantum state is
discarded or measured out, `reconstruction=True` can build a smaller sampling
plan that preserves the joint distribution of the requested classical outputs.
This is particularly effective for magic-state cultivation and distillation
circuits, where global Clifford dilation and projection folding can reduce peak
active width beyond what local HIR passes achieve.

```python
import clifft

program = clifft.compile("H 0\nT 0\nH 0\nM 0", reconstruction=True)
print(program.reconstruction_info(keep_records=True))
result = clifft.sample(program, shots=100, seed=42)
```

Compilation prepares up to two reconstructed plans alongside the original
executable: one that retains all measurement records, detectors, and
observables, and one that retains only detectors and observables. `sample()`
uses the full-output plan, while `sample_survivors()` chooses between them
based on `keep_records`. Output order, reference parities, correlations, and
postselection acceptance probabilities are preserved.

Exact state and probability queries (`get_statevector()`,
`basis_probabilities()`, `record_probabilities()`) and fixed-fault sampling
(`sample_k()`, `sample_k_survivors()`) always use the original executable.
`EXP_VAL` probes disable reconstruction.

`reconstruction_info(keep_records=...)` reports `selected`, `fallback_reason`,
the selected `peak_active_width` and `num_actions`, and
`original_peak_active_width`. Standard `Program` properties such as
`num_qubits`, `peak_active_width`, `noise_site_probabilities`, and `inspect()`
continue to describe the original executable. Each reconstructed candidate is
selected only when it is supported, stays within compiler resource limits, and
strictly reduces peak active width compared to the original plan. Clifford
circuits already at width zero skip the search.

Reconstruction supports Clifford+T circuits with Pauli measurements, resets,
record-controlled Pauli feedback, categorical Pauli noise, and symmetric
readout noise. General-angle rotations, noncomputational instruments,
asymmetric readout noise, and custom HIR passes that alter the sampling
distribution (such as `RemoveNoisePass`) fall back to the original plan.
Reconstructed HIR runs the default HIR optimization pipeline; `hir_passes`
configures only the original pipeline.

Because reconstruction adds compilation time and extra plans, compare end-to-end
compile plus sampling time at your target shot count, especially on circuits
that rely on early detector postselection (reconstructed detector checks run at
the end of the shot). Seeded runs are reproducible for a given compiled program
and `keep_records` setting, though individual trajectories may differ from the
original plan. The option is off by default and applies to `clifft.compile()`
for CPU sampling; `clifft.lower()` and experimental GPU compilation prepare
only the supplied HIR.

## Inspect or customize the pipeline

The following APIs are intended for power users working on compiler behavior,
custom optimization, or intermediate-representation inspection.

### Parse the circuit

`clifft.parse()` returns a `Circuit` without tracing or lowering it:

```python
import clifft

circuit = clifft.parse("H 0\nCNOT 0 1\nM 0 1")
```

Use `clifft.parse_file()` for a file, or the format-specific parsing functions
listed on [Circuit Inputs](circuit-inputs.md) for OpenQASM 2.

### Trace to the Heisenberg IR

`clifft.trace()` absorbs Clifford operations into an offline tableau and emits
explicit non-Clifford operations, measurements, noise, and classical outputs
in the Heisenberg frame:

```python
import clifft

hir = clifft.trace(clifft.parse("H 0\nCNOT 0 1\nT 0\nM 0 1"))
print(hir)
```

### Optimize the HIR

The default pass manager applies the same HIR optimization sequence used by
`clifft.compile()`:

```python
import clifft

hir = clifft.trace(clifft.parse("H 0\nCNOT 0 1\nT 0\nM 0 1"))
clifft.default_hir_pass_manager().run(hir)
```

Construct a focused pipeline when individual passes need to be controlled:

```python
import clifft

hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
passes = clifft.HirPassManager()
passes.add(clifft.PeepholeFusionPass())
passes.run(hir)
```

Pass `hir_passes=None` to `clifft.compile()` to skip HIR optimization while
keeping the one-step API.

### Plan and prepare execution

`clifft.lower()` converts an optimized `HirModule` into the same public
`Program` type returned by `clifft.compile()`:

```python
import clifft

hir = clifft.trace(clifft.parse("H 0\nT 0\nM 0"))
clifft.default_hir_pass_manager().run(hir)
program = clifft.lower(hir)
```

When optimizing for survivor sampling or normalized syndromes, pass
`postselection_mask`, `normalize_syndromes`, `expected_detectors`, or
`expected_observables` to `clifft.trace()`. The resulting `HirModule` carries
those settings through HIR optimization and into `clifft.lower(hir)`
automatically.

Internally, lowering first creates a semantic sampling plan that selects active
coordinates and records expressions, rotations, measurements, width
transitions, and outputs. Executable preparation then creates the immutable
descriptors and dependency tables used by the executor and selects supported
scalar or SIMD kernels.

These decisions happen before execution. Runtime kernels do not evolve a
tableau, choose coordinates, localize Paulis, or discover dependencies. See
[Software Architecture](../theory/architecture.md) for how HIR, semantic
planning, executable preparation, and shot execution fit together.
