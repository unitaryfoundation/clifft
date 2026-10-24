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
`expected_detectors`, and `expected_observables` define the result contract
used during sampling.

For detector-based post-selection, use `normalize_syndromes=True` when the
circuit has fixed detector and observable parities after stochastic and readout
noise are removed. Clifft samples the reference once during compilation without
checking determinism; coherent error rotations remain. For those errors, supply
`expected_detectors` and `expected_observables` from the intended ideal circuit,
as described in [reference selection](simulation.md#detectors-and-observables).
`sample_survivors()` applies the post-selection mask to the normalized detector
values; it does not recompute the reference per call.

Circuits containing `LEAKAGE`, `LOSS`, or `LEVEL_TRANSITION` annotations, or the
status probes `HERALD_LEAKAGE_EVENT` and `HERALD_LOSS_EVENT`, take a
trajectory-specific path. Pass the circuit and model to
`clifft.noncomp.sample()` rather than compiling one fixed program first; see
[Leakage and Loss](leakage-and-loss.md).

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

For repeated CPU sampling, `reconstruction=True` opts in to a bounded compiler
that can reduce active width beyond the ordinary HIR passes. It works from the
supplied circuit, including supported noise, without a ZX dependency or a
circuit-family lookup. It is currently useful on magic-state cultivation
circuits; eligibility alone does not guarantee a runtime improvement.

```python
import clifft

program = clifft.compile("H 0\nT 0\nH 0\nM 0", reconstruction=True)
print(program.reconstruction_info(keep_records=True))
result = clifft.sample(program, shots=100, seed=42)
```

Compilation prepares up to two extra plans: one retaining the joint law of all
measurement records, detectors, and observables, and one retaining detectors
and observables. `sample()` uses the first; `sample_survivors()` chooses based on
`keep_records`. Output order, reference parities, correlations, and acceptance
probability are preserved. There is no per-fault compilation or plan cache.

The circuit and API call determine what must be preserved. `EXP_VAL` probes
disable reconstruction. Exact state/probability queries, replay, and
`sample_k`/`sample_k_survivors` use the original executable, preserving their
existing restrictions and original fault sites. The smaller sampling state is
an internal representation of the requested distribution; it makes no claim
that the circuit's physical output state is a stabilizer state.

`reconstruction_info(keep_records=...)` reports `selected`, `fallback_reason`,
the selected `peak_active_width` and `num_actions`, and
`original_peak_active_width`. Existing properties such as `num_qubits`,
`peak_active_width`, and `noise_site_probabilities`, plus `inspect()`, continue
to describe the original executable. If a candidate is unsupported, exceeds a
compiler limit, or does not strictly lower peak width, ordinary sampling uses
the original. Clifford circuits already at width zero skip the search.

Reconstruction currently supports Clifford+T circuits with Pauli measurements,
resets, record-controlled Pauli feedback, and supported categorical Pauli and
symmetric readout noise. General rotations, instruments, and asymmetric
readout noise fall back. Custom HIR passes that change the distribution, such
as `RemoveNoisePass`, also force fallback. Reconstructed HIR uses its own
default optimization pipeline; `hir_passes` configures the original pipeline.

This option adds compilation time and plan storage. Compare total time at your
shot count, particularly for circuits with early postselection: reconstructed
checks currently occur at the end. Lower width does not ensure faster sampling.
Seeded results repeat with the same compiled program and execution settings,
but may differ when reconstruction or the requested output contract changes.
The option is off by default and applies to the regular CPU compiler;
`lower(hir)` and experimental GPU compilation prepare only the supplied HIR.

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
[Software Architecture](../theory/architecture.md) for the contracts between
HIR, semantic planning, executable preparation, and shot execution.
