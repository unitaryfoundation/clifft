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

## Inspect sampling cost

Active width and action count help explain sampling cost and compare compiler
options. `program.peak_active_width`, `program.num_actions`, `program.inspect()`,
and `program.inspect_action(i)` describe full-output sampling, as used by
`sample()` and `sample_survivors(keep_records=True)`.

For survivor counts, select the matching executable once and use its metrics
and inspection methods together:

```python
import clifft

program = clifft.compile("H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]", postselection_mask=[1])
keep_records = False
plan = program.sampling_plan(keep_records=keep_records)
print(f"Sampling peak active width: {plan.peak_active_width}")
print(f"Sampling actions: {plan.num_actions}")
print(plan.inspect_action(0))
result = clifft.sample_survivors(program, shots=1_000, seed=42, keep_records=keep_records)
```

`sampling_plan()` requires an explicit `keep_records` keyword. Match it to the
sampling call: `sample()` retains records, while `sample_survivors()` defaults
to `keep_records=False`.

| Execution | Metadata and inspection |
| --- | --- |
| Full-output ordinary sampling | `program` or `program.sampling_plan(keep_records=True)` |
| Ordinary survivor counts | `program.sampling_plan(keep_records=False)` |
| Exact queries, replay, or fixed-fault sampling | `program.original_plan` |

Each [PlanView](../reference/python-api.md#clifft.PlanView) is read-only and
keeps its owning program alive. Getting a view uses the already compiled
executable; pass the `Program` itself to sampling and exact-query functions.
The view's `reconstruction_selected` and `fallback_reason` explain whether
reconstruction was selected for that output mode. An explicitly requested
`original_plan` has no fallback reason.

Without reconstruction, all views describe the original executable and the
existing properties and inspection methods behave the same way. With
reconstruction, `repr(program)` identifies the full-output sampling metrics and
shows original or survivor-count widths when they differ. Circuit counts,
hidden measurement counts used by replay, and noise-site probabilities retain
their original meanings.

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

For repeated CPU sampling, `reconstruction=True` can reduce the active quantum
state beyond local optimization, particularly in magic-state cultivation and
distillation circuits. It reorganizes the computation while preserving the
joint distribution of measurement records, detectors, observables, and
postselection acceptance. The option is off by default.

This small example uses `hir_passes=None` to expose a width reduction from two
to one. Keep the default passes for ordinary use; they already achieve width
one on this example.

```python
import clifft

program = clifft.compile(
    "H 0 1\nT 0 1\nH 0 1\nM 0 1", hir_passes=None, reconstruction=True
)
plan = program.sampling_plan(keep_records=True)
assert plan.reconstruction_selected
assert program.peak_active_width == plan.peak_active_width == 1
assert program.original_plan.peak_active_width == 2
print(program)
result = clifft.sample(program, shots=100, seed=42)
```

`sample()` uses the plan retaining all records; `sample_survivors()` selects a
plan according to `keep_records`. Use `sampling_plan(keep_records=...)`
to inspect the selected executable, as described in
[Inspect sampling cost](#inspect-sampling-cost). Unsupported
circuits, resource limits, or insufficient width reduction retain the original
plan. Supported circuits use Clifford+T gates, Pauli noise, and symmetric
readout noise; general-angle rotations and `EXP_VAL` probes fall back.

Custom `hir_passes` configure the original pipeline; reconstructed candidates
use the default passes.

Compare total compilation and sampling time at your intended shot count.
Unsuccessful attempts can still cost several times ordinary compilation, and
the early width guard can miss improvements found by later optimization.
Reconstructed detector checks run at the end of the shot, which can reduce the
benefit of early postselection. Seeded results are reproducible for a fixed
configuration but can differ from the original plan.

See [the reconstruction theory](../theory/stabilizer-optimization.md#closed-output-global-reconstruction)
for the transformation, eligibility, and cost guard, and
[program architecture](../theory/architecture.md#python-bindings) for plan ownership
and execution paths.

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
