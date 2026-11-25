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
`sample()` by default and calls with `keep_records=True` or `"all"`.

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
sampling call, including selective output choices. `sample()` defaults to full
output, while `sample_survivors()` defaults to `keep_records=False`.

| Execution | Metadata and inspection |
| --- | --- |
| Full-output sampling | `program` or `program.sampling_plan(keep_records=True)` |
| Syndrome rows, including `sample_detectors()` | `program.sampling_plan(keep_records="syndromes")` |
| Measurement rows | `program.sampling_plan(keep_records="measurements")` |
| Survivor counts | `program.sampling_plan(keep_records=False)` |
| Exact queries, replay, or fixed-fault sampling | `program.original_plan` |

Each [PlanView](../reference/python-api.md#clifft.PlanView) is read-only and
keeps its owning program alive. Getting a view uses the already compiled
executable; pass the `Program` itself to sampling and exact-query functions.
The view's `reconstruction_selected` and `fallback_reason` explain whether
reconstruction was selected for that output mode. An explicitly requested
`original_plan` has no fallback reason.

These views describe the selected compiled plan before output preparation
removes unused storage and evaluations. The first sampling call for an output
choice prepares and caches that work. Include this first-call cost when
benchmarking workloads that compile many programs and sample only a few shots
from each. Reusing a `Program` amortizes preparation across later calls.

With reconstruction, `repr(program)` shows the original and survivor-count
widths when they differ from the full-output width, and marks unavailable
plans. Circuit and measurement counts still describe the original circuit.

## Active-width scheduling

The default HIR pipeline includes `ActiveWidthSchedulePass` after fusion, phase
reduction, and squeezing. It searches legal operation orders to reduce peak
active width (and, at width 13 or above by default, estimated dense work), while
capping speculative sweeps at `max(65536, num_ops)` and switching to greedy
completion at a fixed classification-probe backstop so compilation stays
bounded.

To customize search parameters or inspect pass statistics, override it in the
default pipeline:

```python
import clifft

schedule = clifft.ActiveWidthSchedulePass(beam_width=16)
pm = clifft.default_hir_pass_manager(overrides=[schedule])
program = clifft.compile("H 0 1 2\nT 0 1 2\nM 0 1 2", hir_passes=pm)
```

Appending an explicit `ActiveWidthSchedulePass` to
`default_hir_pass_manager()` replaces the existing scheduler rather than running
two schedulers back-to-back. In the playground, **ActiveWidthSchedulePass** is
enabled by default under **Passes** and can be toggled off there.

To disable only scheduling, pass a `HirPassManager` containing just the other
passes you want; passing `hir_passes=None` skips all HIR optimization passes.
Reordering preserves sampling distributions but can change samples for a given
random seed.

See [Optimization Passes](../reference/passes.md#activewidthschedulepass) for
options, result statistics, and search-budget limits, and the
[theoretical overview](../theory/overview.md#hir-optimization) for why ordering
affects active width.

## Global reconstruction for sampling

For CPU sampling, global reconstruction reorganizes the computation while
preserving the joint distribution of measurement records, detectors,
observables, and postselection acceptance. This can reduce peak active width
beyond local HIR passes, particularly on magic-state cultivation and
distillation circuits.

`clifft.compile()` defaults to `reconstruction="auto"`, which tries bounded
output-specific reconstruction for both full-record and syndrome-only sampling:

- Takes a reconstructed plan when it lowers peak active width, except on
  postselected circuits of original width at most 16 where a nonzero
  reconstructed width saves at most two qubits and the syndrome plan increases
  actions by more than 10% (such as multi-round coherent-noise memory circuits,
  where extra trajectory splits outweigh the smaller state).
- Breaks equal-width ties by requiring no increase in plan actions and lower
  estimated plan work.
- Uses a smaller scheduling budget when the original plan is already at most
  width 16, and the full budget on wider or overflowing originals.

Automatic reconstruction adds compilation time over compiling the original plan
alone; pass `reconstruction=False` when compile time dominates over sampling, or
`reconstruction=True` to use the full reconstruction budget for any strictly
narrower plan.

This small example uses `hir_passes=None` so automatic reconstruction exposes a
width reduction from two to one (the default HIR passes already achieve width
one on this circuit):

```python
import clifft

program = clifft.compile("H 0 1\nT 0 1\nH 0 1\nM 0 1", hir_passes=None)
plan = program.sampling_plan(keep_records=True)
assert plan.reconstruction_selected
assert program.peak_active_width == plan.peak_active_width == 1
assert program.original_plan.peak_active_width == 2
print(program)
result = clifft.sample(program, shots=100, seed=42)
```

Ordinary and survivor sampling use the plan for the requested `keep_records`.
If the circuit is ineligible or reconstruction does not improve over the
original plan, the sampler uses the original plan when available. See
[Inspect sampling cost](#inspect-sampling-cost) to check which plan was selected.

The CPU planner reaches its dense-state limit when 60 coordinates are active
at once. A circuit can have many more physical qubits. If the original
plan reaches this limit, compilation can still succeed for the output modes
that reconstruction brings below it. Requesting an unavailable mode or the
original plan raises `OverflowError`. Exact queries, replay, fixed-fault
sampling, and `noise_site_probabilities` all need the original plan.

Custom `hir_passes` configure the original pipeline; reconstructed candidates
run the default fusion, phase, rotation, and squeezing passes without
re-running the active-width scheduler.

Seeded results are reproducible for a fixed configuration, though selecting a
reconstructed plan can produce different rows for the same seed than
`reconstruction=False`.

See [the reconstruction theory](../theory/stabilizer-optimization.md#closed-output-global-reconstruction)
for the transformation, eligibility, and cost guard, and
[program architecture](../theory/architecture.md#python-bindings) for plan ownership
and execution paths.

## Runtime Clifford frame for noisy T regions

Noise and random measurement outcomes can flip the signs of T-like rotations.
The original plan then keeps one active qubit for every rotation whose sign
varies between shots, even when the noiseless circuit cancels most of them.
Wide magic-state factories built from transversal `T` or `CCZ` layers reach
active widths in the tens this way, while their noiseless T regions need only
a few. A runtime Clifford frame keeps such a region at its noiseless width:
the plan applies the rotations with fixed signs and tracks every sign flip as
a Clifford update to a frame that each shot maintains as it runs.

The frame is off by default. Request it with `runtime_frame`:

```python
import clifft

circuit = """
H 0
CX 0 1
X_ERROR(0.1) 0
T 0
T_DAG 1
MPP X0*X1
DETECTOR rec[-1]
"""
ordinary = clifft.compile(circuit)
frame = clifft.compile(circuit, runtime_frame=True)
assert ordinary.original_plan.peak_active_width == 1
assert frame.original_plan.uses_runtime_frame
assert frame.original_plan.peak_active_width == 0
result = clifft.sample(frame, shots=1000, seed=42, keep_records="syndromes")
```

On the Bell pair, `T` and `T_DAG` cancel unless the `X` error flips one sign.
The ordinary plan needs an active qubit for that case; the frame plan applies
the leftover Clifford to its frame instead.

- `runtime_frame=True` plans the circuit on a frame wherever a T region is
  narrower that way. A circuit without such a region keeps its ordinary plan,
  and `program.runtime_frame_fallback_reason` says why. Where the corrections
  of every region cancel, the narrower plan needs no frame and runs on every
  executor; the reason says so.
- `runtime_frame="auto"` plans both ways and keeps the frame plan when the
  ordinary original plan is too wide for a dense state, or is at least 16 wide
  and the frame lowers the active-width bound by 4 or more. Below those
  limits the ordinary plan samples faster: updating the frame costs more per
  shot than the coefficients it saves.
- `runtime_frame=False` never uses a frame.

The frame plan replaces the original plan, so sampling, exact queries, replay,
and fixed-fault sampling all use it, and reconstruction is selected against
its width. Check `program.original_plan.uses_runtime_frame` to see the choice.

Things to know when choosing it:

- A frame plan's `peak_active_width` is an upper bound. Each shot decides how
  wide its state becomes, and `"auto"` compares that bound, so it can decline
  a circuit whose shots would stay narrow. Pass `True` for such a circuit.
- Reconstruction still applies. A reconstructed plan replaces the frame plan
  for sampling when it is no wider than the width every shot of the frame
  plan reaches. With `"auto"` it also does when it is narrower than 16, the
  width below which `"auto"` prefers an ordinary plan to a frame.
  `program.sampling_plan(keep_records=...)` shows which plan runs.
- A frame ends once measurements or resets have fixed the state of every
  qubit it covers, and the next T region starts a new one. A factory repeated
  on the same qubits is therefore reduced in every round. While some of those
  qubits stay unmeasured the frame continues, and later T gates on them are
  applied one at a time without the reduction.
- The frame plan is built from the circuit before the HIR passes run, because
  moving measurements earlier splits the regions it reduces. A pass that
  changes the sampling distribution, such as `RemoveNoisePass`, applies to the
  frame plan as well.
- Frame plans run on the scalar and shared CPU executors; see
  [shared trajectory execution](cpu-execution.md#shared-trajectory-execution).
  Circuits with instruments or `EXP_VAL` probes keep their ordinary plan.
- Seeded results differ between a frame plan and an ordinary plan of the same
  circuit.

See [rotations whose signs vary between shots](../theory/stabilizer-optimization.md#rotations-whose-signs-vary-between-shots)
for the identity behind the frame.

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
