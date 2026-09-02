# CPU Execution and Tuning

CPU settings control how a supported sampling workflow uses lanes, threads,
and memory. They do not change circuit semantics or select a different
scientific workflow.

## Automatic selection (default)

First choose the function that returns the result you need. The four fixed-plan
samplers share the same CPU controls:

- `clifft.sample()`
- `clifft.sample_survivors()`
- `clifft.sample_k()`
- `clifft.sample_k_survivors()`

For most workloads, leave `batch_size="auto"` and the expert layout controls
unset. Clifft uses one CPU thread by default; it does not automatically claim
all cores on the machine. Set `threads` when the process should use a larger
worker budget:

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(program, shots=10_000, seed=42, threads=4)
```

Clifft chooses whether that budget is better spent across shots or within a
wide shot. It separately decides whether packing shots into SIMD lanes is
worthwhile. Benchmark before overriding either decision.

## Common controls

| Argument | Default | Meaning |
|---|---|---|
| `threads` | `1` | Total CPU worker budget; `"auto"` uses reported hardware concurrency. |
| `batch_size` | `"auto"` | Packed-lane policy; `1` forces scalar execution. |
| `thread_layout` | `None` | Expert `(shot_workers, intra_shot_workers)` override. |
| `intra_shot_min_active_width` | `None` | Expert threshold for enabling an explicit intra-shot layout. |

The fixed-plan samplers above accept all four controls. The leakage and loss
trajectory API, `clifft.noncomp.sample()`, accepts `threads` but not packing or
intra-shot layouts. Exact probability queries and `get_statevector()` do not
expose these sampling controls.

Three independent mechanisms are involved:

- **Cross-shot workers** run different shots concurrently. Each owns an
  executor and mutable storage.
- **Intra-shot workers** use an OpenMP team within one wide shot. The team
  shares an executor.
- **Packed execution** represents several shots as SIMD lanes within one
  worker.

Packed execution cannot be combined with intra-shot workers.

## Power-user tuning

Use explicit settings only after benchmarking the same circuit, shot count,
sampling function, and output options used in production.

### Thread budgets and layouts

Pass a positive `threads` count to set the total worker budget, or
`threads="auto"` to use implementation-reported hardware concurrency. The
automatic scheduler chooses one layout:

- With at least as many shots as workers, use cross-shot workers.
- With fewer shots and `program.peak_active_width >= 18`, an OpenMP-enabled
  build can spend the budget within shots.
- Otherwise use cross-shot workers bounded by the shot count.

Automatic scheduling does not create a hybrid layout. Builds without OpenMP
and noncomputational trajectories use cross-shot workers only. In containers
with CPU-affinity limits, prefer an explicit count if reported hardware
concurrency exceeds the process quota.

Set `thread_layout=(shot_workers, intra_shot_workers)` to override the
scheduler. The tuple replaces `threads`; keep its product within the CPUs
available to the process. An intra-shot count above one requires OpenMP.

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(
    program,
    shots=8,
    thread_layout=(2, 4),
    intra_shot_min_active_width=17,
)
```

This layout runs two shots concurrently and gives each shot up to four OpenMP
workers after the active width reaches 17. Hybrid layouts require OpenMP
processor binding to be disabled; Clifft rejects one when `OMP_PROC_BIND` is
active.

### Packed batch sampling

The default `batch_size="auto"` considers packed execution only when:

- at least 64 shots were requested;
- the program has no post-selection;
- peak active width is at most 5; and
- estimated packed work and memory stay within automatic budgets.

Long width-5 plans can still use scalar execution. Automatic survivor sampling
also stays scalar because survivor lifetimes cannot be predicted reliably from
the static plan.

Set `batch_size=1` to require scalar execution. A positive integer requests a
capacity of up to 2048 lanes, bounded by the shot count and safety limits:

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
scalar = clifft.sample(program, 100_000, seed=42, batch_size=1)
packed = clifft.sample(program, 100_000, seed=42, batch_size=1024)
```

An explicit capacity can help survivor sampling but should be measured against
scalar execution. Use an explicit cross-shot layout such as
`thread_layout=(4, 1)` when both worker count and packed capacity must be fixed.
Packed execution is unavailable for transition instruments, traps,
continuations, and WebAssembly.

### Reproducibility

Workers dynamically claim contiguous shot ranges. With a fixed seed, changing
`threads` alone produces the same rows and survivor order as one-thread
execution.

Scalar and packed execution use separate random streams, and different packed
capacities can produce different rows. Every supported strategy remains
statistically equivalent. Keep the complete execution configuration fixed when
exact seeded replay is required.

### Memory tradeoffs

Each cross-shot worker owns an executor. Dense coefficient and measurement
scratch storage uses roughly $24 \times 2^k$ bytes at peak active width $k$, in
addition to symbolic state, records, outputs, and metadata. Intra-shot workers
cooperate on one executor and do not replicate this storage.

Packed bit columns add $8 \times \lceil b / 64 \rceil$ bytes per column at lane
capacity $b$, and each packed worker owns a copy. Use fewer cross-shot workers
or a scalar capacity when memory is more constrained than CPU availability.

### OpenMP and process runtimes

Intra-shot execution requires an OpenMP-enabled build. Apple Clang users may
need Homebrew `libomp`. Loading different OpenMP runtimes from Clifft and
another scientific package in one process can conflict, especially on macOS;
process isolation is the robust choice.

On POSIX systems, create process workers before threaded Clifft sampling, or
use the `spawn` or `forkserver` start method. Forking after a threaded sample
and then requesting intra-shot threads in the child can hang in some runtimes.
See [Installation](../getting-started/installation.md#from-source) for build
guidance.

### Benchmark before overriding

Performance depends on the circuit, active width over time, shot count,
post-selection lifetime, outputs, CPU, and memory limits. Compare the defaults
with a small set of representative alternatives such as `batch_size=1`, `256`,
and `1024`, using the production worker budget and result options.

Use `program.peak_active_width` as a first-order cost indicator, but do not
choose a layout from peak width alone.

## Compile-time scheduling

`ActiveWidthSchedulePass` is an opt-in HIR pass, not part of the default
pipeline, that reorders Heisenberg IR operations to reduce peak active
width, then a dense-work estimate; it never leaves a circuit worse than it
found it by peak active width and then by estimated dense work. See
[Active-Width Scheduling](../theory/active-width.md) for the
structural model it searches over. Enable it by building a custom
`HirPassManager` that runs it last, after `PeepholeFusionPass` and
`StatevectorSqueezePass`, and passing that manager to `hir_passes`:

```python
import clifft

pm = clifft.HirPassManager()
pm.add(clifft.PeepholeFusionPass())
pm.add(clifft.StatevectorSqueezePass())
pm.add(clifft.ActiveWidthSchedulePass())

program = clifft.compile("H 0\nT 0\nM 0", hir_passes=pm)
```

On most circuits this costs single-digit milliseconds, but circuits with
many simultaneously-ready, mutually independent non-Clifford rotations can
push compile time higher -- about 600 ms on the largest circuit in the
measured `clifft-paper` QEC corpus, a cost paid once per compiled program
rather than once per shot. See
[Measured Effect](../theory/active-width.md#measured-effect) for the full
per-circuit table.

See [Optimization Passes](../reference/passes.md) for every available pass
and its default.
