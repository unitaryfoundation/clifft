# CPU Execution and Tuning

CPU execution settings control how a supported sampling workflow uses lanes,
threads, and memory. They do not change the circuit semantics or select a
different scientific workflow.

GPU execution is separate and never selected by these controls. See
[Experimental GPU Execution](gpu-execution.md) for the current HIP boundary.

## Defaults First

Start with the sampling function selected in
[Choose a Workflow](../getting-started/choosing-a-workflow.md). For the four
fixed-plan sampling functions, keep `batch_size="auto"`. Leave expert layout
controls unset, and set `threads` only when the process should use a larger CPU
worker budget:

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(program, shots=10_000, seed=42, threads=4)
```

For a fixed-plan sampler, Clifft decides whether that budget is better spent on
several shots or within one wide shot. The automatic batch policy separately
decides whether packing several shots into SIMD lanes is worthwhile.

Use explicit batch capacities or thread layouts only after benchmarking the
same sampling function, circuit, shot count, and output options used in
production.

## Controls at a Glance

| Argument | Default | Meaning |
|---|---|---|
| `threads` | `1` | Total CPU worker budget. A positive integer bounds workers; `"auto"` uses implementation-reported hardware concurrency. |
| `thread_layout` | `None` | Expert `(shot_workers, intra_shot_workers)` override. It replaces the automatic layout and ignores `threads`. |
| `intra_shot_min_active_width` | `None` | Expert active-width threshold for an explicit layout. It requires `thread_layout`; the layout default is 18. |
| `batch_size` | `"auto"` | Packed lane policy. `1` forces scalar execution; a positive integer requests an explicit capacity. |

Threading and packing are independent forms of parallelism:

- **Cross-shot workers** run different shots concurrently. Each worker owns an
  executor and its mutable storage.
- **Intra-shot workers** use an OpenMP team on the active-state work within one
  shot. The team shares one executor.
- **Packed execution** represents multiple shots as SIMD lanes within one
  packed worker. Several packed workers can run concurrently.

Packed execution cannot be combined with intra-shot workers.

## Compatibility by Workflow

| Workflow | Packed `batch_size` | Cross-shot `threads` | Intra-shot layout | Notes |
|---|---|---|---|---|
| `sample()` | Automatic or explicit | Yes | Yes | Rejects a program with post-selection. |
| `sample_survivors()` | Automatic or explicit | Yes | Yes | Automatic packing stays scalar when the program has post-selection; an explicit capacity is supported. |
| `sample_k()` | Automatic or explicit | Yes | Yes | Fixed-fault workflow without post-selection. |
| `sample_k_survivors()` | Automatic or explicit | Yes | Yes | Automatic packing stays scalar when the program has post-selection; an explicit capacity is supported. |
| `noncomp.sample()` | No | Yes | No | Experimental trajectory continuations use cross-shot workers only. |
| `basis_probabilities()` and `record_probabilities()` | No user control | No user control | No user control | Exact-query APIs do not expose sampling execution settings. |
| `get_statevector()` | No | No | No | Dense debugging and validation query, not a shot sampler. |

The fixed-plan rows use the same `Program` and execution controls. Their result
contracts still differ; see [Sampling and Results](simulation.md) and the
[Importance Sampling tutorial](importance-sampling.md).

## Thread Budget and Automatic Scheduling

`sample()`, `sample_survivors()`, `sample_k()`, `sample_k_survivors()`, and
[`noncomp.sample()`](leakage-and-loss.md) accept `threads`. It defaults to `1`.
Pass a positive count to set the total worker budget, or `threads="auto"` to use
implementation-reported hardware concurrency.

For the fixed-plan samplers, the automatic scheduler chooses one layout:

- If the request has at least as many shots as the worker budget, Clifft uses
  up to that many cross-shot workers.
- With fewer shots and `program.peak_active_width >= 18`, an OpenMP-enabled
  build can spend the budget on intra-shot workers instead.
- Otherwise Clifft uses the available cross-shot workers, bounded by the shot
  count.

Automatic scheduling does not create a hybrid layout. Builds without OpenMP
use cross-shot workers only. `noncomp.sample()` also uses cross-shot workers
only because each trajectory can compile different continuations.

In containers or processes with CPU-affinity limits, prefer an explicit count
when reported hardware concurrency exceeds the available CPU quota.

### Explicit and hybrid layouts

Set `thread_layout=(shot_workers, intra_shot_workers)` only when a measured
workload benefits from overriding the scheduler. The tuple replaces `threads`;
keep its product within the CPUs available to the process. An intra-shot count
above one requires an OpenMP-enabled build.

The intra-shot team activates only when the current active width reaches
`intra_shot_min_active_width`, which defaults to 18 for an explicit layout:

<!--pytest.mark.skip-->

```python
result = clifft.sample(
    program,
    shots=8,
    thread_layout=(2, 4),
    intra_shot_min_active_width=17,
)
```

This layout runs two shots at once and gives each shot up to four OpenMP workers
after its active width reaches 17, for at most eight execution threads.

Hybrid layouts require OpenMP processor binding to be disabled. Clifft rejects
a hybrid layout when `OMP_PROC_BIND` is active. Pure cross-shot and pure
intra-shot layouts can use OpenMP binding.

## Packed Batch Sampling

A packed batch stores several shots together so CPU SIMD instructions operate
on their bit columns and active-state data. The four fixed-plan sampling
functions accept `batch_size`.

The default `"auto"` policy considers packed execution only when all of these
conditions hold:

- at least 64 shots were requested;
- the program has no post-selection;
- peak active width is at most 5; and
- estimated packed work and memory remain within the automatic budgets.

Long width-5 plans can still fall back to scalar execution. Automatic survivor
sampling stays scalar for a post-selected plan because survivor lifetimes
cannot be predicted reliably from the static plan.

Set `batch_size=1` to require scalar execution. A positive integer requests an
explicit capacity of up to 2048 lanes, bounded by the shot count and packed
state safety limits:

<!--pytest-codeblocks:cont-->

```python
scalar = clifft.sample(program, 100_000, seed=42, batch_size=1)
packed = clifft.sample(program, 100_000, seed=42, batch_size=1024)
```

An explicit capacity can be useful for post-selected survivor sampling, but it
should be benchmarked against scalar execution. Rejected lanes stop producing
outputs and random work immediately. Packed coefficient kernels can continue
to process their physical positions until repacking is expected to save more
work than it costs.

### Interaction with thread layouts

Packed workers can run across CPU threads, but a packed capacity above one is
incompatible with intra-shot workers. An explicit batch request therefore
fails if the resolved automatic or explicit thread layout uses more than one
intra-shot worker.

Use an explicit cross-shot layout such as `thread_layout=(4, 1)` when both the
worker count and packed capacity must be fixed. Use `batch_size=1` for a pure
intra-shot layout.

Packed execution is unavailable for transition instruments, traps,
continuations, and WebAssembly. `record_probabilities()` is an exact replay API
and does not use packed sampling.

## Reproducibility Across Strategies

Workers dynamically claim contiguous shot ranges. With a fixed seed, changing
`threads` alone produces exactly the same rows and survivor order as a
one-thread run.

Scalar and packed execution use separate random streams, and different packed
capacities can also produce different rows. Every supported strategy remains
statistically equivalent. Keep the workflow and complete execution
configuration fixed when exact seeded replay is required.

## Memory Tradeoffs

Each cross-shot worker owns a separate executor. Dense coefficient and
measurement scratch storage uses roughly $24 \times 2^k$ bytes at peak active
width $k$, in addition to symbolic state, expressions, records, outputs, and
executor metadata. Intra-shot workers cooperate on one executor and do not
replicate this storage.

Packed bit columns add $8 \times \lceil b / 64 \rceil$ bytes per column at lane
capacity $b$, and each packed worker owns a copy. Automatic batching limits
both per-worker and total packed-worker memory. Fixed-fault workers share an
immutable conditioning table but retain independent selection scratch and RNG
state. Noncomputational workers also own their compiled continuations.

Use fewer cross-shot workers or a scalar capacity when memory is more
constrained than CPU availability.

## OpenMP and Process Runtimes

Intra-shot execution requires an OpenMP-enabled build. Apple Clang users may
need Homebrew `libomp`. On macOS, loading different OpenMP runtimes from Clifft
and another scientific package in one process can conflict; process isolation
is the robust choice.

On POSIX systems, create process workers before using threaded Clifft sampling,
or use the `spawn` or `forkserver` start method. Forking after a threaded sample
and then requesting intra-shot threads in the child can hang in some OpenMP
runtimes.

See [Installation](../getting-started/installation.md#from-source) for build and
runtime guidance.

## Benchmark Before Overriding

Performance depends on the circuit, active width over time, shot count,
post-selection lifetime, requested outputs, CPU, and memory limits. Compare the
defaults with a small set of representative alternatives, such as
`batch_size=1`, `256`, and `1024`, using the production thread budget and result
options.

Use `program.peak_active_width` as a first-order cost indicator, but do not
choose a layout from peak width alone. See [Performance](performance.md) for the
broader active-width model and benchmark methodology.
