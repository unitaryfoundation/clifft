# CPU Execution and Tuning

CPU settings control how sampling uses SIMD lanes, threads, and memory without
changing the sampled distribution.

## Automatic selection (default)

First choose the function that returns the result you need. The four standard
samplers share the same CPU controls:

- `clifft.sample()`
- `clifft.sample_survivors()`
- `clifft.sample_k()`
- `clifft.sample_k_survivors()`

For most workloads, leave `batch_size="auto"` and the expert layout controls
unset. Clifft uses one CPU thread by default; set `threads` to use more
workers:

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(program, shots=10_000, seed=42, threads=4)
```

Clifft chooses whether that budget is better spent across shots or within a
wide shot. It separately decides whether packing shots into SIMD lanes is
worthwhile. Benchmark before overriding either decision.

## Reusing survivor sampling work

Repeated postselected count jobs can spend more time preparing noise and evolving
identical trajectories than collecting survivors. `SurvivorSampler` retains
worker storage across calls, rejects shots using necessary detector constraints
before quantum work, and shares a coefficient state until shots need different
rotation signs or measurement outcomes. Remaining selected detectors are checked
as soon as their input records are ready, so rejected shots skip later evolution.

```python
import clifft

program = clifft.compile(
    "H 0 1\nT 0 1\nDEPOLARIZE1(0.001) 0 1\nH 0 1\nM 0 1\n"
    "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]",
    postselection_mask=[1],
    reconstruction=True,
)
sampler = clifft.SurvivorSampler(
    program,
    batch_size=8192,
    threads=2,
    initial_states=128,
    noise_preparation=clifft.NoisePreparation.SITE_SPARSE,
)
result = sampler.sample(100_000, seed=42)
print(result.passed_shots, result.observable_ones)
```

This interface returns aggregate survivor, logical-error, and observable counts;
its per-shot arrays are empty. It uses the program's `keep_records=False` plan,
including reconstruction when selected. Instrument continuations are unsupported.
Use the ordinary sampling functions for retained rows or fixed-fault conditioning.
Plans with zero active width use the packed classical executor with the selected
noise preparation; they need no trajectory groups or separate entrance checks.

The settings are explicit so you can measure the complete workload:

- `batch_size` accepts a positive integer, `"auto"` (the default), or `"tune"`.
  A numeric value sets the classical batch capacity. Larger batches can amortize
  noise preparation but leave too few batches to occupy all workers on small calls.
  Automatic capacity fits the state budget even if every lane becomes distinct.
  An explicit capacity can be larger when sharing keeps the state count small.
- `batch_size="tune"` calibrates capacities on the first nonzero sample call and
  retains the selected capacity for later calls. `tuning_budget_seconds` sets the
  soft deadline, and `result.batch_tuning` retains the calibration report. Trials
  are excluded from returned counts. This tunes capacity with your chosen noise
  and sharing settings; it does not select those strategies for you.
- `initial_states` reserves up to that many coefficient states per worker, within
  the state budget. When a later action needs more states, dispatch suspends and
  grows the pool before resuming with the same sampled noise and branches.
- `max_state_bytes` bounds the combined coefficient and collapse-scratch allocations
  across workers, including page alignment. Its default is 64 MiB; classical
  batches and prepared plans are separate. An explicit capacity whose next action
  could require more states than this budget permits raises `RuntimeError` at a
  suspension boundary.
  Increase the budget or reduce the capacity or worker count before retrying.
  Successfully allocated states stay available for subsequent calls.
- `NoisePreparation.SITE_SPARSE` applies sampled faults directly to expression
  columns. `SITE_ROWS` collects each shot's effects contiguously before
  transposing them, which can help when faults affect many expressions.
  `DENSE` uses the ordinary packed propagation path; `SPARSE` applies direct
  updates in that path's original noise-event order.
- `compact_outputs=True` expands record snapshots and derived symbols into the
  expressions needed for survivor counts. This can reduce noise propagation and
  classical record work, at the cost of additional preparation during construction.
- `group_by_noise=True` groups shots by their effect on future rotation signs
  before evolution. This avoids repeated sign scans but may retain more distinct
  states and split measurement branches earlier. Compare it with the default
  incremental grouping on your workload; it is ignored when `sharing=False`.
- `sharing=False` uses independent coefficient evolution, which can avoid grouping
  overhead when most trajectories differ. `early_postselection=False` skips the
  preparation of entrance constraints and retains the original detector order;
  this can reduce setup costs when few shots will be rejected.

Compare construction time and sustained sampling separately. Small jobs or
circuits with many distinct trajectories may favor ordinary sampling. This
interface uses cross-shot threads. With automatic capacity, changing the worker
budget can change batch size and seeded results. Calibration is timing-dependent,
so pin a numeric capacity to replay a configuration.

A fixed seed, batch size, and noise preparation reproduce counts independently
of worker scheduling and state-pool growth. Different batch sizes or noise
preparations can change seeded results while preserving their distribution.
Calls on one sampler serialize; separate samplers can run concurrently.

## Common controls

| Argument | Default | Meaning |
|---|---|---|
| `threads` | `1` | Total CPU worker budget; `"auto"` uses reported hardware concurrency. |
| `batch_size` | `"auto"` | Packed-lane policy; `1` forces scalar execution and `"tune"` measures candidate capacities. |
| `tuning_budget_seconds` | `None` | Keyword-only soft calibration budget for `batch_size="tune"`; `None` uses 0.25 seconds. |
| `thread_layout` | `None` | Expert `(shot_workers, intra_shot_workers)` override. |
| `intra_shot_min_active_width` | `None` | Expert threshold for enabling an explicit intra-shot layout. |

The four standard sampling functions accept all five controls. `SurvivorSampler`
uses cross-shot workers and accepts the first three controls. For leakage and loss,
`clifft.noncomp.sample()` and `NonComputationalProgram.sample()` accept
cross-shot `threads`, `batch_size` (`"auto"`, `"tune"`, or a positive integer),
and `tuning_budget_seconds`. Compile-once fixed execution supports batching and
batch-size tuning, whereas trajectory execution runs one shot at a time
(`batch_size=1`) and rejects `batch_size > 1` or `"tune"` (see
[Leakage and Loss](leakage-and-loss.md#compiling-and-execution)). Intra-shot
thread layouts are not supported by `clifft.noncomp`, and exact probability
queries and `get_statevector()` do not expose these sampling controls.

Three independent mechanisms are involved:

- **Cross-shot workers** run different shots concurrently. Each owns an
  executor and mutable storage.
- **Intra-shot workers** use an OpenMP team within one wide shot. The team
  shares an executor.
- **Packed execution** represents several shots as SIMD lanes within one
  worker.

Packed execution cannot be combined with intra-shot workers.

## Budgeted batch calibration

The default `batch_size="auto"` uses conservative heuristics and may miss a
faster batch size for your circuit. Set `batch_size="tune"` to briefly measure
batch sizes for your circuit and simulation settings, then run all requested
shots using the fastest measured choice. Calibration adds overhead, so it is
most useful for longer sampling jobs.

```python
import clifft

program = clifft.compile("H 0\nT 0\nH 0\nM 0")
result = clifft.sample(
    program,
    shots=100_000,
    threads=2,
    batch_size="tune",
    tuning_budget_seconds=0.1,
)
print(result.batch_tuning.batch_size)
```

`tuning_budget_seconds` defaults to 0.25 seconds. It is a **soft limit** on
calibration time: in-progress calibration trials can finish after the deadline
before the requested shots run. Keep these points in mind:

- Calibration shots are extra and discarded; all requested shots still run.
  Trial results are excluded from the returned counts and estimates.
- Batch sizes that exceed the worker memory limits are excluded. If none fit,
  Clifft samples without batching and skips calibration.
- If calibration cannot make a useful comparison, Clifft uses `auto` and
  returns the samples with a `RuntimeWarning` suggesting a larger budget.
  A budget overrun alone does not cause a warning.

The selected settings are returned in `result.batch_tuning`. To reuse the
batch size without recalibrating, pass `batch_size=result.batch_tuning.batch_size`
on later calls with the same circuit and simulation settings. Retune when the
workload or machine changes; selections are not cached automatically.

??? note "Calibration report details"

    `result.batch_tuning` is a `BatchTuningReport`; it is `None` for `auto` or
    explicitly sized calls.

    | Field | Meaning |
    |---|---|
    | `batch_size`, `baseline_batch_size` | Selected and automatic batch sizes; `1` means scalar. |
    | `shot_workers`, `intra_shot_workers` | Resolved worker layout. |
    | `elapsed_seconds` | Total calibration time, including setup and cleanup. |
    | `trial_shots` | Extra attempted shots, including warmup. |
    | `sufficient_measurements` | Whether the automatic choice and at least one alternative provided usable timings. False means the automatic choice was retained. |
    | `stop_reason` | Why calibration ended; see below. |
    | `trials` | A `BatchTuningTrial` for each candidate that was prepared. |

    `completed` means every eligible candidate provided usable timing data,
    even if the final trial crossed the soft deadline. `budget_exhausted`
    means time prevented measuring at least one candidate; a partial sweep
    can still provide sufficient measurements. `insufficient_measurements`
    means usable timing data was missing without a candidate being excluded
    for lack of time.

    `zero_shots`, `zero_budget`, and `single_candidate` indicate that no
    calibration was needed or requested. These cases do not issue a warning.

    Each trial records `batch_size`, `shot_workers`, `setup_seconds`,
    `warmup_shots`, `warmup_seconds`, subsequent measured `shots` and
    `elapsed_seconds`, and `shots_per_second`. `used_warmup` indicates that
    throughput comes from the first probe alone; setup time is excluded.

    To reproduce the worker allocation as well as the batch size, pass
    `thread_layout=(report.shot_workers, report.intra_shot_workers)` from a
    nonempty call and preserve any explicit intra-shot threshold.

## Power-user tuning

Use explicit settings only after benchmarking the same circuit, shot count,
sampling function, and output options you plan to run.

### Thread budgets and layouts

Pass a positive `threads` count to set the total worker budget, or
`threads="auto"` to use implementation-reported hardware concurrency. The
automatic scheduler chooses one layout:

- With at least as many shots as workers, use cross-shot workers.
- With fewer shots and a selected executable's peak active width of at least 18,
  an OpenMP-enabled build can spend the budget within shots.
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
- peak active width and post-selection satisfy the policy below; and
- estimated packed work and memory stay within automatic budgets.

| Program | Automatic packed capacity limit |
| --- | --- |
| No postselection, width 0-4 | 2048 lanes |
| Postselected, width 0-4 | 256 lanes |
| No early postselection, width 5 | Up to 1024 lanes when coefficient work is limited or amortized by classical work |
| No early postselection, width 6-8 | Up to 512, 256, or 128 lanes respectively when classical work amortizes the quantum kernels |
| Early postselection at width 5 or above, or width above 8 | Scalar |

An early postselection check can reject a shot before further coefficient work.
Checks after the last quantum action, including reconstructed detector checks,
do not impose that restriction. Expectation-value probes count as coefficient
work only when retaining rows. Width and work refer to the selected executable
plan; use `program.sampling_plan(keep_records=...)` to inspect its width. Narrow
postselected plans retain the 256-lane cap because rejection can also skip
classical work, even at width zero.

For widths 6-8, capacities are also bounded by attempted shots per worker
when each requested worker has at least 64 shots. Shorter requests retain their
packed capacity and use fewer workers. Memory budgets can further reduce capacity
and concurrency. When early detectors reject most shots, packed execution can be
slower than scalar execution. Fixed noncomputational sampling also accounts for
its classical input-generation work when considering wider packed states.

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

With a fixed seed and the same program, shot count, and execution settings,
sampling repeats its results within the same build. Automatic batching considers
the thread budget, so changing `threads` or `thread_layout` can change batch
boundaries and sampled rows. With `threads="auto"`, the available worker count
can differ between machines, so the same seed need not produce the same rows.
To preserve rows and survivor order while changing the number of cross-shot
workers, pin a numeric `batch_size`.

Scalar and packed execution use separate random streams, and different packed
capacities can produce different rows. Every supported strategy remains
statistically equivalent. Keep the complete execution configuration fixed when
exact seeded replay is required.

Timing-based calibration can select a different capacity on repeated calls,
even with the same seed. For debugging, pin the reported numeric batch size
instead of requesting calibration again. Calibration uses separate random
streams from the requested shots. Omitting `seed` continues to use hardware
entropy.

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
need Homebrew `libomp` when building from source. If OpenMP runtimes from
different scientific packages conflict, use separate processes.

On POSIX systems, create process workers before threaded Clifft sampling, or
use the `spawn` or `forkserver` start method. Forking after a threaded sample
and then requesting intra-shot threads in the child can hang in some runtimes.
See [Installation](../getting-started/installation.md#from-source) for build
guidance.

### Benchmark before overriding

Performance depends on the circuit, active width over time, shot count,
post-selection lifetime, outputs, CPU, and memory limits. Compare the defaults
with a small set of representative alternatives such as `batch_size=1`, `256`,
and `1024`, using your target thread count and output options.

Use `program.peak_active_width` as a first-order cost indicator for full-output
sampling. For survivor counts, use
`program.sampling_plan(keep_records=False).peak_active_width`; for fixed-fault
sampling, use `program.original_plan.peak_active_width`. See
[Inspect sampling cost](compilation.md#inspect-sampling-cost) for examples.
Peak width alone does not determine the best layout.

## Compile-time scheduling

For repeated sampling, the opt-in active-width scheduler can reduce the active
state at extra compilation cost. See
[Active-width scheduling](compilation.md#active-width-scheduling) for setup,
playground controls, and how to compare compilation plus sampling time.
