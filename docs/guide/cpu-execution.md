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
| `batch_size` | `"auto"` | Packed-lane policy; `1` forces scalar execution and `"tune"` measures candidate capacities. |
| `tuning_budget_seconds` | `None` | Keyword-only soft calibration budget for `batch_size="tune"`; `None` uses 0.25 seconds. |
| `thread_layout` | `None` | Expert `(shot_workers, intra_shot_workers)` override. |
| `intra_shot_min_active_width` | `None` | Expert threshold for enabling an explicit intra-shot layout. |

The fixed-plan samplers above accept these controls. The leakage and loss
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
calibration time: work already in progress can finish after the deadline. It
does not limit the requested sampling run. Keep these points in mind:

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
    | `shot_workers`, `intra_shot_workers` | Resolved production worker layout. |
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

Timing-based calibration can select a different capacity on repeated calls,
even with the same seed. For debugging, pin the reported numeric batch size
instead of requesting calibration again. Calibration uses separate random
streams from production. Omitting `seed` continues to use hardware entropy.

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
and `1024`, using the production worker budget and result options.

Use `program.peak_active_width` as a first-order cost indicator, but do not
choose a layout from peak width alone.

## Compile-time scheduling

For repeated sampling, the opt-in active-width scheduler can reduce the active
state at extra compilation cost. See
[Active-width scheduling](compilation.md#active-width-scheduling) for setup,
playground controls, and how to compare compilation plus sampling time.
