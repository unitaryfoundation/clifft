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

Set `batch_size="tune"` to calibrate and then run all requested shots in the
same call. No separate preparation call is needed:

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

Calibration sweeps the automatic choice, scalar execution, and capacities
64, 256, 1024, and 2048, removing duplicate and ineligible configurations.
It can consider packed execution above the automatic width cutoff and for
postselected programs. Packed candidates must fit the existing automatic
worker-storage budgets of 8 MiB per worker and 64 MiB across workers;
calibration output buffers are separately limited to 8 MiB. Candidates that
cannot exercise their production worker count within that output limit are
omitted. Threads and compiler passes are not tuned.

Each candidate receives roughly an equal share of the remaining budget.
Worker preparation and the first probe (warmup) count toward the budget.
Probes measure attempted-shot throughput, including output collection. If no
subsequent measurement fits, the first probe's timing is used; otherwise,
selection uses the subsequent measurements to reduce first-touch effects. Calibration
uses the requested sampling function, record-retention setting, fixed-fault
stratum when applicable, and resolved thread layout. Workers are prepared once
per candidate and released before the next candidate. The fastest measured
candidate is used for production. Unless the baseline and at least one
alternative are both measured, the automatic choice is retained. Inconclusive
calibration emits a Python `RuntimeWarning` when returning the production result,
suggesting a larger `tuning_budget_seconds`. The requested production shots still run.

The budget is a **soft wall-clock limit** for extra calibration work. An
allocation or an execution chunk already in progress can exceed it. Once a
packed candidate has been measured, its throughput helps skip larger probes
estimated not to fit the remaining time. Scalar timings do not prevent trying
the first packed candidate. These estimates are heuristics: the first probe or
an unexpectedly slow probe can still overrun the budget. It does not limit
production sampling. Zero budget uses the automatic policy without
trials. Empty requests and layouts with no eligible alternative also skip
calibration, without a warning. In particular, if every packed candidate exceeds
the memory limits, increasing the time budget will not help. Budgets must be finite
and nonnegative, and an explicit budget requires `batch_size="tune"`.

Calibration shots are discarded. They do not enter returned rows,
`total_shots`, survivor counts, or logical-error estimates. Production still
runs exactly the requested number of attempted shots. Tuning adds overhead,
so use it for jobs long enough to benefit, and compare total elapsed time
including calibration. Measured choices can vary with machine load and are
not guaranteed to improve performance.

`result.batch_tuning` is a `BatchTuningReport`; it is `None` for ordinary
automatic or explicitly sized calls. Its fields are:

| Field | Meaning |
|---|---|
| `batch_size`, `baseline_batch_size` | Selected and automatic lane capacities; `1` means scalar. |
| `shot_workers`, `intra_shot_workers` | Resolved production worker layout. |
| `elapsed_seconds` | Total calibration time, including setup, warmup, and cleanup. |
| `trial_shots` | Extra attempted shots, including warmup. |
| `sufficient_measurements` | Whether both the baseline and at least one alternative provided usable timings. False means the automatic choice was retained. |
| `stop_reason` | `completed`, `budget_exhausted`, `insufficient_measurements`, `zero_shots`, `zero_budget`, or `single_candidate`. |
| `trials` | Candidate measurements in sweep order. |

Each `BatchTuningTrial` records `batch_size`, `shot_workers`, `warmup_shots`,
subsequent measured `shots`, `setup_seconds` (worker preparation only),
`warmup_seconds`, subsequent measured `elapsed_seconds`, and `shots_per_second`.
`used_warmup` indicates that throughput comes from the first probe alone. A
trial without any usable probe timing has zero throughput.

`single_candidate` means there was no eligible alternative, so no timing work
was needed. `completed` means every eligible candidate provided usable timing
data. The last candidate using up its share, or an in-progress final probe
crossing the soft deadline, still counts as completion; `elapsed_seconds`
records the actual duration.

`budget_exhausted` means at least one candidate was skipped or could not run a
probe within the available time. It can occur with or without
`sufficient_measurements`: a partial sweep can still compare the baseline with
an alternative. `insufficient_measurements` means some usable timing data was
missing without a candidate being excluded for lack of time.

To reuse a selection, pass `batch_size=result.batch_tuning.batch_size` on
subsequent calls. For the same worker allocation, also pass
`thread_layout=(report.shot_workers, report.intra_shot_workers)` using the
report from a nonempty call and preserve any explicit intra-shot threshold.
Recommendations are specific to the program, sampling mode, output options,
request size, and machine; they are not cached automatically.

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
