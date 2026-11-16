# CPU Execution and Tuning

CPU settings control how sampling uses SIMD lanes, threads, and memory without
changing the sampled distribution.

## Automatic selection (default)

First choose the function that returns the result you need. The four standard
samplers share the same CPU controls:

- `clifft.sample()` and its decoder helper `clifft.sample_detectors()`
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
wide shot. `sample()` and `sample_survivors()` can also share coefficient
evolution between shots that follow the same quantum trajectory. This is useful
for sufficiently large postselected workloads and noiseless circuits with a
shared unitary prefix. Short calls and uncertain cases use ordinary scalar or
packed execution. Requested expectation-value rows use ordinary execution so
every probe is retained.

Calls can run concurrently on the same `Program` and reuse its preparation.
For a fixed seed and execution configuration, results are independent of earlier
calls. [Benchmark before overriding](#benchmark-before-overriding) the defaults.

## Common controls

| Argument | Default | Meaning |
|---|---|---|
| `threads` | `1` | Total CPU worker budget; `"auto"` uses reported hardware concurrency. |
| `batch_size` | `"auto"` | Select execution automatically; `1` forces scalar execution, another integer requests ordinary packed execution, and `"tune"` measures capacities within the selected path. |
| `tuning_budget_seconds` | `None` | Keyword-only soft calibration budget for `batch_size="tune"`; `None` uses 0.25 seconds. |
| `thread_layout` | `None` | Expert `(shot_workers, intra_shot_workers)` override. |
| `intra_shot_min_active_width` | `None` | Expert threshold for enabling an explicit intra-shot layout. |

The four standard sampling functions accept all five controls, and `sample()`,
`sample_detectors()`, and `sample_survivors()` also accept the keyword-only
[shared trajectory controls](#shared-trajectory-execution) (`sharing`,
`group_by_noise`, `noise_preparation`, and `max_worker_state_bytes`). For
leakage and loss,
`clifft.noncomp.sample()` and `NonComputationalProgram.sample()` accept
cross-shot `threads`, `batch_size` (`"auto"`, `"tune"`, or a positive integer),
and `tuning_budget_seconds`. Compile-once fixed execution supports batching and
batch-size tuning, whereas trajectory execution runs one shot at a time
(`batch_size=1`) and rejects `batch_size > 1` or `"tune"` (see
[Leakage and Loss](leakage-and-loss.md#compiling-and-execution)). Intra-shot
thread layouts are not supported by `clifft.noncomp`, and exact probability
queries and `get_statevector()` do not expose these sampling controls.

The execution mechanisms are:

- **Cross-shot workers** run different shots concurrently. Each owns an
  executor and mutable storage.
- **Intra-shot workers** use an OpenMP team within one wide shot. The team
  shares an executor.
- **Packed execution** represents several shots as SIMD lanes within one
  worker.
- **Shared execution** keeps per-shot classical data while reusing coefficient
  states across equivalent trajectories. The default state budget is 64 MiB per
  worker; classical storage and output rows are additional. Automatic and tuned
  capacities fit even when every shot needs its own state, and are capped at
  8192 lanes.

Packed and shared execution cannot be combined with intra-shot workers.

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
- If timing data cannot support a useful comparison, Clifft uses `auto` and
  returns the samples with a `RuntimeWarning` suggesting a larger time budget.
  A budget overrun alone does not cause a warning.
- Shared calibration limits each probe's output size to 8 MiB. If a full batch
  per worker exceeds that limit at the automatic capacity, Clifft keeps `auto`
  and warns that calibration could not fit. Fewer workers or fewer retained
  record types can make calibration fit; a longer time budget does not help.

The selected capacity and timings are returned in `result.batch_tuning`, and the
resolved settings appear on `result.execution`. Calibration compares capacities
within the selected execution path. To reproduce a calibrated run, pass
`batch_size=result.execution.batch_size` together with
`sharing=result.execution.sharing` and any explicit noise or grouping
overrides; a numeric `batch_size` with `sharing="auto"` selects ordinary
execution and does not reproduce a shared call. Retune when the workload or
machine changes; selections are not cached automatically.

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

    `output_budget` means a shared baseline probe exceeded the output-size
    limit. No trials ran, and a `RuntimeWarning` explains the fallback.

    `zero_shots`, `zero_budget`, and `single_candidate` indicate that no
    calibration was needed or requested. These cases do not issue a warning.

    Each trial records `batch_size`, `shot_workers`, `setup_seconds`,
    `warmup_shots`, `warmup_seconds`, subsequent measured `shots` and
    `elapsed_seconds`, and `shots_per_second`. `used_warmup` indicates that
    throughput comes from the first probe alone; setup time is excluded.

    To reproduce the worker allocation as well as the batch size, pass
    `thread_layout=(report.shot_workers, report.intra_shot_workers)` from a
    nonempty call along with `sharing=result.execution.sharing` and preserve
    any explicit intra-shot threshold.

## Shared trajectory execution

Many shots in a noisy or postselected circuit follow the same quantum
trajectory, and noiseless circuits with measurements share their unitary prefix
before branching. Shared execution reuses quantum state evolution across
equivalent shots while keeping per-shot classical records, and discards
postselected shots that fail noise-only entrance checks or completed detector
checks before later evolution.

`sample()`, `sample_detectors()`, and `sample_survivors()` select shared
execution automatically on eligible postselected workloads and noiseless
unitary-prefix circuits when `sharing="auto"` and `batch_size="auto"`. On first
use, `Program` caches the requested output plan, noise metadata, entrance
checks, and any grouping key; later or concurrent calls on the same `Program`
reuse that immutable preparation and allocate their own workers per call. Pass
keyword overrides when benchmarking or pinning shared settings:

```python
import clifft

program = clifft.compile(
    "H 0 1\nT 0 1\nDEPOLARIZE1(0.001) 0 1\nH 0 1\nM 0 1\n"
    "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]",
    postselection_mask=[1],
    reconstruction=True,
)
result = clifft.sample_survivors(
    program,
    shots=100_000,
    seed=42,
    threads=2,
    batch_size=8192,
    sharing=True,
    group_by_noise=True,
    noise_preparation=clifft.NoisePreparation.SITE_ROWS,
)
print(result.passed_shots, result.execution.executor, result.execution.batch_size)
```

Use `sample_k()` or `sample_k_survivors()` for fixed-fault conditioning, and
`clifft.noncomp` for leakage and loss instruments. Shared execution is
incompatible with retained `EXP_VAL` probes (`keep_records=True` on a circuit
containing `EXP_VAL`), `batch_size=1`, and intra-shot thread layouts. Circuits
whose selected plan has zero active width have no quantum state to share; with
`sharing=True`, they run on the packed classical executor under the shared
batching and noise policy (`result.execution.executor == "packed"` and
`result.execution.sharing is True`).

### Batch size and state memory

Because equivalent shots share quantum states, shared execution can run larger
classical batches than ordinary packed execution:

- **`batch_size`**: With `sharing="auto"` or `False`, a numeric `batch_size`
  keeps ordinary execution (clamped to 2048 lanes, or scalar at `1`). Pass
  `sharing=True` with a numeric `batch_size` to request an explicit shared
  batch capacity. With `batch_size="auto"`, shared execution chooses a capacity
  up to 8192 for the call, sized so every lane could split into its own quantum
  state within `max_worker_state_bytes`. Short calls get smaller batches so each
  requested worker receives one, down to 1024 lanes per worker; below that,
  fewer workers run fuller batches. `batch_size="tune"` measures candidate
  capacities up to that safe limit for the call. An explicit shared capacity can
  exceed the safe limit when you know trajectories share states.
- **`max_worker_state_bytes`** (default `64 MiB`): Memory limit per worker for
  quantum coefficient storage, measurement-collapse scratch, and page alignment
  (classical buffers, output rows, and compiled plans are separate). The limit
  applies per worker rather than across all threads, so state storage can reach
  `max_worker_state_bytes * shot_workers` when every lane diverges; states are
  allocated at explicit suspension boundaries as trajectories split. With
  `sharing=True` on nonzero-width plans, the budget must hold at least two
  states. Automatic and tuned capacities fit one state per lane within this
  budget. With a larger explicit capacity, Clifft checks a conservative bound
  on the states an action's possible splits could require; if that bound
  exceeds the budget, the call raises `RuntimeError` and leaves the `Program`
  reusable, even if the sampled outcomes would have needed fewer states. Reduce
  `batch_size` or raise `max_worker_state_bytes` before retrying.

The default budget holds 8192 states through active width 8, 2730 at width 10,
and 341 at width 13. For wider circuits, raise `max_worker_state_bytes` if the
additional per-worker memory is acceptable so `"auto"` and `"tune"` can consider
larger batches.

### Sharing and noise overrides

| Option | Default | Meaning and when to change |
|---|---|---|
| `sharing` | `"auto"` | `"auto"` lets Clifft choose; `True` requires shared execution (or packed classical batches at width zero); `False` keeps ordinary scalar or packed execution, such as when sparse postselection rejects few shots and most trajectories diverge. |
| `group_by_noise` | `"auto"` | `"auto"` resolves to `False`. Set `True` (with `sharing=True`) to partition shots upfront by their effect on future rotation signs, avoiding per-rotation sign scans at the cost of earlier measurement splits. |
| `noise_preparation` | `"auto"` | `"auto"` uses `SITE_SPARSE` on the shared path and `DENSE` on the ordinary packed path, except on [width-zero plans](#noise-preparation-on-width-zero-plans). `SPARSE` shares `DENSE`'s random stream while skipping non-firing faults when few faults occur per batch; `SITE_ROWS` accumulates per-shot fault effects before transposing when individual faults touch many expressions. |
| `max_worker_state_bytes` | `67108864` | Per-worker quantum state pool limit (64 MiB) when `result.execution.executor == "shared"`. |

Inspection of the resolved choices is available on `result.execution`
(`SamplingExecution`), including `executor` (`"scalar"`, `"packed"`, or
`"shared"`), `sharing`, `batch_size`, `shot_workers`, `intra_shot_workers`,
resolved `keep_records` and `bit_packed`, `noise_preparation`, `group_by_noise`,
and `max_worker_state_bytes`.

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

When shared execution is not selected, `batch_size="auto"` considers packed execution only when:

- at least 64 shots were requested;
- peak active width and post-selection satisfy the policy below; and
- estimated packed work and memory stay within automatic budgets.

| Program | Automatic packed capacity limit |
| --- | --- |
| Width 0, or no postselection at width 1-4 | 2048 lanes |
| Postselected, width 1-4 | 256 lanes |
| No early postselection, width 5 | Up to 1024 lanes when coefficient work is limited or amortized by classical work |
| No early postselection, width 6-8 | Up to 512, 256, or 128 lanes respectively when classical work amortizes the quantum kernels |
| Early postselection at width 5 or above, or width above 8 | Scalar |

An early postselection check can reject a shot before further coefficient work.
Checks after the last quantum action, including reconstructed detector checks,
do not impose that restriction. Expectation-value probes count as coefficient
work only when retaining rows. Width and work refer to the selected executable
plan; use `program.sampling_plan(keep_records=...)` to inspect its width.
Postselected plans of width 1-4 keep the 256-lane cap so rejected lanes carry
less work. Width-zero plans have no coefficient work, and their classical cost
per shot falls as batches grow; fixed-fault sampling keeps the 256-lane cap
there.

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

#### Noise preparation on width-zero plans

A width-zero plan, such as a Clifford circuit with Pauli noise, spends all of
its sampling time on classical work, so how faults are drawn sets its
throughput. `noise_preparation="auto"` chooses from the expected number of
faults in one batch:

- **`SPARSE`** while a batch expects no more faults than the plan has noise
  sites. It draws once per fault, which suits low noise rates and short calls.
- **`SITE_SPARSE`** above that. It draws once per site in each batch and
  updates the affected expressions site by site.

The batch here is the smaller of the shot count and the capacity limit, so the
same program can use `SPARSE` for a short call and `SITE_SPARSE` for a long
one. Those two use different random streams. With a numeric `batch_size`,
`"auto"` uses `DENSE` in place of `SITE_SPARSE`; `SPARSE` and `DENSE` return
the same rows, so a numeric `batch_size` keeps seeded rows fixed. Circuits in
which one fault flips many later outcomes, such as long stretches without
resets, can run faster with an explicit `SITE_ROWS`.

### Reproducibility

With a fixed seed and the same program, shot count, and execution settings,
sampling repeats its results within the same build. Automatic batching considers
the thread budget, so changing `threads` or `thread_layout` can change batch
boundaries and sampled rows. With `threads="auto"`, the available worker count
can differ between machines, so the same seed need not produce the same rows.
With a numeric `batch_size`, changing only the number of cross-shot workers
preserves rows and survivor order within ordinary execution.

Untuned selection depends only on the program and call arguments, including the
requested outputs and worker layout. Cold, warm, and concurrent first use make
the same choices. For `sample()` and `sample_survivors()`, automatic calls can
select shared execution; pass `sharing=False` or an ordinary numeric
`batch_size` (with `sharing="auto"`) to select ordinary execution, or pass
`sharing=True` to pin shared execution.

Scalar, packed, and shared execution use separate random streams, and different
capacities or noise preparations can produce different rows (`SITE_SPARSE` and
`SITE_ROWS` share a random stream, and `DENSE` and `SPARSE` share a separate
random stream). Changing `keep_records` can also change individual seeded rows
while preserving the requested output distribution. `bit_packed` only changes
array storage and preserves the sampled bits. Keep the complete execution
configuration fixed when exact seeded replay is required.

Timing-based calibration can select a different capacity on repeated calls,
even with the same seed. To replay a calibrated run for debugging, pin the
settings recorded on `result.execution` (`batch_size=result.execution.batch_size`,
`sharing=result.execution.sharing`, and `noise_preparation` when one is
reported) instead of requesting calibration again; passing a numeric
`batch_size` alone without `sharing=True` selects ordinary execution with the
`DENSE` random stream. Calibration uses separate random streams from the
requested shots. Omitting `seed` uses hardware entropy.

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
