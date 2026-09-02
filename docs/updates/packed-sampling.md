# Packed Sampling in Clifft (v0.10.0, September 2026)

Version 0.8.0 rebuilt Clifft around symbolic-coordinate sampling. Version
0.9.0 then restored multicore execution, letting one worker budget run many
shots concurrently or divide a wide active state across OpenMP workers.

Version 0.10.0 adds a third form of CPU parallelism: packed batch sampling.
Instead of advancing each trajectory through the prepared plan independently,
Clifft can advance a group of low-active-width trajectories together. The goal
is higher throughput when each individual shot is too small to keep modern CPU
execution resources busy on its own.

This is still exact simulation. Packing changes how Clifft organizes work and
random streams, not the circuit distribution being sampled.

## From cores to lanes

The parallelism added in v0.9.0 works at the worker level. Cross-shot workers
own separate executors and claim independent ranges of shots, while intra-shot
workers cooperate on the coefficient array for one wide trajectory.

That model works well when there are enough substantial shots to distribute or
when one active state is large enough to divide across cores. It leaves another
important regime: many shots whose active states stay small.

At active width $k$, Clifft's dense active state contains $2^k$ complex
coefficients. When $k$ is small, the arithmetic for one action can be cheaper
than the surrounding control flow, symbolic expression evaluation, record
updates, and repeated passes over short arrays. Adding more independent worker
objects does not remove that per-shot work.

Packed sampling instead treats several shots as lanes of one executor. Each
prepared action is visited once for the group, and its lane-local effects are
applied across the active trajectories together.

## One plan, many trajectories

Clifft prepares the packed path before entering hot execution:

- stochastic symbols are sampled for the batch and Boolean values are stored
  as packed bit columns;
- record, detector, observable, and symbolic sidecars use the same lane
  organization;
- active-state coefficients are interleaved by basis index so the values for
  neighboring lanes are contiguous;
- rotations, promotions, measurements, conditional variants, and expectation
  values operate directly on that representation; and
- rejected post-selection lanes can be compacted while retaining survivor
  order and the associated sidecars.

The executor therefore reuses one prepared action stream without merging the
trajectories mathematically. Every lane still carries its own random choices,
record history, symbolic state, and active-state coefficients.

The approach was inspired by the packed cross-shot path in
[SymFT](https://arxiv.org/abs/2607.28600), whose low-active-width results helped
motivate bringing a similar execution mode to Clifft. Clifft adapts that idea
to its own prepared sampling plans and supports ordinary rows, retained
post-selected survivors, and fixed-fault workflows through the same public
sampling APIs. Comparisons with SymFT belong on the dedicated Performance page;
this article focuses on what changed between Clifft v0.9.0 and v0.10.0.

## Automatic when the shape is right

Packed execution is not universally faster. It adds transposition, lane
bookkeeping, and memory costs that must be recovered by sharing enough useful
work. The best choice depends on shot count, active width over time, output
mode, post-selection lifetime, worker count, and the machine running the job.

The default `batch_size="auto"` policy is deliberately conservative. It
considers packing for ordinary fixed-plan sampling when at least 64 shots were
requested, the program has no post-selection, peak active width is at most 5,
and estimated work and retained storage fit automatic budgets. A long width-5
program can still stay scalar when the estimate says packing is unlikely to
pay off.

Automatic survivor sampling stays scalar because static planning cannot
predict how long rejected lanes will remain alive. Advanced callers can still
request an explicit packed capacity for measured workloads:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")

automatic = clifft.sample(program, shots=100_000, seed=42)
scalar = clifft.sample(program, shots=100_000, seed=42, batch_size=1)
packed = clifft.sample(program, shots=100_000, seed=42, batch_size=1024)
```

A positive integer requests capacity for up to that many lanes, subject to the
shot count and safety limits. Explicit capacities are power-user controls, not
recommendations that larger batches are always better. The
[CPU execution guide](../guide/cpu-execution.md) documents the current policy,
supported APIs, memory model, and tuning guidance.

Packed execution and intra-shot OpenMP both target the active-state kernels in
different ways, so they cannot be combined. Use cross-shot workers with packed
lanes when both controls must be fixed explicitly.

## Reproducibility across execution modes

Changing only the cross-shot worker count preserves seeded rows and survivor
order. Workers claim deterministic shot ranges from the same scalar or packed
random stream.

Scalar and packed execution intentionally use separate random-stream domains,
and different explicit packed capacities can group draws differently. A fixed
seed therefore reproduces results when the complete execution configuration is
unchanged, but it does not promise identical rows after switching between
scalar and packed modes.

All supported modes sample the same circuit distribution. This distinction
keeps exact replay well defined without coupling future execution strategies to
one historical random-number schedule.

## Reducing the work before packing it

The largest performance wins often come from shrinking the active state before
choosing how to execute it. Clifft's statevector-squeeze pass moves expansions
later when intervening operations permit it, allowing measurements to remove
active coordinates sooner.

Version 0.10.0 extends that motion across expansion convoys: sequences of
expanding operations that previously blocked a safe non-expanding operation
waiting beyond them. The new lookahead preserves stable ordering at blocked
crossings while avoiding repeated scans.

On the coherent QEC integration workloads used during development, this lowers
peak active width from 8 to 5 for the distance-3, three-round circuit and from
24 to 13 for the distance-5, five-round circuit. Since coefficient storage and
kernel work scale with $2^k$, reducing $k$ is complementary to both packed and
threaded execution.

Apple Silicon also gains native NEON kernels for direct and fused rotations and
diagonal active measurements. Clifft selects them automatically when the
operation is large enough to benefit and retains scalar fallbacks below the
measured profitability thresholds.

## Measured against v0.9.0

!!! note "Benchmark figure pending"
    The final article will summarize matched `clifft-bench` measurements of
    PyPI v0.9.0 and the exact v0.10.0 release candidate on the same host and
    workload corpus. This section will contain the headline findings and one
    figure; the dedicated Performance page will own the broader results,
    methodology, raw-data links, and SymFT comparisons.

The comparison will separate improvements from packed execution, compiler
active-width reductions, and architecture-specific kernels where the workload
and hardware make those distinctions meaningful. Workloads that remain scalar
under the automatic policy are part of the result too: avoiding an unprofitable
mode is one of the policy's jobs.

## A more self-contained core

The packed executor is the main user-visible performance change, but v0.10.0
also reduces Clifft's production dependency surface.

Compilation, optimization, planning, exact state queries, Python, and
WebAssembly now use Clifft's native runtime-width Pauli strings and tableaus.
Stim remains valuable as an independent Python test oracle, but production
targets no longer fetch or link it. This separation makes the correctness
boundary clearer and reduces production build time and build-tree size without
claiming a runtime speedup from the dependency change itself.

Clifft also accepts a documented unitary subset of OpenQASM 2 without requiring
Qiskit. The native parser supports standard `qelib1.inc` unitary operations,
register broadcasting, barriers, and finite constant-angle expressions. It
rejects measurements, resets, classical control, and custom gate declarations
rather than assigning them approximate semantics. See
[Circuit Inputs](../guide/circuit-inputs.md) for the supported paths and their
limits.

Together, these changes continue the arc begun in v0.8.0: plan more work ahead
of execution, keep the hot path specialized to the workload shape, and retain
explicit controls when automatic policy cannot know enough about a production
environment.
