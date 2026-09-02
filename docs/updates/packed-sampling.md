# Packed Sampling in Clifft (v0.10.0, September 2026)

Version 0.8.0 rebuilt Clifft around symbolic-coordinate sampling. Version
0.9.0 restored multicore execution, letting one worker budget run many shots
concurrently or divide a wide active state across OpenMP workers.

Version 0.10.0 adds another level of CPU parallelism: packed batch sampling.
Clifft can now advance a group of low-active-width trajectories together,
improving throughput when each individual shot is too small to keep modern CPU
execution resources busy on its own.

This remains exact simulation. Packing changes how Clifft organizes the work,
not the circuit distribution being sampled.

## Why pack shots?

At active width $k$, Clifft's dense active state contains $2^k$ complex
coefficients. Wide states provide enough arithmetic to divide within a shot,
while large shot counts can be spread across independent workers. But many
near-Clifford workloads combine a large number of shots with small active
states.

In that regime, advancing shots one at a time repeatedly evaluates the same
prepared action stream and operates on arrays too short to use the CPU
efficiently. Packed sampling instead treats several shots as lanes of one
executor. Boolean state is stored in packed columns, active-state coefficients
are interleaved across lanes, and each prepared action is applied to the group.
The trajectories remain mathematically independent, but the executor shares
more of the work around them.

The approach was inspired by the packed cross-shot path in
[SymFT](https://arxiv.org/abs/2607.28600), whose results on low-active-width
circuits helped motivate a similar mode in Clifft. Clifft adapts the idea to
its prepared sampling plans and result APIs. The dedicated Performance page
will present the broader comparisons with SymFT; here the focus is the change
from Clifft v0.9.0 to v0.10.0.

## Automatic when useful

Packed execution has its own bookkeeping and memory costs, so it is not faster
for every circuit. The default `batch_size="auto"` policy considers the shot
count, active width, estimated lane work, output mode, and retained storage. It
selects packing conservatively for workloads expected to benefit and keeps the
scalar path otherwise.

The same interface is available on the four fixed-plan samplers:

- `sample()`
- `sample_survivors()`
- `sample_k()`
- `sample_k_survivors()`

Most users can leave the automatic policy alone. Advanced callers can set
`batch_size=1` to require scalar execution or request an explicit packed
capacity after measuring their own workload. Automatic survivor sampling stays
scalar because the planner cannot predict how long post-selected lanes will
remain alive.

Packed execution cannot be combined with intra-shot OpenMP, since both modes
organize the active-state kernels in different ways. The
[CPU execution guide](../guide/cpu-execution.md) documents the selection
policy, supported combinations, and memory tradeoffs.

## Less work before faster execution

The runtime improvements are paired with compiler work that can shrink the
active state before execution begins. Clifft's statevector-squeeze pass delays
expansions when intervening operations permit it, giving measurements an
earlier opportunity to remove active coordinates.

Version 0.10.0 extends that motion across sequences of expanding operations
that previously blocked a safe non-expanding operation. On the coherent QEC
workloads used during development, this reduces peak active width from 8 to 5
for the distance-3, three-round circuit and from 24 to 13 for the distance-5,
five-round circuit. Since storage and kernel work scale with $2^k$, reducing
$k$ complements both packed and threaded execution.

Apple Silicon also gains native NEON kernels for direct and fused rotations and
diagonal active measurements. Clifft selects them automatically when an
operation is large enough to benefit and retains scalar fallbacks for smaller
work.

## Measured against v0.9.0

!!! note "Benchmark figure pending"
    The final article will summarize matched `clifft-bench` measurements of
    PyPI v0.9.0 and the exact v0.10.0 release candidate on the same host and
    workload corpus. It will include the headline findings and one figure. The
    dedicated Performance page will contain the wider results, methodology,
    raw-data links, and SymFT comparisons.

The comparison will cover both workloads accelerated by packing and workloads
that benefit from the compiler or architecture-specific changes instead.
Remaining on the scalar path when packing would lose is also part of the
automatic policy's job.

## A more self-contained core

Version 0.10.0 also reduces Clifft's production dependency surface.
Compilation, optimization, planning, exact state queries, Python, and
WebAssembly now use Clifft's native Pauli strings and tableaus. Stim remains an
independent test oracle, but production targets no longer fetch or link it.
This reduces production build time and build-tree size without claiming a
runtime speedup from the dependency change itself.

Clifft also accepts a documented unitary subset of OpenQASM 2 without requiring
Qiskit. The native parser supports standard `qelib1.inc` unitary operations,
register broadcasting, barriers, and finite constant-angle expressions. It
rejects unsupported dynamic constructs rather than assigning them approximate
semantics. See [Circuit Inputs](../guide/circuit-inputs.md) for the supported
paths and their limits.

Together, these changes keep more work in the prepared plan, choose an
execution strategy suited to the workload, and make the core package more
self-contained.
