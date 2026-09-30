# Less Work per Shot in Clifft (v0.11.0 release candidate)

A near-Clifford experiment often compiles one circuit and samples it many
times. Some shots survive to the end; others are rejected as soon as a detector
fires. Each shot follows the same prepared plan, but the amount of useful work
can vary considerably.

The previous two releases expanded how Clifft shares that work across CPU
resources: v0.9 added parallel sampling, and v0.10 added packed execution for
groups of shots. Version 0.11.0 continues by looking at the work each shot
actually needs. Can the compiler arrange for a smaller active state? Can a
rejected shot stop before drawing noise that will never matter? Can an
experiment collect its statistics without materializing individual results?

These questions connect the new active-width scheduler, changes to
postselected sampling, and Clifft's first Sinter integration. They also help
frame our ongoing experimental GPU work: the shape of a shot matters when
deciding how to execute it.

## Start with a smaller active state

Clifft's dense state contains $2^k$ coefficients, where $k$ is the active width.
Non-Clifford operations can expand that state, while measurements can shrink
it. The order in which independent operations run therefore affects how much
state is live at once, even when the sampled distribution stays the same.

The default compiler already uses this freedom through its statevector-squeeze
pass. The new `ActiveWidthSchedulePass` adds an opt-in search over legal
operation orders. It first looks for a lower peak active width, then for less
estimated dense work at the same peak. It can also move operations across
supported Pauli noise with compiler-prepared sign corrections that preserve
the noise semantics.

This search costs compilation time. Repeated sampling gives that investment a
chance to pay back, but a lower width or work estimate does not guarantee a
shorter run. The pass remains disabled by default; users can enable it after
the default passes in Python or in the Playground and compare compilation plus
sampling time for their own workload. Reordering preserves distributions but
can change the samples produced by a particular seed. See
[Compiling Circuits](../guide/compilation.md#active-width-scheduling) for setup
and the pass's reported statistics.

## Let rejected shots stop working

In postselected experiments, a shot may become unusable long before the end of
the circuit. Preparing every noise event at the start of that shot spends work
on a future it will never reach.

Scalar sampling now defers noise draws until execution reaches their first
use. The planner determines those deadlines ahead of time, so the executor
does not discover dependencies during a shot. If a detector rejects the shot,
the remaining noise work can be skipped.

There is a similar opportunity when producing the final result. Packed
survivor sampling now skips internal columns that are no longer needed when
it compacts the surviving rows. When an application needs only aggregate
counts, it can avoid materializing those rows altogether.

## Carry that through to the experiment

The new optional `clifft.sinter.PerfectionistSampler` connects counts-only
survivor sampling to Sinter's experiment collection tools. It supports
Clifford error-detection experiments that accept a shot only when every
detector is quiet. Any logical observable flip on an accepted shot counts as
an error.

The adapter compiles a Clifft program in each Sinter worker and returns shot,
discard, and error counts. Packed sampling is its default, and Stim and Sinter
remain optional dependencies installed through `clifft[sinter]`. This gives
existing Sinter experiments a way to try Clifft while keeping their collection
and analysis workflow.

This first integration is deliberately limited to all-detector postselection
on supported Clifford circuits. Broader decoding and non-Clifford integration
remain areas for exploration. The [Sinter guide](../guide/sinter.md) provides a
complete example and describes the supported task semantics.

## Explore another place to run the plan

The same compiler and symbolic planner also feed Clifft's experimental GPU
backends. Version 0.11.0 introduces NVIDIA CUDA support and extends AMD HIP with
cooperative execution for wider active states. Small states can use one GPU
thread per shot; wider states can share their coefficient work across a thread
block, using shared or global memory according to the state's size and the
device's capacity.

Both backends support eligible ordinary and postselected sampling workflows
with FP64 or experimental FP32 coefficients. They remain experimental,
require explicit source builds, and are never selected automatically by the
CPU API. Fixed-fault importance sampling, leakage and loss trajectories, and
exact state queries are outside their current scope. The
[CUDA](../development/cuda-backend.md) and [HIP](../development/hip-backend.md)
guides describe hardware requirements and usage.

As these execution choices grow, so does the need to check that they implement
the same sampling behavior. This release expands shared conformance coverage
across CPU modes and both GPU backends, including both coefficient precisions
and cooperative execution. Independent Stim and Qiskit Aer references check
the applicable circuit semantics.

We are continuing to work on performance across the CPU and experimental GPU
paths. The next step is to benchmark the release candidate in
[clifft-bench](https://github.com/unitaryfoundation/clifft-bench) before the
final 0.11.0 release.
