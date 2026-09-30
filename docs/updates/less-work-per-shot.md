# Less Work per Shot in Clifft (v0.11.0, September 2026)

A near-Clifford experiment often compiles one circuit and samples it many
times. The cost of those shots depends on more than the circuit's qubit count:
it depends on the active width throughout execution and, for postselected
experiments, how far each shot runs before it is rejected.

The previous two releases expanded how Clifft uses CPU resources, through
parallel and packed sampling. Version 0.11.0 continues improving CPU sampling
in two ways: searching for better operation schedules and deferring noise
sampling until its first use. This post looks at both changes with local A/B
measurements, then introduces the Sinter integration, ongoing experimental GPU
work, and the test coverage supporting these execution choices.

## Schedule for lower active width

Clifft's dense state contains $2^k$ coefficients, where $k$ is the active width.
Non-Clifford operations can increase that width, while measurements can reduce
it. The order in which independent operations run therefore affects both the
peak width and how much work is performed at larger widths.

The default compiler already uses this freedom through its statevector-squeeze
pass. The new `ActiveWidthSchedulePass` adds an opt-in search over legal
operation orders. It first looks for a lower peak active width, then for less
estimated dense work at the same peak. It can also move operations across
supported Pauli noise with compiler-prepared sign corrections that preserve
the noise semantics.

We compared the default pipeline with the same pipeline followed by the new
pass, using its default settings. The table reports sampling throughput with
the pass enabled relative to disabled, and compilation time separately.
Values above 1 mean faster sampling; values below 1 mean slower sampling.
These are local single-thread scalar measurements on four repository fixtures.

| Circuit | Peak width off / on | Compile ms off / on | Sampling throughput on / off |
| --- | ---: | ---: | ---: |
| Coherent QEC d=3 r=3 | 5 / 4 | 0.49 / 1.34 | 1.28x |
| Coherent QEC d=5 r=5 | 13 / 13 | 4.58 / 13.78 | 2.31x |
| Cultivation d=5 with postselection | 10 / 10 | 13.94 / 19.88 | 0.86x |
| Quantum Volume 10 qubits | 10 / 10 | 4.97 / 5.98 | 1.13x |

The coherent distance-5 example benefits even though its peak width is
unchanged: reducing the work performed at larger widths can matter as much as
reducing the peak. The cultivation example also shows why the decision cannot
be made from peak width alone: its sampling gets slower with this schedule.

Searching adds compilation time, which repeated sampling can amortize. Given
this trade-off, we conservatively keep the pass disabled by default. Users
should consider enabling it on their workloads and experimenting with its
settings to improve performance, comparing compilation plus sampling at the
shot counts they actually need. See
[Compiling Circuits](../guide/compilation.md#active-width-scheduling) for setup
and the pass's reported statistics.

## Sample noise when it is needed

In postselected experiments, a shot may be rejected long before the end of the
circuit. Preparing every noise event at the start of that shot spends work on
events it will never reach.

Scalar sampling now defers noise draws until execution reaches their first
use. The planner determines those deadlines ahead of time. If a detector
rejects the shot, the remaining noise work can be skipped.

In local tests on the circuit corpus, deferred noise increased sampling
throughput with early rejection by up to about 70%.

## A first connection to Sinter

Early testing suggested that Clifft could be competitive with Stim, and
sometimes faster, on some Clifford error-detection circuits. At the same time,
we wanted to explore broader QEC workflows beyond standalone sampling calls.
Connecting to Sinter is a practical first step toward both: existing
experiments can try Clifft while retaining Sinter's collection and analysis
workflow.

The standalone sampler benchmark now provides a concrete Clifford example:
on surface-code d7/r7 with all-detector postselection, Clifft reaches
**4.13 million attempted shots/s**, compared with **1.12 million for Stim**,
a **3.69x** ratio under the same sampling protocol. See the
[Performance guide](../guide/performance.md#clifford-postselection-with-stim)
for the configuration and scope of that comparison.

The new optional `clifft.sinter.PerfectionistSampler` supports experiments
that accept a shot only when every detector is quiet. Any logical observable
flip on an accepted shot counts as an error. Each Sinter worker compiles a
Clifft program and collects aggregate shot, discard, and error counts.

Install `clifft[sinter]` to use the adapter. This first integration supports
all-detector postselection on supported Clifford circuits; broader decoding
and non-Clifford integration remain areas for exploration. The
[Sinter guide](../guide/sinter.md) provides a complete example.

## Experimental GPU execution

We are also exploring workloads that could benefit from the memory capacity
and bandwidth available on GPUs. The experimental AMD HIP backend now supports
cooperative execution for wider active states, and a new NVIDIA CUDA backend
opens another hardware path. Small states can use one GPU thread per shot;
wider states can share coefficient work across a thread block.

Thanks to [Jose Manuel Monsalve Diaz](https://github.com/josemonsalve2) and AMD
for the HIP contributions, and [Farrokh Labib](https://github.com/FarLab) for
the CUDA backend.

Both backends support eligible ordinary and postselected sampling workflows
with FP64 or experimental FP32 coefficients. They remain experimental,
require explicit source builds, and are never selected automatically by the
CPU API. Fixed-fault importance sampling, leakage and loss trajectories, and
exact state queries are outside their current scope. The
[HIP](../development/hip-backend.md) and [CUDA](../development/cuda-backend.md)
guides describe hardware requirements and usage.

## Test features across execution choices

Supporting more execution choices also means checking their interactions with
existing features. We have expanded and reorganized the tests around a shared
suite of sampling behaviors: a test of measurements, detector postselection,
noise, or another supported feature can run through each applicable execution
mode. Adding a backend to that suite gives it the existing checks for the
features it supports. Adding a shared feature test extends coverage across
those backends.

The suite now covers scalar, packed, and threaded CPU sampling alongside
experimental HIP and CUDA sampling in FP64 and FP32. Additional cases exercise
cooperative GPU tiers, while importance sampling and leakage/loss trajectories
share coverage across the CPU modes that support them. Independent Stim and
Qiskit Aer references check the applicable circuit semantics. This structure
helps catch interactions that isolated tests for a new feature or backend
could miss.
