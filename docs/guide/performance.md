# Performance

Clifft is built for fast, exact simulation of near-Clifford circuits. Its
compiler resolves Clifford coordinates and symbolic dependencies before
sampling, then executes only the remaining branch values and dense active-state
operations for each shot.

The measurements below cover three complementary questions: how Clifft
compares with other CPU simulators, how its throughput has
changed across releases, and how it behaves when a circuit becomes fully dense.

## Near-Clifford throughput

The recurring [`clifft-bench`](https://github.com/unitaryfoundation/clifft-bench)
campaign measures attempted shots per second for complete circuits on one
pinned logical CPU. For each workload and simulator, it selects the best batch
size for that workload before collecting the comparison.

Both Clifft release configurations enable
[active-width scheduling](compilation.md#active-width-scheduling) with the
standard settings after the default compiler pipeline. The pass remains opt-in
in ordinary Clifft use. The comparison measures the combined release
configuration, including scheduling and the selected batch size.

![Clifft v0.12, SymFT v0.1.1, xtim v3.1.4, and Stim v1.16 attempted shots per second on their measured QEC workloads](../assets/performance/simulator-throughput-light.png#only-light)
![Clifft v0.12, SymFT v0.1.1, xtim v3.1.4, and Stim v1.16 attempted shots per second on their measured QEC workloads](../assets/performance/simulator-throughput-dark.png#only-dark)

Clifft is faster than SymFT on all seven workloads, by **1.05x to 201x**.
xtim is faster than Clifft on cultivation d3/d5 and 15-to-1 distillation,
by **1.60x to 19.9x**, and essentially tied on the Clifford surface-code
workload. Stim supplies a separate Clifford comparison below. Missing markers
mean no measurement in this campaign; xtim has four measured workloads and
Stim has one.

These are attempted-shot rates, so post-selected shots that are later discarded
still count as simulation work. The run used one placement of an AWS
`m7a.xlarge` with an AMD EPYC 9R14, Ubuntu 24.04, and one pinned logical CPU.
Each reported configuration has five timed samples of at least 30 seconds.
Compilation is measured separately and excluded from sampling throughput,
except for lazy plan construction during sampling. In particular, each xtim
sample starts with an empty plan cache and includes lazy planning; its rate
is a fixed-duration average. All tools return aggregate counts. Stim and xtim
materialize packed records that the adapter reduces to those counts, with
both steps included in their timing.

The Clifft measurements use the published **0.12.0rc1** wheel, with
**0.11.0rc1** as the previous-release baseline. Plot labels show the final
release versions; only documentation has changed since the measured candidate.
SymFT is pinned to **0.1.1** at upstream commit `c89b985`, and xtim to
**3.1.4** at `2945407` with its portable build configuration.
The harness calibrates numeric batch sizes independently; it does not use
Clifft's new `batch_size="tune"` API.

See the immutable
[comparison table](https://github.com/unitaryfoundation/clifft-bench/blob/59f78d280c5fcb90f42cc75cfd73ff3f004ad4e8/results/release-v1/release-v1-20261009-192718/comparisons.csv)
and the
[benchmark contract](https://github.com/unitaryfoundation/clifft-bench/blob/59f78d280c5fcb90f42cc75cfd73ff3f004ad4e8/docs/benchmark-contract.md)
for the exact software identities, timing boundaries, and result semantics.

### Clifford postselection with Stim

The surface-code d7/r7 circuit provides a separate Clifford comparison with
Stim. Clifft reaches **4.35 million attempted shots/s**, compared with
**1.12 million for Stim**, a **3.90x** throughput ratio. Both tools sample the
same circuit, reject shots with any detector event, and count observable errors
among survivors.

This comparison uses the official Stim **1.16.0** wheel with calibrated
chunking. It measures sampling, all-detector postselection, and aggregate
counts on this workload, with compilation excluded. Clifft's ability to reject
shots early is part of the measured execution strategy.

## Changes in v0.12

![Clifft v0.12 throughput changes from v0.11 across seven QEC workloads, ranging from -1.16% to +1.90%](../assets/performance/v012-vs-v011-light.png#only-light)
![Clifft v0.12 throughput changes from v0.11 across seven QEC workloads, ranging from -1.16% to +1.90%](../assets/performance/v012-vs-v011-dark.png#only-dark)

The seven QEC workloads changed by **-1.16% to +1.90%** on this host.
These small differences do not establish a release-wide speedup. Both versions
use their own default compiler passes followed by the opt-in scheduler and
independently calibrated batching.

## Performance over time

Clifft's compiler-like structure provides several independent places to make
simulation faster: circuit optimization, symbolic planning, executable
preparation, and active-state kernels. That structure does not mean every
speedup comes from the compiler, but it lets later releases improve one stage
without moving circuit analysis back into the per-shot execution loop.

![Median Clifft throughput by release relative to v0.1](../assets/performance/performance-over-time-light.png#only-light)
![Median Clifft throughput by release relative to v0.1](../assets/performance/performance-over-time-dark.png#only-dark)

The first broad step arrived in [v0.8](../updates/symbolic-sampling.md), when
symbolic plans replaced the original localized-Pauli virtual machine. Version
[0.10](../updates/packed-sampling.md) combined another compiler improvement
with packed sampling. Version [0.11](../updates/less-work-per-shot.md) builds
on that work with operation scheduling and deferred noise sampling.
Version [0.12](../updates/smaller-active-states.md) preserves that QEC
throughput while adding state-aware compiler simplification.

The history plot reaches a **13.6x median speedup over v0.1**. Every point
uses the same six-workload reporting core. Earlier release posts used eight
workloads; the current core omits the two single-round coherent circuits,
so its historical medians differ from those earlier summaries. The new
15-to-1 workload appears in current comparisons without changing the history's
six-workload core.

The largest v0.10 gain, 837x on coherent `d=5, r=5`, primarily reflects a
compiler rewrite that reduced the circuit's peak active width from 24 to 13.
This is why the release history is best read as the result of the whole
compile-and-execute system, not as a benchmark of one kernel.

The v0.1 through v0.9 points come from a common
[history execution](https://github.com/unitaryfoundation/clifft-bench/tree/59f78d280c5fcb90f42cc75cfd73ff3f004ad4e8/results/clifft-history-v1/clifft-history-v1-20260902).
For each workload, the v0.10 through v0.12 points chain the paired ratios from
their respective release runs onto that history. The plot then takes the
median across workloads at each version. This avoids treating an absolute
difference between two host boots as a product change. The exact execution
chain is recorded in
[reporting sources](https://github.com/unitaryfoundation/clifft-bench/blob/59f78d280c5fcb90f42cc75cfd73ff3f004ad4e8/reporting/sources.json).

## Dense Quantum Volume circuits

Near-Clifford structure is Clifft's main advantage. A dense Quantum Volume
circuit instead drives the active width to the full qubit count, making Clifft
carry a conventional $2^n$ state vector. This deliberately tests Clifft where
its specialized representation offers the least help.

![Execution time for dense Quantum Volume circuits in Clifft, Qiskit Aer, qsim, and Qulacs](../assets/performance/quantum-volume-light.png#only-light)
![Execution time for dense Quantum Volume circuits in Clifft, Qiskit Aer, qsim, and Qulacs](../assets/performance/quantum-volume-dark.png#only-dark)

These are the existing **Clifft 0.10.0rc1** measurements, using 16 physical CPU
cores. Clifft records the shortest median execution time
at QV20 and QV22. At QV28 it completes in 24.7 seconds, compared with 29.9
seconds for Qiskit Aer and 344.9 seconds for Qulacs; qsim leads at 11.4
seconds. Smaller widths favor tools with lower fixed overhead, and qsim leads
from QV24 onward. The result is not that Clifft wins every dense workload, but
that it remains in the leading group even outside its intended near-Clifford
regime.

This experiment reproduces the original Clifft paper's timing boundaries.
Clifft is charged for compilation plus one sample, while Qiskit transpilation
and Qulacs/qsim circuit preparation occur before their timers. It is therefore
not an equal end-to-end latency comparison. The experiment uses three circuit
seeds per width on an AWS `c8i.8xlarge`; see the
[experiment description](https://github.com/unitaryfoundation/clifft-bench/blob/566924f2638a04d87cffa5930a91f3b4b3a48ee7/experiments/qv/README.md)
and
[complete result table](https://github.com/unitaryfoundation/clifft-bench/blob/566924f2638a04d87cffa5930a91f3b4b3a48ee7/experiments/qv/results/qv-0.10.0rc1-20260902/cases.csv).

## Future benchmark scope

The near-Clifford comparison is CPU-only. GPU benchmarking remains ongoing
work, including comparisons with Tsim under a separate hardware and measurement
contract.
