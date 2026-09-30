# Performance

Clifft is built for fast, exact simulation of near-Clifford circuits. Its
compiler resolves Clifford coordinates and symbolic dependencies before
sampling, then executes only the remaining branch values and dense active-state
operations for each shot.

The measurements below cover three complementary questions: how Clifft
compares with another near-Clifford CPU simulator, how its throughput has
changed across releases, and how it behaves when a circuit becomes fully dense.

## Near-Clifford throughput

The recurring [`clifft-bench`](https://github.com/unitaryfoundation/clifft-bench)
campaign measures attempted shots per second for complete circuits on one
pinned logical CPU. For each workload and simulator, it selects the best batch
size for that workload before collecting the comparison.

The Clifft v0.11 configuration also enables
[active-width scheduling](compilation.md#active-width-scheduling) with the
standard settings after the default compiler pipeline. The pass remains opt-in
in ordinary Clifft use. The comparison measures the combined release
configuration, including scheduling and the selected batch size.

![Clifft v0.11 and SymFT v0.1.1 attempted shots per second across six near-Clifford workloads, with Clifft speedup ratios](../assets/performance/clifft-symft-throughput-light.png#only-light)
![Clifft v0.11 and SymFT v0.1.1 attempted shots per second across six near-Clifford workloads, with Clifft speedup ratios](../assets/performance/clifft-symft-throughput-dark.png#only-dark)

Clifft is faster than SymFT on all six workloads. The advantage ranges from
**1.06x to 201x**, with a **1.29x median** across the workload set. The plot
shows both tools' absolute rates on a shared logarithmic axis; the right-hand
column gives Clifft's speedup over SymFT for each workload.

These are attempted-shot rates, so post-selected shots that are later discarded
still count as simulation work. The run used one placement of an AWS
`m7a.xlarge` with an AMD EPYC 9R14, Ubuntu 24.04, and one pinned logical CPU.
Each reported configuration has five timed samples of at least 30 seconds.
Compilation is measured separately and excluded from sampling throughput.

The Clifft measurements use the published **0.11.0rc1** wheel, with
**0.10.0rc1** as the previous-release baseline. Plot labels show the intended
final release versions. SymFT is pinned to version **0.1.1** at upstream commit
`c89b985`.

See the immutable
[comparison table](https://github.com/unitaryfoundation/clifft-bench/blob/566924f2638a04d87cffa5930a91f3b4b3a48ee7/results/release-v1/release-v1-20260930-180134/comparisons.csv)
and the
[benchmark contract](https://github.com/unitaryfoundation/clifft-bench/blob/566924f2638a04d87cffa5930a91f3b4b3a48ee7/docs/benchmark-contract.md)
for the exact software identities, timing boundaries, and result semantics.

### Clifford postselection with Stim

The surface-code d7/r7 circuit provides a separate Clifford comparison with
Stim. Clifft reaches **4.13 million attempted shots/s**, compared with
**1.12 million for Stim**, a **3.69x** throughput ratio. Both tools sample the
same circuit, reject shots with any detector event, and count observable errors
among survivors.

This comparison uses the official Stim **1.16.0** wheel with calibrated
chunking. It measures sampling, all-detector postselection, and aggregate
counts on this workload, with compilation excluded. Clifft's ability to reject
shots early is part of the measured execution strategy.

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
on that work with operation scheduling and deferred noise sampling. Its
median per-workload throughput ratio is **1.20x v0.10**, with gains of
**2.52x and 2.21x** on the two coherent QEC circuits, **1.30x** on distillation,
and **1.10x** on cultivation at distance 5. The other two workloads remain
essentially unchanged.

The history plot reaches a **13.4x median speedup over v0.1**. Every point
uses the same six-workload reporting core. Earlier release posts used eight
workloads; the current core omits the two single-round coherent circuits,
so its historical medians differ from those earlier summaries.

The largest v0.10 gain, 837x on coherent `d=5, r=5`, primarily reflects a
compiler rewrite that reduced the circuit's peak active width from 24 to 13.
This is why the release history is best read as the result of the whole
compile-and-execute system, not as a benchmark of one kernel.

The v0.1 through v0.9 points come from a common
[history execution](https://github.com/unitaryfoundation/clifft-bench/tree/566924f2638a04d87cffa5930a91f3b4b3a48ee7/results/clifft-history-v1/clifft-history-v1-20260902).
For each workload, the v0.10 and v0.11 points chain the paired ratios from
their respective release runs onto that history. The plot then takes the
median across workloads at each version. This avoids treating an absolute
difference between two host boots as a product change.

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
