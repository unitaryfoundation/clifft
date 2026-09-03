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

![Clifft v0.10 and SymFT v0.1 attempted shots per second across eight near-Clifford workloads, with Clifft speedup ratios](../assets/performance/clifft-symft-throughput-light.png#only-light)
![Clifft v0.10 and SymFT v0.1 attempted shots per second across eight near-Clifford workloads, with Clifft speedup ratios](../assets/performance/clifft-symft-throughput-dark.png#only-dark)

Clifft is faster than SymFT on all eight workloads. The advantage ranges from
**1.05x to 87.7x**, with a **1.53x median** across the workload set. The plot
shows both tools' absolute rates on a shared logarithmic axis; the right-hand
column gives Clifft's speedup over SymFT for each workload.

These are attempted-shot rates, so post-selected shots that are later discarded
still count as simulation work. The run used one placement of an AWS
`m7a.xlarge` with an AMD EPYC 9R14, Ubuntu 24.04, and one pinned logical CPU.
Each reported configuration has five timed samples of at least 30 seconds.

See the immutable
[comparison table](https://github.com/unitaryfoundation/clifft-bench/blob/3a9a2eae6c8a1c144699530b806512a579deacdc/results/release-v1/release-v1-20260903-133252/comparisons.csv)
and the
[benchmark contract](https://github.com/unitaryfoundation/clifft-bench/blob/3a9a2eae6c8a1c144699530b806512a579deacdc/docs/benchmark-contract.md)
for the exact software identities, timing boundaries, and result semantics.

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
[0.10](../updates/packed-sampling.md) combines another compiler improvement
with packed sampling: its median throughput is **3.23x v0.9** and **7.0x
v0.1** across the eight workloads.

The largest v0.10 gain, 837x on coherent `d=5, r=5`, primarily reflects a
compiler rewrite that reduced the circuit's peak active width from 24 to 13.
This is why the release history is best read as the result of the whole
compile-and-execute system, not as a benchmark of one kernel.

The v0.1 through v0.9 points come from a common
[history execution](https://github.com/unitaryfoundation/clifft-bench/tree/b1eb8f489b646273538d8a3efcdef5f07a0364d1/results/clifft-history-v1/clifft-history-v1-20260902).
The v0.10 point chains the paired v0.10/v0.9 ratio from the release run onto
that history. This avoids treating an absolute difference between two host
boots as a product change.

## Dense Quantum Volume circuits

Near-Clifford structure is Clifft's main advantage. A dense Quantum Volume
circuit instead drives the active width to the full qubit count, making Clifft
carry a conventional $2^n$ state vector. This deliberately tests Clifft where
its specialized representation offers the least help.

![Execution time for dense Quantum Volume circuits in Clifft, Qiskit Aer, qsim, and Qulacs](../assets/performance/quantum-volume-light.png#only-light)
![Execution time for dense Quantum Volume circuits in Clifft, Qiskit Aer, qsim, and Qulacs](../assets/performance/quantum-volume-dark.png#only-dark)

Using 16 physical CPU cores, Clifft records the shortest median execution time
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
[experiment description](https://github.com/unitaryfoundation/clifft-bench/blob/f02d8496ee9269c9fa25a4cf4bdb982ffd8a28e2/experiments/qv/README.md)
and
[complete result table](https://github.com/unitaryfoundation/clifft-bench/blob/f02d8496ee9269c9fa25a4cf4bdb982ffd8a28e2/experiments/qv/results/qv-0.10.0rc1-20260902/cases.csv).

## Future benchmark scope

The near-Clifford comparison is CPU-only. A future campaign will cover GPU
execution, including Tsim, under a separate hardware and measurement contract.
A pure-Clifford comparison with Stim will likewise be added after the relevant
Clifft execution changes and benchmark cases land.
