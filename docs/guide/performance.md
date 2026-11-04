# Performance

Clifft is built for fast, exact simulation of near-Clifford circuits. Its
compiler resolves Clifford coordinates and symbolic dependencies before
sampling, then executes only the remaining branch values and dense active-state
operations for each shot.

The measurements below cover three complementary questions: how Clifft
compares with other CPU simulators, how its throughput has
changed across releases, and how it behaves when a circuit becomes fully dense.

## CPU simulator throughput

The recurring [`clifft-bench`](https://github.com/unitaryfoundation/clifft-bench)
campaign measures attempted shots per second for complete circuits on one
pinned logical CPU. For each workload and simulator, it selects the best batch
size for that workload before collecting the comparison.

Both Clifft release configurations enable
[active-width scheduling](compilation.md#active-width-scheduling) with the
standard settings after the default compiler pipeline. The pass remains opt-in
in ordinary Clifft use. The comparison measures the combined release
configuration, including scheduling and the selected batch size.

![Clifft v0.13, SymFT v0.1.1, xtim v3.2.2, Stim v1.16, and Tsim v0.1.5 attempted shots per second across fourteen CPU workloads](../assets/performance/simulator-throughput-light.png#only-light)
![Clifft v0.13, SymFT v0.1.1, xtim v3.2.2, Stim v1.16, and Tsim v0.1.5 attempted shots per second across fourteen CPU workloads](../assets/performance/simulator-throughput-dark.png#only-dark)

The four panels cover magic-state preparation, surface-code memory and coherent
noise, arithmetic and code switching, and dense Quantum Volume circuits.
**Each panel has its own logarithmic scale**; compare horizontal positions
within a panel. Missing markers indicate pairings without measurements.

Clifft reaches **597 million attempted shots/s** on 15-to-1 distillation and
**83.6 million** on the measured Reed–Muller circuit, leading the measured
tools on both. xtim and Tsim are faster on injection/cultivation d3 and d5.
The separate end-to-end cultivation d5 circuit reaches **39.1 thousand
shots/s** in Clifft and **135 thousand** in Tsim. Circuit scope matters when
comparing these rates.

These are attempted-shot rates, so post-selected shots that are later discarded
still count as simulation work. The collections used AWS `m7a.xlarge`
instances with an AMD EPYC 9R14, Ubuntu 24.04, one pinned logical CPU, one
thread, and a 12 GiB memory ceiling. Each reported configuration has five
timed samples of at least 30 seconds in one fresh worker process.
Compilation is measured separately and excluded from sampling throughput,
except for lazy plan construction during sampling. In particular, each xtim
sample starts with an empty plan cache and includes lazy planning; its rate
is a fixed-duration average. All tools return aggregate counts. Stim and xtim
materialize packed records that the adapter reduces to those counts, with
both steps included in their timing.

The Clifft measurements were collected on **November 4, 2026**, using the
published **0.13.0rc2** wheel and **0.12.0rc1** baseline. Their locked
environments use NumPy 2.5.4 and 2.5.2, respectively, so the comparison includes
those dependency versions. Figure labels use the intended final versions
0.13.0 and 0.12.0. The harness calibrates numeric batch sizes independently;
it does not use Clifft's `batch_size="tune"` API.

Peer results are retained from **November 3, 2026**: SymFT **0.1.1**
(`c89b985`), xtim **3.2.2** (`edf0dd0`), Stim **1.16.0**, and Tsim **0.1.5**
(`e9406ed`). The report validates matching hardware, protocols, circuit
digests, selection semantics, and call sizes before combining their absolute
rates with RC2. These are separate collections; the peer comparisons are
not fresh paired measurements. All Clifft release ratios below use only the
November 4 execution.

The immutable [RC2 results and environment](https://github.com/unitaryfoundation/clifft-bench/tree/dce2f9e8ca5eb9311c074fb3c8c0793d469f4b1c/results/release-clifft/0.13.0rc2-20261104-140631),
[peer results](https://github.com/unitaryfoundation/clifft-bench/tree/dce2f9e8ca5eb9311c074fb3c8c0793d469f4b1c/results/release-v1/0.13.0rc1-20261103-183623),
and [generated review](https://github.com/unitaryfoundation/clifft-bench/blob/dce2f9e8ca5eb9311c074fb3c8c0793d469f4b1c/results/release-clifft/0.13.0rc2-20261104-140631/report/review.md)
record every case and exact software identity. The
[benchmark contract](https://github.com/unitaryfoundation/clifft-bench/blob/dce2f9e8ca5eb9311c074fb3c8c0793d469f4b1c/docs/benchmark-contract.md)
defines the timing boundaries and result semantics.

### Clifford postselection with Stim

The surface-code d7/r7 circuit provides a separate Clifford comparison with
Stim. Clifft reaches **4.22 million attempted shots/s**, compared with
**1.12 million for Stim**, a **3.77x** throughput ratio across the two collections.
Both tools sample the
same circuit, reject shots with any detector event, and count observable errors
among survivors.

This comparison uses the official Stim **1.16.0** wheel with calibrated
chunking. It measures sampling, all-detector postselection, and aggregate
counts on this workload, with compilation excluded. Clifft's ability to reject
shots early is part of the measured execution strategy.

## Changes in v0.13

![Clifft v0.13 throughput changes from v0.12: 223x for 15-to-1 distillation, 16.2x for Reed–Muller, and -0.52% to +8.21% for the other seven paired workloads](../assets/performance/release-change-light.png#only-light)
![Clifft v0.13 throughput changes from v0.12: 223x for 15-to-1 distillation, 16.2x for Reed–Muller, and -0.52% to +8.21% for the other seven paired workloads](../assets/performance/release-change-dark.png#only-dark)

The nine measured release pairs show **223x** throughput on 15-to-1
distillation, **16.2x** on Reed–Muller, and **8.21% higher throughput** on the
fixed-input Draper adder. Injection/cultivation d3 improves by **2.96%** and
coherent d3/r3 by **4.74%**. The other four changes are within the benchmark's
**±1.1% inconclusive band**, which is a reporting threshold, not a confidence
interval. Both versions use their default compiler passes followed by the
opt-in scheduler and independently calibrated batching, so these are changes
to the complete release configuration.

**Cultivation d5 still has substantial variation between worker processes.**
This campaign measured **+0.18%**, with both releases selecting batch 1 after
full calibration. Earlier independent-worker measurements found variation in
both releases and repeated RC1 losses with direct batch-1 preparation. Five
samples from one worker per version cannot resolve that behavior; the
[follow-up investigation](https://github.com/unitaryfoundation/clifft/issues/656)
remains open. The +0.18% result describes this calibrated run, rather than a
settled release-wide percentage for d5.

## Performance over time

Clifft's compiler-like structure provides several independent places to make
simulation faster: circuit optimization, symbolic planning, executable
preparation, and active-state kernels. That structure does not mean every
speedup comes from the compiler, but it lets later releases improve one stage
without moving circuit analysis back into the per-shot execution loop.

![Median Clifft throughput from v0.1 through v0.12 relative to v0.1](../assets/performance/performance-over-time-light.png#only-light)
![Median Clifft throughput from v0.1 through v0.12 relative to v0.1](../assets/performance/performance-over-time-dark.png#only-dark)

The first broad step arrived in [v0.8](../updates/symbolic-sampling.md), when
symbolic plans replaced the original localized-Pauli virtual machine. Version
[0.10](../updates/packed-sampling.md) combined another compiler improvement
with packed sampling. Version [0.11](../updates/less-work-per-shot.md) builds
on that work with operation scheduling and deferred noise sampling.
Version [0.12](../updates/smaller-active-states.md) preserves that QEC
throughput while adding state-aware compiler simplification.

This historical series ends at v0.12 with a **13.6x median speedup over v0.1**. Every point
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

## Dense Quantum Volume circuits on 16 cores

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
work, including comparisons with Tsim.
