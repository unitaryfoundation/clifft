# Performance

Clifft is built for fast, exact simulation of near-Clifford circuits. Its
compiler resolves Clifford coordinates and symbolic dependencies before
sampling, then executes only the remaining branch values and dense active-state
operations for each shot.

The measurements below cover three complementary questions: how Clifft
compares with other CPU simulators, how its throughput has
changed across releases, and how it behaves when a circuit becomes fully dense.

## CPU simulator throughput

The [`clifft-bench`](https://github.com/unitaryfoundation/clifft-bench)
suite measures attempted shots per second for complete circuits on one
pinned logical CPU. Each simulator uses its best calibrated batch size for each
workload, and Clifft enables opt-in
[active-width scheduling](compilation.md#active-width-scheduling) with standard
settings after the default compiler pipeline.

![Clifft v0.13, SymFT v0.1.1, xtim v3.2.2, Stim v1.16, and Tsim v0.1.5 attempted shots per second across fourteen CPU workloads](../assets/performance/simulator-throughput-light.png#only-light)
![Clifft v0.13, SymFT v0.1.1, xtim v3.2.2, Stim v1.16, and Tsim v0.1.5 attempted shots per second across fourteen CPU workloads](../assets/performance/simulator-throughput-dark.png#only-dark)

The four panels group workloads by circuit family, and **each panel uses its own
logarithmic scale**. Missing markers indicate simulator–workload pairs that are
not measured in this suite:

- **Magic-state preparation:** Clifft reaches **597 million attempted shots/s**
  on 15-to-1 distillation and **83.6 million** on the measured Reed–Muller
  circuit, where stabilizer recovery and phase simplification collapse the
  active state. On the injection/cultivation circuits (`d=3` and `d=5`), xtim
  and Tsim are faster. The separate end-to-end cultivation `d=5` workload
  continues through escape into a distance-15 surface code, with postselection
  on cultivation-stage detectors, reaching
  **39.1 thousand shots/s** in Clifft and **135 thousand** in Tsim.
- **Surface-code memory and coherent noise:** Clifft approaches Tsim's throughput
  on coherent `d=3, r=3` (**2.78 million** vs. **2.89 million shots/s**) and leads on
  coherent `d=5, r=5` (**12.8 thousand shots/s** vs. **3.03 thousand** in Tsim
  and **63.3** in SymFT), where compiled Pauli rotations and bounded active
  width keep per-shot work small.
- **Arithmetic and code switching:** Clifft reaches **199 million shots/s** on
  the fixed-input 16-bit Draper adder (vs. **25.8 million** in Tsim) and
  samples the measured `BT27` and `BT81` code-switching circuits at **247
  thousand** and **831 thousand shots/s**.
- **Quantum Volume (1 core):** Dense circuits drive the active width to the full
  qubit count. Tsim's active-statevector backend is faster at `QV10`
  (**480 thousand** vs. **23.2 thousand shots/s** in Clifft), whereas at `QV20`
  Clifft's dense statevector execution reaches **5.85 shots/s** compared with
  **0.274** in SymFT and **0.0938** in Tsim.

### Clifford postselection with Stim

On the Clifford `Surface code d7, r7` workload in the second panel, all tools
sample the same circuit, reject any shot with a detector event, and count
logical observable errors among survivors. Clifft reaches **4.22 million
attempted shots/s** compared with **1.12 million** for Stim (**3.77x**), while
xtim and Tsim reach **4.85 million** and **5.58 million**. Clifft can reject
failed shots early and accumulate survivor counts directly. xtim and Tsim also
support early rejection, but return records that their adapters reduce to
counts. The Stim baseline materializes packed detector and observable records
before filtering survivors in the adapter. The comparison includes these
output-processing costs.

### Measurement setup

Rates count attempted shots, so postselected shots that are later discarded
still count as simulation work. Ahead-of-time compilation is excluded; lazy
plan construction during sampling (such as xtim's per-sample plan cache fill)
and reduction of packed outputs to aggregate survivor counts are included in
the timed window.

Benchmarks run on AWS `m7a.xlarge` instances (AMD EPYC 9R14, Ubuntu 24.04)
using one pinned logical CPU, one thread, a 12 GiB memory limit, and five timed
samples of at least 30 seconds per case. The comparison covers **Clifft 0.13**,
**SymFT 0.1.1**, **xtim 3.2.2**, **Stim 1.16.0**, and **Tsim 0.1.5**; see the
[Clifft v0.13 results](https://github.com/unitaryfoundation/clifft-bench/tree/dce2f9e8ca5eb9311c074fb3c8c0793d469f4b1c/results/release-clifft/0.13.0rc2-20261104-140631),
[peer simulator results](https://github.com/unitaryfoundation/clifft-bench/tree/dce2f9e8ca5eb9311c074fb3c8c0793d469f4b1c/results/release-v1/0.13.0rc1-20261103-183623),
[review summary](https://github.com/unitaryfoundation/clifft-bench/blob/dce2f9e8ca5eb9311c074fb3c8c0793d469f4b1c/results/release-clifft/0.13.0rc2-20261104-140631/report/review.md),
and [benchmark contract](https://github.com/unitaryfoundation/clifft-bench/blob/dce2f9e8ca5eb9311c074fb3c8c0793d469f4b1c/docs/benchmark-contract.md)
for exact software commits, environments, and timing boundaries.

## Changes in v0.13

![Clifft v0.13 throughput changes from v0.12: 223x for 15-to-1 distillation, 16.2x for Reed–Muller, and -0.52% to +8.21% for the other seven paired workloads](../assets/performance/release-change-light.png#only-light)
![Clifft v0.13 throughput changes from v0.12: 223x for 15-to-1 distillation, 16.2x for Reed–Muller, and -0.52% to +8.21% for the other seven paired workloads](../assets/performance/release-change-dark.png#only-dark)

Across the nine paired workloads, throughput reaches **223x** the v0.12 rate on
15-to-1 distillation and **16.2x** on Reed–Muller. Peak active width falls from
**5 to 0** and **5 to 1**, respectively, reducing the dense state carried during
sampling. The release's improved
[stabilizer recovery and phase simplification](../updates/sampling-through-leakage-and-loss.md#stabilizer-recovery-and-phase-simplification)
eliminate more non-Clifford work before sampling. Packed sampling also skips
noise-sampling work when a compiled plan has no quantum-noise sites.

Among the remaining workloads, the fixed-input 16-bit Draper adder improves by
**8.21%**, coherent `d=3, r=3` by **4.74%**, and cultivation `d=3` by
**2.96%**. The other four circuits fall within the **±1.1% inconclusive band**.
Both releases use their default compiler pipeline followed
by [active-width scheduling](compilation.md#active-width-scheduling) and
independently calibrated batch sizes.

Cultivation d5 varies between processes in both releases, so this run does not
establish a reliable release-change percentage; see
[#656](https://github.com/unitaryfoundation/clifft/issues/656).

## Performance over time

While the comparison above measures v0.13 against v0.12 on the expanded
benchmark suite, the historical series below tracks a consistent six-workload
QEC core from v0.1 through v0.12. Clifft's compiler-and-executor separation
allows each release to improve circuit optimization, symbolic planning,
executable preparation, or active-state kernels without moving circuit analysis
back into the per-shot execution loop.

![Median Clifft throughput from v0.1 through v0.12 relative to v0.1](../assets/performance/performance-over-time-light.png#only-light)
![Median Clifft throughput from v0.1 through v0.12 relative to v0.1](../assets/performance/performance-over-time-dark.png#only-dark)

The first major step arrived in [v0.8](../updates/symbolic-sampling.md), when
symbolic plans replaced the original localized-Pauli virtual machine. Version
[0.10](../updates/packed-sampling.md) combined compiler width reduction with
packed sampling—including an 837x gain on coherent `d=5, r=5` after reducing
peak active width from 24 to 13. Version [0.11](../updates/less-work-per-shot.md)
added operation scheduling and deferred noise sampling, and version
[0.12](../updates/smaller-active-states.md) preserved that QEC throughput while
adding state-aware compiler simplification, reaching a **13.6x median speedup
over v0.1**.

The v0.1 through v0.9 points come from a common
[history execution](https://github.com/unitaryfoundation/clifft-bench/tree/59f78d280c5fcb90f42cc75cfd73ff3f004ad4e8/results/clifft-history-v1/clifft-history-v1-20260902),
and v0.10 through v0.12 chain paired release ratios onto that baseline as
recorded in
[reporting sources](https://github.com/unitaryfoundation/clifft-bench/blob/59f78d280c5fcb90f42cc75cfd73ff3f004ad4e8/reporting/sources.json).

## Dense Quantum Volume circuits on 16 cores

Whereas the single-core Quantum Volume rows above compare near-Clifford
samplers at `QV10` and `QV20`, this multi-core study compares Clifft against
dedicated statevector simulators (**Qiskit Aer**, **qsim**, and **Qulacs**) up
to `QV28` ($2^{28}$ amplitudes) where Clifft's near-Clifford representation
offers the least help.

![Execution time for dense Quantum Volume circuits in Clifft, Qiskit Aer, qsim, and Qulacs](../assets/performance/quantum-volume-light.png#only-light)
![Execution time for dense Quantum Volume circuits in Clifft, Qiskit Aer, qsim, and Qulacs](../assets/performance/quantum-volume-dark.png#only-dark)

Using 16 physical CPU cores on an AWS `c8i.8xlarge` (**Clifft 0.10**), Clifft
records the shortest median execution time at `QV20` and `QV22`. At `QV28` it
completes in 24.7 seconds, compared with 29.9 seconds for Qiskit Aer and 344.9
seconds for Qulacs; qsim leads from `QV24` onward and finishes `QV28` in 11.4
seconds. Smaller widths favor tools with lower fixed startup overhead. Even
outside its intended near-Clifford regime, Clifft remains competitive with
dedicated statevector simulators.

Following the original Clifft paper's protocol, Clifft's timer includes
compilation plus one sample, while Qiskit transpilation and Qulacs/qsim circuit
preparation occur before their timers. Each width uses three circuit seeds; see
the [experiment description](https://github.com/unitaryfoundation/clifft-bench/blob/566924f2638a04d87cffa5930a91f3b4b3a48ee7/experiments/qv/README.md)
and [complete result table](https://github.com/unitaryfoundation/clifft-bench/blob/566924f2638a04d87cffa5930a91f3b4b3a48ee7/experiments/qv/results/qv-0.10.0rc1-20260902/cases.csv).
