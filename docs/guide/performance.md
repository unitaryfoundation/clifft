# Performance

Clifft targets the regime between pure stabilizer simulation and fully dense
statevector simulation: large circuits with mostly Clifford structure and
localized non-Clifford effects.

The main quantity to watch is the peak active width $k$. The active state has
$2^k$ amplitudes, so cost depends on both the peak width and the time spent at
each width. Total qubit count and non-Clifford gate count alone are poor
performance predictors.

## Headline CPU results

!!! tip "Clifft is fast on real near-Clifford QEC circuits"

    On one pinned CPU core, Clifft 0.9 sustained 1.20M attempted shots/s on
    distance-3 cultivation and 576k on an 85-qubit distillation circuit. It
    exceeded one million attempted shots/s on two workloads in the current
    eight-circuit campaign.

These are medians over three fresh placements on an AWS `m7a.xlarge` with an
AMD EPYC 9R14 CPU and Ubuntu 24.04. Each placement used one pinned core. The
`vs 0.8` column compares the median rate across placements for each release.

| Circuit | Qubits | Peak $k$ | Clifft 0.9 attempted shots/s | vs 0.8 |
|---|---:|---:|---:|---:|
| Surface code `d=7, r=7`, $p=10^{-3}$ | 118 | 0 | 484k | +4% |
| Cultivation `d=3` | 15 | 4 | **1.20M** | +8% |
| Cultivation `d=5` | 42 | 10 | 182k | +33% |
| Distillation | 85 | 5 | **576k** | +6% |
| Coherent surface code `d=3, r=1` | 26 | 4 | **1.88M** | +37% |
| Coherent surface code `d=3, r=3` | 26 | 7 | 461k | +42% |
| Coherent surface code `d=5, r=1` | 64 | 12 | 14.6k | +1% |
| Coherent surface code `d=5, r=5` | 64 | 22 | 7.0 | +10% |

Clifft 0.9 has a higher median than 0.8 on all eight workloads and is faster
in 23 of the 24 individual placement/workload pairs. Browse the canonical
[`current-tools-v1` execution](https://github.com/unitaryfoundation/clifft-bench/tree/main/results/current-tools-v1/current-tools-v1-20260824-r1)
for raw JSON and the complete derived tables.

Sampling throughput counts every attempted shot, including shots discarded by
detector postselection. The wide coherent `d=5, r=5` case is retained as a
stress case; its low rate should not be generalized to circuits with the same
qubit count or peak width.

## Choosing a simulator

| Regime | Recommended tool | Why |
|---|---|---|
| Pure Clifford QEC | Stim | Stim packs stabilizer-frame work across shots and is the specialist when active width is always zero. |
| Low-magic or near-Clifford circuits | Clifft | Clifft is most effective when non-Clifford effects stay localized and measurements repeatedly shrink the active state. |
| Dense universal circuits | A statevector simulator or Clifft | When $k=n$, Clifft is exact but has the same exponential state-size limit as dense statevector simulation. |

Clifft is not intended to replace Stim for fully Clifford workloads. Its main
target is the middle regime, where exact non-Clifford effects matter without
forcing the entire physical system into one dense state.

## Reading the active-width regimes

### Active width zero

At active width zero, coefficient-array work disappears, but Clifft still
evaluates sampling actions and symbolic expressions. A Clifford-only engine
such as Stim can instead pack stabilizer-frame work from many shots into each
SIMD instruction.

### Low and moderate active width

Cultivation and distillation are Clifft's main target regime. Their physical
circuits contain many more qubits than the active state, and measurements can
remove active coordinates during execution. Clifft performs dense work over
the smaller $2^k$ active state instead of the full $2^n$ physical state.

The same peak width does not imply the same throughput. Circuit length, width
at each action, measurement and detector work, postselection timing, and active
Pauli shape all affect the cost of an attempted shot.

### Larger active width and the dense limit

As $k$ grows, active-state work dominates and Clifft approaches ordinary dense
statevector scaling. Parallel execution can help wide individual shots, but it
does not remove the exponential memory limit.

## Cross-tool and release comparisons

The same single-core QEC campaign compares Clifft 0.9 with SymFT single-shot
and batched modes. SymFT's fastest applicable mode is faster on seven of the
eight workloads; Clifft is 1.33x faster on distance-5 cultivation. The winning
SymFT batch size varies by circuit, so the
[per-workload table](https://github.com/unitaryfoundation/clifft-bench#current-single-core-qec-results)
is more useful than one aggregate ranking.

![QEC workload throughput ratio between Clifft and the fastest measured SymFT mode](https://raw.githubusercontent.com/unitaryfoundation/clifft-bench/8d6a70d47c7f7fa596d87170375c1583dbfca499/figures/current-tools-v1-20260824-r1.png)

The release-history campaign runs Clifft 0.1 through 0.9 on the same corpus and
hardware epoch. Clifft 0.9 is faster than 0.1 on all eight workloads, with a
median per-workload speedup of 1.93x and a range of 1.18x to 16.89x. See the
[release-history results](https://github.com/unitaryfoundation/clifft-bench#clifft-release-history).

![Clifft throughput across releases](https://raw.githubusercontent.com/unitaryfoundation/clifft-bench/8d6a70d47c7f7fa596d87170375c1583dbfca499/figures/clifft-history-v1-20260825-r1.png)

### Multicore Quantum Volume

#### Current-tool latency

![Quantum Volume latency by simulator](https://raw.githubusercontent.com/unitaryfoundation/clifft-bench/8d6a70d47c7f7fa596d87170375c1583dbfca499/figures/qv-multicore-v1-2026082-current-tools.png)

On 16 physical cores, Clifft 0.9 has the lowest median single-shot latency of
the four measured tools at QV20 and QV22. Tool ordering changes with circuit
width, so the full curve is more informative than one aggregate ranking.

#### Clifft strong scaling

![Clifft Quantum Volume strong scaling](https://raw.githubusercontent.com/unitaryfoundation/clifft-bench/8d6a70d47c7f7fa596d87170375c1583dbfca499/figures/qv-multicore-v1-2026082-clifft-scaling.png)

Clifft's paired median QV24 speedup is 10.17x from 1 to 16 cores. Points show
paired medians across three deterministic seeds; whiskers show the seed range.

This is a separate exploratory campaign on an AWS `c8i.8xlarge` with three
deterministic circuit seeds per point. Its latency numbers must not be mixed
with the single-core QEC throughput table. See the
[QV methodology and data](https://github.com/unitaryfoundation/clifft-bench/blob/main/docs/qv-multicore.md).

## Measurement scope

The QEC campaign runs each tool version in an isolated pinned environment. A
fresh process performs setup and warmup, followed by five timed samples that
total about 150 seconds per workload and placement. Three fresh placements
reduce the risk that one boot or one point in time determines the current-tool
comparison. Compilation and sampling are recorded separately, and every result
is consumed.

The circuit corpus, software locks, host identity, boot IDs, raw measurements,
and derived comparisons live in
[`clifft-bench`](https://github.com/unitaryfoundation/clifft-bench). Raw JSON is
authoritative. Absolute values belong to their named hardware epoch and are
representative measurements, not performance guarantees.

## Continuous microbenchmarks

Clifft also runs short C++ and Python benchmarks every day on GitHub-hosted
runners. Those checks are useful for spotting implementation-level drift, but
their shared hardware and shorter cases do not make them a source of canonical
absolute performance numbers. See [Benchmark History](../development/benchmark-history.md)
for the dashboards and a direct comparison of the two benchmark layers.

## Historical results

The [original Clifft preprint](https://arxiv.org/abs/2604.27058) and its
companion repository report the earlier localized-Pauli SVM and include GPU and
broad Quantum Volume comparisons. Those results explain the original method,
but they are not measurements of the current symbolic-coordinate sampler.

The [symbolic sampling update](../updates/symbolic-sampling.md) gives the
migration history and a matched comparison of the last legacy SVM, current
Clifft, and SymFT's CPU execution modes.
