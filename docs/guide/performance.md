# Performance

Clifft targets circuits between pure stabilizer simulation and fully dense
statevector simulation: large circuits with mostly Clifford structure, localized
non-Clifford effects, noise, measurements, detectors, and observables.

The main quantity to watch is the peak active width $k$. Non-Clifford operations
can increase $k$, while measurements can reduce it. The active state contains
$2^k$ amplitudes, so both the peak width and the time spent at each width matter.
Total qubit count and non-Clifford gate count alone are poor performance
predictors.

## Choosing a simulator

| Regime | Recommended tool | Why |
|---|---|---|
| Pure Clifford QEC | Stim | Stim packs stabilizer-frame work across shots and remains the better specialist when the active width is always zero. |
| Low-magic or near-Clifford circuits | Clifft | Clifft is most effective when non-Clifford effects stay localized and measurements repeatedly shrink the active state. |
| Dense universal circuits | A statevector simulator or Clifft | When $k=n$, Clifft is exact but has the same exponential state-size limit as dense statevector simulation. |

Clifft is not intended to replace Stim for fully Clifford workloads. Its main
target is the middle regime, where exact non-Clifford effects matter without
forcing the entire physical system into one dense state.

## Representative CPU results

The following results use the current Clifft sampler at commit
[`04c4fe6`](https://github.com/unitaryfoundation/clifft/commit/04c4fe662d9b42d06817450096dbb56a541e709d).
They report median single-threaded CPU performance from 12 balanced process-level
runs. Sampling throughput counts all attempted shots, including shots rejected by
detector postselection. The percentage in parentheses is the relative median
absolute deviation.

| Circuit | Qubits | Peak $k$ | Compile | Attempted shots/s |
|---|---:|---:|---:|---:|
| Surface code `d=7, r=7`, $p=10^{-3}$ | 118 | 0 | 12.2 ms | 453k (1.3%) |
| Cultivation `d=3` | 15 | 4 | 1.38 ms | 1.07M (0.5%) |
| Cultivation `d=5` | 42 | 10 | 11.3 ms | 125k (3.6%) |
| Distillation | 85 | 5 | 2.18 ms | 534k (1.2%) |
| Coherent surface code `d=3, r=1` | 26 | 4 | 0.360 ms | 1.33M (0.9%) |
| Coherent surface code `d=3, r=3` | 26 | 7 | 0.634 ms | 394k (1.0%) |
| Coherent surface code `d=5, r=1` | 64 | 12 | 1.24 ms | 14.1k (1.2%) |

The coherent `d=5, r=5` circuit reaches peak width 22. Under the explicit reference
postselection used here, every attempted shot was discarded and each process
completed only a handful of shots. It is therefore retained as a setup and API
guard in the campaign, not reported as a meaningful throughput result.

## Reading the active-width regimes

### Active width zero

At active width zero, coefficient-array work disappears, but Clifft still evaluates
the circuit's sampling actions and symbolic expressions. Stim has an architectural
advantage in this regime: a Clifford-only engine can pack work from many shots into
each SIMD instruction, while Clifft retains machinery that also supports
non-Clifford variants.

### Low and moderate active width

Cultivation and distillation are Clifft's main target regime. Their physical
circuits contain many more qubits than the active state, and measurements can
remove active coordinates during execution. Clifft therefore performs dense work
over the smaller $2^k$ active state instead of the full $2^n$ physical state.

The same peak width does not imply the same throughput. Circuit length, the width
at each action, measurement and detector work, postselection timing, and the shape
of active Pauli operations all affect the amount of work per attempted shot.

### Larger active width and the dense limit

As $k$ grows, coefficient-array work dominates and Clifft approaches ordinary
dense-statevector scaling. A controlled width-12 diagnostic with 512 active
operations reached 62.9k attempted shots/s on the benchmark host. This diagnostic
isolates sustained active-state work; it is not intended to represent an application
workload.

Quantum Volume circuits provide dense execution guards rather than a broad
cross-simulator claim in this campaign. The current sampler reached 19.5k attempted
shots/s on the checked QV-10 fixture. QV performance is sensitive to the generated
circuit's mix of fused, directly vectorized, and scalar operations, so one fixture
should not be treated as a universal QV scaling result. QV-20 took roughly 0.2
seconds per shot and completed only four shots per timed process, which is too few
for a precise rate comparison.

## ISA portability check

The Linux wheels use an `x86-64-v2` baseline and select ISA-specific kernels at
runtime. The primary table forced AVX-512 so that its execution target was
unambiguous. In a separate eight-round forced-AVX2 check, the real QEC workload
medians ranged from 0.97x to 1.16x their AVX-512 rates. This is a portability guard,
not evidence that one ISA is universally faster: individual circuits exercise
different mixtures of specialized and scalar kernels.

## Methodology

The campaign ran on one pinned core of an AMD EPYC 9554P (Zen 4) KVM host. Clifft
used GCC 13.3, CMake `Release` (`-O3 -DNDEBUG`), an `x86-64-v2` baseline, and
runtime ISA dispatch. Each timed cell ran in a fresh process with one thread, a
warmup excluded from timing, and enough shots to target approximately 1.5 seconds.
The 12 rounds used distinct seeds, and the process schedule was balanced to reduce
temporal drift.

Each circuit used an explicit noiseless detector and observable reference.
Postselected circuits used the same detector mask. The benchmark view retained only
observable 0; QV fixtures without an observable declaration used the final measured
bit as observable 0. Sampling returned survivor counts rather than raw records, and
every result was consumed. Compilation and sampling were timed separately; the
reference calculation was excluded from compilation time.

The QEC circuits come from the
[Clifft paper corpus at `db7dc9f`](https://github.com/unitaryfoundation/clifft-paper/tree/db7dc9f13a2c2854690e92390c779048a1ac1400/qec_bench).
The QV fixtures are pinned in the Clifft repository at the measured commit.

These numbers are representative benchmark points, not performance guarantees.
Hardware, compiler, ISA, noise model, postselection rate, and circuit structure all
matter.

## Cross-simulator benchmarks

The public [clifft-bench](https://github.com/unitaryfoundation/clifft-bench)
project is developing the canonical, reproducible comparison of Clifft with other
near-Clifford simulators. Its initial phase is CPU-only and records Clifft and
SymFT compilation, warmup, correctness, and sampling separately on an immutable
circuit corpus. Stim comparisons belong on compatible Clifford workloads; GPU
simulators require a separately scoped hardware comparison. In particular, the
published Tsim results used GPU hardware and should not be mixed into a CPU-only
table.

Until that project selects its canonical host and publishes result sets, this page
focuses on Clifft's own performance model and representative current measurements
instead of maintaining a second cross-simulator leaderboard.

## Historical results

The [original Clifft preprint](https://arxiv.org/abs/2604.27058) and its companion
repository report the earlier localized-Pauli SVM and include GPU and broad
Quantum Volume comparisons. Those results remain useful for understanding the
original method, but they should not be read as measurements of the current
symbolic-coordinate sampling pipeline.

The [symbolic sampling update](../updates/symbolic-sampling.md) gives the
release-oriented migration history and a matched comparison of the last legacy
SVM, current Clifft, and SymFT's CPU execution modes.
