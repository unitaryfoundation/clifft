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

## Release-target CPU results

The following results use the current Clifft sampler at commit
[`04c4fe6`](https://github.com/unitaryfoundation/clifft/commit/04c4fe662d9b42d06817450096dbb56a541e709d).
They report median single-threaded CPU performance from 12 balanced process-level
runs. Sampling throughput counts all attempted shots, including shots rejected by
detector postselection. The percentage in parentheses is the relative median
absolute deviation.

| Circuit | Qubits | Peak $k$ | Compile | Attempted shots/s | Versus legacy SVM |
|---|---:|---:|---:|---:|---:|
| Surface code `d=7, r=7`, $p=10^{-3}$ | 118 | 0 | 12.2 ms | 453k (1.3%) | 1.53x |
| Cultivation `d=3` | 15 | 4 | 1.38 ms | 1.07M (0.5%) | 1.46x |
| Cultivation `d=5` | 42 | 10 | 11.3 ms | 125k (3.6%) | 1.86x |
| Distillation | 85 | 5 | 2.18 ms | 534k (1.2%) | 5.86x |
| Coherent surface code `d=3, r=1` | 26 | 4 | 0.360 ms | 1.33M (0.9%) | 0.98x |
| Coherent surface code `d=3, r=3` | 26 | 7 | 0.634 ms | 394k (1.0%) | 1.14x |
| Coherent surface code `d=5, r=1` | 64 | 12 | 1.24 ms | 14.1k (1.2%) | 0.93x |

The historical column compares the current sampler with the last Clifft revision
whose default was the legacy SVM, using matched circuit, reference, postselection,
shot, and output contracts. Ratios above one favor the current sampler. It wins on
five of the seven real workloads, is within 2% on coherent `d=3, r=1`, and is 7%
slower on coherent `d=5, r=1` on this AVX-512 host.

The coherent `d=5, r=5` circuit reaches peak width 22. Under the matched reference
postselection used here, every attempted shot was discarded and each process
completed only a handful of shots. It is therefore retained as a setup and API
guard in the raw campaign, not reported as a meaningful throughput result.

## Reading the active-width regimes

### Active width zero

The surface-code result shows that the current sampler substantially reduced the
fixed overhead present in the legacy SVM. Stim still has an architectural
advantage: a Clifford-only engine can pack work from many shots into each SIMD
instruction, while Clifft retains machinery that also supports non-Clifford
variants.

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
dense-statevector scaling. A controlled width-12 diagnostic improved 2.65x over
the legacy SVM, showing that the current SIMD kernels help even outside the small
active-width regime.

Quantum Volume circuits provide dense execution guards rather than a broad
cross-simulator claim in this campaign. The current sampler reached 19.5k
attempted shots/s on the checked QV-10 fixture, 0.88x the legacy AVX-512 result.
QV performance is sensitive to the generated circuit's mix of fused, directly
vectorized, and scalar operations, so one fixture should not be treated as a
universal QV scaling result. QV-20 took roughly 0.2 seconds per shot; the resulting
five shots/s median has too few observations for a precise comparison.

## ISA portability check

The Linux wheels use an `x86-64-v2` baseline and select ISA-specific kernels at
runtime. The primary table forced AVX-512 so that its execution target was
unambiguous. A separate eight-round forced-AVX2 check reproduced the current
sampler's QEC wins: 1.50x on surface `d=7`, 1.44x and 1.99x on cultivation `d=3`
and `d=5`, 5.81x on distillation, 1.38x on coherent `d=3, r=3`, and 1.69x on
coherent `d=5, r=1`, all relative to the legacy SVM under AVX2. QV-10 was at
0.98x parity.

## Methodology and raw data

The campaign ran on one pinned core of an AMD EPYC 9554P (Zen 4) KVM host. Clifft
used GCC 13.3, CMake `Release` (`-O3 -DNDEBUG`), an `x86-64-v2` baseline, and
runtime ISA dispatch. Each timed cell ran in a fresh process with one thread, a
warmup excluded from timing, and enough shots to target approximately 1.5 seconds.
Backend order rotated each round, and a new seed was paired across backend
positions.

Every implementation received the same explicit noiseless detector and observable
reference. Postselected circuits used the same detector mask. The benchmark view
retained only observable 0; QV fixtures without an observable declaration used the
final measured bit as observable 0. Sampling returned survivor counts rather than
raw records, and every result was consumed. Compilation and sampling were timed
separately; the shared reference calculation was excluded from compilation time.

The full campaign also includes the corrected SymFT CPU implementation at
[`c89b985`](https://github.com/haoliri0/SOFT/commit/c89b98514a919240b8afa53a271e08d926d3c987),
in both non-batched and batched modes. Batched SymFT is an architectural throughput
comparison, not the same execution model as Clifft's current per-shot sampler.

- [Summary, revisions, hashes, and dispersion](../assets/benchmarks/symbolic-sampler-2026-08/summary.json)
- [All primary AVX-512 samples](../assets/benchmarks/symbolic-sampler-2026-08/raw-avx512.json)
- [All forced-AVX2 samples](../assets/benchmarks/symbolic-sampler-2026-08/raw-avx2.json)
- [Canonical paper benchmark corpus](https://github.com/unitaryfoundation/clifft-paper/tree/db7dc9f13a2c2854690e92390c779048a1ac1400/qec_bench)

These numbers are representative benchmark points, not performance guarantees.
Hardware, compiler, ISA, noise model, postselection rate, and circuit structure all
matter.

## Historical results

The [original Clifft preprint](https://arxiv.org/abs/2604.27058) and its companion
repository report the earlier localized-Pauli SVM and include GPU and broad
Quantum Volume comparisons. Those results remain useful for understanding the
original method, but they should not be read as measurements of the current
symbolic-coordinate sampling pipeline.
