# Issue 280 active-width follow-up

Date: 2026-08-09

This is a measurement checkpoint, not an implementation proposal. It refreshes
the active-width evidence after the scalar rotation specialization, constant
rotation fusion, and AVX-512 fused-U4 specialization merged to `main`.

## Reproduction context

- Clifft: `48b84ff024471c102b80df9943312ab272ff9197`
- QV generator: `unitaryfoundation/clifft-paper` at
  `db7dc9f13a2c2854690e92390c779048a1ac1400`
- Qiskit: 2.3.1 from the paper benchmark's frozen environment
- QV widths and circuit seeds: 10 and 20 qubits, seeds 42 through 51
- Build: Release, `-O3 -DNDEBUG -g -fno-omit-frame-pointer`, native CPU
  baseline, fast-math, runtime ISA `avx512`
- Host: AMD EPYC 9554P; one OpenMP thread pinned to CPU 3
- Timings: one warm-up batch, then three batches with paired sampling seeds;
  selected reverse-order checks used five batches
- Profiles: `cycles:u`, 999 Hz, frame-pointer call graphs, zero lost samples

The scratch census ran `parse`, `trace`, the default HIR passes, reference
normalization, and `plan_sampling()`. It reproduced the production
`prepare_fused_rotation_run()` lowering loop. Coefficient visits are weighted
by `2^active_width`; identity rotations count as actions but not coefficient
visits. Direct-rotation pairing pivots use `PreparedPauli::pair_selector`, the
highest set X bit.

## QV circuit sensitivity

The QV-10 result is topology-sensitive. A single circuit is not a reliable
summary, although every circuit contains the same three execution buckets.

| Width | Symbolic / legacy median across circuits | Per-circuit range | AVX fused visits | Scalar fused visits | Direct visits |
|---|---:|---:|---:|---:|---:|
| QV-10, 5,000 shots | 1.028x | 0.814x-1.597x | 39.6%-73.5%, median 62.2% | 1.3%-10.7%, median 5.8% | 15.7%-51.7%, median 32.0% |
| QV-20, 1 shot | 0.749x | 0.492x-1.118x | 72.9%-96.5%, median 87.5% | 0.1%-4.3%, median 1.0% | 2.9%-23.2%, median 11.9% |

QV-10 AVX coverage and symbolic/legacy runtime have a -0.904 Pearson
correlation across these ten circuits. This is a prioritization signal, not a
performance model: legacy work also changes with circuit topology.

| QV-10 seed | Legacy seconds | Symbolic seconds | Ratio | AVX visits | Scalar fused | Direct |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.249 | 0.335 | 1.346x | 45.1% | 8.2% | 46.7% |
| 43 | 0.296 | 0.285 | 0.960x | 70.4% | 3.2% | 26.4% |
| 44 | 0.199 | 0.318 | 1.597x | 39.6% | 8.8% | 51.7% |
| 45 | 0.253 | 0.276 | 1.091x | 62.4% | 4.6% | 33.0% |
| 46 | 0.275 | 0.284 | 1.035x | 65.0% | 3.9% | 31.0% |
| 47 | 0.311 | 0.271 | 0.873x | 62.1% | 7.0% | 30.9% |
| 48 | 0.309 | 0.280 | 0.907x | 73.1% | 7.4% | 19.5% |
| 49 | 0.281 | 0.311 | 1.107x | 56.4% | 1.3% | 42.3% |
| 50 | 0.319 | 0.326 | 1.020x | 54.4% | 3.4% | 42.1% |
| 51 | 0.333 | 0.271 | 0.814x | 73.5% | 10.7% | 15.7% |

Reversing backend order for seeds 42, 44, 48, and 51 preserved the ordering
and direction: the ratios were 1.308x, 1.402x, 0.917x, and 0.822x.

QV-20's one-shot contract includes executor/state construction and is noisy as
a kernel diagnostic. Three-shot batches made the selected results stable:
seed 47, with 96.5% AVX visits, was 0.708x legacy; seed 49, with the largest
direct share at 23.2%, was 0.883x legacy. Seeds 48 and 50 were 0.625x and
0.829x legacy. Retain the one-shot case as an API guard, but use a short
multi-shot batch when attributing dense-kernel throughput.

Fusion reduces the direct-lowering rotation visits by 88.4%-93.2% on QV-10
and 92.3%-93.7% on QV-20. No circuit hit selector-overflow fallback. The
remaining direct QV work is almost entirely SIMD-friendly under the existing
eight-lane cutoff:

| Direct shape, as share of all executed rotation visits | QV-10 median | QV-20 median |
|---|---:|---:|
| Diagonal | 17.5% | 6.6% |
| Non-diagonal, pairing pivot at least 3 | 14.2% | 5.2% |
| Non-diagonal, pairing pivot below 3 | 0.5% | 0.0% |

Dynamic-sign actions supply a median 28.2% of all QV-10 rotation visits and
9.6% of QV-20 visits. A dynamic sign prevents a fixed fused matrix, but it
does not prevent a prepared direct-rotation SIMD kernel: evaluate the sign
once per action and select the sine sign before entering the coefficient loop.

## Representative profiles

### QV-10 seed 44

This is the largest observed QV-10 regression and has the lowest AVX coverage.

| Symbolic self cost | Share |
|---|---:|
| Direct `apply_rotation` | 34.13% |
| Scalar fused rotation | 28.82% |
| AVX-512 fused U4 | 26.37% |
| Measurement probability | 5.28% |
| Measurement collapse | 1.51% |

The scalar fused path consumes 28.82% of cycles for only 8.8% of executed
rotation visits. Normalized by visits, it costs about 5x as much as either the
direct specialized scalar path or the AVX-512 fused path in this profile. Its
visits are 4,008 rank-two low-pivot and 928 rank-one coefficients.

The legacy profile is 71.23% AVX-512 U4, 20.37% AVX-512 U2, and 6.76% executor.
Symbolic uses 1.542x the cycles and 2.394x the instructions, despite higher IPC
(4.58 versus 2.95). This is direct evidence that the unresolved issue is
vectorized kernel coverage, not dispatch or expression evaluation.

### Coherent d3 r3

At 300,000 attempted shots the current symbolic/legacy median is 1.133x,
improved from the earlier pre-specialization 1.5x-class profile. All 4,434
coefficient visits are dynamic-sign direct rotations: 3,376 use pairing pivot
at least 3 and 1,058 use a lower pivot.

| Symbolic self cost | Share |
|---|---:|
| Direct `apply_rotation` | 55.77% |
| Measurement probability | 14.16% |
| Measurement collapse | 6.80% |
| Executor loop | 6.67% |
| Promotion | 2.90% |
| Expression lookup | 2.86% |

The generated direct-rotation assembly is scalar (`vmovsd` and scalar FMA),
including the regular high-pivot case. Symbolic uses 1.125x the cycles and
1.220x the instructions while sustaining higher IPC (4.93 versus 4.55).

### Regime k12 L512

At 20,000 shots symbolic is now 0.769x legacy. It remains useful as an active
kernel guard rather than a regression target. All 26,624 rotation visits are
direct: 18,432 high-pivot and 8,192 low-pivot; 24,576 come from short constant
runs and 2,048 from a dynamic sign.

Rotation is 54.05% of symbolic cycles. Measurement probability and collapse
are another 17.38% and 12.29%. Thus it will still respond to active-kernel
work, but an optimization must preserve the existing end-to-end win.

## Recommended active-kernel sequence

1. Add a separately compiled AVX-512 direct-rotation specialization for
   diagonal descriptors and non-diagonal descriptors with pairing pivot at
   least 3. Keep `PreparedRotation` and scalar execution ISA-neutral, prepare
   optional lane permutations and parity signs outside dispatch, and pass the
   per-shot sign into the kernel. These two shapes cover essentially all
   direct QV visits, 76% of coherent rotation visits, and 69% of k12 rotation
   visits. A 2x direct-kernel improvement has Amdahl ceilings of about 1.21x
   for QV-10 seed 44 and 1.39x for coherent d3 r3.

2. Extend fused SIMD coverage to rank-two low-pivot U4 and rank-one U2. This
   is narrower but disproportionately expensive: matching the present AVX-512
   cost per visit projects roughly a 1.30x end-to-end improvement for QV-10
   seed 44. Keep this independent from direct rotation SIMD so attribution is
   clear.

3. Re-profile before broadening rotation shapes. If direct low-pivot work is
   still material, add it next. Otherwise move to active measurement
   probability and collapse, already 21% of coherent and 30% of k12 cycles.

AVX2 and Apple-specific kernels remain necessary portability follow-ups, but
the AVX-512 experiments can establish descriptor shape and attainable
throughput first. OpenMP and packed-shot work are independent of these
single-state kernel gaps.

Suggested acceptance set for the next prototype:

- QV-10 seeds 42, 44, and 51: mixed, regression, and winning topologies.
- QV-20 seeds 47 and 49 in both one-shot and three-shot batches: high-AVX and
  high-direct guards.
- Coherent d3 r3: primary dynamic-sign direct-rotation target.
- Regime k12 L512: short constant-run and active-measurement guard.
- Forced scalar/AVX2 ISA smoke tests: fallback safety, not performance gates.
