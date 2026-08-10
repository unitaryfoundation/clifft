# Post-#289/#290 backend inventory

This checkpoint measures commit `47506231a9d382f276f6d118c4d4a82c57e589cc`
after the forward planner and lane-paired direct-rotation kernel merged. It is a
measurement report, not an implementation proposal.

## Method

- AMD EPYC 9554P VM, one pinned core, `OMP_NUM_THREADS=1`.
- Release (`-O3 -DNDEBUG`) with debug symbols and frame pointers.
- Native results use runtime AVX-512 dispatch. The portability slice uses a
  separate wheel-like `x86-64-v2` build with `CLIFFT_FORCE_ISA=avx2`.
- Legacy and symbolic programs use the same default HIR optimization,
  noiseless detector/observable normalization, postselection mask, shot count,
  and output mode.
- Execution medians come from three paired blocks unless noted. Each block uses
  a new seed shared by backend positions. Checksums consume outputs; differing
  checksums are expected because RNG schedules differ.
- Ratios are symbolic / legacy, so values below one favor symbolic.

## Native AVX-512 execution

### Paper and controlled workloads

| case | shots | legacy s | symbolic s | ratio |
|---|---:|---:|---:|---:|
| surface d7 r7 | 300,000 | 0.996 | 0.661 | 0.66 |
| cultivation d3 | 500,000 | 0.665 | 0.468 | 0.70 |
| cultivation d5 | 100,000 | 1.564 | 0.771 | 0.49 |
| distillation | 300,000 | 3.136 | 0.573 | 0.18 |
| coherent d3 r1 | 1,000,000 | 0.741 | 0.679 | 0.92 |
| coherent d3 r3 | 300,000 | 0.859 | 0.748 | 0.87 |
| coherent d5 r1 | 10,000 | 0.675 | 0.812 | **1.20** |
| coherent d5 r5 setup/slow run | 3 | 1.972 | 0.791 | 0.40 |
| k12/L512 | 20,000 | 0.924 | 0.460 | 0.50 |
| EXP_VAL 20q/200 probes | 100,000 | 0.185 | 0.144 | 0.78 |

The other committed benchmark shapes also favor symbolic: target QEC is
0.83x, deep Clifford 50q/5000 is 0.11x, surface d5 r5 at p=0.05 is 0.64x,
and surface d11 r11 at p=0.001 is 0.44x.

### QV seed sensitivity

| QV-10 seed | legacy s, 5k shots | symbolic s | ratio |
|---:|---:|---:|---:|
| 42 | 0.252 | 0.265 | **1.05** |
| 43 | 0.316 | 0.224 | 0.71 |
| 44 | 0.209 | 0.236 | **1.13** |
| 45 | 0.239 | 0.214 | 0.89 |
| 46 | 0.279 | 0.238 | 0.85 |
| 47 | 0.309 | 0.240 | 0.78 |
| 48 | 0.319 | 0.253 | 0.79 |
| 49 | 0.288 | 0.235 | 0.82 |
| 50 | 0.309 | 0.255 | 0.83 |
| 51 | 0.332 | 0.262 | 0.79 |

Symbolic wins eight of ten QV-10 circuits; the median ratio across seeds is
about 0.82. QV-20 also favors symbolic for all three checked circuits: seed 42
is 0.79x, seed 47 is 0.76x, and seed 49 is 0.82x.

The current QV-10 seed-44 profile is 35.3% AVX-512 fused rotation, 35.1%
scalar fused rotation, 13.8% AVX-512 direct rotation, 6.1% measurement
probability, and 3.5% collapse. The remaining tail is therefore a low-pivot
fused-kernel coverage issue, not expression evaluation or dispatch.

## Importance sampling is a real legacy win

| case | legacy s | symbolic s | ratio |
|---|---:|---:|---:|
| surface d7 r7 forced k=0, 100k | 0.806 | 0.382 | 0.47 |
| surface d7 r7 forced k=4, 100k | 0.219 | 0.726 | **3.32** |
| surface d7 r7 forced k=8, 100k | 0.194 | 0.812 | **4.20** |
| surface d7 r7 forced k=12, 100k | 0.192 | 0.868 | **4.52** |
| cultivation d5 forced k=4, 20k | 0.183 | 0.253 | **1.38** |

The profile and executor structure identify the mechanism. Symbolic currently
activates and propagates every selected quantum-fault symbol before entering
the action stream. Most forced-k surface shots discard early, but they have
already paid downstream true-symbol fanout for faults that occur after that
postselection point. Legacy consumes its forced sites in circuit order and
therefore benefits from early termination. At surface k=8, symbolic spends
81.7% self time in `Executor::run_shot(KFaultSampler&)`; forced assignment and
its visible callees account for additional work before ordinary actions begin.

A focused pre-batching fix is to consume selected quantum sites at their
existing prepared noise boundaries, just as readout sites are consumed in
circuit order. Initial pre-action sites still need assignment before dispatch.
This should preserve the current k=0 win while restoring early-postselection
work avoidance.

## Noncomputational path

Five paired blocks on the committed d17 r5 benchmark give:

| case | legacy s | symbolic s | ratio |
|---|---:|---:|---:|
| lossless, 2,000 shots | 0.00276 | 0.00289 | **1.05** |
| p=0.01 leakage/loss, 100 shots | 0.0422 | 0.274 | **6.49** |

The symbolic leakage profile is primarily execution, not the planner:

- measurement probability: 31.3%;
- instrument no-fire transform: 17.3%;
- measurement collapse: 11.0%;
- continuation-planning tableau work: roughly 20% in aggregate.

Eliminating continuation planning cannot close a 6.5x gap. The next broad
active-state target is measurement/instrument probability and collapse,
preferably with the same prepared-Pauli shape specialization and ISA boundary
used by rotations. This also attacks the roughly 25.5% measurement/collapse
share in coherent d5 r1.

## Portable x86-64-v2 / AVX2 slice

The symbolic rotation specializations are currently AVX-512-only. A separate
`x86-64-v2` Release build confirms that this is a deprecation blocker rather
than only a theoretical portability concern:

| case | shots | legacy AVX2 s | symbolic s | ratio |
|---|---:|---:|---:|---:|
| surface d7 r7 | 100,000 | 0.309 | 0.219 | 0.71 |
| cultivation d5 | 20,000 | 0.310 | 0.349 | **1.13** |
| distillation | 100,000 | 1.052 | 0.193 | 0.18 |
| coherent d3 r3 | 100,000 | 0.299 | 0.384 | **1.28** |
| coherent d5 r1 | 3,000 | 0.326 | 0.627 | **1.92** |
| QV-10 seed 44 | 2,000 | 0.0899 | 0.308 | **3.43** |
| QV-20 seed 47 | 1 | 0.337 | 1.056 | **3.13** |
| k12/L512 | 5,000 | 0.236 | 0.212 | 0.90 |

Thus the native AVX-512 results are not sufficient to remove legacy for the
wheel population. The proven fused and direct rotation shapes need AVX2
implementations, and Apple Silicon needs its own performance validation, before
the symbolic backend can claim broad execution parity.

## Planner after #289

| case | legacy total ms | symbolic total ms | symbolic plan ms | prepare ms |
|---|---:|---:|---:|---:|
| surface d7 r7 | 6.94 | 48.67 | 43.22 | 1.80 |
| cultivation d3 | 0.69 | 2.05 | 1.50 | 0.20 |
| cultivation d5 | 4.40 | 21.70 | 17.19 | 2.46 |
| distillation | 2.63 | 19.26 | 18.26 | 0.12 |
| coherent d3 r3 | 0.56 | 1.71 | 1.40 | 0.02 |
| coherent d5 r1 | 1.14 | 7.60 | 6.93 | 0.03 |
| coherent d5 r5 | 3.48 | 28.02 | 25.69 | 0.09 |
| QV-10 seed 44 | 2.36 | 2.15 | 0.64 | 1.08 |
| QV-20 seed 47 | 10.89 | 17.27 | 3.57 | 11.32 |
| k12/L512 | 2.61 | 4.95 | 2.85 | 0.13 |

The planner is not exhausted, but it is no longer the first throughput target.
Surface and distillation profiles are dominated by Stim tableau scatter,
invariant checks, bit access, and symbolic-frame application rather than the
suffix rewriting removed by #289. QV-20 is different: fused-run preparation is
39.8% of total compile cycles and explains most of the 11.7 ms preparation
stage. These are worthwhile compile-latency targets later, especially for
repeated continuations, but their absolute tens-of-milliseconds cost is
amortized in the current many-shot runs. In the leakage profile, all observed
continuation-planning work is only about one fifth of time.

## Component factoring check

SOFT's planner enables components only for coherent d5 r1 and d5 r5 in this
corpus. Its estimated dense/component coefficient work is 229,402/95,936 for
d5 r1 and 640.5M/156.5M for d5 r5. It rejects components for cultivation d5
(77,610/82,960), QV-10 and QV-20 (nearly equal work), and the other small-width
paper workloads. k12/L512 has a lower raw component estimate (65,534/37,676)
but still fails SOFT's profitability gate.

Pinned SOFT single-shot execution with corrected reference normalization is
slower than Clifft symbolic on surface, cultivation d5, distillation, and
coherent d3 r3. It is 2.5x faster only on coherent d5 r1, where components are
enabled. This supports deferring component factoring as a larger, specialized
feature instead of treating it as the next general optimization.

## Recommendation

1. **Fix circuit-order forced-fault activation first.** It is narrow,
   pre-batching, and directly explains the largest ordinary API regression.
2. **Prototype active measurement/instrument kernels next.** Use coherent d5
   r1 and noncomputational d17 r5 as primary cases, with QV and k12 guards.
3. **Port the proven rotation kernels to AVX2 before deprecating legacy.** This
   is independent of batching and required for the wheel baseline. Validate
   Apple Silicon separately.
4. **Optionally add low-pivot fused U2/U4 SIMD only if eliminating the two
   QV-10 tail losses is a release goal.** It is not the broadest current win.
5. **Then re-profile and decide on batching.** Return to tableau/planner and
   QV fused-preparation latency when compile-once or continuation latency is
   the active product constraint.

On AVX-512, ordinary `sample`/survivor execution is already ready from a
performance perspective on nearly all measured circuits. Full legacy
deprecation is premature: importance sampling, leakage trajectories, and
AVX2-only active-state workloads still have material measured legacy wins.
None of those three requires batching to investigate or improve.
