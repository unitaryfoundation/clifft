# Post-merge backend checkpoint

## Scope

This checkpoint identifies the remaining performance opportunities after the
incremental expression evaluator in #281 and the scalar Pauli-rotation kernel
in #282 merged.

- Clifft: `b80225c911f736fef56bd2fddbac389c6621cdbf`.
- Paper corpus: `db7dc9f13a2c2854690e92390c779048a1ac1400`.
- Build: one Release configuration with `-O3 -DNDEBUG -g
  -fno-omit-frame-pointer`, native ISA, and fast math.
- Runtime: one OpenMP thread pinned to CPU 3.
- Timing: three balanced blocks, six fresh-process samples per backend, paired
  seeds, matched circuit and output semantics.
- Profiling: `cycles:u` at 999 Hz with frame-pointer call graphs and zero lost
  samples. Counter tables contain three executions per arm.

The A/A control was not repeated. The earlier same-machine controls bounded
ordinary timing noise to approximately five percent, while every regression
profiled here is near tenfold. Checksums prove that outputs were consumed;
outcome parity uses aggregate discard and logical statistics rather than
cross-backend checksum equality.

## Compact checkpoint

`symbolic/legacy` above one means the symbolic backend is slower.

| Case | Legacy | Symbolic | Ratio | Absolute gap |
|---|---:|---:|---:|---:|
| surface d7 r7 aggregate, 200k shots | 0.549 s | 0.410 s | 0.754x | -0.139 s |
| cultivation d3 aggregate, 500k shots | 0.610 s | 0.525 s | 0.861x | -0.085 s |
| distillation aggregate, 200k shots | 1.928 s | 0.375 s | 0.195x | -1.553 s |
| QV-10 raw, 10k shots | 0.518 s | 4.513 s | 8.896x | +3.994 s |
| QV-20 raw, one shot | 0.240 s | 2.452 s | 10.028x | +2.212 s |
| noncomputational d17 r5 lossless, 500k shots | 0.599 s | 0.432 s | 0.722x | -0.167 s |
| noncomputational d17 r5 low leak, 1k shots | 0.407 s | 4.013 s | 9.828x | +3.606 s |

Surface discard was 91.19% for legacy and 91.20% for symbolic. Cultivation
discard was 31.36% and 31.31%; distillation was 92.02% and 92.04%. Logical
statistics remained consistent within uncertainty.

The expression-heavy real workloads remain at parity or better. QV is now the
largest ordinary sampling gap. Lossless noncomputational sampling winning
while low-leak sampling loses separates the base driver from transition,
continuation, and instrument work.

## QV attribution

### Counters and hot paths

QV-20 takes about three seconds to compile in either backend, so its sampling
counters subtract a separate matched compile run from the full process and
divide the remainder across the two warm-up shots plus one timed shot.

| Case | Cycle ratio | Instruction ratio | Branch ratio | IPC legacy/symbolic |
|---|---:|---:|---:|---:|
| QV-10 | 8.680x | 15.651x | 107.214x | 3.29 / 5.93 |
| QV-20, compile-adjusted | 11.701x | 25.018x | 215.446x | 2.42 / 5.17 |

The symbolic backend again sustains higher IPC but performs much more work.
QV-10 spends 97.20% of symbolic self cycles in `sampling::apply_rotation()`.
Legacy spends 75.91% in its fused AVX-512 U4 kernel and 15.25% in U2. The
sample-dominated QV-20 symbolic record spends 79.61% of whole-process cycles
in `apply_rotation()`; the remaining record share includes the common compile
and reference preamble, so rotation accounts for effectively all sampled
execution. The sample-dominated legacy record is led by U4 and U2 kernels.

### Work-count explanation

| Metric | QV-10 | QV-20 |
|---|---:|---:|
| symbolic actions | 937 | 3,723 |
| symbolic rotations | 917 | 3,683 |
| symbolic predicted dense passes | 937 | 3,723 |
| symbolic predicted coefficient visits | 577,361 | 2,992,955,803 |
| legacy instructions | 130 | 398 |
| legacy `ARRAY_ROT` | 10 | 20 |
| legacy `ARRAY_U2` | 35 | 107 |
| legacy `ARRAY_U4` | 45 | 191 |
| symbolic rotations / legacy dense rotation ops | 10.19x | 11.58x |

The work-count ratios closely match the measured runtime ratios. The scalar
butterfly is no longer the primary problem: symbolic executes every localized
rotation as a separate coefficient sweep, while the SVM's single-axis and tile
fusion passes combine long runs into U2 and U4 sweeps.

Dynamic signs do not block a useful first prototype. QV-10 has 191 dynamic-sign
rotations, but they represent only 4.75% of predicted dense visits. QV-20 has
353, representing only 0.65%. A scratch projection that considers only
constant-sign consecutive rotations, bounds each fused orbit to rank two, and
uses the SVM's minimum-three-array-operation threshold removes 89.40% of QV-10
rotation coefficient visits and 93.54% of QV-20 visits.

### Recommended QV prototype

Add one focused execution-lowering transformation, not a general pass
framework:

1. In `ExecutablePlan` lowering, find consecutive constant-sign rotations at
   the same active width whose X masks span rank one or two.
2. Require at least three rotations, following the existing SVM fusion
   threshold.
3. Precompute the resulting 2x2 or 4x4 matrix and a bounded orbit descriptor.
4. Apply it in one coefficient sweep using the applicable SVM alignment,
   explicit-complex-arithmetic, and ISA-dispatch patterns.
5. Leave dynamic-sign rotations on the current scalar path initially.

Primary acceptance cases are QV-10 raw and QV-20 raw. Coherent d3 r3 and
k12/L512 guard the scalar path; cultivation and distillation guard real
end-to-end wins. Only consider dynamic-sign fusion after this isolated
prototype is measured.

## Compilation

| Case | Legacy | Symbolic | Ratio | Absolute gap |
|---|---:|---:|---:|---:|
| surface d7 r7 | 9.315 ms | 447.110 ms | 47.997x | +437.794 ms |
| cultivation d3 | 0.976 ms | 18.826 ms | 19.297x | +17.850 ms |
| distillation | 3.545 ms | 28.685 ms | 8.092x | +25.140 ms |
| QV-10 | 3.449 ms | 3.824 ms | 1.109x | +0.375 ms |
| QV-20 | 3.063 s | 3.062 s | 1.000x | -0.0005 s |

The surface gap is unchanged. Current symbolic surface compilation spends
98.39% inclusively in `plan_sampling()`, 70.02% in
`transform_future_operations()`, and 71.57% under measurement processing.
Major self costs are Stim tableau scatter at 32.89%, conditional-Pauli
propagation at 18.02%, affine XOR at 5.78%, allocation at 5.34%, and `memmove`
at 4.76%.

This remains a real second optimization track. QV-20 compile parity also shows
that it is not a general frontend problem: the gap appears on output-heavy QEC
plans whose remaining suffix is repeatedly transformed across measurements.

## Noncomputational sampling

The refreshed low-leak counter ratios are 9.021x cycles, 13.295x instructions,
and 13.156x branches. The attribution reproduces the earlier result:

- continuation compilation: 57.74% inclusive;
- `plan_sampling()`: 56.32% inclusive;
- future-operation transformation: 43.42% inclusive;
- `Executor::run_shot()`: 39.68% inclusive;
- measurement probability: 21.60% self;
- Stim tableau scatter: 16.59% self;
- instrument no-fire application: 11.28% self;
- measurement collapse: 7.80% self;
- `memmove`: 7.11% self.

Eliminating continuation compilation has only a 2.37x Amdahl ceiling and
cannot close the 9.83x gap alone. Planner/suffix work should be measured as a
shared compile and continuation opportunity; measurement, collapse, and
instrument kernels remain an independent later track.

## Priority

1. Prototype constant-sign rank-one/rank-two symbolic rotation fusion for QV.
2. Investigate repeated suffix transformation for surface compilation and
   continuation planning.
3. Re-profile low-leak noncomputational execution after planner work, then
   isolate measurement/collapse/instrument improvements.

The residual coherent gap is smaller than these targets. Packed expressions,
dynamic-sign fusion, and broad component fusion are not justified before the
focused QV result is measured.

Raw checkpoint output and perf data were kept outside the repository under
`/tmp/clifft-issue280-postmerge-checkpoint` and
`/tmp/clifft-issue280-postmerge-profiles`. The committed harness, circuit
hashes, fixed shot counts, and commands in the protocol make the measurements
reproducible without checking in machine-specific binary profiles.
