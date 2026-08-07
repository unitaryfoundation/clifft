# Scalar incremental expression A/B

Date: 2026-08-07

Branch: `codex/issue-280-incremental-ab`

Base: measurement commit `cf9cec7`, whose production base is main commit
`3fdafa411e8e4eca812ff17df2a8f30c584fdc03`.

## Result

A minimal eager scalar expression evaluator removes the 10-12x symbolic
regressions on surface d7 r7 and cultivation d3 in a same-build A/B. It does
so without packed shots, expression interning, graph rewriting, action fusion,
or SIMD changes. The remaining coherent d3 r3 and k=12/L=512 time is dominated
by active rotation and measurement kernels.

This is a prototype result, not a production-change recommendation. In
particular, the prototype does not support noncomputational continuation plans.
The executor design and continuation behavior need review before any production
implementation is proposed.

## Prototype

The direct arm retains the current per-use affine term scan. The incremental
arm adds only the following mechanism:

1. Prepare a reverse dependency list from each symbol to every expression use
   containing it.
2. Preallocate one byte accumulator for each prepared expression occurrence.
3. At the start of a shot, reset the accumulators to their expression constants.
4. When a symbol is assigned true, toggle its dependent accumulators.
5. Replace an expression term scan with one accumulator lookup.

Expressions are not interned: identical expressions still have separate
accumulators and reverse edges. There is no hybrid cost model and no prospective
graph-optimizer machinery. False symbols do no propagation work.

The build is the same Release configuration used for both arms:
`-O3 -DNDEBUG -g -fno-omit-frame-pointer`, with the existing native ISA and
fast-math settings. Timed runs disable the census counters. Each case uses
three balanced ABBA/BAAB blocks, giving six samples per evaluator and six
paired seeds. The two evaluator positions receive the same seed; later pairs
receive fresh seeds.

## Same-build timing

The ratio is computed within each paired seed and then summarized by its
median. All 42 pairs produced identical consumed-output checksums, passed-shot
counts, and logical-error counts. Equality is expected here because the two
expression evaluators do not change the RNG schedule.

| Case | Shots | Direct median | Incremental median | Incremental/direct | Speedup | Pair range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| surface d7 r7 ordinary | 200,000 | 7.184 s | 0.506 s | 0.0706 | 14.17x | 0.0690-0.0728 |
| surface d7 r7 forced k=0 | 200,000 | 23.308 s | 0.912 s | 0.0392 | 25.48x | 0.0385-0.0413 |
| cultivation d3 ordinary | 500,000 | 7.215 s | 0.638 s | 0.0884 | 11.31x | 0.0870-0.0903 |
| cultivation d3 forced k=0 | 400,000 | 7.120 s | 0.633 s | 0.0896 | 11.17x | 0.0810-0.0983 |
| distillation ordinary | 200,000 | 1.088 s | 0.425 s | 0.3882 | 2.58x | 0.3701-0.3986 |
| coherent d3 r3 ordinary | 300,000 | 1.491 s | 1.458 s | 0.9772 | 1.02x | 0.8911-1.0635 |
| regime k=12/L=512 | 20,000 | 1.204 s | 1.031 s | 0.8568 | 1.17x | 0.8205-0.9325 |

The direct arm in this harness is 2-12% slower than the earlier canonical
symbolic medians, depending on the case. The table is therefore the primary
evidence; applying its paired multiplier to a different run is only an
estimate. As a directional comparison, applying the multipliers to the
canonical symbolic medians predicts symbolic/legacy ratios of approximately
0.75 for surface ordinary, 0.90 for cultivation ordinary, 1.12 for cultivation
k=0, 0.18 for distillation, 1.56 for coherent d3 r3, and 1.09 for k=12/L=512.
This suggests expression evaluation is sufficient to close the large gaps,
but a canonical backend rerun is required before claiming absolute parity.

## Executed-work census

The runtime census counts only expressions reached before early postselection.
The final column is a deliberately simple work proxy:
`direct term visits / (expression resets + true-symbol fanout)`. A byte reset,
a term load, and a reverse-edge toggle do not have equal cost, so it predicts
direction and regime rather than elapsed time.

| Case | Expressions evaluated/shot | Direct terms/shot | True fanout/shot | Resets/shot | Work proxy | Discard fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| surface ordinary | 372.4 | 62,982.9 | 102.8 | 819 | 68.3x | 0.911 |
| surface forced k=0 | 819.0 | 214,342.0 | 109.0 | 819 | 231.0x | 0 |
| cultivation ordinary | 95.7 | 25,257.1 | 5.7 | 112 | 214.5x | 0.313 |
| cultivation forced k=0 | 112.0 | 31,256.0 | 0 | 112 | 279.1x | 0 |
| distillation | 190.6 | 6,171.6 | 213.9 | 225 | 14.1x | 0.921 |
| coherent d3 r3 | 102.2 | 194.1 | 6.2 | 156 | 1.20x | 0.910 |
| regime k=12/L=512 | 566.0 | 18,866.0 | 683.2 | 566 | 15.1x | 0 |

This explains the timing regimes directly. Surface and cultivation have very
large expressions but few true faults. Coherent d3 r3 has so little affine
work that resetting all 156 accumulators nearly equals the avoided 194 term
visits. The k=12 fixture reduces expression work substantially, but dense
active work still sets its total runtime.

## Minimal static census

The static analyzer describes the plan that the executor actually prepares;
for measurements it excludes the branch symbol removed by executor lowering.

| Case | Max active width | Expression uses | Terms/full shot | Duplicate term visits | Rotations | Rotation coefficient visits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| surface d7 r7 | 0 | 819 | 214,342 | 1,972 | 0 | 0 |
| cultivation d3 | 4 | 112 | 31,256 | 6,186 | 23 | 348 |
| distillation | 5 | 225 | 12,877 | 0 | 5 | 160 |
| coherent d3 r3 | 7 | 156 | 353 | 54 | 69 | 4,434 |
| regime k=12/L=512 | 12 | 566 | 18,866 | 0 | 12 | 26,624 |

Exact duplicate removal could eliminate less than 1% of surface terms and
about 20% of cultivation terms. It cannot by itself explain or close the
observed 10-12x gaps. The incremental result does not require it.

The active-operation census also identifies the likely next kernel targets:

- Cultivation executes 336 of 348 rotation coefficient visits at width four;
  all rotation signs are shot-dependent and 332 visits use multi-X-bit Paulis.
- Coherent d3 r3 executes 3,840 of 4,434 rotation visits at width seven;
  3,040 visits use multi-X-bit Paulis, 58 adjacent rotation pairs share a
  width, and active measurements account for 1,084 predicted coefficient
  visits.
- k=12/L=512 executes 26,624 rotation visits, including 22,528 at width 11,
  plus 8,192 non-diagonal active-measurement visits.

## Post-change perf attribution

For each row, direct and incremental profiles use the same circuit, shots, and
seed. Profiles use `cycles:u`, frequency 999 Hz, frame-pointer call graphs, and
recorded zero lost samples. Percentages below are flat self-cost over the full
profiled process. Compilation is included, but a zero-shot timing was 0.00-0.08
seconds versus 0.66-7.36 seconds of sampling, so it does not change the main
conclusion.

| Case | Evaluator self-cost | Direct | Incremental |
| --- | --- | ---: | ---: |
| cultivation d3 | affine `evaluate` | 87.77% | 9.49% |
|  | rotation | 3.84% | 25.79% |
|  | measurement probability + collapse | 1.40% | 9.18% |
|  | true-symbol propagation | below 0.4% | 3.22% |
| coherent d3 r3 | affine `evaluate` | 7.30% | 3.74% |
|  | rotation | 66.92% | 67.71% |
|  | measurement probability + collapse | 12.78% | 14.79% |
|  | true-symbol propagation | below 0.4% | 0.71% |
| regime k=12/L=512 | affine `evaluate` | 16.16% | 1.56% |
|  | rotation | 54.29% | 62.22% |
|  | measurement probability + collapse | 16.08% | 21.21% |
|  | true-symbol propagation | below 0.4% | 0.98% |

The cultivation hot path moves from affine scanning to rotation, action
dispatch, measurement, and shot-loop overhead. Coherent is unchanged within
the paired-run spread because it was already kernel-bound. The k=12 residual
is likewise a rotation/measurement problem. These profiles support keeping
expression and active-kernel work as independent changes.

## Validation and artifacts

- Two focused tests compare every visible and hidden record, symbol, detector,
  observable, expectation value, discard state, active state coefficient, and
  relevant census invariant for 1,000 ordinary noisy/postselected shots and
  1,000 forced-zero-fault shots.
- All 1,242 C++ tests pass in the Release profiling build.
- The balanced-run driver writes results incrementally and retains paired
  seeds and per-sample outputs. The local raw JSON is
  `tools/bench/backend_comparison/results/20260807T_incremental_ab/expression_ab.json`.
- The six local `perf.data` files are in the sibling `profiles/` directory.
  Generated result directories remain gitignored; the branch contains the
  harness, census, tests, and this complete numeric summary.

Reproduction:

```bash
cmake --build build-profile --target profile_expression_ab profile_expression_census -j

python3 tools/bench/backend_comparison/run_expression_ab.py \
  --binary build-profile/profile_expression_ab \
  --circuits-dir tools/bench/backend_comparison/results/20260807T202928Z/circuits \
  --output tools/bench/backend_comparison/results/20260807T_incremental_ab/expression_ab.json \
  --blocks 3 --cpu 3 --seed-base 280000
```

## Review question

The evidence supports reviewing a production-quality scalar incremental
evaluator before designing expression graph optimizers or packed-shot
execution. The design review should focus on continuation compatibility,
whether to retain a hybrid direct path for short expressions, storage/lifetime
of reverse dependencies, and eliminating prototype-only public controls. Active
kernel work can then be measured and landed separately against coherent d3 r3,
k=12/L=512, and QV-10.
