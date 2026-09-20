# Complete reconstructed f7 cultivation attempts

Date: 2026-09-20. Following the requested priority, f7 was tested before
automatic fallback or production consolidation. The hybrid now samples the
complete reconstructed f7 fold-transversal surface-code cultivation circuit.
With PR 471 enabled, p=0.001 attempts take approximately 0.45 s without early
rejection and 0.17 s with early rejection. A fresh native process peaks at
157.5 MiB. Ordinary Clifft's optimized plan reaches active width 44, implying
a 256 TiB coefficient array alone, so its executor was not allocated.

This demonstrates access to a larger circuit within the reconstructed family;
there is no measured f7 speedup ratio against an executed dense baseline.

## Exactly what protocol is included

The circuit follows the f7 sequence in
[Takada, Bartlett and Williamson, Appendix A and Table 3](https://arxiv.org/html/2609.16929v1#A1):

1. Physical T-state injection into the rotated d3 code and growth to regular d3.
2. d3 stabilizer extraction and two noisy folded logical checks.
3. Growth to d5, stabilizer extraction and two noisy d5 logical checks.
4. Unitary growth to d7, stabilizer extraction and two noisy d7 logical checks.
5. Noisy d7 stabilizer extraction, then noiseless fictitious error detection
   (FED) and the final folded logical measurement.

It includes 85 data qubits, 14 ancillas, 351 visible measurements, 436 hidden
reset outcomes, 348 acceptance detectors and 2,189 physical noise sites.
Physical T/T_DAG and CCZ gates are retained. No ideal encoded input is
substituted for the preceding cultivation stages. The noise model is the
same gate-level Pauli model used for f3/f5, with no idle noise.

Thus this is the whole reconstructed **cultivation stage plus FED evaluation**,
not only the large gadget. It is not a complete factory pipeline with
decoder-based fictitious error correction (FEC), escape, or storage. It is
also not an author-supplied circuit: the sequential stabilizer gate order
remains an explicit reconstruction choice. Correct sampling of this circuit
does not by itself establish its claimed fault distance or reproduce the
paper's logical-error estimates.

The generated [physical circuit](../fixtures/folded_msc/f7_fed_p1e-3.stim)
has SHA-256 `fff72e8f48a0509a00d52133d8cb334a1743fc7c74ce1f40eeeccc9785909545`.

## Additions needed at f7

`folded_msc_f7.py` adds the centered d5-to-d7 growth, using the four CNOT
layers of [Higgott et al., Figure 2](https://quantum-journal.org/papers/q-2021-08-05-517/).
The Hadamard-dual version matches the generator's X/Z-check convention.
Stim checks confirm that the growth preserves all three logical Pauli
eigenstates and produces the complete d7 stabilizer group.

The eight-qubit cat and six-qubit verification cat follow Takada Figure 19c,
including the published controlled-factor assignment. Both uniform
verification outcomes are accepted. Five pairwise-parity detectors compare
each later verification readout with the first; treating all six as
postselect-on-zero would incorrectly discard half the ideal preparations.

The compiler eliminates the extra verification-cat branch by conditioning
on its first, fair readout. It symbolically proves that the other records
are independent of the surviving coherent data-cat branch and precomputes
the selected Pauli faults' effects on that substitution. At runtime, the
sampler draws one verification bit per check and updates only preallocated
phase tables. Its conditional log probability includes both factors of 1/2.
No runtime tableau or topology analysis is introduced.

Physical masks in this GCC/Clang research harness now use fixed 128-bit
integers, accommodating 85 data qubits. Contraction gather addresses remain
compiler-precomputed. The old f3/f5 bundles remain readable, and production
Clifft and Stim are unchanged.

For this first f7 test, ordinary Clifft executes the entire prefix through
d7 pre-checks, including the earlier f5 logical checks. Only the largest
two-check block and its continuation use the contraction sampler. This
prefix peaks at width 22 and retains its 64 MiB coefficient allocation.
We have not yet connected native f5 output through growth into native f7.

## Correctness

- Twenty-four fresh complete histories pass: eight each at p=0, 0.001 and
  0.03. They include fault realizations with up to 74 faults and both
  accepted and rejected attempts. Each sampled physical fault and every
  visible/hidden record is retained in the raw artifact.
- The independent coherent-Clifford oracle evaluates the whole physical
  f7 circuit. Clifft separately replays the prefix. Their conditional log
  probabilities agree with the native continuation to at most 7.11e-15.
  Dense full-circuit Clifft replay is deliberately not used at width 44.
- All noiseless histories accept with logical output zero. Their two
  verification readouts account for a complete suffix probability of 1/4,
  with all four uniform-pattern combinations valid.
- The cat test checks both ideal verification patterns, the fairness of
  the first verification measurement, and every single-site Pauli channel
  in cat preparation. Every accepted single-fault case has at most one
  data-cat bit error modulo the global cat flip. This is a component check,
  not an exhaustive full-circuit fault-distance proof.
- Thirty-five tests pass, covering f7 growth/cats/complete native histories,
  f3/f5 regressions, independent dense-Aer checks of noisy controlled factors,
  analytic logical projectors, brute-force factor sums and native sampling.
  Formatting, lint, type and hygiene checks pass.

The f7 oracle is independent of the fixed-contraction executor, but shares
the reconstructed physical circuit. It verifies simulation correctness;
comparison with the authors' actual circuits is still valuable for checking
the reconstruction itself.

## Runtime and memory

The build is study base `41a18d455e855650f3c5dffa75db301f67ff2ded` plus
PR 471 head `8f69b3c305303483e5dd33dcf4d13dbbebae1fa9`. The default passes
run first and the scheduler is explicitly enabled on both prefix and full
ordinary plans. PR 472 adds documentation only. Both bounded and unbounded
scheduling leave the full plan at width 44.

AMD EPYC 9554P, GCC 13.3 native Release, one worker pinned to CPU 0. Timings
are medians of three batches of 16 fresh attempts after 16 warmups.
Verification workers ran on other cores. They include prefix execution,
fresh noise, binding, contraction, all records and final evaluation, but
exclude compilation and loading.

| Scheduler | Full attempt | Early rejection |
| --- | ---: | ---: |
| Bounded | 450 ms | 177 ms |
| Unbounded | 450 ms | 171 ms |

The unbounded early-rejection batches range from 162 to 189 ms/attempt;
these small batches are throughput measurements, not acceptance-rate or
logical-error estimates. Bundle preparation takes about 30.2 s once.
Native bundle loading plus prefix construction takes 0.32-0.57 s. Compiling
the full ordinary plan without allocating its executor takes 0.15 s with
bounded scheduling and 3.85 s with unbounded scheduling.

| Resource | f5 contraction experiment | f7 contraction experiment |
| --- | ---: | ---: |
| X-generator variables | 20 | 42 |
| Largest intermediate table | 256 complex entries | 65,536 complex entries |
| Fixed-plan numeric payload | 1.77 MiB | 79.06 MiB |
| Reused bound-input buffer | 154.5 KiB | 326.5 KiB |

The f7 plans comprise one amplitude plan and 43 marginal plans, with about
1.60 million elimination-table entries visited across the marginal plans
before counting coherent term pairs. The largest table alone substantially
understates total work and storage.

A fresh shell-launched native process, including prefix execution and full
ordinary-plan compilation but no ordinary full executor, peaks at 157.5 MiB.
That includes the 64 MiB ordinary prefix buffer and the contraction plans.
The sampler's earlier RSS checkpoint precedes warmups and must not be used
as its complete footprint. Python-launched `ru_maxrss` values also inherit
the Python parent's high-water mark; the separate shell measurement is the
memory evidence here.

The ordinary width-44 coefficient requirement is a planner-derived size,
not an observed allocation or timing. The harness now rejects attempts to
allocate a dense reference above its explicit profiling limit.

## What this establishes and what remains

The approach extends to the larger verified cat and to complete f7
cultivation attempts without a new simulator architecture. This is stronger
evidence of useful reach than the earlier isolated gadget or f5 kernel
timings. However, precomputed-plan payload grows about 45-fold from f5 to
f7; this experiment does not establish polynomial scaling or arbitrary
distance support.

No fallback-selection work was added. Before consolidation, the useful
remaining engineering questions are whether to share/compress the marginal
plans and how to pass the native f5 logical state through growth without
the ordinary width-22 prefix. Those affect both memory and runtime. Author
circuits remain the independent protocol-fidelity check. Importance sampling
would address rare logical-error estimation, not the ability to execute
these complete attempts.

## Memory-layout audit and next steps

Implemented follow-up: the [memory experiment](folded_memory.md) reduces
native f7 numeric payload to 10.43 MiB with shared workspaces, narrower
indices and exact blocked lookup tables. The estimates below describe the
initial audit; the follow-up records actual variants and validation.

A follow-up audit of the generated f3/f5/f7 `FactorPlan` objects separates
immutable lookup tables from mutable execution buffers. At f7 the current
79.06 MiB numeric payload consists of:

| Allocation | MiB |
| --- | ---: |
| Expanded 64-bit gather addresses | 59.050 |
| Complex scratch arrays summed across plans | 13.419 |
| Complex product arrays summed across plans | 6.008 |
| Expanded 64-bit leaf labels | 0.584 |
| Output addresses | 0.00034 |

These counts exclude container overhead and the ordinary prefix. The sampler
evaluates its 44 contractions sequentially. One preallocated workspace per
worker can therefore replace the summed scratch/product arrays: maximum
scratch plus maximum product is only 2.422 MiB. That alone would reduce
numeric payload to 62.06 MiB. The largest scratch has 93,179 entries, so
32-bit addresses suffice; all leaf labels are in 0..3 and fit in a byte.
Combining workspace sharing, 32-bit gather/output addresses and byte leaf
labels gives a calculated payload of 32.02 MiB at f7 and 0.631 MiB at f5.
These are layout estimates, not implemented or measured RSS improvements.
Complex arithmetic remains double precision.

The recommended next experiment is to implement these exact layout changes
and rerun complete-history reference checks, throughput measurements and
fresh-process peak RSS with the same PR-471-enabled baseline. There is no
need to change contraction order or introduce runtime topology planning.

Further candidates are compiler-precomputed bit-projection descriptors in
place of expanded gathers, and compiler-assigned scratch reuse after each
intermediate's last use. Descriptor indexing trades memory traffic for
arithmetic and needs a timing comparison. These improve representation;
they do not establish better asymptotic contraction-width scaling.

Separately, the current f7 run retains a 64 MiB ordinary f5-containing prefix.
A native f5-to-f7 handoff through the physical growth and pre-check circuit
could remove that buffer and may improve runtime. It needs a new validated
boundary transfer: the f7 sequence omits the standalone f5 post-checks, so
the existing terminal f5 sampler cannot simply be chained unchanged.

## Artifacts and reproduction

- [Timing batches and source/binary hashes](folded_protocol_f7_benchmark.json)
- [Fault draws, complete records and reference checks](folded_protocol_f7_validation.json)
- [Circuit counts, plan payload, build provenance and process memory](folded_protocol_f7_context.json)

Use the existing isolated PR-471 build from the
[baseline report](scheduled_baseline.md), rebuilding `sample_folded_protocol`
with the updated profiling sources.

```bash
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=tools/profile uv run --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/benchmark_folded_protocol.py --distances 7 --f5-shots 16 \
  --sampler /tmp/clifft-pr471-baseline/build-study/sample_folded_protocol \
  --bundles /tmp/folded-f7-bench --output /tmp/f7-benchmark.json
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=tools/profile uv run --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/audit_folded_protocol.py --distances 7 --cases 8 \
  --oracle-cases 8 --schedule unbounded \
  --sampler /tmp/clifft-pr471-baseline/build-study/sample_folded_protocol \
  --reference /tmp/clifft-pr471-baseline/build-study/replay_cultivation \
  --output /tmp/f7-validation.json
```

The historical `--f5-shots` option also controls f7 batch size. No dense f7
reference samples are requested by this driver. For the separate memory
measurement, compile an f7 p=0 bundle and launch the native CLI directly from
a shell under `/usr/bin/time -f %M`, with arguments
`bundle 1 0 512 0 unbounded 0`.
