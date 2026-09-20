# Complete f7 attempts with native f5-to-f7 growth

Date: 2026-09-20. The certified
[growth handoff](folded_growth_handoff.md) is now connected to the native
f5 and f7 kernels. On the same reconstructed physical circuit, complete
attempts are about 2.5 times faster than the previous hybrid, early-rejected
attempts about 5.2 times faster, and peak process memory falls from about
105 MiB to 40 MiB. Both builds include PR 471 with unbounded scheduling;
PR 472 adds documentation only.

## What changed

Ordinary Clifft now stops before the d5 folded checks, at a prefix with peak
active width 8. The native f5 kernel samples the two physical logical checks
and the syndrome sector that the later d7 checks resolve. A compiled Clifford
transfer propagates that sector, the two logical amplitudes and classical
ancillas through the actual noisy growth and d7 pre-check circuit. The native
f7 kernel then executes the two larger checks and their existing continuation.

The compiler proves that the 40 d5 stabilizers plus 44 newly prepared qubits
map to the full 84-generator d7 stabilizer group. This justifies resolving
the d5 syndrome early without adding physical post-checks. The standalone
f5 final evaluation is not executed. The transfer preserves all actual
measurement and hidden reset records.

The adapter converts the f5 contraction's computational-coset convention
to physical logical axes before applying the growth Pauli frame, then uses
canonical d7 Z-syndrome duals for the next contraction. This includes the
logical-X correction when the input coset offset anticommutes with logical Z.

All topology analysis, fault propagation and transfer packing happen before
execution. The hot path uses preallocated storage, fixed parity/XOR maps,
and the existing contraction kernels. It performs no tableau evolution or
allocation. Production Clifft and Stim sources are unchanged.

## Same physical workload

The circuit SHA-256 remains
`fff72e8f48a0509a00d52133d8cb334a1743fc7c74ce1f40eeeccc9785909545`.
It contains 351 visible measurements, 436 hidden reset outcomes and 2,189
physical noise sites. The compiler checks that the new partition covers
each noise site exactly once:

| Stage | Noise sites |
| --- | ---: |
| Ordinary pre-f5 prefix | 535 |
| Native f5 checks | 134 |
| Physical growth and d7 pre-checks | 694 |
| Native f7 checks and continuation | 826 |

This remains the complete reconstructed cultivation stage plus FED and
logical evaluation. It does not add escape, storage, decoder-based FEC or
author-circuit validation. The sequential extraction ordering and noise
conventions are unchanged.

## Correctness

Twenty-four fresh complete physical histories pass independent verification:
eight each at p=0, 0.001 and 0.03, including histories with up to 73 faults.
The coherent-Clifford oracle evaluates the entire physical circuit, while
ordinary Clifft separately replays the shortened prefix. Maximum absolute
conditional-log-probability disagreement is 7.11e-15. All noiseless histories
accept with logical output zero. Physical records, acceptance parities and
logical output also agree with replay.

Raw traces now retain faults from the f5 and growth stages separately, in
addition to prefix and f7-suffix faults. The staged sampler uses separate
deterministic RNG streams, so the same seed does not imply the same history
as the older dense-prefix sampler. Validation compares probabilities for
each actually generated history, not seed-by-seed output equality.

All 25 regression tests pass, including complete f3/f5/f7 histories with
both f7 prefix modes, native terminal/dense-projector checks and contraction
workspace tests. Formatting, lint, type and file-hygiene checks also pass.

## Runtime and memory

AMD EPYC 9554P, GCC 13.3 native Release, one worker pinned to CPU 0, p=0.001,
seed 9134. Each process runs 16 warmups followed by three batches: 32 full
attempts per batch or 128 early-rejected attempts per batch. Correctness
jobs ran on other cores. The table reports median time per attempted shot
and fresh-process peak RSS from `/usr/bin/time`.

| Measurement | Previous dense-prefix hybrid | Native growth |
| --- | ---: | ---: |
| Full attempt | 476.6 ms | 194.3 ms |
| Early rejection | 226.3 ms | 43.8 ms |
| Peak RSS, full attempts | 105.18 MiB | 40.00 MiB |
| Peak RSS, early rejection | 105.23 MiB | 40.01 MiB |
| Prefix peak active width | 22 | 8 |
| Prefix coefficient allocation | 64 MiB | 4 KiB |
| Contraction numeric payload | 10.43 MiB | 11.06 MiB |
| Native loading and prefix setup | 0.56 s | 0.26 s |

The extra f5 contraction tables slightly increase contraction payload; the
large saving comes from removing the dense prefix coefficient buffer.
The growth transfer adds about 244 KiB of packed numeric tables, plus its
metadata. Process RSS includes the harness's full ordinary-plan compilation,
but no ordinary full executor is allocated. That full plan still reaches
width 44, implying a 256 TiB coefficient array alone.

These are speedups over the previous hybrid, not measured speedups against
an executed ordinary f7 baseline. Early-rejection batches vary with the
sampled histories; the native batches range from 34.6 to 54.4 ms per attempt.
The run sizes do not establish acceptance or logical-error rates. Compilation
of Python contraction plans and bundles is outside the timed shot loop.

## Use and artifacts

The new path is opt-in in the research compiler; existing bundles continue
to use the dense-prefix path. Generate a staged bundle with:

```bash
PYTHONPATH=tools/profile uv run --offline --frozen --group dev \
  --with cirq-core==1.6.1 python tools/profile/compile_folded_protocol.py \
  --distance 7 --probability 0.001 --native-growth --output /tmp/f7-native-growth

taskset -c 0 /tmp/clifft-pr471-baseline/build-study/sample_folded_protocol \
  /tmp/f7-native-growth 32 0 9134 0 unbounded 0
```

The audit and benchmark drivers also accept `--native-growth`. The generated
`f5/` subdirectory is an intermediate component, not a standalone complete
protocol bundle. Recompiling without the flag removes the growth marker so
a reused output directory cannot accidentally retain the staged execution mode.

- [Timing batches, commands and native process peaks](folded_native_growth_benchmark.json)
- [Complete physical histories and independent checks](folded_native_growth_validation.json)
- [Source/build hashes and fault partition](folded_native_growth_context.json)

The integration is a useful extension of the existing Clifft-based prototype.
Consolidating its research-only interfaces and comparing with author-supplied
circuits remain separate work; this result does not establish scaling at
arbitrary distance or solve rare-logical-error estimation.
