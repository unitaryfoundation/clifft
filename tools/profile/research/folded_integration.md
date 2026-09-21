# Optional folded-region integration: bounded benchmarks

2026-09-21. Implementation revision `6a348a9a` on
`codex/folded-mainline-integration`. This measures the actual public
`compile(..., specialize_folded=True)` and `sample` / `sample_survivors` path,
not the standalone research harness.

The result is useful acceleration on the reconstructed f5 circuit and
accessible f7 sampling, but a substantial slowdown at f3. Keep the option
experimental and disabled by default. Pause generalization and merging until
independent protocol circuits or another concrete block family are available.

## Sampling results

Same physical f3/f5/f7 fixtures at p=0.001, one CPU worker, scalar batches.
Times are medians of three batches, excluding compilation. Postselection
selects every detector and keeps survivor records. These bounded runs measure
throughput, not logical error rates or statistical acceptance agreement.

| Circuit | Mode | PR 471 optimized ordinary | Integrated folded | Ratio ordinary / folded |
| --- | --- | ---: | ---: | ---: |
| f3 | Full sampling | 4.20 us/attempt | 226 us/attempt | 0.019x |
| f3 | Postselection | 3.84 us/attempt | 187 us/attempt | 0.021x |
| f5 | Full sampling | 340 ms/attempt | 3.96 ms/attempt | 86x |
| f5 | Postselection | 188 ms/attempt | 2.55 ms/attempt | 74x |
| f7 | Full sampling | Not allocated | 835 ms/attempt | -- |
| f7 | Postselection | Not allocated | 632 ms/attempt | -- |

Ordinary f7 still compiles to peak active width 44: 256 TiB for its coefficient
array alone. It was compiled but never sampled. Integrated widths are 1, 8,
and 22 for f3, f5, and f7 respectively.

## Compilation and memory

Compilation is one cold invocation per fresh process; the intervals below
span the full-sampling and postselection configurations, not repeated-trial
confidence intervals. RSS is the **whole process high-water mark**, including
Python, NumPy, compilation temporaries, plans, one worker's scratch and
coefficients, and output storage. It is not a per-worker allocation estimate.

| Circuit | Ordinary compile | Folded compile | Ordinary peak RSS | Folded peak RSS |
| --- | ---: | ---: | ---: | ---: |
| f3 | 9-11 ms | 45-50 ms | 53-54 MiB | 41 MiB |
| f5 | 232-239 ms | 381-386 ms | 109-110 MiB | 50-51 MiB |
| f7 | 4.02-4.18 s | 5.79-5.80 s | Compile only: 92-93 MiB | 189-193 MiB |

Batch sizes differ to keep each run bounded: ordinary f3 uses 131072 attempts,
ordinary f5 uses 16, and folded f3/f5/f7 use 2048/128/8. In particular, the f3
RSS difference includes substantially different output sizes and should not
be interpreted as a kernel-memory saving. For folded f7, compilation alone
peaked at about 134 MiB, before sampling raised the high-water mark.

## Disabled-option control

The identical f3 default pipeline was measured on parent revision `795d0b35`
and on the integrated revision with `specialize_folded=False`, using 131072
attempts per batch. The first comparison was 6.63 versus 6.86 us/attempt.
An interleaved parent/current/current/parent repeat reversed that difference:
individual batches ranged from 6.49-7.64 us for the parent and 6.50-6.95 us for
the integrated build. There is no consistent regression in this small control;
it does not establish a universal zero-overhead guarantee.

## Comparison contract and limits

- GCC 13.3 Release builds, assertions off, native CPU target, OpenMP available;
  AMD EPYC 9554P, pinned CPU 0, one worker, `batch_size=1`.
- The ordinary comparator is the exact package from the
  [optimized-baseline study](scheduled_baseline.md). Its extension hash matches
  that study. It incorporates PR 471 head
  `8f69b3c305303483e5dd33dcf4d13dbbebae1fa9`, with unbounded active-width
  scheduling, noise transparency, and neutral-rotation sinking enabled.
  PR 472 adds documentation, not runtime optimizations.
- The integrated build uses its current default prefix passes; it does not
  contain PR 471's scheduler. This compares the actual branch to the stronger
  ordinary baseline. It is not a claim about every possible thread/batch setup.
- The initial integration omits the research harness's native f5-to-f7 growth
  handoff and rejection inside the specialized region. Consequently these f7
  timings are slower than the standalone native-growth results. The ordinary
  width-22 prefix remains a substantial cost.
- These are our reconstructed cultivation circuits through fictitious error
  detection and a noiseless final logical measurement. They exclude escape,
  decoding, and fictitious error correction; they are not author-supplied
  protocol artifacts. Recognition is deliberately strict.
- Implementation validation previously passed 936 native and 216 Python tests,
  including an independent Aer comparison. Hot-path allocation checks passed
  at all three distances. This benchmark adds performance evidence, not a new
  independent correctness oracle.

## Reproduction

Install separate Release packages for the integrated revision, its parent,
and the optimized baseline. Do not load these through an editable-import hook.
The driver launches isolated child interpreters, verifies import paths, pins
the CPU, and records extension/circuit hashes and all batches.

```bash
.venv/bin/python tools/profile/benchmark_folded_integration.py \
  --current /tmp/folded-bench-current \
  --scheduled /tmp/clifft-pr471-package \
  --parent /tmp/folded-bench-parent \
  --output tools/profile/research/folded_integration_benchmark.json

.venv/bin/python tools/profile/benchmark_folded_integration.py \
  --current /tmp/folded-bench-current \
  --parent /tmp/folded-bench-parent --controls-only \
  --output tools/profile/research/folded_integration_controls.json
```

Raw results: [sampling and resources](folded_integration_benchmark.json),
[interleaved disabled-option controls](folded_integration_controls.json).
