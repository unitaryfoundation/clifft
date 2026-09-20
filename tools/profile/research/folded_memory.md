# Exact memory reductions for the folded sampler

Date: 2026-09-20. This follows the complete reconstructed
[f7 cultivation experiment](folded_protocol_f7.md). The native research
sampler now shares execution buffers across contractions, narrows lookup
indices, and stores large address tables as two smaller tables. Arithmetic
remains `complex<double>`; the physical circuit, sampling algorithm and
contraction order are unchanged.

## Storage changes

| f7 numeric allocation | Original | Selected layout |
| --- | ---: | ---: |
| Lookup arrays, including leaf labels and outputs | 59.63 MiB | 8.00 MiB |
| Scratch and product workspaces | 19.43 MiB | 2.42 MiB |
| Total | 79.06 MiB | 10.43 MiB |

These are allocated numeric-vector capacities, excluding container objects,
the bound-input buffer, physical noise model and ordinary Clifft prefix.
The final native sampler reports lookup and workspace capacities separately.
At f5, payload falls from 1.77 MiB to 0.631 MiB; at f3 it falls from
66.0 KiB to 19.3 KiB. Small-case process RSS need not fall proportionally,
because plan arrays are only part of the process footprint.

The 44 f7 plans are used sequentially. They now hold immutable lookup
arrays, while the sampler owns one scratch/product workspace sized to the
largest requirements across all plans. Independent workers need separate
workspaces. This experiment does not yet share plan objects across workers.

Every gather address is below the loader's two-million-entry limit and fits
in `uint32_t`. Leaf labels are validated as 0..3 before conversion to
`uint8_t`. Invalid labels cannot wrap into valid values.

For gathers larger than 256 entries, the loader checks whether each address
can be written as `low[j % 256] + high[j / 256]`. If every address agrees,
only these two arrays are retained. Otherwise the original expanded
32-bit table remains. The loader validates the original addresses before
compression, so both forms access exactly the same initialized storage.
Execution uses fixed block loops with no allocation, tableau evolution,
dependency discovery or changes to the contraction graph.

The simpler workspace-sharing plus narrowing variant requires 32.02 MiB at
f7. More aggressive 32-entry blocks can reduce payload to 5.24 MiB, or
8.11 MiB when small gathers stay expanded. They were not selected: extra
block-loop overhead worsened runtime, especially when small f5 tables were
also compressed. The selected 256-entry blocks leave all f3/f5 gathers
expanded and provide most of the useful f7 memory reduction.

## Validation

The selected PR-471-enabled binary has SHA-256
`d37abb983ab6c28ab174e552218b79d0b2b43957a099bc2e7a78ceaa089f22b8`.
Its 24 f7 histories, eight each at p=0, 0.001 and 0.03, pass the independent
coherent-Clifford oracle and ordinary-Clifft prefix replay. Maximum absolute
conditional-log-probability error is 7.11e-15. Every saved physical fault,
measurement, detector, logical result and reported conditional probability
matches the corresponding pre-compression history exactly.

The final layout passes 23 tests covering complete f3/f5/f7 attempts,
native terminal replay and dense physical projectors, folded logical Born
probabilities, and lookup/workspace regressions. Formatting, lint, type and
file-hygiene checks pass.

The native lookup regression checks repeated workspace use with fresh
complex input tables against expanded NumPy contractions at ranks 3, 9
and 12. It includes irregular address tables that cannot be compressed and
an invalid wide leaf label that must be rejected before narrowing.
AddressSanitizer and UndefinedBehaviorSanitizer checks also pass on these
paths. Leak detection was disabled because LeakSanitizer cannot operate
under the sandbox's ptrace environment.

## Measured memory and runtime

The final paired comparison uses p=0.001, seed 9134, one worker pinned to
CPU 0, 16 warmups, then three batches of 32 attempts at f7 or 1,024 attempts
at f3/f5. The selected binary runs first in each pair; the original binary
then runs the same workload. Correctness jobs ran on other cores. Values
below are median elapsed time per attempted shot and fresh-process peak RSS.

| Workload | Original time | Selected time | Original peak RSS | Selected peak RSS |
| --- | ---: | ---: | ---: | ---: |
| f7 full | 446 ms | 491 ms | 173.54 MiB | 105.27 MiB |
| f7 early rejection | 212 ms | 233 ms | 173.67 MiB | 105.13 MiB |
| f5 full | 3.87 ms | 3.97 ms | 12.90 MiB | 11.90 MiB |
| f5 early rejection | 2.15 ms | 2.25 ms | 12.85 MiB | 11.79 MiB |
| f3 full | 205 us | 202 us | 6.43 MiB | 6.35 MiB |
| f3 early rejection | 175 us | 178 us | 6.39 MiB | 6.46 MiB |

This is a measured memory improvement with a runtime tradeoff: about 39%
less process memory at f7, but about 10% longer attempts in this final
comparison. Initial shorter batches showed a 4-5% full-attempt penalty;
neither comparison establishes a speedup. f5 changes by 3-5%, and f3 by
about 2%. The raw artifact retains all batches rather than only the best
trial. Early-rejection timings depend on the seed and rejection histories;
they must not be compared directly with differently seeded older reports.
Accepted and failed counts match between the original and selected binaries
in every paired f3/f5/f7 batch. These are throughput runs, not logical-error
or acceptance-rate estimates.

The intermediate 32.02 MiB layout using only shared workspaces and narrowed
indices had essentially unchanged f7 full-attempt runtime in its shorter
comparison: 447 ms before and 443 ms after, with peak RSS reduced from
173.47 MiB to 126.48 MiB. This remains a useful alternative when runtime
matters more than minimizing memory. The research kernel currently retains
the more compact 256-entry blocked layout; no production dispatch policy
has been introduced.

## Scope and next decision

These changes reduce native execution memory. The Python planner still
constructs expanded plans, and their text serialization is unchanged.
They do not establish improved asymptotic contraction-width scaling.

The f7 ordinary prefix still retains a 64 MiB coefficient buffer, now much
larger than the contraction payload. The next useful algorithmic experiment
is a certified native f5-to-f7 handoff through the actual physical growth
and pre-check circuit. The standalone f5 output cannot simply be chained:
the f7 protocol omits its terminal f5 post-checks. Author-supplied circuits
remain the protocol-fidelity check; this memory experiment does not change
the reconstruction's cultivation-plus-FED scope.

A [bounded handoff experiment](folded_growth_handoff.md) now certifies the
signed-code growth relation and validates a compiled noisy bridge against
Stim. The subsequent [native integration](folded_native_growth.md) removes
the dense prefix allocation and measures complete f7 attempts.

## Artifacts

- [All layout trials and measured process peaks](folded_memory_benchmark.json)
- [Selected-layout independent history validation](folded_memory_validation.json)
- [Implementation and validation context](folded_memory_context.json)

The benchmark artifact preserves exact commands and binary hashes for each
variant. Both the original and modified samplers use the same study base
plus PR 471, with unbounded scheduling after the default passes; PR 472
adds documentation only. A fresh native child is measured with
`/usr/bin/time`, launched by a small standard-library-only Python driver.
Its parent high-water mark is below all f7 peaks. The in-program earlier
RSS checkpoint is not used as peak execution memory.
