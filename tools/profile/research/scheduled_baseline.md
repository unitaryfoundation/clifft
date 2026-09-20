# Cultivation comparisons with active-width scheduling enabled

Date: 2026-09-20. PR 471 materially improves the baseline. It approximately
halves the complete f5 reconstruction's sampling cost, without reducing peak
width or memory. The synthetic d7 gadget still wins by about 4.1x against
the stronger baseline; ordinary Clifft still wins at d3/d5. These results
supersede the default-pipeline-only performance comparisons in the earlier
reports. They do not demonstrate a production speedup on complete f5 MSC:
that gadget reduction remains an offline probability reference.

## Build and comparison contract

The isolated source tree combines study revision
`41a18d455e855650f3c5dffa75db301f67ff2ded` with the complete
[PR 471](https://github.com/unitaryfoundation/clifft/pull/471) head
`8f69b3c305303483e5dd33dcf4d13dbbebae1fa9`, including its planner prerequisites.
This preserves the study revision's more recent sampling fixes. The only
merge conflict was a benchmark-canary test's expected workload count; the
resolution uses the PR's dynamic count. No production source conflicted.
[PR 472](https://github.com/unitaryfoundation/clifft/pull/472) is documentation
stacked on this exact optimizer revision, with no additional runtime changes.

All three configurations use the same merged core, compiler, and machine:

- **Off:** the default peephole-fusion and statevector-squeeze passes.
- **Budgeted:** those passes followed by `ActiveWidthSchedulePass`, with its
  default beam width 8 and search budget 16.
- **Unbounded:** the same beam width with `search_budget=None`.

Noise-transparent scheduling and neutral-rotation sinking remain enabled.
Unbounded search is still a beam-search heuristic, not an optimality proof.
GCC 13.3, Release `-O3 -DNDEBUG -ffast-math -march=native -mtune=native`,
OpenMP AUTO, one worker, CPU 0 of the AMD EPYC 9554P. The native and Python
executables link the same merged core. Python import location and extension
hash are checked and recorded; an isolated environment avoids the original
editable install intercepting imports.

Physical circuit hashes match the previous synthetic sweep, and the folded
f3/f5 p=0.001 fixtures retain their recorded hashes. Only compiler scheduling
changes. Sampling timings exclude compilation and process startup. Folded
measurements use the public Python sampler; synthetic measurements use the
same scalar native complete-attempt harness as the earlier sweep.

## Complete folded protocols

At p=0.001, medians of three batches:

| Protocol and sampling mode | Off | Budgeted | Unbounded | Off / unbounded |
| --- | ---: | ---: | ---: | ---: |
| f3, full attempts | 6.709 us | 4.689 us | 4.180 us | 1.60x |
| f3, early rejection | 5.706 us | 3.612 us | 3.578 us | 1.59x |
| f5, full attempts | 770.5 ms | 396.7 ms | 340.6 ms | 2.26x |
| f5, early rejection | 445.2 ms | 230.1 ms | 207.9 ms | 2.14x |

The f3 runs use 262,144 attempts per batch at p=0, 0.001, 0.003 and 0.01.
The f5 runs retain the previous eight attempts per batch at p=0 and 0.001.
Both early-rejection settings are tested in all three configurations: 36
timed cases in total. Seeds, warmups, worker count, detector masks and output
retention match the preceding folded benchmark. Full f5 p=0 also improves
from 764.6 ms to 405.3 ms and 364.7 ms, so the gain is not an artifact of
rejecting a fortunate selection of noisy attempts.

Each f5 p=0.001 configuration accepted seven of the 24 attempts and reported
zero accepted logical failures. These small counts are throughput checks,
not estimates of protocol survival or logical-error rates. The f3 acceptance
counts agree closely across configurations; identical seeds need not give
identical streams after early rejection or compiler reordering.

Compilation at p=0.001 with early rejection:

| Protocol | Off | Budgeted | Unbounded |
| --- | ---: | ---: | ---: |
| f3 | 2.51 ms | 5.56 ms | 8.95 ms |
| f5 | 20.6 ms | 38.6 ms | 222.8 ms |

These are individual compilation measurements, separated from the timing
batches. At f5 the extra unbounded compilation over bounded search is paid
back after roughly nine attempts using these early-rejection timings. The
extra f3 early-rejection gain is small enough that bounded search is a
reasonable choice for shorter jobs. Future high-shot comparisons should
retain both configurations and account for compilation explicitly.

The f5 planner's dense-work proxy decreases from 851,405,646 to 464,967,224
and 425,250,888. Peak active width remains 22; f3 remains 8. Reduced work at
the same peak, rather than smaller state storage, explains the gain.

## Synthetic geometric gadget family

At p=0.001, all entries below are microseconds per complete attempt:

| d | Ordinary off | Ordinary budgeted | Ordinary unbounded | Gadget alongside unbounded | Unbounded / gadget |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.413 | 0.415 | 0.432 | 15.83 | 0.027x |
| 5 | 5.510 | 4.520 | 4.406 | 107.17 | 0.041x |
| 7 | 2734.7 | 2270.1 | 2227.7 | 537.2 | 4.15x |
| 9 | Skipped | Skipped | Skipped | 2223.4 | Not measured |

All distances run at p=0, 0.001 and 0.01, with all three scheduling settings:
36 cases. Batch sizes remain 65,536, 8,192, 256 and 256 respectively, three
repetitions after 16 warmup attempts. There is no early rejection. The gadget
path and its prefix compilation stay fixed; only ordinary full-circuit
compilation receives the scheduler. Eight gadget histories per circuit are
identical across all three settings. The native harness measures both paths
in each process, preserving the earlier timing boundary.

At p=0.001, ordinary compilation times in milliseconds are:

| d | Off | Budgeted | Unbounded |
| --- | ---: | ---: | ---: |
| 3 | 0.258 | 0.423 | 0.435 |
| 5 | 0.666 | 1.714 | 1.842 |
| 7 | 1.559 | 5.555 | 7.072 |
| 9 | 1.811 | 15.700 | 26.095 |

The native compilation timer includes ordinary executor construction when
execution is enabled. Gadget preparation and native loading are recorded
separately in the raw results. This remains the ideal-input, unflagged,
terminal-only synthetic family described in [its report](gadget_family.md),
not a full cultivation protocol. The previous 5.2x historical d7 result
should now be replaced by the approximately 4.1x optimized-baseline result.

## Memory

All three scheduling settings have identical peak active widths:

| Circuit | Peak width | Coefficient array per lane | Measured total process peak RSS |
| --- | ---: | ---: | ---: |
| Folded f3 | 8 | 4 KiB | 43.0-43.1 MiB |
| Folded f5 | 22 | 64 MiB | 114.4-114.6 MiB |
| Synthetic d3 | 4 | 256 B | 42.7-42.8 MiB |
| Synthetic d5 | 10 | 16 KiB | 42.8-43.0 MiB |
| Synthetic d7 | 19 | 8 MiB | 51.1-51.4 MiB |
| Synthetic d9 | 31 | 32 GiB | Sampling skipped |

RSS is measured separately in a fresh Python process for every configuration,
using Linux `ru_maxrss`, compilation followed by one full shot at p=0.001.
It includes Python, imports, the plan and scratch; it is not the timed
multi-shot batches' memory or an isolated executor allocation measurement.
The f5 compile-only peaks are about 52.2-52.6 MiB. The d9 compile-only peaks
are about 43.6-43.7 MiB; these do not represent d9 execution memory. All
dense sampling above width 24 is skipped before allocating coefficient state.

The gadget's previously reported contraction numeric payloads are unchanged:
11.35 KiB, 113.70 KiB, 614.16 KiB and 2.54 MiB for d3/d5/d7/d9. Those exclude
other gadget storage and therefore must not be compared directly with total
process RSS. The optimizer does not erase the gadget's coefficient-storage
advantage on the larger synthetic cases.

## Correctness and retained next step

Follow-up: the [complete native folded sampler](folded_protocol.md) now
implements and validates the experiment proposed below. Its timings use
this PR 471 baseline with both bounded and unbounded search enabled.

- The merged build passes 31 Python scheduler, squeeze-integration and
  benchmark-canary tests. The existing native prototype tests also pass
  (five tests) after extending the profiling CLI with a scheduling option.
- All 24 saved f3 histories, independently checked against dense Aer, and
  all 24 saved f5 histories, checked against the coherent-Clifford oracle,
  replay successfully in all three modes: 144 checks, maximum absolute
  log-probability difference 4.89e-15. These fix physical Pauli faults before
  scheduling; they complement the existing stochastic scheduler tests.
- Synthetic circuit hashes match all twelve original inputs. Changing the
  baseline scheduler preserves all captured gadget histories and the native
  normalization checks. This is a baseline refresh, not a new independent
  validation of the gadget algebra.
- Targeted formatting, lint, type and file-hygiene checks pass.

The retained next experiment is a compiler-planned folded-check reduction
with continuation support, validated on complete f5. Its comparison must use
this optimized baseline, around 208 ms per attempt with early rejection in
this run. The offline coherent-Clifford oracle's timing is not a comparable
sampler timing. The memory bottleneck and remaining f5 execution cost still
make the experiment worth pursuing, but no complete-protocol speedup or
distance-seven MSC simulation has yet been demonstrated.

## Artifacts and reproduction

- [Folded measurements](scheduled_folded_baseline.json)
- [Synthetic measurements](scheduled_gadget_baseline.json)
- [Isolated memory measurements](scheduled_baseline_memory.json)
- [Physical replay audit](scheduled_baseline_validation.json)

The JSON files retain build/extension hashes, optimizer options, source
hashes and individual timing batches. The folded driver was formatted after
its run; its recorded driver hash identifies the pre-format version. The
audit also received a type annotation after its run. Neither changed the
executed algorithm.

Build the merged source in an isolated checkout, retain the current research
`sample_terminal_gadget.cpp`, `replay_cultivation.cpp`,
`gadget_contraction_kernel.h` and `study_schedule.h`, and link the Python
extension and native tools against that same core. The working isolated build
for this run is `/tmp/clifft-pr471-baseline/build-study`; the installed Python
package is `/tmp/clifft-pr471-package`. An isolated Python interpreter loads
that package before other site packages. Verify `clifft.__file__` and the
presence of `clifft.ActiveWidthSchedulePass` before benchmarking.

With that environment (and cirq-core 1.6.1 for synthetic bundle compilation):

```bash
python tools/profile/benchmark_scheduled_baseline.py --family folded \
  --build-source /tmp/clifft-pr471-baseline --output /tmp/scheduled_folded.json
python tools/profile/benchmark_scheduled_baseline.py --family synthetic \
  --build-source /tmp/clifft-pr471-baseline \
  --sampler /tmp/clifft-pr471-baseline/build-study/sample_terminal_gadget \
  --output /tmp/scheduled_gadgets.json
python tools/profile/measure_scheduled_memory.py --output /tmp/scheduled_memory.json
python tools/profile/audit_scheduled_baseline.py \
  --reference /tmp/clifft-pr471-baseline/build-study/replay_cultivation \
  --output /tmp/scheduled_validation.json
```

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; the drivers pin themselves
to one allowed CPU. The comparison driver expects the isolated checkout to
retain its uncommitted merge's `MERGE_HEAD`, which records the optimizer revision.
