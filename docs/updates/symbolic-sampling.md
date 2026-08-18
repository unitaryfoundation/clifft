# Symbolic Sampling in Clifft

August 2026

Clifft has replaced its original localized-Pauli Schrodinger virtual machine
(SVM) with a symbolic-coordinate compiler and sampler. The main Python
sampling APIs and their output contracts remain, but the compiled program,
planning model, and execution kernels have changed.

This update explains the method's lineage, what changed for users, and how the
new sampler compares with the last legacy Clifft SVM and the current CPU paths
in SymFT. The [theoretical overview](../theory/overview.md) and
[software architecture](../theory/architecture.md) remain the evergreen
references for the current design.

## From the original Clifft method to symbolic coordinates

The [original Clifft paper](https://arxiv.org/abs/2604.27058) introduced the
factored active-state representation. It stores the non-stabilizer part of a
trajectory in a dense vector with $2^k$ amplitudes, where $k$ is the active
width, rather than allocating a full $2^n$ statevector for $n$ physical
qubits. The original implementation localized each active Pauli to a virtual
axis by executing Clifford transformations on that dense vector.

[SymFT](https://arxiv.org/abs/2607.28600), by Wang Fang, Huazhe Lou, and
Riling Li, is the second-generation successor to
[SOFT](https://arxiv.org/abs/2512.23037). Its paper describes a planner that
combines SOFT's generalized-stabilizer simulation with Clifft's dense
active-state representation. It adds symbolic Clifford-Pauli-frame
factorization, adaptive stabilizer-coordinate planning, and direct
multi-coordinate instructions that avoid runtime Pauli localization.

The current Clifft sampler adopts those SymFT developments in a Clifft-specific
compiler and execution stack:

- `SamplingPlan` records target-independent symbolic expressions,
  active-coordinate actions, width transitions, outputs, and continuation
  boundaries.
- Executable preparation fixes storage, transposes symbolic dependencies,
  combines supported rotation runs, and selects scalar or SIMD kernels once
  for the host.
- Each shot assigns event symbols and incrementally updates only the affected
  affine expressions. It does not evolve a tableau or localize Paulis.
- Direct Pauli kernels operate on the active coordinates. Rotation fusion and
  explicit AVX2 and AVX-512 kernels reduce repeated coefficient sweeps.
- Instruments and trajectory-specific continuations extend the same symbolic
  state model to leakage and loss workflows.

The active-state factorization is therefore continuous with the original
Clifft method, while the symbolic frame, planner, and direct execution model
come from the later SymFT architecture. Clifft's HIR optimization,
`SamplingPlan` boundary, executable lowering, continuation machinery, public
APIs, and kernel implementation are its own integration of those ideas.

## What changed for users

The usual workflow is unchanged:

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(program, shots=100_000, seed=42)
```

`sample()`, `sample_survivors()`, `sample_k()`,
`sample_k_survivors()`, `record_probabilities()`,
`basis_probabilities()`, `get_statevector()`, and `clifft.noncomp.sample()`
all use the symbolic sampler. Detector and observable normalization,
postselection, record layout, survivor accounting, importance strata, and
expectation-value outputs retain their public meanings. A fixed seed remains
reproducible, but exact sampled rows are not promised to match the removed SVM
because the two executors consume randomness differently.

The intentional low-level changes are:

- `clifft.compile()` now returns a prepared symbolic `Program`, not iterable
  VM bytecode. Use `program.inspect()` for diagnostic executable-plan text.
- `program.peak_active_width` names the width of the dense active state.
  `peak_rank` remains temporarily as a deprecated alias.
- `num_actions` replaces `num_instructions`. The old `Opcode`, `Instruction`,
  bytecode pass manager, and instruction-list interfaces were removed.
- HIR passes remain the supported customization boundary. The post-lowering
  `bytecode_passes` argument was removed from `compile()`.
- The old mutable `State` plus `execute()` inspection path was removed.
  `get_statevector(program)` remains available for eligible final states.
- SVM-specific backend and OpenMP controls were removed. The current sampler
  selects its scalar, AVX2, or AVX-512 executor when preparing a program and is
  single-threaded today.

These low-level interfaces had no cross-version stability guarantee. The
[compilation guide](../guide/compilation.md) documents the current inspection
and HIR-pass workflow.

## Matched CPU comparison

The tables below report medians from 12 balanced process-level runs on one
pinned core of an AMD EPYC 9554P (Zen 4) KVM host. Each process used one thread,
a fresh paired seed, an excluded warmup, and enough shots to target about 1.5
seconds. Clifft used GCC 13.3 `Release` builds (`-O3 -DNDEBUG`) with an
`x86-64-v2` baseline and the AVX-512 runtime path forced. SymFT used its native
CPU build at the corrected reference-normalization commit.

Every arm received the same circuit view, explicit noiseless detector and
observable references, detector postselection, and aggregate survivor-output
contract. Shot counts were calibrated independently per arm to give comparable
timed durations and are recorded in the raw data. Throughput counts attempted
shots, including shots discarded by postselection. The SymFT batched column
uses its packed cross-shot counts path; it is an architectural throughput
comparison, not the same execution mode as Clifft's current one-shot-at-a-time
sampler.

### Sampling throughput

| Circuit | Peak $k$ | Legacy Clifft | Current Clifft | SymFT single | SymFT batch |
|---|---:|---:|---:|---:|---:|
| Surface code `d=7, r=7` | 0 | 295k/s | 453k/s | 190k/s | 3.44M/s |
| Cultivation `d=3` | 4 | 735k/s | 1.07M/s | 726k/s | 2.58M/s |
| Cultivation `d=5` | 10 | 67.3k/s | 125k/s | 117k/s | 162k/s |
| Distillation | 5 | 91.2k/s | 534k/s | 272k/s | 1.74M/s |
| Coherent `d=3, r=1` | 4 | 1.36M/s | 1.33M/s | 976k/s | 2.55M/s |
| Coherent `d=3, r=3` | 7 | 346k/s | 394k/s | 425k/s | 664k/s |
| Coherent `d=5, r=1` | 12 | 15.2k/s | 14.1k/s | 32.3k/s | 30.5k/s |

Current Clifft is faster than the legacy SVM on five of these seven real
workloads. It is within 2% on coherent `d=3, r=1` and about 7% on coherent
`d=5, r=1`. Relative median absolute deviation for the current-Clifft cells
ranged from 0.5% to 3.6%.

Against SymFT's non-batched path, current Clifft leads on five rows, trails by
about 8% on coherent `d=3, r=3`, and trails by 2.29x on coherent `d=5, r=1`.
SymFT selected exact product-component execution only for the latter circuit;
Clifft currently uses one monolithic active vector. This is a useful measured
case for the separate [product-component investigation](https://github.com/unitaryfoundation/clifft/issues/314),
not evidence that every width-12 circuit has the same gap.

SymFT batching is most valuable when per-shot active-state work is small enough
for packed symbolic and output work to dominate. It provides the largest gains
on the active-width-zero and low-width rows and much less on cultivation
`d=5`. On coherent `d=5, r=1`, batching adds no gain over SymFT's component
path. Clifft tracks packed cross-shot execution separately rather than folding
it into this cutover.

### Compilation time

| Circuit | Legacy Clifft | Current Clifft | SymFT single | SymFT batch |
|---|---:|---:|---:|---:|
| Surface code `d=7, r=7` | 7.78 ms | 12.2 ms | 22.1 ms | 21.6 ms |
| Cultivation `d=3` | 0.902 ms | 1.38 ms | 3.24 ms | 3.95 ms |
| Cultivation `d=5` | 5.02 ms | 11.3 ms | 42.2 ms | 42.4 ms |
| Distillation | 2.93 ms | 2.18 ms | 2.76 ms | 3.36 ms |
| Coherent `d=3, r=1` | 0.389 ms | 0.360 ms | 0.790 ms | 1.10 ms |
| Coherent `d=3, r=3` | 0.661 ms | 0.634 ms | 1.71 ms | 2.02 ms |
| Coherent `d=5, r=1` | 1.23 ms | 1.24 ms | 3.24 ms | 3.77 ms |

Symbolic planning does more work than legacy localization on some circuits,
especially the Clifford-heavy surface and cultivation cases. Compilation still
finishes in milliseconds here, and current Clifft compiles every row faster
than either tested SymFT preparation mode. Compilation and sampling were timed
separately; a shared reference-syndrome calculation was excluded.

## Reproduction and limits

The measured revisions were:

- current Clifft
  [`04c4fe6`](https://github.com/unitaryfoundation/clifft/commit/04c4fe662d9b42d06817450096dbb56a541e709d),
  after the sampler cutover and legacy removal;
- legacy Clifft
  [`aa7e7a3`](https://github.com/unitaryfoundation/clifft/commit/aa7e7a3d3e03d0414bb4f5757d9a7204b082539c),
  the last legacy-default SVM revision; and
- SymFT
  [`c89b985`](https://github.com/haoliri0/SOFT/commit/c89b98514a919240b8afa53a271e08d926d3c987),
  including corrected CPU reference normalization.

The QEC inputs are pinned to the
[Clifft paper corpus at `db7dc9f`](https://github.com/unitaryfoundation/clifft-paper/tree/db7dc9f13a2c2854690e92390c779048a1ac1400/qec_bench).
The complete artifacts record circuit and benchmark-view SHA-256 hashes,
expected detector and observable strings, compiler flags, calibration,
per-round shots, times, counts, memory peaks, medians, dispersion, and a
forced-AVX2 portability cross-check:

- [summary and all cell statistics](../assets/updates/symbolic-sampler-2026-08/summary.json)
- [all primary AVX-512 process samples (gzip-compressed JSON)](../assets/updates/symbolic-sampler-2026-08/raw-avx512.json.gz)
- [all forced-AVX2 process samples](../assets/updates/symbolic-sampler-2026-08/raw-avx2.json)

The full campaign also includes controlled active-width-12, QV-10, QV-20,
and coherent `d=5, r=5` guards. QV-20 and coherent `d=5, r=5` completed too
few shots per process for precise throughput claims, so they are deliberately
absent from the headline table. These measurements describe one CPU, compiler,
and corpus; they are not general performance guarantees. The public
[clifft-bench](https://github.com/unitaryfoundation/clifft-bench) project is
the planned home for ongoing cross-simulator benchmarks.

## What remains deliberately separate

The symbolic sampler cutover does not depend on these follow-up features:

- [packed single-threaded cross-shot execution](https://github.com/unitaryfoundation/clifft/issues/313);
- [exact product-component active states](https://github.com/unitaryfoundation/clifft/issues/314);
- [intra-shot parallel kernels and NUMA placement](https://github.com/unitaryfoundation/clifft/issues/312);
- [cross-shot worker parallelism](https://github.com/unitaryfoundation/clifft/issues/343); and
- [Apple Silicon-specific kernels](https://github.com/unitaryfoundation/clifft/issues/299).

They target different workload regimes and can be evaluated independently
without retaining the legacy SVM.
