# Symbolic Sampling in Clifft (v0.8.0, August 2026)

Clifft has replaced its original localized-Pauli Schrodinger virtual machine
(SVM) with a symbolic-coordinate compiler and sampler. The main Python
sampling APIs and their output contracts remain, but the machinery beneath
them is new.

This update answers the main questions about the rewrite. The
[theoretical overview](../theory/overview.md) and
[software architecture](../theory/architecture.md) remain the evergreen
references for the current design.

## What changed?

The original and current samplers both keep the non-stabilizer part of a
trajectory in a dense active state with $2^k$ amplitudes, where $k$ is the
active width. What changed is how Clifft represents and executes everything
around that state.

The old compiler lowered a circuit to public SVM bytecode. For each active
Pauli, lowering chose a Clifford localization and emitted the corresponding
array operations followed by a narrow state-vector operation. A shot did not
discover how to localize the Pauli; it executed those precomputed Clifford
array transformations, whose runtime cost could include several dense passes.

The rewrite incorporates two separate insights from SymFT. First, a prepared
kernel can apply a multi-coordinate Pauli directly, so localization is no
longer required to obtain a regular dense operation. Second, the effects of
stochastic Pauli noise, measurements, and feedback can be propagated during
planning as symbolic Boolean expressions. Each shot supplies the event values
and evaluates the affected signs instead of evolving a mutable Pauli frame.

Together, these changes let the compiler resolve the Clifford basis, active
coordinates, branch-dependent structure, and Pauli shapes ahead of time. A
shot performs no tableau evolution, localization analysis, commutation
analysis, or dependency discovery, and it executes no localization-induced
Clifford array operations.

The current pipeline is:

```text
Circuit -> HIR -> SamplingPlan -> ExecutablePlan -> Executor -> results
```

`SamplingPlan` is the target-independent semantic plan. Executable preparation
then fixes storage, transposes symbolic dependencies, combines supported
rotation runs, and selects a scalar, AVX2, or AVX-512 executor for the host.
The resulting action tape is a private prepared representation, not public or
stable VM bytecode.

## What stays the same for users?

The usual workflow is unchanged:

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(program, shots=100_000, seed=42)
```

`sample()`, `sample_survivors()`, `sample_k()`,
`sample_k_survivors()`, `record_probabilities()`,
`basis_probabilities()`, `get_statevector()`, and `clifft.noncomp.sample()`
all use the new sampler. Detector and observable normalization,
postselection, record layout, survivor accounting, importance strata, and
expectation-value outputs retain their public meanings.

A fixed seed remains reproducible, but exact sampled rows are not promised to
match the removed SVM because the two executors consume randomness
differently.

## What disappeared under the hood?

The SVM and its bytecode were an elegant fit for the original model: lowering
translated simulation into a compact instruction vocabulary, bytecode passes
combined useful patterns, and several interpreters implemented that same
contract for different CPU targets.

The symbolic plan needs richer prepared data: active-Pauli masks, affine
dependencies, measurement descriptors, fused matrices, and optional
ISA-specific sidecars. In this design, the old bytecode's small instructions
and uniform structure were no longer important sources of speed. Keeping them
would preserve a second representation and force direct symbolic actions back
through the vocabulary of localization.

The rewrite therefore retires the old execution stack instead of carrying two
backends:

- The SVM and its separate scalar, AVX2, and AVX-512 interpreters are gone.
- The localized VM `Opcode`, `Instruction`, and `CompiledModule` bytecode types
  are gone.
- Bytecode optimization passes and the `bytecode_passes` argument to
  `compile()` are gone. HIR passes remain the supported compiler-customization
  boundary.
- Runtime frame and localization instructions are gone. Their work is either
  resolved during planning or replaced by direct active-Pauli actions.
- The mutable `State` plus `execute()` inspection path is gone.
  `get_statevector(program)` remains available for eligible final states.
- SVM-specific backend and OpenMP controls are gone. Parallel sampling and
  intra-shot parallel kernels are future work rather than compatibility layers
  around the removed SVM.

`clifft.compile()` now returns a prepared symbolic `Program`. It is no longer
an iterable bytecode sequence: use `program.inspect()` for diagnostic
executable-plan text. `num_actions` replaces `num_instructions`, and
`program.peak_active_width` names the dense state's width. `peak_rank` remains
temporarily as a deprecated alias.

These low-level interfaces had no cross-version stability guarantee. The
[compilation guide](../guide/compilation.md) documents the current inspection
and HIR-pass workflow.

This does not rule out a bytecode or another compact command stream in the
future. A GPU or another execution target may benefit from one, but it should
be designed around that target and the symbolic plan rather than preserve the
removed SVM's instruction set.

## Why is Pauli localization no longer necessary?

An active Pauli can involve several active coordinates. The old lowerer handled
this by finding a Clifford change of basis that made the Pauli a simple
virtual-axis operation, then emitting that change as SVM array instructions.
The localization decision was made at compile time. During every shot,
however, the SVM still executed the planned transformations on the dense
array, potentially requiring several full coefficient sweeps before reaching
the final rotation or measurement kernel.

The symbolic planner already knows the Pauli's active-coordinate $X$ and $Z$
masks. A direct kernel can therefore do the same operation without changing
basis:

- a diagonal Pauli changes each amplitude using the parity selected by its
  $Z$ mask; and
- a non-diagonal Pauli updates amplitude pairs whose indices differ by its
  $X$ mask, with phases determined by the $Z$ mask.

For example, a rotation about $X_0 Z_2$ can pair each basis index $a$ directly
with $a \mathbin{\mathrm{xor}} 001$ and obtain the sign from bit 2. The old
compiled path instead executed the already-planned transformations that mapped
the Pauli to a designated axis, then applied a simple kernel there.

A general dense Pauli rotation must already visit essentially every active
coefficient. Direct diagonal and paired-amplitude kernels make that necessary
sweep the operation itself, rather than adding localization sweeps around it.
Prepared rotation fusion can also combine compatible runs into one traversal.
Localization can still be useful when a layout change is amortized across many
later operations, but it is no longer required as the universal execution
mechanism. Once direct prepared actions covered rotations, measurements,
instruments, and output semantics, retaining the SVM and its localization
bytecode added a second implementation without providing a needed capability.

## What does "symbolic" mean here?

Noise events, measurement branches, and record-controlled feedback vary from
shot to shot, but their possible effects are known during compilation. The
planner gives the relevant binary events Boolean symbols and expresses a
branch-dependent sign as an affine formula such as

$$
c \oplus s_2 \oplus s_5.
$$

It then propagates these formulas through the Clifford-Pauli frame and attaches
the resulting expressions to rotations, measurements, records, detectors, and
observables. The plan therefore fixes which events can affect each action even
though it cannot know their values yet.

During a shot, the executor assigns each sampled symbol once and incrementally
updates only the expressions that depend on it. This replaces per-shot tableau
evolution and physical-qubit Pauli-frame updates with Boolean assignments and
prepared active-state actions. The structure is compiled once; only the
branch values remain dynamic.

## Where did the new method come from?

The [original Clifft paper](https://arxiv.org/abs/2604.27058), by Bradley A.
Chase and Farrokh Labib, introduced Clifft's factored active-state
representation and describes the earlier localized-Pauli SVM.

[SymFT](https://arxiv.org/abs/2607.28600), by Wang Fang, Huazhe Lou, and
Riling Li, is the second-generation successor to
[SOFT](https://arxiv.org/abs/2512.23037). Its paper describes a planner that
combines SOFT's generalized-stabilizer simulation with Clifft's dense
active-state representation. It adds symbolic Clifford-Pauli-frame
factorization, adaptive stabilizer-coordinate planning, and direct
multi-coordinate instructions.

The current Clifft sampler adopts those SymFT developments. Clifft's HIR
optimization, `SamplingPlan` boundary, executable lowering, incremental
expression execution, rotation fusion, scalar and SIMD kernels, instruments
and continuations, and public APIs are its own integration of those ideas.

The active-state factorization is therefore continuous with the original
Clifft method, while the symbolic frame, planner, and direct execution model
come from the later SymFT architecture.

## Is the new sampler faster?

Across the seven real workloads below, current Clifft is faster than the last
legacy SVM on five and remains close on the other two. The same campaign also
compares SymFT's non-batched CPU sampler and its packed cross-shot counts path.
The latter is an architectural throughput comparison, not the same execution
mode as Clifft's current one-shot-at-a-time sampler.

### Sampling throughput

These are median attempted shots per second, including shots discarded by
postselection. Higher is better.

| Circuit | Peak $k$ | Legacy Clifft | Current Clifft | SymFT single | SymFT batch |
|---|---:|---:|---:|---:|---:|
| Surface code `d=7, r=7` | 0 | 295k/s | 453k/s | 190k/s | 3.44M/s |
| Cultivation `d=3` | 4 | 735k/s | 1.07M/s | 726k/s | 2.58M/s |
| Cultivation `d=5` | 10 | 67.3k/s | 125k/s | 117k/s | 162k/s |
| Distillation | 5 | 91.2k/s | 534k/s | 272k/s | 1.74M/s |
| Coherent `d=3, r=1` | 4 | 1.36M/s | 1.33M/s | 976k/s | 2.55M/s |
| Coherent `d=3, r=3` | 7 | 346k/s | 394k/s | 425k/s | 664k/s |
| Coherent `d=5, r=1` | 12 | 15.2k/s | 14.1k/s | 32.3k/s | 30.5k/s |

Current Clifft is within 2% of the legacy SVM on coherent `d=3, r=1` and
about 7% on coherent `d=5, r=1`. Relative median absolute deviation for the
current-Clifft cells ranged from 0.5% to 3.6%.

Against SymFT's non-batched path, current Clifft leads on five rows, trails by
about 8% on coherent `d=3, r=3`, and trails by 2.29x on coherent `d=5, r=1`.
SymFT selected exact product-component execution only for the latter circuit;
Clifft currently uses one monolithic active vector. This is a useful measured
case for the [product-component investigation](https://github.com/unitaryfoundation/clifft/issues/314),
not evidence that every width-12 circuit has the same gap.

SymFT batching provides its largest gains on the active-width-zero and
low-width rows and much less on cultivation `d=5`. On coherent `d=5, r=1`,
batching adds no gain over SymFT's component path.

### Compilation time

These are median end-to-end compilation times. Lower is better.

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
than either tested SymFT preparation mode.

??? note "Benchmark protocol, revisions, and raw data"
    The campaign used 12 balanced process-level runs on one pinned core of an
    AMD EPYC 9554P (Zen 4) KVM host. Each process used one thread, a fresh
    paired seed, an excluded warmup, and an independently calibrated shot count
    targeting about 1.5 seconds. Clifft used GCC 13.3 `Release` builds
    (`-O3 -DNDEBUG`) with an `x86-64-v2` baseline and the AVX-512 runtime path
    forced. SymFT used its native CPU build.

    Every arm received the same circuit view, explicit noiseless detector and
    observable references, detector postselection, and aggregate
    survivor-output contract. Compilation and sampling were timed separately;
    a shared reference-syndrome calculation was excluded. Shot counts and all
    process results are recorded in the raw data.

    The measured revisions were current Clifft
    [`04c4fe6`](https://github.com/unitaryfoundation/clifft/commit/04c4fe662d9b42d06817450096dbb56a541e709d),
    the legacy-default Clifft SVM
    [`aa7e7a3`](https://github.com/unitaryfoundation/clifft/commit/aa7e7a3d3e03d0414bb4f5757d9a7204b082539c),
    and SymFT
    [`c89b985`](https://github.com/haoliri0/SOFT/commit/c89b98514a919240b8afa53a271e08d926d3c987),
    including corrected CPU reference normalization. The QEC inputs are pinned
    to the
    [Clifft paper corpus at `db7dc9f`](https://github.com/unitaryfoundation/clifft-paper/tree/db7dc9f13a2c2854690e92390c779048a1ac1400/qec_bench).

    - [Summary and all cell statistics](../assets/updates/symbolic-sampler-2026-08/summary.json)
    - [All primary AVX-512 process samples (gzip-compressed JSON)](../assets/updates/symbolic-sampler-2026-08/raw-avx512.json.gz)
    - [All forced-AVX2 process samples](../assets/updates/symbolic-sampler-2026-08/raw-avx2.json)

    The artifacts include circuit hashes, expected detector and observable
    strings, compiler flags, calibration, shots, seeds, times, counts, memory
    peaks, medians, and dispersion. They also retain controlled
    active-width-12, QV-10, QV-20, and coherent `d=5, r=5` guards. QV-20 and
    coherent `d=5, r=5` completed too few shots per process for precise
    throughput claims, so they are absent from the headline table.

    These measurements describe one CPU, compiler, and corpus; they are not
    performance guarantees. The public
    [clifft-bench](https://github.com/unitaryfoundation/clifft-bench) project is
    the planned home for ongoing cross-simulator benchmarks.

## What future work was deferred from this rewrite?

As of this rewrite, Clifft deliberately deferred several independent
performance approaches to focused follow-ups:

- [packed single-threaded cross-shot execution](https://github.com/unitaryfoundation/clifft/issues/313);
- [exact product-component active states](https://github.com/unitaryfoundation/clifft/issues/314);
- [intra-shot parallel kernels and NUMA placement](https://github.com/unitaryfoundation/clifft/issues/312);
- [cross-shot worker parallelism](https://github.com/unitaryfoundation/clifft/issues/343); and
- [Apple Silicon-specific kernels](https://github.com/unitaryfoundation/clifft/issues/299).

The cutover establishes the single-shot symbolic baseline first. These
features target different workload regimes and can now be measured and added
without retaining the legacy SVM.
