# Smaller Plans, Shared Trajectories (v0.14.0)

**Release date:** November 24, 2026

Clifft 0.14.0 makes exact near-Clifford CPU sampling substantially faster out
of the box—achieving an **11.1x median speedup** on one core and **9.83x** on
eight cores across our nine paired QEC circuits, without per-circuit tuning.

Two ideas drive most of that jump, both inspired by QuEra's
[Tsim](../acknowledgments.md#tsim):

1. **Global sampling reconstruction** reorganizes a circuit for the specific
   outputs you ask for, shrinking the active quantum state beyond what local
   compiler passes can achieve alone.
2. **Shared trajectory execution** lets shots that follow the same quantum path
   share statevector work while keeping independent classical records and
   dropping failed postselected shots early.

Both features are now on by default in `clifft.compile()` and the CPU sampling
functions, alongside automatic active-width scheduling, runtime AVX2/AVX-512
batch dispatch, and Stim-compatible bit-packed decoder outputs. Thanks to
Rafael Haenel (`@rafaelha`) for proposing an initial ZX-calculus reconstruction
prototype in [#606](https://github.com/unitaryfoundation/clifft/pull/606), and
to David Ryan (`@ddri`) for inverted measure-reset absorption and OpenQASM 2
parser hardening in [#675](https://github.com/unitaryfoundation/clifft/pull/675)
and [#673](https://github.com/unitaryfoundation/clifft/pull/673).

## Global sampling reconstruction

Clifft's sampling cost scales with $2^k$, where $k$ is the *active width*—the
number of non-Clifford degrees of freedom live in the dense quantum state at
once. Local compiler passes (fusion, phase-polynomial reduction, squeezing, and
scheduling) already shrink $k$ substantially, but they preserve the circuit's
sequential gate-and-measurement structure.

When you only need sampled outcomes—measurements, detectors, observables, or
postselected survivor counts—you do not need to replay the circuit's original
gate-by-gate timeline. Global reconstruction steps outside that sequential
order: it folds the circuit's Clifford gates, noise propagation, and any
pure-Clifford tail (such as cultivation's surface-code escape stage) into a
compact sampling problem over the initial $|0\dots 0\rangle$ state, then
reorders the surviving quantum operations measurement-first to keep active
width as small as possible.

Because Clifft tracks only the exact outputs you requested—either full
measurement records or just detectors and observables—through that folded
frame, it can prove algebraically which intermediate details cannot change the
output distribution and drop them safely:

- **Dead faults and fixed axes:** Pauli faults on qubits discarded before any
  measurement, and coordinates that remain fixed in $|0\rangle$, are removed.
- **Unobservable commuting checks:** Intermediate projections that commute with
  every remaining non-Clifford rotation and every requested output have no
  effect on the sampled outcomes and are dropped.
- **Merged noise channels:** Independent noise sites that flip the exact same
  set of rotations and outputs are combined into a single channel.

This shares the same closed-output idea and measurement-first scheduling that
[Tsim](https://github.com/QuEraComputing/tsim/pull/203) pioneered with
ZX-calculus diagrams. Whereas Tsim converts a circuit into a ZX graph, applies
ZX rewrites to expose a smaller set of phase gadgets, and extracts a narrower
circuit, Clifft performs the analogous reduction natively in its own
stabilizer-tableau algebra—avoiding a separate ZX engine and feeding the result
straight back into Clifft's existing optimizer and executor.

![Four-stage global sampling reconstruction pipeline: 1 Original circuit in fixed gate order, 2 Fold Cliffords and pure-Clifford tails into a stabilizer tableau frame over |0...0>, 3 Drop dead faults, fixed |0> axes, commuting checks, and merge equivalent noise channels while keeping exact output distributions, 4 Reschedule surviving operations measurement-first into a narrower HIR sampling plan](../assets/updates/reconstruction-pipeline-v014-light.svg#only-light)
![Four-stage global sampling reconstruction pipeline: 1 Original circuit in fixed gate order, 2 Fold Cliffords and pure-Clifford tails into a stabilizer tableau frame over |0...0>, 3 Drop dead faults, fixed |0> axes, commuting checks, and merge equivalent noise channels while keeping exact output distributions, 4 Reschedule surviving operations measurement-first into a narrower HIR sampling plan](../assets/updates/reconstruction-pipeline-v014-dark.svg#only-dark)

On `Cultivation d5` (`p = 1e-3`, postselected survivor counts), that rewrite
drops peak active width from **10 to 6**—cutting the peak dense state from
1,024 amplitudes to 64 (**16x smaller**) and lowering estimated dense work by
**17.8x**:

![Active width across quantum operations on Cultivation d5 before and after global reconstruction: peak width falls from 10 (1,024 amplitudes) to 6 (64 amplitudes), a 16x reduction in peak state size](../assets/updates/reconstruction-width-v014-light.svg#only-light)
![Active width across quantum operations on Cultivation d5 before and after global reconstruction: peak width falls from 10 (1,024 amplitudes) to 6 (64 amplitudes), a 16x reduction in peak state size](../assets/updates/reconstruction-width-v014-dark.svg#only-dark)

`clifft.compile()` now defaults to `reconstruction="auto"`, preparing narrower
plans for full-record and syndrome-only sampling when they improve on the
original plan. If a circuit's original plan overflows the 60-coordinate dense
limit, compilation can still succeed for output modes that reconstruction
brings below the limit. You can inspect both plans on the compiled `Program`:

```python
import clifft

# hir_passes=None highlights reconstruction on a minimal two-qubit example
program = clifft.compile("H 0 1\nT 0 1\nH 0 1\nM 0 1", hir_passes=None)
plan = program.sampling_plan(keep_records=True)

assert plan.reconstruction_selected
assert plan.peak_active_width == 1
assert program.original_plan.peak_active_width == 2
```

Pass `reconstruction=False` if compilation time dominates a very short job;
exact probability queries, replay, and fixed-fault `sample_k()` continue to use
`program.original_plan`. See
[Global reconstruction for sampling](../guide/compilation.md#global-reconstruction-for-sampling)
and the [reconstruction theory](../theory/stabilizer-optimization.md#closed-output-global-reconstruction)
for eligibility rules and details.

## Shared trajectory execution

Even after compilation shrinks the active state, ordinary batched sampling
still evolves a separate statevector for every shot. In low-noise QEC circuits,
postselected magic-state preparation, and noiseless circuits with measurements
at the end, most shots follow the exact same quantum trajectory—only a small
fraction split off when a noise fault flips a later rotation sign or an active
measurement takes a different branch.

Shared trajectory execution groups shots that share the same quantum state into
a single evolution, splitting groups only when a fault or active measurement
causes trajectories to diverge, and pruning shots as soon as a postselected
detector fails:

![Ordinary sampling versus shared trajectory execution: ordinary sampling evolves 8 independent statevectors for 8 shots, while shared trajectory execution evolves one shared trunk, splits only on diverging faults, and drops failed postselected shots early](../assets/updates/shared-trajectories-v014-light.svg#only-light)
![Ordinary sampling versus shared trajectory execution: ordinary sampling evolves 8 independent statevectors for 8 shots, while shared trajectory execution evolves one shared trunk, splits only on diverging faults, and drops failed postselected shots early](../assets/updates/shared-trajectories-v014-dark.svg#only-dark)

Each shot still keeps its own independent classical noise, measurement,
detector, and observable records. On nonzero-width plans with
`batch_size="auto"` and `sharing="auto"`, Clifft enables shared execution
automatically for:

- postselected plans in `sample_survivors()` (at 64+ shots);
- noiseless unitary-prefix plans such as Quantum Volume (at width 8+ and 16+
  shots); and
- unpostselected noisy plans whose shots stay on few trajectories (at width 3+
  and 64+ shots).

Under the hood, v0.14 packs shared states into contiguous 64-byte-aligned
storage, reuses idle worker pools across calls on a `Program`, and splits
groups by visiting only the lanes that disagree. You can inspect what the
automatic selector chose on `result.execution`:

```python
import clifft

program = clifft.compile(
    "H 0 1\nT 0 1\nDEPOLARIZE1(0.001) 0 1\nH 0 1\nM 0 1\n"
    "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]",
    postselection_mask=[1],
)
result = clifft.sample_survivors(program, shots=10_000, seed=42)

assert result.execution.executor == "shared"
assert result.execution.sharing is True
assert result.passed_shots + result.discards == 10_000
```

Pass `sharing=True` or `sharing=False` to override automatic selection. See
[Shared trajectory execution](../guide/cpu-execution.md#shared-trajectory-execution)
for memory budgets and tuning options.

## Selective and bit-packed decoder outputs

QEC decoding pipelines rarely need the full unpacked measurement record for
every shot—they need detector syndromes and logical observables in compact
bit-packed arrays. Clifft 0.14 adds `keep_records` (`True`/`"all"`,
`"syndromes"`, `"measurements"`, or `False` for survivor counts) and
`bit_packed=True` (Stim-compatible little-endian `b8` format, eight columns per
byte), along with the decoder helper `clifft.sample_detectors()`:

```python
import clifft

program = clifft.compile(
    "H 0\nT 0\nH 0\nDEPOLARIZE1(0.01) 0\nM 0\n"
    "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
)
detectors, observables = clifft.sample_detectors(
    program,
    shots=1_000,
    seed=42,
    separate_observables=True,
    bit_packed=True,
)
assert detectors.shape == (1_000, 1)
assert observables.shape == (1_000, 1)
```

Selecting only the outputs you consume lets both reconstruction and sampling
skip unused classical records and reduces output memory. See
[Select and pack outputs](../guide/simulation.md#select-and-pack-outputs).

## Faster defaults and upgrading

Two additional changes round out the out-of-the-box speedups in v0.14:

- **Default active-width scheduling:** `clifft.compile()` now includes a
  bounded `ActiveWidthSchedulePass` by default, so you no longer need to wire
  up a custom pass manager to get width-aware scheduling.
- **Runtime AVX2 and AVX-512 batch dispatch:** Packed CPU batch kernels now
  dispatch to AVX2 or AVX-512 at runtime inside the standard prebuilt wheels,
  giving a 1.4x–2.5x boost on ordinary packed workloads on modern x86 CPUs.

All default changes preserve exact sampling distributions, though compiled
plans, compile time, and fixed-seed rows can differ from v0.13. To restore
v0.13-style compilation and execution choices, pass `reconstruction=False` and
an explicit `HirPassManager` without `ActiveWidthSchedulePass` to `compile()`,
and `sharing=False` with a numeric `batch_size` when sampling. Circuits using
inverted `MR`/`MRX`/`MRY` targets also follow the new Clifford-frame handling
described in `CHANGELOG.md`; pin `clifft==0.13.0` if exact cross-version replay
is required.

## Release benchmark results

![Clifft v0.14 survivor-count throughput ratios over v0.13 on one and eight cores side by side: gains on all nine paired circuits, ranging from 1.08x to 65.0x](../assets/updates/v014-vs-v013-light.png#only-light)
![Clifft v0.14 survivor-count throughput ratios over v0.13 on one and eight cores side by side: gains on all nine paired circuits, ranging from 1.08x to 65.0x](../assets/updates/v014-vs-v013-dark.png#only-dark)

On the AWS `m7a.4xlarge` CPU release suite using Clifft's default compiler and
sampling settings:

- Across all nine paired survivor-count circuits, v0.14 delivers an **11.1x
  median speedup** on one core (`1.74x`–`65.0x`) and **9.83x** on eight cores
  (`1.08x`–`58.2x`) over v0.13.
- Across the fixed six-circuit QEC history tracked since v0.1, v0.14 reaches a
  **134x median chained speedup**.
- Against peer simulators on the expanded 18-circuit suite, Clifft leads Tsim
  1.0.0 on **16/16** packed circuits on eight cores (**12/16** on one core) and
  **10/10** survivor-count circuits, SymFT 2026.10.8 on **7/7** survivor-count
  circuits, and xtim 4.1.2 on **4/4** eight-core survivor-count circuits.

Explore the interactive throughput chart, per-circuit compile times, and
eight-core scaling in the [Performance guide](../guide/performance.md), and see
the [release notes](https://github.com/unitaryfoundation/clifft/blob/main/CHANGELOG.md)
for the complete list of changes and compatibility details.
