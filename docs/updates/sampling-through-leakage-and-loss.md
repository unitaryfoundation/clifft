# Improved Leakage and Loss Sampling Throughput (v0.13.0)

**Release date:** November 4, 2026

Since v0.7, Clifft has supported sampling non-Clifford circuits with leakage
and loss. Leakage moves a qubit outside its computational subspace; loss
removes the physical carrier from its site. These errors can change which
subsequent gates act and what later measurements report.

Leakage and loss complicate Clifft's compilation model. Normally, Clifft
compiles a circuit once and reuses the resulting plan across many shots. If a
qubit is lost, gates involving it may disappear from the remaining circuit,
invalidating that plan. Until now, Clifft handled such events by compiling a continuation
for the rest of the shot. Repeated compilation can dominate sampling time.

Clifft 0.13 adds two execution modes to avoid that cost: fixed execution
reuses one compiled plan for eligible models, while online execution applies
the circuit directly as each shot runs. Separately, the ordinary circuit
compiler now recovers more known stabilizers after measurement and feedback,
and uses postselection to simplify more non-Clifford operations.

## Fixed and batched execution

For some noise models, the leakage/loss statuses and associated noise can be
sampled independently of the quantum state. Clifft can generate those choices
as classical inputs to one fixed quantum plan, including processing many
shots together in packed batches.

Two-qubit gates need special treatment. Suppose a lost qubit fully depolarizes
its partner whenever they would interact. Applying that depolarization after
the gate gives the same reduced state on the remaining qubits as dropping
the gate and depolarizing the partner. The compiled plan can therefore keep
the gate, with the partner noise conditioned on loss. Here is a model that
allows this, with random readout on lost sites:

```python
from clifft import noncomp

circuit = """
    H 0
    T 0
    LOSS(0.01) 0
    CX 0 1
    H 0 1
    M 0 1
"""
classifier = noncomp.Classifier(
    [[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]]
)
model = noncomp.Model(
    classifier=classifier,
    gate_partner_effects={
        "lost": noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25)),
    }
)
program = noncomp.compile(circuit, model)
assert program.strategy == "fixed"

first = program.sample(shots=10_000, seed=42)
second = program.sample(shots=10_000, seed=43)
```

The new `noncomp.compile()` API returns a reusable program. Its default
`execution="auto"` selects fixed execution when the circuit and model permit
it, and trajectories otherwise. `program.explain()` reports the selection
and any fallback reason; the [execution guide](../guide/leakage-and-loss.md#fixed-execution-vs-trajectories)
gives the eligibility conditions.

## Online execution without recompilation

A model that drops gates after loss without disturbing the remaining partner
cannot use the fixed plan above. State-dependent leakage rates also require
resolving transitions against the quantum state during sampling.

The new online backend updates the Clifford frame and active state as it
executes each shot. When a qubit leaks or is lost, it changes which subsequent
gates apply without compiling a continuation. To use it on the same circuit
with gates simply dropped after loss:

<!--pytest-codeblocks:cont-->
```python
online = noncomp.compile(
    circuit, noncomp.Model(classifier=classifier), execution="online"
)
result = online.sample(shots=10_000, seed=42)
```

Online execution pays for Clifford-frame updates on every shot and skips the
compiler's optimization passes. Its active state can be larger, and it
currently processes one shot at a time per worker. The measurements below
show both a gain from avoiding recompilation and cases where trajectories
remain faster. Online execution requires explicit `execution="online"`;
see the [guide](../guide/leakage-and-loss.md#online-execution) for its controls.

## Local sampling measurements

On 33-qubit repetition-code circuits with five rounds, fixed execution was
about **600x faster** than v0.12 for the two loss models with full partner
depolarization. When loss instead dropped gates without disturbing their
partners, explicit online execution was about **14x faster**.

The loss rows use 1% loss per data qubit per round. The state-dependent rows
use 0.8% leakage and 0.2% loss from the excited state after each S gate.

| Model | v0.12 time (ms) | v0.13 auto time (ms) | v0.13 online time (ms) |
| --- | ---: | ---: | ---: |
| Loss, full partner depolarization | 609 | 1.02 | 34.4 |
| Same, with coherent X rotations | 692 | 1.11 | 46.5 |
| Loss, dropped gates, coherent X rotations | 636 | 624 | 45.6 |
| State-dependent leakage and loss | 765 | 775 | 1,280 |
| Same, with coherent X rotations | 803 | 805 | 1,263 |

Automatic execution chose fixed plans for the first two rows and trajectories
for the others. Online execution was **1.6-1.7x slower** on the state-dependent
cases, while the default retained roughly v0.12 performance.

These are local measurements on an AMD EPYC 9554P VM, using one pinned vCPU
and matching Clang 18 Release builds of v0.12.0 and the v0.13 development
commit [23ff24f](https://github.com/unitaryfoundation/clifft/commit/23ff24f542bb9708f24b22ea5e5c5d2a3f4f827f).
Each entry is the median of five calls producing **2,000 shots**, including
preparation and compilation, with all outputs and final statuses retained.
Circuit parsing and model construction happen before timing. The
[circuits, models, settings, and individual timings](../assets/updates/noncomp-sampling-v013.json)
are available alongside this post.

## Stabilizer recovery and phase simplification

For ordinary circuits compiled with `clifft.compile()`, measurement and
feedback can restore known Pauli eigenvalues after non-Clifford gates. In
this example, `M 0` followed by `CX rec[-1] 0` prepares qubit 0 in `|0>`.
Consequently, `Z0*Z1` and `Z1` act identically on the state, and the two
opposite rotations on qubit 1 cancel:

```python
import clifft

source = """
    H 0
    T 0
    H 0
    M 0
    CX rec[-1] 0
    H 1
    R_Z(0.137) 1
    R_PAULI(-0.137) Z0*Z1
    H 1
    M 1
"""
simplifier = clifft.RotationSimplificationPass()
passes = clifft.default_hir_pass_manager(overrides=[simplifier])
program = clifft.compile(source, hir_passes=passes)
assert simplifier.rotations_removed == 2
```

The rotation pass in v0.12 retained both rotations; the new pass removes them
using the recovered `+Z0` stabilizer. The earlier T gate still affects the
first measurement's distribution. Removing the feedback instruction prevents
the cancellation because qubit 0 then has an outcome-dependent Z eigenvalue.

Postselection can also fix Pauli eigenvalues on surviving shots. The phase
pass now uses supported postselected measurements to recover those
constraints, and collects larger groups of commuting T rotations to expose
cancellations across a complete group.

[Merlin's description of measured code-switching circuits](../acknowledgments.md#merlin)
using affine support and phase polynomials motivated these extensions to
Clifft's Pauli constraint analysis and TOHPE synthesis. Its BT27 circuit
exercises stabilizer recovery through measurement and feedback. The modified
BT81 example contains a commuting phase region with 87 independent variables,
beyond the previous default collection limit of 32; the new default allows
up to 128. The [pass reference](../reference/passes.md#constraints-from-measurement-and-feedback)
describes the conditions under which these rewrites apply.

## Release benchmark results

On the single-core CPU benchmark suite, these compiler and sampling
improvements increase throughput over v0.12 by **223x** on 15-to-1
distillation, **16.2x** on Reed–Muller, and **8.21%** on the fixed-input
16-bit Draper adder. See the
[performance guide](../guide/performance.md#changes-in-v013) for the full
release comparison and simulator benchmarks.

See the [release notes](https://github.com/unitaryfoundation/clifft/blob/main/CHANGELOG.md)
for changed defaults and compatibility details when upgrading.
