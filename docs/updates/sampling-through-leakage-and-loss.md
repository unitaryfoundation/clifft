# Sampling Through Leakage and Loss (v0.13.0)

**Release date:** Pending

A lost qubit can remove gates from the rest of a circuit. Leakage can disturb
its gate partners, change later readout, and trigger feedback. Simulating
these effects means following both the quantum state and which operations
still apply on each shot.

Clifft 0.12 expanded the ways to describe those effects. Version 0.13 adds
execution choices to make them practical to sample: reusable, batched plans
for eligible models, and an online backend that responds to events as they
happen. The compiler also finds more simplifications using what measurement
and postselection reveal about the state.

## Compile once and sample in batches

Clifft's trajectory sampler recompiles the remaining circuit when a leakage
or loss event changes later operations. That supports detailed models, but
repeated compilation can become a substantial part of the cost of collecting
many shots.

Some models allow the same compiled quantum circuit to serve every shot.
Clifft can sample their leakage/loss statuses and associated noise separately,
then supply those choices to a fixed plan. This opens up packed execution,
where one worker processes several shots together.

For example, suppose a lost qubit fully depolarizes its partner at a two-qubit
gate. With that noise applied, keeping the gate in the compiled circuit gives
the same output statistics as dropping it and fully randomizing the partner.
That lets this model use fixed execution:

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

The new `noncomp.compile()` API prepares a reusable program, so repeated
sampling calls share the preparation work. Here, the classifier assigns a
random readout bit to a lost site. With the default
`execution="auto"`, Clifft chooses fixed execution when the model permits it
and trajectories otherwise. `program.explain()` reports the choice and any
reason for falling back. The [execution guide](../guide/leakage-and-loss.md#fixed-execution-vs-trajectories)
describes which physical assumptions allow a fixed plan.

Automatic batching accounts for the classical work and memory as well as the
quantum state. For longer jobs, fixed execution also supports
[`batch_size="tune"`](../guide/leakage-and-loss.md#tuning-batch-size): it tries
several batch sizes and reports one you can reuse on subsequent calls.

## Follow changes during a shot

Other models need more flexibility. A lost qubit might cause a gate to be
skipped while leaving its partner undisturbed, or a transition probability
might depend on the qubit's quantum state. Recompiling after each event can
be costly when many such events occur.

The new `execution="online"` option updates the Clifford frame and compact
active state as each shot runs. It can therefore skip gates, handle
state-dependent transitions, and continue after recapture without compiling
a new continuation.

For the same circuit, a model that simply drops gates involving lost qubits
can use online execution explicitly:

<!--pytest-codeblocks:cont-->
```python
online = noncomp.compile(
    circuit, noncomp.Model(classifier=classifier), execution="online"
)
result = online.sample(shots=1000, seed=42, collect_stats=True)
assert result.stats.online_shots == 1000
assert result.stats.continuation_compilations == 0
```

This is a useful choice when trajectory recompilation dominates sampling
time. Online execution runs scalar shots, with optional parallel workers,
and skips compiler optimization passes. Its active state can therefore be
larger, and each extra active qubit doubles the dense-state storage. Compare
sampling time and peak active width for your circuit; automatic selection
continues to choose between fixed execution and trajectories. See
[Online Execution](../guide/leakage-and-loss.md#online-execution) for controls
and supported operations.

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
cases, while the default retained roughly v0.12 performance. Avoiding
recompilation helps some models substantially; it does not make online
execution the fastest choice for every circuit.

These are local measurements on an AMD EPYC 9554P VM, using one pinned vCPU
and matching Clang 18 Release builds of v0.12.0 and the v0.13 development
commit [23ff24f](https://github.com/unitaryfoundation/clifft/commit/23ff24f542bb9708f24b22ea5e5c5d2a3f4f827f).
Each entry is the median of five calls producing **2,000 shots**, including
preparation and compilation, with all outputs and final statuses retained.
Circuit parsing and model construction happen before timing. The
[circuits, models, settings, and individual timings](../assets/updates/noncomp-sampling-v013.json)
are available alongside this post.

## Use what measurements tell us

For circuits compiled with `clifft.compile()`, this release also simplifies
more non-Clifford operations using known properties of the state. Earlier
non-Clifford gates need not prevent the compiler from learning something
useful later. An ideal Z measurement followed by an X correction
when the result is one, for example, prepares `|0>` regardless of the earlier
state. The phase and rotation passes can now recover that fact and use it to
simplify subsequent operations.

Phase optimization also uses supported postselected measurements to reason
about the shots that survive. It searches larger groups of commuting T
rotations and removes rotations that cannot affect any remaining recorded
output. These reductions can save work on every shot, at the cost of more
compilation effort. The new `default_hir_pass_manager(overrides=[...])` API
lets you adjust one default pass while keeping the rest of the pipeline.

See the [pass reference](../reference/passes.md#constraints-from-measurement-and-feedback)
for the conditions and controls, and [Optimization for Simulation](../theory/stabilizer-optimization.md)
for the reasoning. The [Merlin research and examples](../acknowledgments.md#merlin)
helped inform this work.

## Check the histories behind the samples

Rare leakage and loss histories are easy to miss in random sampling. The new
validation runs selected histories directly, accounts for their probabilities,
and compares their physical states and joint outputs with an independent
reference. It also checks the state change when a state-dependent transition
does *not* occur, since that filtering can affect later interference.

Shared structured circuit tests now exercise fixed, trajectory, and online
execution, using Qiskit Aer and Stim references where applicable. Together,
these checks cover both the physical evolution along a history and the
observable results across execution choices. See
[Testing](../development/testing.md#weighted-noncomputational-histories) for
the validation approach, and the
[release notes](https://github.com/unitaryfoundation/clifft/blob/main/CHANGELOG.md)
for compatibility changes when upgrading.
