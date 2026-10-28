# Leakage and Loss

Pauli noise acts within a qubit's two-dimensional computational subspace. Real
hardware can instead drive the state out of that subspace through *leakage*, or
lose the physical carrier from its site entirely through *loss*. Neither
process is a Pauli channel, so the site no longer holds an ordinary qubit
state.

Leakage and loss can persist until the qubit relaxes, the carrier is
recaptured, or the site is reset. One event can therefore affect a sequence of
later gates, measurements, and detector outcomes. Clifft tracks each site's
status alongside its ordinary coherent simulation. A computational site
remains in the quantum state and can be in a superposition or entangled; a
leaked or lost site occupies one definite noncomputational level in each shot.

`noncomp.sample` automatically chooses an exact fixed plan when the circuit,
model, and requested outputs permit it. Such plans compile once and can sample
shots in packed batches, including supported non-Clifford circuits. Other
requests use general trajectories, rewriting and compiling continuations as
jumps change subsequent operations. Both strategies preserve the same model
and output semantics.

This page is the API walkthrough. The model and its simulation semantics
are described in [Noncomputational States](../theory/noncomputational.md).
The [Partner Interactions guide](partner-interactions.md) explains optional
Pauli noise and leakage spreading onto computational partners.
The [Delayed Loss tutorial](delayed-loss.md) introduces the API with a
surface-code memory experiment. The advanced [Logical Shor Noise Sweep
tutorial](neutral-atom-leakage.md) applies the complete five-level model to a
published logical experiment.

## A model and its outputs

A `noncomp.Model` defines how qubit sites enter, leave, and are observed
outside the computational subspace. It contains an initial level distribution,
transition matrices, a measurement classifier, optional partner interaction
effects, and policies such as whether a reset restores a lost site. The
classifier maps a known level to probabilities
for a binary measurement result and, optionally, a herald.

Clifft models five `Level` values: the computational levels `g` and `e`,
identified with $\lvert 0\rangle$ and $\lvert 1\rangle$; the leaked levels
`leak_g` and `leak_e`; and `lost`. These levels index transition-matrix rows and
columns and classifier probability columns.

`QubitStatus` describes a site's status in a sampled shot. Both `g` and `e`
correspond to `COMPUTATIONAL` because the site remains in the coherent quantum
state rather than occupying a known basis level. The statuses `LEAK_G`,
`LEAK_E`, and `LOST` name definite noncomputational levels. An initial weight
on `g` or `e` prepares the corresponding computational basis state; later
evolution may put the site in a superposition or entangle it with other sites.

`noncomp.sample` returns the usual measurement, detector, and observable
arrays and `heralds` for each measurement. Detailed `final_status` is opt-in;
it is `None` unless `final_status=True` is requested.
When updating existing calls that consume final status, add `final_status=True`.

### A minimal model

This example randomizes the initial state across all five levels and shows how
the classifier maps those levels to measurement outputs. It has no transitions
between levels after sampling begins.

```python
import numpy as np

from clifft import noncomp

Level = noncomp.Level

# Classifier P[symbol][level]: g and leak_g read 0, e and leak_e read 1,
# lost heralds (third symbol).
classifier = noncomp.Classifier(
    [
        [1, 0, 1, 0, 0],  # P(record 0 | level)
        [0, 1, 0, 1, 0],  # P(record 1 | level)
        [0, 0, 0, 0, 1],  # P(herald   | level)
    ]
)

model = noncomp.Model(
    initial_state=[0.92, 0.03, 0.02, 0.01, 0.02],  # P(g, e, leak_g, leak_e, lost)
    classifier=classifier,
)

r = noncomp.sample("M 0", model, shots=20_000, seed=1, final_status=True)

# final_status: (shots, num_qubits) of QubitStatus values.
status = r.final_status[:, 0]
lost = (status == noncomp.QubitStatus.LOST).mean()
assert abs(lost - 0.02) < 0.005

# heralds: (shots, num_measurements); 1 where the classifier heralded.
assert (r.heralds[:, 0] == (status == noncomp.QubitStatus.LOST)).all()

# symbols() reports the herald as value 2 instead of the placeholder bit.
symbols = r.symbols()
assert np.array_equal(symbols[:, 0] == 2, r.heralds[:, 0])
```

The output arrays have the same layout as ordinary sampling: every measurement
still occupies its record slot. On a computational site, Clifft performs the
requested quantum measurement. For `M` and `MR`, the classifier's `g` and `e`
columns can then add computational-basis readout confusion. On a leaked or lost
site, the classifier supplies the result in place of a quantum measurement and
applies to `M`, `MX`, and `MY` alike.

!!! note "Computational readout confusion is Z-basis only"
    The classifier's `g` and `e` columns apply to `M` and `MR`. On a
    computational site, `MX`, `MY`, `MRX`, and `MRY` perform their ordinary
    quantum measurements without consulting those columns. Use measurement
    noise such as `MX(p)` or `MY(p)` for readout errors in another basis. If
    the site is leaked or lost, the classifier supplies the result regardless
    of measurement basis.

When the classifier emits its herald symbol, `heralds` marks the slot and the
binary `measurements` entry holds a uniformly drawn placeholder. Detectors and
observables use that binary bit as usual; the herald remains separate.
`symbols()` folds the herald back in as a third value per slot (0, 1, or 2) for
comparison with tools that report loss in-band.

`heralds[shot, slot]` means that the classifier emitted its third symbol at
that measurement slot. It does not identify when or where the underlying
transition occurred, so it is not an exact spacetime erasure flag. The herald
is side information for downstream analysis or adaptive decoding.

Omitting `initial_state` starts every site in `g`, which matches the standard
Clifft convention that all qubits start in $\lvert 0 \rangle$. A model capable
of leakage or loss requires a classifier whenever the circuit contains any
physical-qubit measurement, even if the measured site cannot itself become
noncomputational. `MPAD` and the status probes below are exempt because their
record bits do not measure the site's computational state.

## Nondestructive status checks

`HERALD_LEAKAGE_EVENT` and `HERALD_LOSS_EVENT` report the status at a circuit
position without changing the site's quantum state or occupation. Each plain
qubit target appends one ordinary record bit, in target order:

| Status | `HERALD_LEAKAGE_EVENT` | `HERALD_LOSS_EVENT` |
|--------|------------------------|---------------------|
| Computational | 0 | 0 |
| `LEAK_G` or `LEAK_E` | 1 | 0 |
| `LOST` | 0 | 1 |

Status probes are perfect and take no arguments. Targets must be
plain qubit indices without inversion.

The result is available to `DETECTOR`, `OBSERVABLE_INCLUDE`, and record-controlled
feedback. These probes require no classifier and leave the classifier `heralds`
sidecar zero at their record slots. They report status at the time of the check,
not the time of the underlying jump.

```python
from clifft import noncomp

result = noncomp.sample(
    """
    LEAKAGE(1) 0
    LOSS(1) 1
    HERALD_LEAKAGE_EVENT 0 1
    HERALD_LOSS_EVENT 0 1
    CX rec[-1] 2
    DETECTOR rec[-4]
    OBSERVABLE_INCLUDE(0) rec[-1]
    """,
    noncomp.Model(),
    shots=16,
    seed=1,
)
assert (result.measurements == [1, 0, 0, 1]).all()
assert result.detectors.all()
assert result.observables.all()
assert not result.heralds.any()
```

To model detection errors, apply `READOUT_NOISE` to the probe's record bit.
This example uses a 2% false-positive rate and a 5% false-negative rate:

```text
HERALD_LEAKAGE_EVENT 0
READOUT_NOISE(0.02, 0.05) rec[-1]
```

Use `READOUT_NOISE(0, p)` for missed detections only, or `READOUT_NOISE(p)` for
symmetric errors. These errors affect the reported bit, leaving the site's
quantum state and status unchanged.

Like transition annotations, status probes require `noncomp.sample`;
ordinary `clifft.compile` rejects them.

## Transitions: hooks and inline annotations

A transition matrix `T[to][from]` gives the probability of a transition event
from one source level to one destination level. The destination may equal the
source; a column's deficit below 1 is the no-jump probability.

Transitions act as noise associated with specific circuit positions. There are
three ways to place them:

- **Gate hooks.** A `transitions` key that names a gate, such as `"CZ"` or `"S"`,
  evaluates its matrix after every occurrence of that gate. Keys naming
  recognized instructions that cannot be hooked are rejected at model
  construction. This includes noise channels and annotations, instructions
  that never produce a circuit node (`MXX`/`MYY`/`MZZ`,
  `CH`/`CCX`/`CCZ`), and identity no-ops. Use a hook to apply the same
  transition after every occurrence of a gate.
- **Inline references.** `LEVEL_TRANSITION[name] 0` evaluates the named matrix
  from the `transitions` mapping on site 0 at that circuit position. Any
  arbitrary transition name or hookable gate name can be referenced
  explicitly. Use an inline reference at selected circuit positions or with a
  transition whose name is not a gate.
- **Inline leakage.** `LEAKAGE(p) 0` moves `g` to `leak_g` and `e` to
  `leak_e` with probability `p`. An already leaked or lost site is unchanged.
  Use it for the common source-preserving leakage channel without defining a
  five-by-five matrix.
- **Inline loss.** `LOSS(p) 0` loses the carrier at site 0 with probability
  `p` from any occupied level. Use it for a self-contained loss probability
  that needs no matrix in the model.

Gate hooks are shorthand for inline references. Before sampling, Clifft
expands each hook into a `LEVEL_TRANSITION[name]` annotation for every physical
site operand of the hooked gate. Classical feedback (`CX rec[-1] 0`) receives
no annotation: a record-conditioned Pauli is a frame update, not a physical
execution of the hooked gate.

A transition back to `g` or `e` can represent relaxation or recapture into
the computational subspace; a reset represents active re-preparation.

### Combining hooks and inline annotations

This example applies one transition after every `CZ`, reuses the same matrix
at one explicitly selected position, and adds an independent inline loss.

```python
from clifft import noncomp

Level = noncomp.Level


def T(entries):
    """Build a 5x5 transition matrix T[to][from] from {(to, from): p}."""
    m = [[0.0] * 5 for _ in range(5)]
    for (to, frm), p in entries.items():
        m[to][frm] = p
    return m


classifier = noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])
leak = T({(Level.LEAK_E, Level.G): 0.01, (Level.LEAK_E, Level.E): 0.01})

model = noncomp.Model(
    transitions={
        "CZ": leak,  # hook: evaluated on each site after every CZ
        "manual_leak": leak,  # evaluated only where explicitly referenced
    },
    classifier=classifier,
)

circuit = """
    H 0
    CZ 0 1
    LEAKAGE(0.01) 0
    LOSS(0.005) 1
    LEVEL_TRANSITION[manual_leak] 0
    M 0 1
"""
r = noncomp.sample(circuit, model, shots=1000, seed=7)
assert r.measurements.shape == (1000, 2)
```

## What happens on a leaked or lost site

Most unitary gates, ordinary noise channels, and classical corrections that
touch a leaked or lost site are skipped as a whole. By default, this leaves
every computational partner unchanged. For physical two-qubit unitary gates,
`gate_partner_effects` and `InteractionRule` can replace this identity with
Pauli noise and optional leakage spreading on the computational partner.
Leaked and lost sources are configured separately. See
[Partner Interactions](partner-interactions.md) for the configuration API,
explicit circuit annotations, and their ordering with transition hooks. A
single-qubit measurement is different: it keeps its record position, but the classifier
rather than the measurement basis supplies its result (`M`, `MX`, and `MY`
alike).

Correlated-error instructions (`E`/`CORRELATED_ERROR` and
`ELSE_CORRELATED_ERROR`) are the exception. Clifft retains them to preserve
chain conditioning; their Paulis still act on computational operands but are
inert on a noncomputational site.

### Loss from an entangled pair

Losing one half of an entangled pair leaves the partner in the reduced
state. For a Bell pair, that reduced state is maximally mixed, so its
computational-basis measurement is unbiased:

```python
import numpy as np

from clifft import noncomp

Level = noncomp.Level

lose_all = [[0.0] * 5 for _ in range(5)]
lose_all[Level.LOST][Level.G] = 1.0
lose_all[Level.LOST][Level.E] = 1.0

model = noncomp.Model(
    transitions={"S": lose_all},  # the hooked S loses its qubit with certainty
    classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]),
)

r = noncomp.sample("H 0\nCX 0 1\nS 0\nM 0\nM 1", model, shots=20_000, seed=2, final_status=True)

survivor = r.measurements[:, 1]
assert abs(survivor.mean() - 0.5) < 0.01  # the survivor's Z readout is unbiased
assert (r.final_status[:, 0] == noncomp.QubitStatus.LOST).all()
assert (r.final_status[:, 1] == noncomp.QubitStatus.COMPUTATIONAL).all()
```

### Classifier results feed detectors

The classifier supplies the binary record bit before detectors and
observables are evaluated, so a noncomputational readout can produce
detector events:

```python
import numpy as np

from clifft import noncomp

Level = noncomp.Level

leak_up = [[0.0] * 5 for _ in range(5)]
leak_up[Level.LEAK_E][Level.G] = 1.0
leak_up[Level.LEAK_E][Level.E] = 1.0

model = noncomp.Model(
    transitions={"S": leak_up},
    classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]),  # leak_e reads 1
)

# A reference measurement, then the leaked one; the detector XORs them.
circuit = "M 1\nH 0\nS 0\nM 0\nDETECTOR rec[-1] rec[-2]"
r = noncomp.sample(circuit, model, shots=1000, seed=4)
assert (r.detectors[:, 0] == 1).all()  # the leaked qubit always reads 1
```

## State-dependent rates resolve at runtime

If a transition's rate differs between `g` and `e`, the jump probability for
a computational site in a superposition depends on the amplitudes at that
point in the shot. The draw happens at sample time against the live state, so
the statistics are exact. In the example below, the `e` component of
$\lvert + \rangle$ contributes probability $\tfrac12$ to reading 1. The `g`
component has weight $\tfrac12$, and a fraction $p$ of it leaks to a level
that also reads 1. The total is therefore $\tfrac12 + \tfrac{p}{2}$.

```python
from clifft import noncomp

Level = noncomp.Level

p = 0.4
leak_from_g = [[0.0] * 5 for _ in range(5)]
leak_from_g[Level.LEAK_E][Level.G] = p

model = noncomp.Model(
    transitions={"S": leak_from_g},
    classifier=noncomp.Classifier([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]),
)

r = noncomp.sample("H 0\nS 0\nM 0", model, shots=20_000, seed=5)
assert abs(r.measurements[:, 0].mean() - (0.5 + p / 2)) < 0.01
```

Evaluating a transition against the quantum state at that point also preserves
correlations. When the destination depends on the source level, the leaked
site's classified readout stays correlated with its entangled partner. The
semantics and active-width cost are described in
[Noncomputational States](../theory/noncomputational.md).

## Policy knobs

- **`reset_restores_lost`** (default `False`): whether a reset on a lost site
  re-prepares it or is dropped. A reset always restores a *leaked* site; a
  measure-and-reset always produces its classifier record, while this option
  controls only whether the reset half restores a lost site.
- **`damping`** (default `"exact"`): source-dependent total jump rates on a
  coherent site require no-jump back-action. Exact simulation may add one unit
  of peak active width for each affected coherent site; each additional unit
  doubles the active-state dimension. Repeated transition positions on a site
  that remains active do not increase the width further. `"neglect"` omits
  this cost and the no-jump back-action, introducing an error of order
  $\lvert p_g - p_e \rvert$ per transition position. There is no error when
  $p_g = p_e$, so `LEAKAGE(p)` and `LOSS(p)` are always exact. The cost doubles
  with each additional unit of active width because the active-state dimension
  is $2^k$.
- **`seed`**: same contract as ordinary sampling: a fixed seed and execution
  configuration are reproducible, `None` uses hardware entropy.
- **`threads`** (default `1`): number of cross-shot workers. Use a positive
  integer to bound resource use or `"auto"` for the implementation-reported
  hardware concurrency.
  With the same program, seed, shot count, and batch configuration, results
  are independent of worker count. Each worker owns its executor and scratch
  storage, so memory use grows with the worker count. Trajectory workers also
  own any continuations they compile.
- **`max_active_width`**: caps the compiled peak active width before allocating or
  growing the state for that module.
  The cap applies to each compiled module, including branches a given shot
  never takes, so it is conservative.

## Automatic execution and reusable preparation

Existing `noncomp.sample(circuit, model, ...)` calls automatically select an
execution strategy. To reuse preparation, keep a program:

```python
from clifft import noncomp

model = noncomp.Model(
    classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]])
)
program = noncomp.compile("H 0\nLEAKAGE(0.01) 0\nM 0", model)
assert program.strategy == "fixed"
print(program.explain())

result = program.sample(shots=10_000, seed=42)
assert result.final_status is None
second = program.sample(shots=10_000, seed=43)
```

Preparation owns the circuit/model snapshot and draws no shot randomness.
Each sampling call owns its workers and RNG, so a program can be reused
concurrently. The convenience function repeats preparation on each call;
there is no cross-call cache.

| Option | Behavior |
| --- | --- |
| `execution="auto"` | Select an exact supported fixed plan, otherwise use trajectories. |
| `execution="fixed"` | Require fixed execution; reject unsupported requests during preparation. |
| `execution="trajectory"` | Force the general trajectory driver. |
| `final_status=False` | Omit detailed status; `result.final_status` is `None`. |
| `final_status=True` | Return existing `QubitStatus` values; unresolved leaked labels can require trajectories. |
| `batch_size="auto"` | Use shared batching heuristics for fixed execution, scalar execution for trajectories. |
| `batch_size="tune"` | Calibrate fixed execution for this call; reject if the selected strategy is trajectory. |
| `batch_size=1` | Use scalar execution, including a compile-once fixed plan when eligible. |
| `batch_size=N`, N > 1 | Request packing under existing capacity limits; reject if the selected strategy is trajectory. |

Strategy and final-status requirements are fixed by `compile`; batch size and
cross-shot `threads` are sampling options. The default remains one worker.
Survivor/importance sampling and custom pass managers are not supported by
this API.

### Tuning batch size

For fixed execution, `batch_size="tune"` measures several capacities before
sampling. It times the complete sampling work, including classical inputs,
worker setup, quantum execution, and output generation. This can find useful
packed configurations beyond the conservative active-width cutoff in `"auto"`.
The default auto policy and fixed-execution eligibility are unchanged.

<!--pytest-codeblocks:cont-->
```python
result = program.sample(10000, seed=42, batch_size="tune", tuning_budget_seconds=0.25)
# Reuse the selected capacity without paying for another calibration.
next_result = program.sample(10000, seed=43, batch_size=result.batch_tuning.batch_size)
```

Calibration has a soft time budget, defaulting to 0.25 seconds; an individual
probe can overrun it. The returned `clifft.BatchTuningReport` records the selected
capacity, timings, and discarded trial shots even without `collect_stats=True`.
Candidate workers obey the shared memory budgets, including packed classical
storage, and probe output storage is bounded, including heralds and requested
final statuses. A zero budget uses auto without probes; zero shots also skip
calibration. Inconclusive measurements warn and use auto. Trajectory programs
reject `"tune"`, just as they reject an explicit packed capacity.

Tuning is per call and leaves the prepared program unchanged. Its overhead can
exceed the time saved on a short call, so reuse the reported capacity for repeated
workloads. Trial shots never enter the returned results or advance their random
streams. Timing can select a different capacity on another call; pin the reported
capacity to replay results for the same seed and shot count.

### Fixed eligibility

The initial fixed subset supports ground-state initialization, inline leakage
and loss, supported readout/reset/feedback, and reachable inactive/live gates
with full partner depolarization. Non-Cliffords can qualify. Named transitions,
spreading, level-sensitive classifiers, and other unsupported combinations
fall back. Default loss with an identity partner effect can therefore still
use trajectories. Eligibility depends on the whole request, including output
requirements. Detailed status can remain eligible if restoring resets erase
the unresolved leaked labels. See
[fixed execution](../theory/noncomputational.md#fixed-execution)
for details on which circuits and noise combinations are eligible.

With `max_active_width`, automatic selection can reject an over-cap fixed
candidate and retain trajectories. Each continuation must still fit the cap;
an error may therefore arise during sampling. Invalid circuit/model contracts
always raise instead of silently triggering fallback.

### Inspecting execution

`program.explain()` formats the selected strategy, fallback reason, available
source locations, and known width/storage estimates without running shots.
For performance investigations, pass `collect_stats=True` to see what actually
ran: whether shots were packed, how many workers were used, and how often the
trajectory driver compiled continuations.

<!--pytest-codeblocks:cont-->
```python
result = program.sample(shots=10_000, seed=42, collect_stats=True)
assert result.stats is not None
assert result.stats.packed_shots == 10_000
assert result.stats.continuation_compilations == 0
```

These diagnostics are optional; ordinary calls leave `result.stats` as `None`.
They help distinguish recompilation costs from scalar execution or limited
batching. Preparation counts and timing refer to the program's preparation,
not repeated work on each call. See
[NonComputationalStats](../reference/python-api.md#clifft.noncomp.NonComputationalStats)
for the counters, resource estimates, and timing definitions.

### Trajectory continuations

When a request is ineligible for fixed execution, or `execution="trajectory"`
is requested, the prepared program may still compile during sampling.
Preparation validates and resolves the circuit/model once. When a sampled
jump changes subsequent operations, the driver compiles a continuation and
resumes the shot after the jump. Reusing preparation does not eliminate these
compilations; `continuation_compilations` reports how many occurred.

Continuations must preserve the already-executed prefix and its measurement
correlations. The trajectory driver therefore uses a restricted internal
compiler pipeline and checks each prefix before resuming. See
[Optimization Passes](../reference/passes.md#trajectory-safety-metadata)
for the pass restrictions and
[Noncomputational States](../theory/noncomputational.md#how-this-composes-with-clifft)
for the execution model.

## Limits

- **Leakage mobility is not modeled.** Configurable
  [partner effects](partner-interactions.md) support Pauli noise and spreading;
  spreading leaves the original leaked operand leaked. Correlated transfer
  that restores the source requires a joint transition and remains unsupported.
- **Ordinary multi-qubit noise retains the drop policy.** `DEPOLARIZE2/3` and
  `PAULI_CHANNEL_2/3` touching a noncomputational site drop as whole operations.
  Correlated-error chains keep their existing exception. See the
  [noise policy](partner-interactions.md#ordinary-noise-keeps-its-existing-policy)
  before substituting otherwise equivalent noise representations.
- **Coherent leakage is outside the trajectory model.** Jumps into a
  noncomputational level are treated as incoherent, definite occupations.
- **`MPP` is not supported** under a model that can leak or lose qubits:
  a parity of levels outside the qubit subspace has no faithful single-bit
  record. Expand the parity readout into an explicit ancilla circuit; the
  ancilla's ladder gates then follow the configured partner-interaction policy.
- **`EXP_VAL` is not supported.** `NonComputationalSample` has no
  expectation-value output, so circuits containing these probes are rejected
  before sampling.
- **A classifier is required** whenever a model capable of leakage or loss
  meets a circuit containing any physical-qubit measurement. This is a
  model-level capability check, not a per-site reachability analysis.
