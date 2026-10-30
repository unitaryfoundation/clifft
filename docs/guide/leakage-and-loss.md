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

Use `clifft.noncomp.sample()` to simulate a circuit and model in one call, or
`clifft.noncomp.compile()` to compile them once for repeated sampling.

This page walks through the `clifft.noncomp` API. Related pages cover:

- [Partner Interactions](partner-interactions.md): Pauli noise and leakage
  spreading on computational qubits paired with a leaked or lost qubit.
- [Noncomputational States](../theory/noncomputational.md): mathematical
  definition of the five-level model, quantum-jump back-action, and fixed
  execution.
- Tutorials: [Delayed Loss in a Surface Code](delayed-loss.md) (introductory)
  and [Logical Shor Noise Sweep](neutral-atom-leakage.md) (full five-level
  neutral-atom model).

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

`noncomp.sample` returns the usual `measurements`, `detectors`, and
`observables` arrays, plus a `heralds` array for each measurement slot. To
also record each qubit's end-of-shot `QubitStatus`, pass `final_status=True`
(by default, `result.final_status` is `None`).

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
noncomputational. `MPAD` and the status checks below are exempt because they do
not perform a physical qubit measurement.

## Nondestructive status checks

`HERALD_LEAKAGE_EVENT` and `HERALD_LOSS_EVENT` report the status at a circuit
position without changing the site's quantum state or occupation. Each plain
qubit target appends one ordinary record bit, in target order:

| Status | `HERALD_LEAKAGE_EVENT` | `HERALD_LOSS_EVENT` |
|--------|------------------------|---------------------|
| Computational | 0 | 0 |
| `LEAK_G` or `LEAK_E` | 1 | 0 |
| `LOST` | 0 | 1 |

These instructions take no probability arguments and accept plain qubit indices
(no `!` target inversion). Their record bits work with `DETECTOR`,
`OBSERVABLE_INCLUDE`, and classical feedback (`rec[...]`), require no
classifier, and leave `result.heralds` set to `0` at those slots. Each check
reports whether the qubit is leaked or lost at that point in the circuit, not
when the jump happened.

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

Status checks are noiseless by default. To model detection errors, apply
`READOUT_NOISE` to the resulting record bit. For example, a 2% false-positive
rate and a 5% false-negative rate:

```text
HERALD_LEAKAGE_EVENT 0
READOUT_NOISE(0.02, 0.05) rec[-1]
```

Use `READOUT_NOISE(0, p)` for missed detections only, or `READOUT_NOISE(p)` for
symmetric errors. `READOUT_NOISE` flips only the recorded bit; the qubit's
quantum state and status are unchanged.

Like transition annotations, status checks require `clifft.noncomp`;
ordinary `clifft.compile` rejects them.

## Transitions: hooks and inline annotations

A transition matrix `T[to][from]` gives the probability of a transition event
from one source level to one destination level. The destination may equal the
source; a column's deficit below 1 is the no-jump probability.

Transitions act as noise associated with specific circuit positions. There are
four ways to place them:

- **Gate hooks.** Give a `transitions` entry the name of a physical gate, such
  as `"CZ"` or `"S"`, to apply that matrix to each operand after every
  occurrence of the gate. Noise instructions, annotations, identity no-ops, and
  composite instructions that expand during parsing (`MXX`/`MYY`/`MZZ`,
  `CH`/`CCX`/`CCZ`) cannot be hooked and raise `ValueError`.
- **Inline references.** Write `LEVEL_TRANSITION[name] 0` in the circuit to
  apply a named matrix from `Model(transitions={...})` to qubit 0 at that
  specific line. Use this for custom transition names (such as `"idle_decay"`)
  or one-off placements.
- **Inline leakage.** Write `LEAKAGE(p) 0` in the circuit to move `g -> leak_g`
  and `e -> leak_e` with probability `p` without defining a 5x5 matrix. An
  already leaked or lost qubit is unchanged.
- **Inline loss.** Write `LOSS(p) 0` in the circuit to move any occupied level
  to `lost` with probability `p` without defining a 5x5 matrix.

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

## Model policies

Two `noncomp.Model` options control how resets and state-dependent transition
rates behave:

- **`reset_restores_lost`** (default `False`): whether reset instructions
  (`R`, `RX`, `RY`, and the reset half of `MR`, `MRX`, `MRY`) re-prepare a lost
  site into the computational subspace. Resets always restore a *leaked* site,
  and measure-and-reset instructions always record a classified measurement bit
  even when the reset half is dropped.
- **`damping`** (default `"exact"`): how Clifft handles the no-jump state update
  when `g` and `e` have different total jump rates ($p_g \neq p_e$). When a
  qubit in superposition survives a transition without jumping, its state shifts
  slightly toward the less leaky level. `"exact"` simulates that shift
  accurately, which can increase the active width $k$ by one (doubling the
  $2^k$ active-state size) when the affected superposition is not already in the
  active state. Transitions reuse existing active-state axes when possible,
  though intervening gates (such as `SWAP` or basis changes on entangled qubits)
  can cause later transitions on the same physical qubit to add width again.
  `"neglect"` skips the no-jump update to avoid that extra width, introducing an
  error of order $\lvert p_g - p_e \rvert$ at each transition. When $p_g = p_e$,
  as in `LEAKAGE(p)` and `LOSS(p)`, there is no state bias and both settings are
  exact.

## Compiling and execution

`noncomp.sample(circuit, model, ...)` compiles and samples in a single call.
When running multiple shot counts, seeds, or worker settings against the same
circuit and model, compile once with `noncomp.compile()` and reuse the returned
`NonComputationalProgram`:

```python
from clifft import noncomp

model = noncomp.Model(
    classifier=noncomp.Classifier([[1, 0, 0.5, 0.5, 0.5], [0, 1, 0.5, 0.5, 0.5]]),
    gate_partner_effects={
        "leaked": noncomp.PartnerEffect(pauli=(0.25, 0.25, 0.25)),
    },
)
program = noncomp.compile("H 0\nLEAKAGE(0.01) 0\nCX 0 1\nM 0 1", model)
assert program.strategy == "fixed"

result = program.sample(shots=10_000, seed=42)
assert result.final_status is None
second = program.sample(shots=10_000, seed=43, threads=2)
```

The compiled program stores a copy of the circuit and model and can be sampled
repeatedly, including concurrently from multiple threads. `noncomp.sample()`
accepts all of the `compile()` and `sample()` options below and compiles on
every call.

### Fixed execution vs. trajectories

Clifft has two execution strategies for leakage and loss, with identical model
and output semantics:

- **Fixed execution (`strategy == "fixed"`):** compiles the quantum circuit
  once up front and can process shots in batches, including circuits with
  non-Clifford gates.
- **Trajectory execution (`strategy == "trajectory"`):** supports the full
  five-level model by stepping through one shot at a time and recompiling the
  remaining circuit whenever a jump changes later operations.

By default (`execution="auto"`), `noncomp.compile()` uses fixed execution when
possible and falls back to trajectories otherwise. Eligibility is checked
against the qubit statuses that can actually reach each circuit position, so
unused model rules or unreachable pairings do not force trajectories. The main
checks are:

| Requirement for `"fixed"` | Uses `"trajectory"` when... |
| --- | --- |
| All qubits start in `g` (`initial_state` omitted or `[1, 0, 0, 0, 0]`) | `initial_state` puts nonzero probability on `e`, `leak_g`, `leak_e`, or `lost` |
| Active transitions use inline `LEAKAGE(p)` and `LOSS(p)` | Circuit invokes a nonzero named `transitions` matrix (via `LEVEL_TRANSITION[...]` or a gate hook) |
| No reachable leakage spreading (`spread_probability == 0`) | A partner effect with `spread_probability > 0` can see a leaked source paired with a computational partner |
| **Full partner depolarization** `PartnerEffect(pauli=(0.25, 0.25, 0.25))` wherever a two-qubit gate can pair a leaked or lost qubit with a computational one | Such a pairing uses the default no-op (drop) or partial Pauli noise |
| Classifier treats `leak_g` and `leak_e` identically at every leaked measurement | Classifier assigns different probabilities to `leak_g` and `leak_e` on a qubit that can be leaked when measured |
| `final_status=False`, or resets/loss clear any leaked state before circuit end | `final_status=True` and a qubit could end the circuit leaked (since `final_status` distinguishes `LEAK_G` from `LEAK_E`) |
| Compiled fixed program fits within `max_active_width` | Fixed program exceeds `max_active_width` (Clifft falls back to trajectories, where each compiled continuation must still fit within `max_active_width`) |

A few other operations also require trajectories when a target can be
noncomputational, such as multi-qubit Pauli-product gates (`SPP`, `TPP`,
`R_PAULI`). Call `program.explain()` to see the selected strategy and any
fallback reason.

!!! tip "Why two-qubit gates need full partner depolarization for fixed execution"
    Fixed execution keeps two-qubit gates in the compiled quantum circuit even
    when one qubit has leaked or been lost. That produces the exact same result
    as dropping the gate only when the computational partner is completely
    depolarized (`pauli=(0.25, 0.25, 0.25)`). Standalone `LEAKAGE_INTERACTION`
    and `LOSS_INTERACTION` instructions do not attach to a two-qubit gate and
    support arbitrary Pauli probabilities in fixed execution. See
    [Fixed execution](../theory/noncomputational.md#fixed-execution) for the
    derivation.

Invalid circuits or models always raise `ValueError` during `compile()`;
automatic fallback applies only to valid requests outside the fixed subset.

### Compilation and sampling options

Options passed to `noncomp.compile()` (and forwarded by `noncomp.sample()`):

| Argument | Default | Meaning |
| --- | --- | --- |
| `final_status` | `False` | When `True`, populate `result.final_status` with per-qubit `QubitStatus` values. |
| `execution` | `"auto"` | `"auto"` selects `"fixed"` when eligible and `"trajectory"` otherwise; `"fixed"` raises `ValueError` if ineligible; `"trajectory"` forces trajectories. |
| `max_active_width` | `None` | Cap on peak active width. Under `"auto"`, an over-cap fixed program falls back to trajectories; trajectory continuations exceeding the cap raise during sampling. |

Options passed to `program.sample()` (and `noncomp.sample()`):

| Argument | Default | Meaning |
| --- | --- | --- |
| `shots` | *(required)* | Number of shots to sample. |
| `seed` | `None` | Integer seed for reproducible results, or `None` for hardware entropy. |
| `threads` | `1` | Number of cross-shot CPU workers, or `"auto"` for hardware concurrency. Memory use grows with worker count. |
| `batch_size` | `"auto"` | Batching mode (`"auto"`, `"tune"`, `1`, or `N > 1`); see below. |
| `tuning_budget_seconds` | `None` | Soft time limit in seconds for `batch_size="tune"` trials (defaults to `0.25` s). |
| `collect_stats` | `False` | Populate `result.stats` with a `NonComputationalStats` execution report. |

`batch_size` controls how many shots a worker processes together:

- **`"auto"`** (default): choose a batch size automatically; trajectories run
  one shot at a time.
- **`"tune"`**: try several batch sizes to find a faster setting. Available only
  for fixed execution.
- **`1`**: process shots individually. Fixed execution still reuses its compiled
  program.
- **`N` (`N > 1`)**: request up to `N` shots per batch. Available only for fixed
  execution.

Leave `batch_size="auto"` for normal use. Clifft chooses whether to process
shots individually or in batches based on your circuit, shot count, and memory
needs, without spending time on trial runs. For repeated sampling of the same
circuit, `batch_size="tune"` can help find a faster setting, as shown below.

Changing the batch size can change individual samples, even with the same seed.
With the same program, seed, shot count, and batch size, changing `threads`
does not change the results.

Post-selected survivor sampling, fixed-fault importance sampling, intra-shot
thread layouts, and custom optimization pass managers are not supported by
`clifft.noncomp`.

### Tuning batch size

For fixed execution, `batch_size="tune"` tries several batch sizes and chooses
one based on the estimated time to produce your requested samples. Tuning adds
time to the call, so reuse the chosen batch size on later calls to benefit
without repeating the trials:

<!--pytest-codeblocks:cont-->
```python
result = program.sample(10_000, seed=42, batch_size="tune", tuning_budget_seconds=0.25)
# Reuse the chosen batch size without repeating the trials.
next_result = program.sample(10_000, seed=43, batch_size=result.batch_tuning.batch_size)
```

`tuning_budget_seconds` sets a soft time limit on the trial runs (defaulting to
0.25 seconds), and `result.batch_tuning` records the chosen `batch_size` and
timings. Trial shots are discarded and use a separate random stream, so they
never enter your returned results. Because live timings can pick a different
batch size on another run, pass the numeric `batch_size` explicitly when you
need exact seeded replay. See
[Budgeted batch calibration](cpu-execution.md#budgeted-batch-calibration) for
more on how batch tuning works across Clifft.

### Inspecting execution

Call `program.explain()` to check which strategy was chosen, see the circuit
line and reason if fixed execution fell back to trajectories, and view peak
active width estimates without running any shots. To see what happened during a
sampling run (such as how many shots ran in batches versus one at a time, how
many workers were used, and how often trajectory execution recompiled), pass
`collect_stats=True`:

<!--pytest-codeblocks:cont-->
```python
print(program.explain())

result = program.sample(shots=10_000, seed=42, collect_stats=True)
assert result.stats is not None
assert result.stats.packed_shots == 10_000
assert result.stats.continuation_compilations == 0
```

See [`NonComputationalStats`](../reference/python-api.md#clifft.noncomp.NonComputationalStats)
for all counter, memory, and timing fields.

### Trajectory continuations

When `program.strategy == "trajectory"`, `noncomp.compile()` validates the
circuit and expands model hooks once, while quantum compilation still happens
during `sample()`. Whenever a jump changes which later operations apply, the
worker compiles a continuation for the remaining circuit and resumes the shot
(`result.stats.continuation_compilations` counts how many occurred).

To resume mid-shot without disturbing earlier measurement correlations,
trajectory execution uses a restricted internal compiler pipeline. See
[Optimization Passes](../reference/passes.md#trajectory-safety-metadata)
for the pass restrictions and
[Noncomputational States](../theory/noncomputational.md#how-this-composes-with-clifft)
for how continuations compose with Clifft's simulator.

## Limits

- **Leakage and loss do not move between sites.**
  [Partner effects](partner-interactions.md) can apply Pauli noise or spread
  leakage to a computational partner, but the original leaked qubit stays
  leaked (and `SWAP` gates touching a leaked or lost site are dropped rather
  than moving it). Processes where one qubit relaxes while transferring its
  leakage to a neighbor are not supported.
- **Multi-qubit Pauli noise channels drop as a whole.** `DEPOLARIZE2/3` and
  `PAULI_CHANNEL_2/3` touching a leaked or lost site are skipped completely,
  whereas `CORRELATED_ERROR` / `ELSE_CORRELATED_ERROR` chains still apply Pauli
  terms to their remaining computational targets. Noise encodings that are
  equivalent on ordinary qubits can therefore behave differently after leakage
  or loss; see
  [Multi-qubit noise vs. partner effects](partner-interactions.md#multi-qubit-noise-vs-partner-effects).
- **Leakage is incoherent.** Jumps out of the computational subspace produce
  definite classical occupations (`leak_g`, `leak_e`, or `lost`), not coherent
  superpositions between computational and leaked levels.
- **`MPP` is not supported** when the model can leak or lose qubits, because
  multi-qubit parity has no well-defined single-bit classifier outcome once an
  operand leaves the qubit subspace. Decompose parity checks into explicit
  ancilla gates and single-qubit measurements instead.
- **`EXP_VAL` is not supported.** Circuits containing `EXP_VAL` are rejected
  before sampling because `clifft.noncomp` does not compute expectation values.
- **Physical measurements require a classifier** whenever the model can produce
  leakage or loss, even if the measured qubit itself never leaks or is lost in
  the circuit.
