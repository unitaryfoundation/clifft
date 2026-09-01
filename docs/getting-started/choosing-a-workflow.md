# Choose a Workflow

Clifft offers several ways to provide a circuit, obtain results, and tune
performance. This page walks through those three choices so you can select a
workflow from the result you need rather than from Clifft's implementation
details.

!!! tip "First time using Clifft?"
    Start with the [Quick Start](quickstart.md) to compile and sample your first
    circuit, then return here when you need a different result or execution
    strategy.

## 1. Provide a circuit

For most users, the starting point is a Stim circuit passed to
`clifft.compile()`. Existing Stim circuits compile directly, and Clifft adds
non-Clifford gates for circuits that need them.

If your circuit already uses another format:

- pass supported unitary OpenQASM 2 text with `input_format="qasm2"`
- use the separately released `clifft-qiskit` adapter for supported Qiskit
  circuits
- use the separately released `clifft-cirq` converter or sampler for supported
  Cirq circuits

These alternatives have format-specific restrictions. See
[Circuit Inputs](../guide/circuit-inputs.md) before using one. The input
path does not select the simulation workflow or execution backend.

## 2. What do you want to know?

### Sample circuit outcomes

- **I need an output row for every requested shot.** Use `clifft.compile()` and
  `clifft.sample()` to obtain measurement, detector, observable, and expectation
  value results. Start with the [Simulation guide](../guide/simulation.md).
- **I need samples conditioned on post-selection.** Compile with a
  `postselection_mask`, then use `clifft.sample_survivors()`. It returns survivor
  counts and can optionally retain each survivor's records. See the
  [post-selection workflow](../guide/simulation.md#post-selected-survivor-sampling).
- **I need rare-event estimates conditioned on exactly `k` faults.** Use
  `clifft.sample_k()` or `clifft.sample_k_survivors()`, then combine the
  conditional results with the corresponding fault-count probabilities. This
  is an advanced workflow covered by the
  [Importance Sampling guide](../guide/importance-sampling.md).

### Calculate exact probabilities

- **My unitary circuit has no measurements.** Use
  `clifft.basis_probabilities()` to query selected computational-basis outcomes
  without constructing every output probability.
- **My noiseless circuit includes measurements or classical feedback.** Use
  `clifft.record_probabilities()` to query exact probabilities of selected
  measurement records.

Both APIs have circuit restrictions. The
[Strong Simulation tutorial](../guide/strong-simulation.md) explains when to
use each one.

### Model leakage or loss

- **My circuit includes leakage or loss.** Use `clifft.noncomp.sample()` for
  supported noncomputational transitions. It accepts a circuit and model
  together and compiles continuations internally. See
  [Leakage and Loss](../guide/leakage-and-loss.md) for the model and its limits.

## 3. Choose performance options

Choose performance settings only after selecting the workflow that produces
the right result.

### CPU

The regular compilation and sampling APIs execute on the CPU. Their automatic
batching and single-worker defaults are appropriate starting points for most
users.

Advanced callers can tune cross-shot workers, intra-shot OpenMP workers, hybrid
layouts, and explicit packed capacities. These settings change how a supported
workflow executes, not its statistical meaning. See
[CPU Execution and Tuning](../guide/cpu-execution.md).

### HIP (experimental)

The HIP backend is experimental and is never selected automatically. It uses
the separate `clifft.experimental.hip` API, supports only its documented
subset, and requires an explicit source build. See the
[HIP Backend](../development/hip-backend.md) documentation before using it.
