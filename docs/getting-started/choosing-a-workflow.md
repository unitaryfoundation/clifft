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

For most users, the starting point is Stim-compatible circuit text passed to
`clifft.compile()`. This is Clifft's native input and supports the complete
circuit workflow.

If your circuit already uses another format:

- pass supported unitary OpenQASM 2 text with `input_format="qasm2"`
- use the separately released `clifft-qiskit` adapter for supported Qiskit
  circuits
- use the separately released `clifft-cirq` converter or sampler for supported
  Cirq circuits

These alternatives have format-specific restrictions. See
[Circuit Inputs and Integrations](integrations.md) before using one. The input
path does not select the simulation workflow or execution backend.

## 2. What do you want to know?

### Sample circuit outcomes

{% for workflow in workflow_contracts['workflows'] if workflow['group'] == 'sampling' %}
- **{{ workflow['prompt'] }}** {{ workflow['route'] }} See
  [{{ workflow['guide_label'] }}](../{{ workflow['guide'] }}).
{% endfor %}

### Calculate exact probabilities

{% for workflow in workflow_contracts['workflows'] if workflow['group'] == 'exact' %}
- **{{ workflow['prompt'] }}** {{ workflow['route'] }} See
  [{{ workflow['guide_label'] }}](../{{ workflow['guide'] }}).
{% endfor %}

### Model leakage or loss

{% for workflow in workflow_contracts['workflows'] if workflow['group'] == 'specialized' %}
- **{{ workflow['prompt'] }}** {{ workflow['route'] }} See
  [{{ workflow['guide_label'] }}](../{{ workflow['guide'] }}).
{% endfor %}

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

### GPU (experimental)

GPU execution is experimental and is never selected automatically. The current
HIP backend uses the separate `clifft.experimental.hip` API, supports only its
documented subset, and requires an explicit HIP-enabled source build. See the
[Experimental GPU Execution](../guide/gpu-execution.md) guide before using
it.
