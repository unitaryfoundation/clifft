# Choose a Workflow

Choose the result you need before choosing performance settings. Circuit input,
simulation workflow, CPU execution strategy, and hardware backend are separate
decisions in Clifft.

If you are unsure, start with the ordinary CPU workflow:

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(program, shots=1000)
print(result.measurements[:5])
```

This uses the stable CPU backend and its default execution settings. Change the
workflow only when the result or circuit semantics require it. Tune execution
only after measuring a representative workload.

## Choose the result you need

| Goal or circuit requirement | Use | Result and important constraints |
|---|---|---|
| Draw measurement outcomes, detector events, observables, or expectation values | `clifft.compile()` then `clifft.sample()` | Returns one row per requested shot. The compiled program must not have a post-selection mask. |
| Discard shots when selected detectors fire | Compile with `postselection_mask`, then use `clifft.sample_survivors()` | Returns survivor counts. Pass `keep_records=True` when per-survivor rows are also needed. |
| Query exact computational-basis probabilities for a unitary circuit with no measurements | `clifft.basis_probabilities()` | Queries selected bitstrings without constructing every output probability. Noise, measurements, detectors, observables, and post-selection are not supported. |
| Query exact probabilities of measurement records in a noiseless circuit | `clifft.record_probabilities()` | Supports measurement records and classical feedback. Noise, detectors, observables, post-selection, and hidden reset records are not supported. |
| Inspect the complete final pure state of a small unitary circuit | `clifft.get_statevector()` | Debugging and validation path. Expands all physical qubits and is currently limited to 10 qubits. |
| Estimate rare events by conditioning on exactly `k` faults | `clifft.sample_k()` or `clifft.sample_k_survivors()` | Advanced statistical workflow. Results are conditional on `K = k` and must be combined with the corresponding fault-count probabilities. |
| Model leakage, loss, or other supported noncomputational transitions | `clifft.noncomp.sample()` | Experimental trajectory workflow. It accepts a circuit and model together and compiles continuations internally. |

Use the [Simulation guide](../guide/simulation.md) for ordinary and
post-selected sampling. The
[Strong Simulation tutorial](../guide/strong-simulation.md) covers exact
queries, the [Importance Sampling tutorial](../guide/importance-sampling.md)
covers fixed-fault strata, and the
[Leakage and Loss guide](../guide/leakage-and-loss.md) covers noncomputational
trajectories.

## Choose the circuit input separately

The input path does not select the simulation workflow or execution backend.

| Starting point | Path |
|---|---|
| Stim-compatible text with Clifft extensions | Pass the text to `clifft.compile()`. This is the native path and supports the full stable circuit workflow. |
| Unitary OpenQASM 2 text | Pass `input_format="qasm2"` to `clifft.compile()`. Only the documented unitary subset is supported. |
| Qiskit `QuantumCircuit` | Use the separately released `clifft-qiskit` adapter for its supported terminal-measurement workflow. |
| Cirq `cirq.Circuit` | Use the separately released `clifft-cirq` converter or sampler for its supported circuit subset. |

See [Circuit Inputs and Integrations](integrations.md) for installation,
examples, and current limitations.

## Choose execution only after the workflow

The stable `clifft.compile()` and sampling APIs execute on the CPU. Automatic
packed-batch selection is enabled by default where supported, while sampling
uses one worker by default. Most users should keep the automatic batch policy
and use `threads` only to provide a larger CPU worker budget.

Advanced CPU callers can control cross-shot workers, intra-shot OpenMP workers,
hybrid layouts, and explicit packed capacities. These settings change how a
supported workflow executes; they do not change its statistical meaning. See
[Packed Batch Sampling](../guide/simulation.md#packed-batch-sampling) and
[Parallel Sampling](../guide/simulation.md#parallel-sampling).

GPU execution is experimental and is never selected automatically. The current
HIP backend uses the separate `clifft.experimental.hip` API, supports only its
documented subset, and requires an explicit HIP-enabled source build. See the
[Experimental HIP Sampling Backend](../development/hip-backend.md) before using
it. Additional GPU backends will follow the same explicit experimental boundary
until their support contracts are promoted.

## Terminology

- **Circuit input** describes how a circuit reaches Clifft: native
  Stim-compatible text, OpenQASM 2 text, or a framework adapter.
- **Simulation workflow** describes the mathematical result and sampling
  semantics: ordinary samples, survivors, exact queries, fixed-fault strata, or
  noncomputational trajectories.
- **CPU execution strategy** describes how CPU work is scheduled or packed:
  cross-shot, intra-shot, hybrid, scalar, or packed execution.
- **Hardware backend** describes where a prepared workload runs. The stable
  default is CPU execution; HIP is currently an explicit experimental backend.

These choices can interact through documented capability limits, but they are
not interchangeable modes. Start with the workflow, keep its defaults, and use
backend or execution controls only when the workload and hardware justify them.
