# User Guide

Clifft separates four decisions that are easy to conflate: how a circuit enters
Clifft, how it is compiled, which scientific result to request, and how that
work runs on the available hardware. Start with the result you need; execution
tuning comes last.

See [Performance](performance.md) for near-Clifford comparisons, release
history, dense Quantum Volume results, and the measurement contracts behind
them.

## 1. Provide a circuit

Clifft's primary circuit format follows Stim and adds non-Clifford gates and
additional noise models. Clifft also accepts supported OpenQASM 2 text and
circuits from Qiskit or Cirq companion packages.

[Circuit Inputs](circuit-inputs.md) compares these paths, their examples, and
their limitations.

## 2. Compile it

For the usual workflow, `clifft.compile()` turns the input into a reusable
`Program`. The compiler resolves Clifford coordinates, symbolic dependencies,
and active-state actions before sampling begins.

[Compiling Circuits](compilation.md) explains the default path first, then the
lower-level inspection and customization APIs for power users.

## 3. Choose a workflow

The workflow determines the meaning and shape of the result:

| Question | Guide |
|---|---|
| I need ordinary shots or post-selected survivors. | [Sampling and Results](simulation.md) |
| I need rare-event estimates conditioned on fault count. | [Importance Sampling](importance-sampling.md) |
| I need to model leakage or loss outside the computational subspace. | [Leakage and Loss](leakage-and-loss.md) |
| I need exact probabilities for selected states or measurement records. | [Exact Probabilities](strong-simulation.md) |

The [workflow chooser](../getting-started/choosing-a-workflow.md) provides more
detail when the appropriate API is not obvious.

## 4. Choose execution settings

Most users should keep the automatic batching policy and the single-thread
default. [CPU Execution and Tuning](cpu-execution.md) documents worker budgets,
packed sampling, reproducibility, and memory tradeoffs for workloads that need
explicit tuning.

The AMD [HIP backend](../development/hip-backend.md) is a separate,
source-build-only experiment. It is never selected by the regular CPU API.
