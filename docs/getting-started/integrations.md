# Circuit Inputs and Integrations

Clifft's primary input is Stim-compatible circuit text with Clifft extensions
for non-Clifford operations. Other input paths accept OpenQASM 2 text or circuit
objects from another quantum software framework.

OpenQASM 2 support is built into the core `clifft` package. Framework companion
packages are maintained separately and released on their own schedule; use
their READMEs as the source of truth for current limitations.

## Choose an input

| Starting point | Package | Details |
|---|---|---|
| Stim-compatible text | [`clifft`](https://pypi.org/project/clifft/) | [Native input](stim-input.md) with the broadest Clifft feature support. |
| Unitary OpenQASM 2 text | [`clifft`](https://pypi.org/project/clifft/) | Native parsing of the documented [OpenQASM 2 subset](openqasm2.md). |
| Qiskit `QuantumCircuit` | [`clifft-qiskit`](https://github.com/unitaryfoundation/clifft-qiskit) | [Qiskit adapter](qiskit.md) for supported terminal-measurement circuits. |
| Cirq `cirq.Circuit` | [`clifft-cirq`](https://github.com/unitaryfoundation/clifft-cirq) | [Cirq converter and sampler](cirq.md) for supported qubit circuits. |

Use native Stim-compatible text for detector annotations, observables,
post-selection, or importance sampling. Use OpenQASM 2 for its supported unitary
interchange subset, or an adapter when circuit construction, decomposition, and
the surrounding workflow already live in Qiskit or Cirq.

Choosing an input does not select a CPU execution strategy or experimental GPU
backend. After the circuit reaches Clifft, use
[Choose a Workflow](choosing-a-workflow.md) to select the result you need.

## Stim-compatible text

This is the native path and the right default for new Clifft-specific code. See
[Stim-Compatible Text](stim-input.md).

## OpenQASM 2

Core Clifft accepts a unitary subset without requiring Qiskit. See
[OpenQASM 2 Input](openqasm2.md).

## Qiskit

Use the separately released adapter for supported `QuantumCircuit` objects. See
[Qiskit](qiskit.md).

## Cirq

Use the separately released converter or sampler for supported `cirq.Circuit`
objects. See [Cirq](cirq.md).
