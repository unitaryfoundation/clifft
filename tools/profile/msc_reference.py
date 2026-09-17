"""Independent elementary-gate and record-parity checks for the MSC study."""

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator


def classical_reference(program, history, outcomes):
    """Let Stim evaluate the original parity declarations on reported records."""
    circuit = stim.Circuit()
    records = []
    event = 0
    observables = set()
    for op in program.instructions:
        if op.name in {"M", "MX", "MPP", "R", "RX"}:
            for offset in range(len(op.targets)):
                if op.name not in {"R", "RX"}:
                    records.append(
                        outcomes[event] ^ int(program.site_at.get((op.line, offset)) in history)
                    )
                    circuit.append("M", [0])
                event += 1
        elif op.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            targets = [stim.target_rec(int(t[4:-1])) for t in op.targets]
            circuit.append(op.name, targets, op.args if op.name == "OBSERVABLE_INCLUDE" else [])
            if op.name == "OBSERVABLE_INCLUDE":
                observables.add(int(op.args[0]))
    detectors, obs = circuit.compile_m2d_converter(skip_reference_sample=True).convert(
        measurements=np.asarray([records], dtype=np.bool_), separate_observables=True
    )
    return {
        "records": records,
        "detectors": detectors[0].astype(int).tolist(),
        "observables": {str(k): int(obs[0, k]) for k in sorted(observables)},
    }


def materialize(program, history):
    """Expose reset outcomes and bind noise, without rewriting any T gadgets."""
    faults = program.faults(history)
    records = []
    events = 0
    lines = []
    for op in program.instructions:
        if op.name in {"R", "RX", "M", "MX", "MPP"}:
            for offset, word in enumerate(op.targets):
                if op.name in {"R", "RX"}:
                    lines.append(f"{'MX' if op.name == 'RX' else 'M'} {word}")
                    lines.append(f"{'CZ' if op.name == 'RX' else 'CX'} rec[-1] {word}")
                else:
                    lines.append(f"{op.name} {word}")
                    records.append((events, program.site_at.get((op.line, offset)) in history))
                events += 1
        elif op.name in {"CX", "CZ"}:
            for offset in range(0, len(op.targets), 2):
                a, b = op.targets[offset : offset + 2]
                if a.startswith("rec["):
                    event, flipped = records[int(a[4:-1])]
                    if flipped:
                        lines.append(f"{'X' if op.name == 'CX' else 'Z'} {b}")
                    a = f"rec[{event - events}]"
                lines.append(f"{op.name} {a} {b}")
        elif op.name in {"H", "X", "Y", "Z", "S", "S_DAG", "T", "T_DAG"}:
            lines.append(op.name + " " + " ".join(op.targets))
        else:
            lines.extend(f"{axis} {q}" for axis, q in faults.get(op.line, []))
    return "\n".join(lines) + "\n"


def reference(program, history, seed, method="statevector"):
    faults = program.faults(history)
    circuit = QuantumCircuit(program.width, len(program.measurements))
    records = []
    detectors = []
    observables: dict[int, list[tuple[int, bool]]] = {}
    events = 0
    for op in program.instructions:
        if op.name in {"M", "MX", "MPP", "R", "RX"}:
            for offset, word in enumerate(op.targets):
                pauli = (
                    [(p[0], int(p[1:])) for p in word.split("*")]
                    if op.name == "MPP"
                    else [("X" if op.name.endswith("X") else "Z", int(word))]
                )
                circuit.save_expectation_value(
                    Pauli("".join(axis for axis, _ in reversed(pauli))),
                    [q for _, q in pauli],
                    label=f"p{events}",
                )
                for axis, q in pauli:
                    if axis == "Y":
                        circuit.sdg(q)
                    if axis in {"X", "Y"}:
                        circuit.h(q)
                pivot = pauli[-1][1]
                for _, q in pauli[:-1]:
                    circuit.cx(q, pivot)
                circuit.measure(pivot, events)
                for _, q in reversed(pauli[:-1]):
                    circuit.cx(q, pivot)
                for axis, q in pauli:
                    if axis in {"X", "Y"}:
                        circuit.h(q)
                    if axis == "Y":
                        circuit.s(q)
                if op.name in {"R", "RX"}:
                    with circuit.if_test((circuit.clbits[events], 1)):
                        getattr(circuit, "z" if op.name == "RX" else "x")(pivot)
                else:
                    records.append((events, program.site_at.get((op.line, offset)) in history))
                events += 1
        elif op.name in {"CX", "CZ"}:
            for offset in range(0, len(op.targets), 2):
                a, b = op.targets[offset : offset + 2]
                if a.startswith("rec["):
                    event, flipped = records[int(a[4:-1])]
                    with circuit.if_test((circuit.clbits[event], 1 ^ flipped)):
                        getattr(circuit, "x" if op.name == "CX" else "z")(int(b))
                else:
                    getattr(circuit, op.name.lower())(int(a), int(b))
        elif op.name in {"H", "X", "Y", "Z", "S", "S_DAG", "T", "T_DAG"}:
            for target in op.targets:
                getattr(circuit, {"T_DAG": "tdg", "S_DAG": "sdg"}.get(op.name, op.name.lower()))(
                    int(target)
                )
        elif op.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            deps = [records[int(t[4:-1])] for t in op.targets]
            if op.name == "DETECTOR":
                detectors.append(deps)
            else:
                observables.setdefault(int(op.args[0]), []).extend(deps)
        else:
            for axis, q in faults.get(op.line, []):
                getattr(circuit, axis.lower())(q)
    if events != len(program.measurements):
        raise ValueError("reference event count differs")
    options = (
        {"matrix_product_state_truncation_threshold": 0} if method == "matrix_product_state" else {}
    )
    simulator = AerSimulator(method=method, max_parallel_threads=1, **options)
    result = simulator.run(circuit, shots=1, memory=True, seed_simulator=seed).result()
    if not result.success:
        raise ValueError(result.status)
    outcomes = [int(b) for b in reversed(result.get_memory()[0])]
    data = result.data()
    return {
        "outcomes": outcomes,
        "probabilities": [
            (1 + (-1) ** b * float(data[f"p{k}"])) / 2 for k, b in enumerate(outcomes)
        ],
        "records": [b ^ outcomes[k] for k, b in records],
        "detectors": [sum(b ^ outcomes[k] for k, b in deps) % 2 for deps in detectors],
        "observables": {
            str(obs): sum(b ^ outcomes[k] for k, b in deps) % 2 for obs, deps in observables.items()
        },
    }
