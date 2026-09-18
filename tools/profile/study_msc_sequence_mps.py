"""Bounded original-gate Aer MPS comparator for synthetic five-check controls."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from msc_factor_sequence import Sequence
from msc_protocol import Program
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, pauli_error


def original_circuit(sequence, *, expectations=False):
    program = Program(sequence.circuit())
    circuit = QuantumCircuit(sequence.width, sequence.record_count)
    event = 0
    p = sequence.noise
    fault = pauli_error([("I", 1 - p), ("X", p / 3), ("Y", p / 3), ("Z", p / 3)]).to_instruction()
    for op in program.instructions:
        if op.name == "DEPOLARIZE1":
            for target in op.targets:
                circuit.append(fault, [int(target)])
        elif op.name == "MPP":
            for word in op.targets:
                pauli = [(term[0], int(term[1:])) for term in word.split("*")]
                if expectations:
                    circuit.save_expectation_value(
                        Pauli("".join(axis for axis, _ in reversed(pauli))),
                        [q for _, q in pauli],
                        label=f"p{event}",
                    )
                for axis, q in pauli:
                    if axis == "Y":
                        circuit.sdg(q)
                    if axis in "XY":
                        circuit.h(q)
                pivot = pauli[-1][1]
                for _, q in pauli[:-1]:
                    circuit.cx(q, pivot)
                circuit.measure(pivot, event)
                event += 1
                for _, q in reversed(pauli[:-1]):
                    circuit.cx(q, pivot)
                for axis, q in pauli:
                    if axis in "XY":
                        circuit.h(q)
                    if axis == "Y":
                        circuit.s(q)
        elif op.name == "CX":
            for a, b in zip(op.targets[::2], op.targets[1::2], strict=True):
                circuit.cx(int(a), int(b))
        elif op.name in {"H", "S", "S_DAG", "T", "T_DAG", "X", "Y", "Z"}:
            for target in op.targets:
                getattr(circuit, {"S_DAG": "sdg", "T_DAG": "tdg"}.get(op.name, op.name.lower()))(
                    int(target)
                )
        elif op.name not in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            raise ValueError("unsupported elementary operation")
    if event != sequence.record_count:
        raise AssertionError("measurement count differs")
    noise = NoiseModel()
    noise.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return circuit, noise


def benchmark(distance, shots, seed, gib):
    resource.setrlimit(resource.RLIMIT_AS, (gib << 30, gib << 30))
    sequence = Sequence(distance)
    circuit, noise = original_circuit(sequence)
    simulator = AerSimulator(
        method="matrix_product_state",
        max_parallel_threads=1,
        matrix_product_state_truncation_threshold=0.0,
        noise_model=noise,
    )
    start = time.perf_counter()
    result = simulator.run(circuit, shots=shots, seed_simulator=seed, memory=True).result()
    if not result.success:
        raise RuntimeError(result.status)
    records = np.array(
        [[int(c) for c in reversed(row)] for row in result.get_memory()], dtype=np.uint8
    )
    detectors = np.array(
        [
            records[:, r * sequence.stride + k] ^ records[:, (r - 1) * sequence.stride + k]
            for r in range(1, 5)
            for k in range(1, sequence.stride)
        ]
    )
    checksum = int(records.sum() + detectors.sum() + records[:, -1].sum())
    seconds = time.perf_counter() - start
    return dict(
        distance_label=distance,
        shots=shots,
        seconds=seconds,
        microseconds_per_shot=seconds * 1e6 / shots,
        checksum=checksum,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        address_space_limit_gib=gib,
        metadata=result.results[0].metadata,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, default=773)
    parser.add_argument("--gib", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.distance, args.shots, args.seed, args.gib)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
