"""Independent Aer trajectory checks for the pinned small T-cultivation circuits.

The translator accepts only gates present in the author fixtures and fails on
anything else. MPP is expanded with a reusable parity ancilla. Depolarizing
channels retain the Stim probability of a nonidentity Pauli, not Aer's alternate
depolarizing-channel parameterization.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer
import stim
from cultivation_study import CORPUS, corpus, variant_text
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error

import clifft


def to_aer(proxy, probability):
    circuit = stim.Circuit(variant_text(proxy, probability, "S"))
    qc = QuantumCircuit(circuit.num_qubits + 1, circuit.num_measurements)
    ancilla = circuit.num_qubits
    record = 0

    def measure(qubit, args, slot):
        if args and args[0]:
            p = args[0]
            # A flipped parity ancilla changes the recorded bit while retaining
            # the true post-measurement data state, including later feedforward.
            if qubit != ancilla:
                qc.reset(ancilla)
                qc.cx(qubit, ancilla)
            qc.append(pauli_error([("I", 1 - p), ("X", p)]).to_instruction(), [ancilla])
            qc.measure(ancilla, slot)
        else:
            qc.measure(qubit, slot)

    for op in circuit.flattened():
        name = op.name
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if name in {"TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        if any(t.is_inverted_result_target for t in targets):
            raise ValueError("inverted results are unsupported")
        if name in {"S", "S_DAG"}:
            for target in targets:
                (qc.t if name == "S" else qc.tdg)(target.value)
        elif name in {"R", "RX"}:
            for target in targets:
                qc.reset(target.value)
                if name == "RX":
                    qc.h(target.value)
        elif name in {"M", "MX"}:
            for target in targets:
                if name == "MX":
                    qc.h(target.value)
                measure(target.value, args, record)
                if name == "MX":
                    qc.h(target.value)
                record += 1
        elif name in {"CX", "CZ"}:
            for control, target in zip(targets[::2], targets[1::2], strict=True):
                if control.is_measurement_record_target:
                    with qc.if_test((qc.clbits[record + control.value], 1)):
                        (qc.x if name == "CX" else qc.z)(target.value)
                else:
                    (qc.cx if name == "CX" else qc.cz)(control.value, target.value)
        elif name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}:
            p = args[0]
            if not p:
                continue
            width = 2 if name == "DEPOLARIZE2" else 1
            if name.startswith("DEPOLARIZE"):
                labels = ["".join(v) for v in itertools.product("IXYZ", repeat=width)][1:]
            else:
                labels = [name[0]]
            channel = pauli_error(
                [("I" * width, 1 - p)] + [(label, p / len(labels)) for label in labels]
            ).to_instruction()
            for offset in range(0, len(targets), width):
                qc.append(channel, [t.value for t in targets[offset : offset + width]])
        elif name == "MPP":
            for group in op.target_groups():
                qc.reset(ancilla)
                for target in group:
                    if target.is_x_target:
                        qc.h(target.value)
                    elif not target.is_z_target:
                        raise ValueError("only X/Z products are supported")
                    qc.cx(target.value, ancilla)
                measure(ancilla, args, record)
                for target in reversed(group):
                    if target.is_x_target:
                        qc.h(target.value)
                record += 1
        else:
            raise ValueError(f"unsupported upstream gate: {name}")
    assert record == circuit.num_measurements
    return qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("shots must be positive")
    corpus()
    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_truncation_threshold=0,
        max_parallel_threads=1,
    )
    report = dict(qiskit=qiskit.__version__, aer=qiskit_aer.__version__, cases=[])
    for path in sorted(CORPUS.glob("d3*.stim")):
        source = path.read_text()
        for probability in (0, 0.001):
            qc = to_aer(source, probability)
            count = 128 if probability == 0 else args.shots
            result = simulator.run(qc, shots=count, memory=True, seed_simulator=951).result()
            if not result.success:
                raise AssertionError(result.status)
            records = np.array(
                [[int(bit) for bit in bits.replace(" ", "")[::-1]] for bits in result.get_memory()],
                dtype=np.bool_,
            )
            proxy = stim.Circuit(source)
            detectors, observables = proxy.compile_m2d_converter().convert(
                measurements=records, separate_observables=True
            )
            plan = clifft.compile(variant_text(source, probability))
            sampled = clifft.sample(plan, count, seed=82, threads=1)
            actual = np.column_stack(
                [sampled.detectors, sampled.observables, ~sampled.detectors.any(axis=1)]
            )
            expected = np.column_stack([detectors, observables, ~detectors.any(axis=1)])
            if probability == 0 and (detectors.any() or observables.any()):
                raise AssertionError("Aer finds nontrivial noiseless outputs")
            pooled = (actual.mean(axis=0) + expected.mean(axis=0)) / 2
            tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / count) + 2 / count
            differences = np.abs(actual.mean(axis=0) - expected.mean(axis=0))
            if np.any(differences > tolerance):
                raise AssertionError("Aer and Clifft output marginals disagree")
            row = dict(
                circuit=path.name,
                probability=probability,
                shots=count,
                max_output_marginal_difference=float(differences.max()),
                aer_accepted=int((~detectors.any(axis=1)).sum()),
                clifft_accepted=int((~sampled.detectors.any(axis=1)).sum()),
                aer_accepted_errors=int(observables[~detectors.any(axis=1)].any(axis=1).sum()),
            )
            report["cases"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
