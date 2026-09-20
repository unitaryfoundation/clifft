"""Independent physical trajectories and low-weight fault checks for folded MSC."""

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from folded_msc import Operation, make_circuit


def noise_labels(operation):
    if operation.probability is None:
        raise ValueError("expected a noise operation")
    if operation.name == "X_ERROR":
        return ("X",)
    if operation.name not in {"DEPOLARIZE1", "DEPOLARIZE2", "DEPOLARIZE3"}:
        raise ValueError(f"unsupported noise operation {operation.name}")
    return tuple(
        "".join(label) for label in itertools.product("IXYZ", repeat=len(operation.qubits))
    )[1:]


def bind_faults(circuit, faults):
    sites = {i: op for i, op in enumerate(circuit.operations) if op.probability is not None}
    for i, label in faults.items():
        if i not in sites or label not in noise_labels(sites[i]):
            raise ValueError("fault must identify a noise site and a supported nonidentity Pauli")
    output = replace(circuit, operations=[], measurements=circuit.measurements.copy())
    for i, op in enumerate(circuit.operations):
        if op.probability is None:
            output.operations.append(op)
        elif i in faults:
            for q, axis in zip(op.qubits, faults[i], strict=True):
                if axis != "I":
                    output.operations.append(Operation(axis, (q,), op.stage, op.source))
    return output


def sample_faults(circuit, rng):
    result = {}
    for i, op in enumerate(circuit.operations):
        if op.probability is not None and rng.random() < op.probability:
            labels = noise_labels(op)
            result[i] = labels[rng.integers(len(labels))]
    return result


def aer_trajectory(circuit, seed, *, method="statevector", mps_threshold=1e-14):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    visible = len(circuit.measurements)
    hidden = sum(op.name == "R" for op in circuit.operations)
    qubits = circuit.manifest()["num_qubits"]
    if method == "statevector" and qubits > 20:
        raise ValueError("use the MPS reference for more than twenty qubits")
    qc = QuantumCircuit(qubits, visible + hidden)
    m, r = 0, visible
    for op in circuit.operations:
        if op.probability is not None:
            raise ValueError("bind physical faults before calling the trajectory reference")
        if op.name in {"M", "R"}:
            index = m if op.name == "M" else r
            qc.save_probabilities(list(op.qubits), label=f"p{index}")
            qc.measure(op.qubits[0], index)
            if op.name == "M":
                m += 1
            else:
                # Expose the reset's discarded outcome to compare the same
                # complete branch, not a marginal over hidden measurements.
                qc.reset(op.qubits[0])
                r += 1
        elif op.name == "CCZ" and method == "matrix_product_state":
            # Aer MPS supports CCX, but not the equivalent native CCZ opcode.
            qc.h(op.qubits[2])
            qc.ccx(*op.qubits)
            qc.h(op.qubits[2])
        else:
            getattr(qc, {"T_DAG": "tdg"}.get(op.name, op.name.lower()))(*op.qubits)
    result = (
        AerSimulator(
            method=method,
            max_parallel_threads=1,
            matrix_product_state_truncation_threshold=mps_threshold,
        )
        .run(qc, shots=1, memory=True, seed_simulator=seed)
        .result()
    )
    records = result.get_memory()[0][::-1]
    log_probability = sum(
        math.log(float(result.data()[f"p{i}"][int(bit)])) for i, bit in enumerate(records)
    )
    return records, log_probability


def native_replay(circuit, records, executable, *, seed=0):
    with tempfile.TemporaryDirectory() as tmp:
        source, forced = Path(tmp) / "circuit.stim", Path(tmp) / "records.txt"
        source.write_text(circuit.text())
        command = [str(executable.resolve()), str(source), str(seed)]
        if records is not None:
            forced.write_text(records)
            command.append(str(forced))
        return json.loads(subprocess.check_output(command, text=True))


def validate_trajectories(circuit, executable, *, count=24, seed=703):
    rng = np.random.default_rng(seed)
    results = []
    for case in range(count):
        faults = sample_faults(circuit, rng) if case else {}
        physical = bind_faults(circuit, faults)
        records, expected = aer_trajectory(physical, seed + case)
        actual = native_replay(physical, records, executable)
        error = abs(actual["log_probability"] - expected)
        if not actual["reachable"] or error > 1e-10:
            raise AssertionError(
                f"physical branch disagreement in case {case}: {actual}, {expected}"
            )
        results.append(
            dict(
                faults=faults,
                records=records,
                aer_log_probability=expected,
                native_log_probability=actual["log_probability"],
                absolute_log_error=error,
            )
        )
    return results


def single_fault_check(circuit, executable, *, workers=4):
    if (
        not all(m["postselect"] for m in circuit.measurements[:-1])
        or not circuit.measurements[-1]["logical"]
    ):
        raise ValueError("expected the FED circuit with final logical observable")
    # Every preceding outcome is forced to zero, so all reset outcomes are
    # zero as well. The last bit selects accepted logical failure exactly.
    hidden = sum(op.name == "R" for op in circuit.operations)
    failure = "0" * (len(circuit.measurements) - 1) + "1" + "0" * hidden
    choices = [
        (i, label)
        for i, op in enumerate(circuit.operations)
        if op.probability is not None
        for label in noise_labels(op)
    ]

    def check(choice):
        i, label = choice
        result = native_replay(bind_faults(circuit, {i: label}), failure, executable)
        if result["reachable"]:
            return dict(
                operation=i, pauli=label, failure_probability=math.exp(result["log_probability"])
            )
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        witnesses = [result for result in pool.map(check, choices) if result is not None]
    return dict(faults_checked=len(choices), accepted_failure_witnesses=witnesses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("build-study/replay_cultivation"))
    parser.add_argument("--traces", type=int, default=24)
    parser.add_argument("--single-faults", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    circuit = make_circuit(0.02)
    report = dict(
        circuit=circuit.manifest(),
        native_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        trajectories=validate_trajectories(circuit, args.reference, count=args.traces),
    )
    if args.single_faults:
        report["single_fault_check"] = single_fault_check(circuit, args.reference)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                traces=len(report["trajectories"]),
                max_log_error=max(r["absolute_log_error"] for r in report["trajectories"]),
                single_fault_check=report.get("single_fault_check"),
            )
        )
    )
    if report.get("single_fault_check", {}).get("accepted_failure_witnesses"):
        raise SystemExit("accepted single-fault logical failures found; inspect witnesses")


if __name__ == "__main__":
    main()
