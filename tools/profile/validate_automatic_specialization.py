"""Independent and exhaustive bounded checks of the specialization host."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import stim
from automatic_specialization import (
    ANNOTATIONS,
    FaultModel,
    History,
    analyze,
    exact_record_probabilities,
    instruction_text,
    record_count,
    specialize_prefix,
)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error
from study_automatic_specialization import candidates, measurement_control, sample_parities

import clifft
import clifft._clifft_core as core

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/python"))
stim_to_qiskit_noiseless = importlib.import_module("utils_qiskit").stim_to_qiskit_noiseless


def all_histories(model: FaultModel) -> list[tuple[History, float]]:
    result = []
    choices = [tuple(i for i, p in enumerate(s.probabilities) if p) for s in model.sites]
    if math.prod(map(len, choices)) > 1024:
        raise ValueError("Noise enumeration exceeds the validation budget")
    for outcomes in product(*choices):
        history = tuple((i, outcome) for i, outcome in enumerate(outcomes) if outcome)
        probability = math.prod(s.probabilities[o] for s, o in zip(model.sites, outcomes))
        result.append((history, probability))
    return result


def aer_probabilities(circuit: QuantumCircuit, records: list[int]) -> np.ndarray:
    circuit.save_probabilities(records)
    result = AerSimulator(method="density_matrix", max_parallel_threads=1).run(circuit).result()
    if not result.success:
        raise AssertionError("Aer reference failed")
    return np.asarray(result.data()["probabilities"])


def record_pauli(
    circuit: QuantumCircuit, product_: list[tuple[str, int]], record: int, *, inverted: bool = False
) -> None:
    for axis, q in product_:
        if axis == "Y":
            circuit.sdg(q)
        if axis != "Z":
            circuit.h(q)
        circuit.cx(q, record)
        if axis != "Z":
            circuit.h(q)
        if axis == "Y":
            circuit.s(q)
    if inverted:
        circuit.x(record)


def validate_channels() -> dict[str, Any]:
    two = [0.0] * 15
    two[0], two[5], two[12] = 0.019, 0.037, 0.041
    source = (
        "H 0 1\nPAULI_CHANNEL_1(0.03,0.07,0.11) 0\n"
        + "PAULI_CHANNEL_2("
        + ",".join(map(str, two))
        + ") 0 1\n"
        + "MPP(0.13) !Y0*X1\nCX rec[-1] 1\nMX(1) !0\nM(0) 1\nMPAD(0.2) 1\n"
        + "DETECTOR rec[-4] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
    )
    model = FaultModel(source)
    mixture = np.zeros(1 << model.num_records)
    histories = all_histories(model)
    for history, probability in histories:
        fixed = model.render(history)
        program = clifft.compile(fixed)
        mixture += probability * np.asarray(exact_record_probabilities(program, model.num_records))
        sample_parities(source, clifft.sample(program, shots=5, seed=73, threads=1))
    circuit = QuantumCircuit(6)
    circuit.h([0, 1])
    circuit.append(
        pauli_error([("I", 0.79), ("X", 0.03), ("Y", 0.07), ("Z", 0.11)]).to_instruction(), [0]
    )
    labels = ["".join(p) for p in product("IXYZ", repeat=2)][1:]
    circuit.append(
        pauli_error(
            [("II", 1 - sum(two))] + [(p[::-1], w) for p, w in zip(labels, two) if w]
        ).to_instruction(),
        [0, 1],
    )
    record_pauli(circuit, [("Y", 0), ("X", 1)], 2, inverted=True)
    circuit.append(pauli_error([("I", 0.87), ("X", 0.13)]).to_instruction(), [2])
    circuit.cx(2, 1)
    record_pauli(circuit, [("X", 0)], 3, inverted=True)
    circuit.x(3)
    record_pauli(circuit, [("Z", 1)], 4)
    circuit.x(5)
    circuit.append(pauli_error([("I", 0.8), ("X", 0.2)]).to_instruction(), [5])
    reference = aer_probabilities(circuit, [2, 3, 4, 5])
    np.testing.assert_allclose(mixture, reference, atol=1e-12, rtol=0)
    sampled = stim.Circuit(source).compile_sampler(seed=913).sample(131072)
    packed = sampled.astype(np.uint64) @ (1 << np.arange(model.num_records, dtype=np.uint64))
    frequencies = np.bincount(packed.astype(int), minlength=len(reference)) / len(sampled)
    tolerance = 7 * np.sqrt(reference * (1 - reference) / len(sampled)) + 2 / len(sampled)
    if np.any(abs(frequencies - reference) > tolerance):
        raise AssertionError("Stim stochastic Clifford check failed")
    # The host's categorical RNG is checked against the same independent joint law.
    rng = random.Random(19493)
    keys = [history for history, _ in histories]
    index = {h: i for i, h in enumerate(keys)}
    weights = [p for _, p in histories]
    counts = np.zeros(len(keys))
    for _ in range(131072):
        counts[index[model.draw(rng)]] += 1
    delta = abs(counts / counts.sum() - weights)
    bound = (
        7 * np.sqrt(np.asarray(weights) * (1 - np.asarray(weights)) / counts.sum())
        + 2 / counts.sum()
    )
    if np.any(delta > bound):
        raise AssertionError("Categorical history probabilities failed")
    rejected = []
    for unsupported in ("E(0.1) X0 Y1", "HERALDED_ERASE(0.1) 0"):
        try:
            FaultModel(unsupported)
        except (ValueError, core.ParseError):
            rejected.append(unsupported)
        else:
            raise AssertionError("Unsupported channel was silently accepted")
    return {
        "histories": len(histories),
        "records": model.num_records,
        "maximum_aer_error": float(max(abs(mixture - reference))),
        "stim_shots": len(sampled),
        "categorical_draws": int(counts.sum()),
        "unsupported_rejected": rejected,
    }


def validate_measurement_branches() -> dict[str, Any]:
    rows = []
    for feedback in (False, True):
        model = FaultModel(measurement_control(noise=True, feedback=feedback))
        mixture = np.zeros(8)
        branch_counts = []
        maximum_error = 0.0
        for history, weight in all_histories(model):
            fixed = model.render(history)
            original = np.asarray(
                exact_record_probabilities(clifft.compile(fixed, hir_passes=None), 3)
            )
            branches: dict[int, str] = {}
            for seed in range(64):
                specialized, info = specialize_prefix(fixed, seed)
                branches[info["outcomes"][0]] = specialized
                if len(branches) == 2:
                    break
            if len(branches) != 2:
                raise AssertionError("Did not exercise both measurement branches")
            joined = np.zeros(8)
            for bit, text in branches.items():
                mask = np.arange(8) % 2 == bit
                probability = sum(original[mask])
                hir, info = analyze(text)
                if info["output_t"] != 0 or info["peak_width"] != 0:
                    raise AssertionError("Conditional coset failed to become Clifford")
                program = clifft.lower(hir)
                conditional = np.asarray(exact_record_probabilities(program, 3))
                expected = original * mask / probability
                maximum_error = max(maximum_error, float(max(abs(conditional - expected))))
                np.testing.assert_allclose(conditional, expected, atol=1e-12, rtol=0)
                joined += probability * conditional
                sample_parities(model.source, clifft.sample(program, shots=17, seed=731, threads=1))
                branch_counts.append(info["output_t"])
            np.testing.assert_allclose(joined, original, atol=1e-12, rtol=0)
            mixture += weight * joined
        circuit = QuantumCircuit(5)
        circuit.h([0, 1])
        circuit.append(
            pauli_error(
                [
                    ("I", 0.947),
                    ("X", 0.013),
                    ("Y", 0.017),
                    ("Z", 0.023),
                ]
            ).to_instruction(),
            [0],
        )
        record_pauli(circuit, [("Z", 0), ("Z", 1)], 2)
        circuit.append(pauli_error([("I", 0.969), ("X", 0.031)]).to_instruction(), [2])
        circuit.t(0)
        circuit.tdg(1)
        if feedback:
            circuit.cx(2, 0)
        record_pauli(circuit, [("X", 0)], 3)
        record_pauli(circuit, [("X", 1)], 4)
        reference = aer_probabilities(circuit, [2, 3, 4])
        np.testing.assert_allclose(mixture, reference, atol=1e-12, rtol=0)
        rows.append(
            {
                "feedback": feedback,
                "branches": len(branch_counts),
                "maximum_conditional_error": maximum_error,
                "maximum_aer_mixture_error": float(max(abs(mixture - reference))),
            }
        )
    return {"cases": rows}


def terminal_unitary(source: str) -> tuple[str, list[tuple[int, bool]]]:
    """Defer only readouts whose qubits are never subsequently acted on."""
    lines = []
    touched: set[int] = set()
    measured: set[int] = set()
    readouts = []
    parsed = clifft.parse(source)
    records = 0
    for node in parsed.nodes:
        gate = node.gate.name
        if gate in ANNOTATIONS:
            continue
        if any(t.is_rec for t in node.targets):
            raise ValueError("Terminal-readout oracle cannot discard feedback")
        targets = [t.value for t in node.targets]
        if measured.intersection(targets):
            raise ValueError("A measured qubit is used again")
        if gate in {"R", "RX"}:
            if touched.intersection(targets):
                raise ValueError("Only initial resets on untouched qubits are supported")
            if gate == "RX":
                lines.append("H " + " ".join(map(str, targets)))
        elif gate in {"M", "MX", "MY"}:
            for target in node.targets:
                if gate == "MY":
                    lines.append(f"S_DAG {target.value}")
                if gate != "M":
                    lines.append(f"H {target.value}")
                readouts.append((target.value, repr(target).startswith("!")))
            measured.update(targets)
        else:
            lines.append(instruction_text(node, records))
        touched.update(targets)
        records += record_count(node)
    if len(readouts) != parsed.num_measurements:
        raise ValueError("The terminal oracle must retain every physical record")
    return "\n".join(lines), readouts


def remap(source: str) -> str:
    parsed = clifft.parse(source)
    lines, records = [], 0
    for node in parsed.nodes:
        text = instruction_text(node, records)
        records += record_count(node)
        if node.gate.name not in ANNOTATIONS | {"MPAD"}:
            gate, targets = text.split(" ", 1)
            rewritten = [
                f"rec[{t.value - records + record_count(node)}]"
                if t.is_rec
                else re.sub(r"\d+$", str(parsed.num_qubits - 1 - t.value), repr(t))
                for t in node.targets
            ]
            targets = ("*" if node.gate.name in {"MPP", "R_PAULI", "EXP_VAL"} else " ").join(
                rewritten
            )
            text = gate + " " + targets
        lines.append(text)
    return "\n".join(lines) + "\n"


def visible_probability(program: Any, bits: list[int]) -> float:
    try:
        first = core._replay_record(program, bits)
    except ValueError as error:
        message = str(error)
        if "expected " not in message:
            raise
        total = int(message.split("expected ")[1].split(",")[0])
        hidden = total - len(bits)
        if hidden > 8:
            raise ValueError("Hidden-record replay exceeds the validation budget")
        probability = 0.0
        for value in range(1 << hidden):
            replay = core._replay_record(program, bits + [(value >> i) & 1 for i in range(hidden)])
            if replay["reachable"]:
                probability += math.exp(replay["log_probability"])
        return probability
    return math.exp(first["log_probability"]) if first["reachable"] else 0.0


def validate_distillation(panel: dict[str, tuple[str, str]]) -> dict[str, Any]:
    rows = []
    simulator = AerSimulator(method="statevector", max_parallel_threads=1)
    for name in ("15to1_scored", "15to1_direct_x", "bh_scored", "bh_direct_x"):
        model = FaultModel(panel[name][1])
        rng = random.Random(8931)
        histories: list[History] = [()]
        for weight in (1, 5, 12):
            histories.append(
                tuple(
                    (site, rng.randrange(1, len(model.sites[site].replacements)))
                    for site in sorted(rng.sample(range(len(model.sites)), weight))
                )
            )
        maximum_error, support_count = 0.0, 0
        counts = []
        for history in histories:
            fixed = model.render(history)
            unitary, readouts = terminal_unitary(fixed)
            reference_circuit = stim_to_qiskit_noiseless(unitary)
            reference_circuit.save_statevector()
            reference = np.asarray(simulator.run(reference_circuit).result().data()["statevector"])
            labels = np.arange(len(reference), dtype=np.uint64)
            packed = np.zeros(len(reference), dtype=np.uint64)
            for index, (q, inverted) in enumerate(readouts):
                packed |= (
                    ((labels >> np.uint64(q)) & np.uint64(1)) ^ np.uint64(inverted)
                ) << np.uint64(index)
            expected = np.bincount(
                packed.astype(int), weights=abs(reference) ** 2, minlength=1 << len(readouts)
            )
            hir, info = analyze(fixed)
            program = clifft.lower(hir)
            # Matching every positive-probability reference record exhausts its
            # normalized law, including correlations between checks and outputs.
            support = np.flatnonzero(expected > 1e-12)
            probability_sum = 0.0
            for outcome in support:
                bits = [int(outcome >> q & 1) for q in range(len(readouts))]
                probability = visible_probability(program, bits)
                probability_sum += probability
                maximum_error = max(maximum_error, abs(probability - expected[outcome]))
            if abs(probability_sum - 1) > 1e-10 or maximum_error > 1e-10:
                raise AssertionError("Complete distillation record law mismatch")
            support_count += len(support)
            changed = remap(fixed)
            _, renamed = analyze(changed)
            if renamed["output_t"] != info["output_t"]:
                raise AssertionError("Wire renaming changed this panel's algebraic reduction")
            changed_program = clifft.lower(analyze(changed)[0])
            for outcome in support:
                bits = [int(outcome >> q & 1) for q in range(len(readouts))]
                if abs(visible_probability(changed_program, bits) - expected[outcome]) > 1e-10:
                    raise AssertionError("Renamed circuit changed its joint record law")
            counts.append(info["output_t"])
        rows.append(
            {
                "case": name,
                "histories": histories,
                "output_t": counts,
                "positive_probability_records_checked": support_count,
                "maximum_joint_probability_error": float(maximum_error),
                "wire_renaming": "passed",
            }
        )
    return {"cases": rows}


def validate_negative(panel: dict[str, tuple[str, str]]) -> dict[str, Any]:
    model = FaultModel(panel["noncommuting_control"][1])
    rng = random.Random(7118)
    histories: list[History] = [
        (),
        tuple(
            (site, rng.randrange(1, len(model.sites[site].replacements)))
            for site in sorted(rng.sample(range(len(model.sites)), 9))
        ),
    ]
    maximum_error = 0.0
    for history in histories:
        fixed = model.render(history)
        unitary, _ = terminal_unitary(fixed)
        circuit = stim_to_qiskit_noiseless(unitary)
        circuit.save_statevector()
        reference = np.asarray(
            AerSimulator(method="statevector", max_parallel_threads=1)
            .run(circuit)
            .result()
            .data()["statevector"]
        )
        state = clifft.get_statevector(clifft.compile(unitary))
        if abs(1 - abs(np.vdot(reference, state)) ** 2) > 1e-12:
            raise AssertionError("Negative control statevector mismatch")
        for source in (fixed, remap(fixed)):
            hir, info = analyze(source)
            if info["output_t"] != 24:
                raise AssertionError("The unreduced control changed")
            actual = np.asarray(exact_record_probabilities(clifft.lower(hir), 4))
            expected = abs(reference) ** 2
            maximum_error = max(maximum_error, float(max(abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    return {
        "histories": histories,
        "maximum_aer_probability_error": maximum_error,
        "retained_t": 24,
        "wire_renaming": "passed",
    }


def validate_cultivation(panel: dict[str, tuple[str, str]]) -> dict[str, Any]:
    merlin = importlib.import_module("merlin")

    shots = 8192
    rows = []
    for name in ("cultivation_d3", "cultivation_d5"):
        model = FaultModel(panel[name][1])
        rng = random.Random(18182)
        history = tuple(
            (site, rng.randrange(1, len(model.sites[site].replacements)))
            for site in sorted(rng.sample(range(len(model.sites)), 5))
        )
        for fixed in ((), history):
            source = model.render(fixed)
            candidate = clifft.sample(clifft.compile(source), shots=shots, seed=333, threads=1)
            reference = merlin.CircuitSampler(source, seed=331).sample(shots)
            sample_parities(model.source, candidate)
            columns = [[i] for i in range(model.num_records)]
            columns += [
                rng.sample(range(model.num_records), min(5, model.num_records)) for _ in range(64)
            ]
            # Declared parities join the individual records and random correlations.
            columns += [
                [t.value for t in node.targets]
                for node in clifft.parse(source).nodes
                if node.gate.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}
            ]
            features = [
                np.column_stack(
                    [
                        1.0 - 2.0 * np.logical_xor.reduce(records[:, col], axis=1, initial=False)
                        for col in columns
                    ]
                )
                for records in (candidate.measurements, reference.measurements)
            ]
            a, b = features
            variance = a.var(axis=0, ddof=1) / shots + b.var(axis=0, ddof=1) / shots
            delta = np.maximum(0, abs(a.mean(axis=0) - b.mean(axis=0)) - 4 / shots)
            scores = delta / np.maximum(np.sqrt(variance), 1e-12)
            if max(scores) > 7:
                raise AssertionError("Cultivation fixed-history moment check failed")
            rows.append(
                {
                    "case": name,
                    "history": fixed,
                    "shots_per_backend": shots,
                    "features": len(columns),
                    "maximum_score": float(max(scores)),
                }
            )
    return {
        "scope": "Bug checks of fixed-history moments, not distribution equivalence",
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = candidates(args.merlin_checkout)
    result = {
        "channels": validate_channels(),
        "measurement_branches": validate_measurement_branches(),
        "distillation": validate_distillation(panel),
        "negative_control": validate_negative(panel),
        "cultivation": validate_cultivation(panel),
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("automatic_specialization.py"))
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
