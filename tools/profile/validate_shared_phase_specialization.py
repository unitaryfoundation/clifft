"""Check shared phase analysis against full circuits and independent references."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import stim
from automatic_specialization import (
    ANNOTATIONS,
    History,
    exact_record_probabilities,
    instruction_text,
    specialize_prefix,
)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scored_clifford_core import affine_forms
from shared_phase_specialization import SharedPhase, bits, parity
from study_automatic_specialization import candidates, sample_parities, stress_histories
from validate_automatic_specialization import (
    aer_probabilities,
    all_histories,
    record_pauli,
    remap,
    stim_to_qiskit_noiseless,
    terminal_unitary,
    visible_probability,
)
from validate_bt27_scored_equivalence import exact_law as raw_exact_law

import clifft


def exact_law(
    circuit: stim.Circuit, order: list[int], visible: int, flips: int = 0
) -> tuple[int, ...]:
    # The installed Stim flow oracle reverses the signs within batched MPAD.
    # Equivalent isolated measurements avoid that oracle path without changing
    # the sampler, record order, or the immutable Stim dependency.
    expanded = stim.Circuit()
    auxiliary = circuit.num_qubits
    for instruction in circuit:
        if instruction.name != "MPAD":
            expanded.append(instruction)
        else:
            if instruction.gate_args_copy():
                raise ValueError("Exact laws require deterministic padding")
            for target in instruction.targets_copy():
                expanded.append("R", [auxiliary])
                expanded.append("M", [stim.target_inv(auxiliary) if target.value else auxiliary])
    return raw_exact_law(expanded, order, visible, flips)


def padding_oracle_control() -> dict[str, Any]:
    for mask in range(256):
        values = [(mask >> q) & 1 for q in range(8)]
        circuit = stim.Circuit("MPAD " + " ".join(map(str, values)))
        expected = tuple((1 << q) | (values[q] << 8) for q in reversed(range(8)))
        if exact_law(circuit, list(range(8)), 8) != expected:
            raise AssertionError("Padding-law oracle failed its literal record control")
        np.testing.assert_array_equal(circuit.compile_sampler().sample(1)[0], values)
    witness = stim.Circuit("MPAD 0 1")
    return {
        "patterns": 256,
        "stim_version": stim.__version__,
        "raw_flow_witness": list(map(str, witness.flow_generators())),
        "witness_sample": witness.compile_sampler().sample(1).astype(int).tolist(),
    }


def normalized_lines(shared: SharedPhase, history: History) -> list[str]:
    lines = shared.lines.copy()
    for site_id, outcome in history:
        site = shared.model.sites[site_id]
        if site.gate == "READOUT_NOISE":
            gate, target = lines[site.line].split(" ", 1)
            target = (
                str(1 - int(target))
                if gate == "MPAD"
                else target[1:]
                if target.startswith("!")
                else "!" + target
            )
            lines[site.line] = gate + " " + target
        else:
            lines[site.line] = site.replacements[outcome]
    return lines


def record_only(circuit: stim.Circuit) -> stim.Circuit:
    return stim.Circuit("\n".join(str(op) for op in circuit if op.name not in ANNOTATIONS))


def prefix_law(shared: SharedPhase, history: History) -> tuple[tuple[int, ...], int]:
    total = shared.prefix.num_measurements
    flips = 0
    for site, outcome in history:
        flips ^= shared.prefix_responses[site][outcome]
    ideal = record_only(shared.prefix)
    prefix = stim.Circuit("\n".join(normalized_lines(shared, history)[: shared.first]))
    actual = record_only(prefix + shared.probes)
    expected = exact_law(ideal, list(range(total)), total, flips)
    observed = exact_law(actual, list(range(total)), total)
    if expected != observed:
        raise AssertionError(
            "Prefix fault responses changed the complete record and state-sign law"
        )
    return exact_law(ideal, list(range(total)), total), total


def prefix_values(shared: SharedPhase) -> list[int]:
    law, total = prefix_law(shared, ())
    forms = affine_forms(law, total)
    free = bits(sum(1 << q for q, form in enumerate(forms) if form == 1 << q))
    if len(free) > 8:
        raise ValueError("Exhaustive prefix validation exceeds its budget")
    values = []
    for assignment in range(1 << len(free)):
        sample = (1 << total) | sum(((assignment >> i) & 1) << q for i, q in enumerate(free))
        values.append(sum(parity(form, sample) << i for i, form in enumerate(forms)))
    return values


def aer_records(source: str) -> np.ndarray:
    parsed = clifft.parse(source)
    n = parsed.num_qubits
    circuit = QuantumCircuit(n + parsed.num_measurements)
    records = 0
    for node in parsed.nodes:
        gate = node.gate.name
        if gate in ANNOTATIONS:
            continue
        targets = list(node.targets)
        if gate in {"M", "MX", "MY", "MPP"}:
            if gate == "MPP":
                terms = [(repr(t).lstrip("!")[0], t.value) for t in targets]
            else:
                terms = [({"M": "Z", "MX": "X", "MY": "Y"}[gate], targets[0].value)]
            inverted = sum(repr(t).startswith("!") for t in targets) % 2 != 0
            record_pauli(circuit, terms, n + records, inverted=inverted)
            records += 1
        elif gate == "MPAD":
            if targets[0].value:
                circuit.x(n + records)
            records += 1
        elif gate in {"R", "RX", "RY"}:
            for t in targets:
                circuit.reset(t.value)
                if gate != "R":
                    circuit.h(t.value)
                if gate == "RY":
                    circuit.s(t.value)
        elif any(t.is_rec for t in targets):
            if gate not in {"CX", "CY", "CZ"} or not targets[0].is_rec:
                raise ValueError("Unsupported reference feedback")
            getattr(circuit, gate.lower())(n + targets[0].value, targets[1].value)
        else:
            piece = stim_to_qiskit_noiseless(instruction_text(node, records))
            circuit.compose(piece, qubits=list(range(piece.num_qubits)), inplace=True)
    if records != parsed.num_measurements:
        raise AssertionError("Reference lost records")
    return aer_probabilities(circuit, list(range(n, n + records)))


def small_cases() -> list[str]:
    cases = []
    for seed in range(12):
        rng = random.Random(3010 + seed)
        lines = ["H 0 1 2"]
        for _ in range(6):
            a, b = rng.sample(range(3), 2)
            lines += [
                f"{rng.choice(('H', 'S', 'S_DAG'))} {a}",
                f"{rng.choice(('CX', 'CZ'))} {a} {b}",
                f"DEPOLARIZE2(.03) {a} {b}",
            ]
        if seed % 3 != 0:
            lines += ["MPP(.07) Y0*Z1", "CX rec[-1] 2"]
        if seed % 3 == 2:
            lines += ["R 1", "H 1", "M(.02) 0", "CZ rec[-1] 2"]
        for _ in range(5):
            a, b = rng.sample(range(3), 2)
            lines += [
                f"{rng.choice(('T', 'T_DAG'))} {a}",
                f"DEPOLARIZE1(.09) {a}",
                f"{rng.choice(('CX', 'CZ'))} {a} {b}",
                f"{rng.choice(('S', 'X', 'Y', 'Z'))} {b}",
            ]
        lines += [
            "H 0",
            "M(.03) !0",
            "MY(.04) 1",
            "MX(.05) 2",
            "DETECTOR rec[-1] rec[-3]",
            "OBSERVABLE_INCLUDE(0) rec[-2]",
        ]
        cases.append("\n".join(lines) + "\n")
    return cases


def validate_small(panel: dict[str, tuple[str, str]]) -> dict[str, Any]:
    maximum = 0.0
    histories_checked = branches_checked = 0
    originals = [panel[name][1] for name in ("measured_coset", "measured_feedback")] + small_cases()
    sources = originals + [remap(source) for source in originals]
    for case, source in enumerate(sources):
        shared = SharedPhase(source)
        values = prefix_values(shared)
        rng = random.Random(12310 + case)
        if case % len(originals) < 2:
            histories = [h for h, _ in all_histories(shared.model)]
        else:
            histories = [
                (),
                tuple(
                    (site, rng.randrange(1, len(shared.model.sites[site].replacements)))
                    for site in sorted(rng.sample(range(len(shared.model.sites)), 5))
                ),
            ]
        for history in histories:
            prefix_law(shared, history)
            original = shared.model.render(history)
            reference = aer_records(original)
            observed = np.zeros_like(reference)
            for value in values:
                controls = shared.controls(history, value)
                text = shared.render(controls, None)
                program = clifft.lower(clifft.trace(clifft.parse(text)))
                observed += np.asarray(
                    exact_record_probabilities(program, shared.model.num_records)
                ) / len(values)
                sample_parities(source, clifft.sample(program, shots=3, seed=192, threads=1))
                branches_checked += 1
            maximum = max(maximum, float(max(abs(reference - observed))))
            np.testing.assert_allclose(observed, reference, atol=1e-11, rtol=0)
            histories_checked += 1
        print(f"small case {case} passed", flush=True)
    return {
        "circuits": len(sources),
        "wire_renamed_circuits": len(originals),
        "histories": histories_checked,
        "conditional_branches": branches_checked,
        "maximum_aer_probability_error": maximum,
    }


def validate_distillation(panel: dict[str, tuple[str, str]]) -> dict[str, Any]:
    rows = []
    simulator = AerSimulator(method="statevector", max_parallel_threads=1)
    for name in ("15to1_scored", "15to1_direct_x", "bh_scored", "bh_direct_x"):
        shared = SharedPhase(panel[name][1])
        values = prefix_values(shared)
        if len(values) != 1:
            raise AssertionError("Unexpected random unitary preparation")
        histories = stress_histories(shared.model)
        maximum, count = 0.0, 0
        for history in histories:
            prefix_law(shared, history)
            source, readouts = terminal_unitary(shared.model.render(history))
            circuit = stim_to_qiskit_noiseless(source)
            circuit.save_statevector()
            state = np.asarray(simulator.run(circuit).result().data()["statevector"])
            labels = np.arange(len(state), dtype=np.uint64)
            packed = np.zeros_like(labels)
            for index, (q, inverted) in enumerate(readouts):
                packed |= (
                    ((labels >> np.uint64(q)) & np.uint64(1)) ^ np.uint64(inverted)
                ) << np.uint64(index)
            expected = np.bincount(
                packed.astype(int), weights=abs(state) ** 2, minlength=1 << len(readouts)
            )
            text = shared.render(shared.controls(history, values[0]), None)
            program = clifft.lower(clifft.trace(clifft.parse(text)))
            mass = 0.0
            for outcome in np.flatnonzero(expected > 1e-12):
                probability = visible_probability(
                    program, [int(outcome >> q & 1) for q in range(len(readouts))]
                )
                maximum = max(maximum, abs(probability - expected[outcome]))
                mass += probability
                count += 1
            if abs(mass - 1) > 1e-10 or maximum > 1e-10:
                raise AssertionError("Shared distillation law differs from Aer")
        rows.append(
            {
                "case": name,
                "histories": histories,
                "support_records_checked": count,
                "maximum_probability_error": float(maximum),
                **shared.metadata(),
            }
        )
        print(f"{name} passed", flush=True)
    return {"cases": rows}


def validate_bt(panel: dict[str, tuple[str, str]], binary: Path) -> dict[str, Any]:
    shared = SharedPhase(panel["bt27_scored"][1])
    histories = stress_histories(shared.model)
    rng = random.Random(281719)
    histories += [shared.model.draw(rng) for _ in range(8)]
    checked = 0
    hashes = []
    with tempfile.TemporaryDirectory(prefix="shared-phase-check-") as directory:
        input_path, output_path = (
            Path(directory) / "reference.stim",
            Path(directory) / "export.stim",
        )
        for history in histories:
            prefix_law(shared, history)
            lines = normalized_lines(shared, history)
            full = "\n".join(lines) + "\n"
            prefix = stim.Circuit("\n".join(lines[: shared.first]))
            for seed in (13, 391):
                simulator = stim.TableauSimulator(seed=seed)
                simulator.set_num_qubits(shared.model.num_qubits)
                simulator.do(prefix)
                values = list(map(int, simulator.current_measurement_record()))
                for row in shared.probe_rows:
                    expectation = simulator.peek_observable_expectation(row)
                    if expectation not in (-1, 1):
                        raise AssertionError(
                            "Conditional prefix did not have the certified stabilizers"
                        )
                    values.append(int(expectation == -1))
                faults = 0
                for site, outcome in history:
                    faults ^= shared.outcome_controls[site][outcome]
                controls = (
                    1
                    | (sum(bit << q for q, bit in enumerate(values)) << 1)
                    | (faults << shared.fault_base)
                )
                candidate = stim.Circuit(shared.render(controls, None))
                observed = exact_law(
                    candidate, list(range(shared.model.num_records)), shared.model.num_records
                )
                reference, info = specialize_prefix(full, seed)
                if info["outcomes"] != values[: shared.prefix_records]:
                    raise AssertionError(
                        "The conditional references selected different prefix records"
                    )
                input_path.write_text(reference)
                subprocess.run(
                    [str(binary.resolve()), "--export-source", str(input_path), str(output_path)],
                    check=True,
                    capture_output=True,
                )
                exported_text = output_path.read_text()
                order = [
                    int(line.split()[2])
                    for line in exported_text.splitlines()
                    if line.startswith("# RECORD ")
                ]
                expected = exact_law(stim.Circuit(exported_text), order, shared.model.num_records)
                if observed != expected:
                    raise AssertionError(
                        f"Shared BT conditional joint law differs: {history=} {seed=}"
                    )
                hashes.append(hashlib.sha256(str(observed).encode()).hexdigest())
                checked += 1
    return {
        "histories": histories,
        "conditional_joint_laws": checked,
        "law_hashes": hashes,
        "scope": (
            "Exact prefix laws and exact conditional full-output laws "
            "against the existing whole-circuit optimizer"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=Path("build-profile/profile_bt27_scored_sampling")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = candidates(args.merlin_checkout)
    result = {
        "padding_oracle": padding_oracle_control(),
        "small": validate_small(panel),
        "distillation": validate_distillation(panel),
        "bt27": validate_bt(panel, args.binary),
    }
    result["source_hashes"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name("shared_phase_specialization.py"))
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("all shared-phase validations passed", flush=True)


if __name__ == "__main__":
    main()
