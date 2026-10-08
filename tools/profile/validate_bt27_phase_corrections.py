"""Check fault relocation with Aer Choi states and complete BT27 shot outputs.

Aer independently checks small operator identities, including entangled
spectators. The large-fixture check is a Clifft differential comparison, not a
second independent simulator or full tomography.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import stim
from analyze_bt27_phase_corrections import correction, moved_source, phase_polynomial, region
from profile_bt27_fault_specialization import FIXTURE, Faults, source
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from validate_bt27_fault_specialization import compare, probes

import clifft


def append_gates(circuit: Any, lines: list[str]) -> None:
    for line in lines:
        name, *targets = line.split()
        gate = {"T_DAG": "tdg", "S_DAG": "sdg"}.get(name, name.lower())
        getattr(circuit, gate)(*(int(q) for q in targets))


def parity_rotation(support: list[int], dagger: bool) -> list[str]:
    target = support[-1]
    compute = [f"CX {q} {target}" for q in support[:-1]]
    return compute + [f"{'T_DAG' if dagger else 'T'} {target}"] + list(reversed(compute))


def choi_pair(width: int, gates: list[str], faults: Faults) -> tuple[Any, Any, list[str]]:
    poly, coordinates = phase_polynomial(gates, width)
    if coordinates != [1 << q for q in range(width)]:
        raise AssertionError("Generated operator is not diagonal")
    shifted, _, _ = correction(poly, faults)
    initial = QuantumCircuit(2 * width)
    for q in range(width):
        initial.h(q)
        initial.cx(q, q + width)
    original = initial.copy()
    relocated = initial.copy()
    append_gates(original, [f"{p} {q}" for q, p in faults] + gates)
    append_gates(relocated, gates + shifted)
    return original, relocated, shifted


def validate_aer() -> dict[str, Any]:
    ccz = [
        gate
        for degree in (1, 2, 3)
        for support in combinations(range(3), degree)
        for gate in parity_rotation(list(support), degree == 2)
    ]
    pairs = [
        choi_pair(3, ccz, tuple((q, p) for q, p in enumerate(paulis) if p != "I"))
        for paulis in product("IXYZ", repeat=3)
    ]
    rng = random.Random(426091)
    # General parity rotations exercise S and S_DAG corrections, which the
    # homogeneous BT27 phase does not need.
    for _ in range(64):
        gates = [
            gate
            for _ in range(16)
            for gate in parity_rotation(
                rng.sample(range(4), rng.randint(1, 4)), rng.choice([False, True])
            )
        ]
        faults = tuple((q, p) for q in range(4) if (p := rng.choice("IXYZ")) != "I")
        pairs.append(choi_pair(4, gates, faults))
    circuits = []
    counts: Counter[str] = Counter()
    for left, right, shifted in pairs:
        counts.update(line.split()[0] for line in shifted)
        for circuit in (left, right):
            circuit.save_statevector()
            circuits.append(circuit)
    # Deliberately omit the quadratic-in-fault-bits Z term for two X faults.
    # This catches a falsely passing oracle restricted to single faults.
    negative, incomplete, shifted = choi_pair(3, ccz, ((0, "X"), (1, "X")))
    if "Z 2" not in shifted:
        raise AssertionError("Two-fault control did not produce its required Z term")
    incomplete.z(2)
    negative.save_statevector()
    incomplete.save_statevector()
    circuits += [negative, incomplete]
    simulation = AerSimulator(method="statevector", max_parallel_threads=1)
    results = simulation.run(circuits).result()
    maximum_error = 0.0
    for i in range(len(pairs)):
        left = np.asarray(results.get_statevector(2 * i))
        right = np.asarray(results.get_statevector(2 * i + 1))
        overlap = np.vdot(left, right)
        aligned = right * (overlap.conjugate() / abs(overlap)) if abs(overlap) else right
        error = float(np.max(np.abs(left - aligned)))
        maximum_error = max(maximum_error, error)
        np.testing.assert_allclose(left, aligned, atol=1e-12, rtol=0)
    overlap = abs(
        np.vdot(
            results.get_statevector(len(circuits) - 2), results.get_statevector(len(circuits) - 1)
        )
    )
    if overlap > 1e-12:
        raise AssertionError("The deliberately incomplete correction was not detected")
    return {
        "qiskit_aer_version": importlib.metadata.version("qiskit-aer"),
        "exhaustive_ccz_paulis": 64,
        "random_four_qubit_diagonal_circuits": 64,
        "maximum_choi_amplitude_error_up_to_global_phase": maximum_error,
        "correction_gate_counts": dict(counts),
        "missing_quadratic_fault_term_negative_control_overlap": float(overlap),
    }


def validate_physical_frames(study: dict[str, Any]) -> dict[str, Any]:
    _, _, _, poly = region(FIXTURE.read_text())

    def frame(faults: Faults) -> Any:
        gates, _, _ = correction(poly, faults)
        # Keep identity and sparse cases in the same 81-qubit coordinate space.
        return stim.Tableau.from_circuit(stim.Circuit("\n".join(gates + ["I 80"])))

    identity = stim.Tableau(81)
    generators = {(q, p): frame(((q, p),)) for q in range(81) for p in "XZ"}
    for generator in generators.values():
        if generator.then(generator) != identity:
            raise AssertionError("A physical Pauli correction is not an involution")
    checked = 0
    for row in study["results"].values():
        if "faults" not in row:
            continue
        faults = tuple((q, p) for q, p in row["faults"])
        composed = identity
        for q, p in faults:
            for axis in "XZ" if p == "Y" else p:
                composed = composed.then(generators[q, axis])
        if composed != frame(faults):
            raise AssertionError("Physical correction composition disagrees with phase translation")
        checked += 1
    return {"involutive_generators": len(generators), "composed_patterns_checked": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=512)
    args = parser.parse_args()
    if args.shots < 16:
        parser.error("at least 16 shots are required")
    started = perf_counter()
    aer = validate_aer()
    print(f"Aer checked 128 operator identities: {aer}", flush=True)
    study = json.loads(args.study.read_text())
    physical_frames = validate_physical_frames(study)
    print(f"Stim checked physical correction composition: {physical_frames}", flush=True)
    original = FIXTURE.read_text()
    clifford = stim.Circuit(
        "\n".join(line for line in original.splitlines() if not line.startswith(("T ", "T_DAG ")))
    )
    converter = clifford.compile_m2d_converter(skip_reference_sample=True)
    reference = study["reference"]
    probe_text = "".join(f"EXP_VAL {p}\n" for p in probes())
    rows = {}
    for key, row in study["results"].items():
        if "faults" not in row:
            continue
        faults = tuple((q, p) for q, p in row["faults"])
        programs = [
            clifft.compile(
                circuit + probe_text,
                expected_detectors=reference["detectors"],
                expected_observables=reference["observables"],
            )
            for circuit in (source(faults), moved_source(original, faults))
        ]
        widths = [program.peak_active_width for program in programs]
        if max(widths) > 16:
            raise AssertionError("Fault relocation exceeded the study width budget")
        samples = [
            clifft.sample(program, shots=args.shots, seed=seed, threads=1, batch_size=1)
            for program, seed in zip(programs, (145831, 977131))
        ]
        result = compare(samples[0], samples[1], converter, reference)
        # Check the relocated circuit's detector references too.
        det, obs = converter.convert(
            measurements=samples[1].measurements.astype(bool), separate_observables=True
        )
        np.testing.assert_array_equal(samples[1].detectors, det ^ np.array(reference["detectors"]))
        np.testing.assert_array_equal(
            samples[1].observables, obs ^ np.array(reference["observables"], dtype=bool)
        )
        rows[key] = {"peak_widths_with_probes": widths, **result}
        if len(rows) % 32 == 0:
            print(f"Checked {len(rows)} full-fixture fault relocations", flush=True)
    output = {
        "study_sha256": hashlib.sha256(args.study.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "aer_operator_validation": aer,
        "stim_physical_frame_validation": physical_frames,
        "fixture_validation": {
            "method": "Clifft differential comparison with independent shot seeds",
            "shots_per_pattern_per_side": args.shots,
            "pauli_probes": len(probes()),
            "limitations": (
                "Finite probe and record moments, not tomography or independent BT27 simulation"
            ),
            "results": rows,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Validated {len(rows)} full-fixture patterns", flush=True)


if __name__ == "__main__":
    main()
