"""Audit the terminal experiment against the pinned author corpus.

Recognition is checked on each complete, unchanged physical T circuit. Isolated
blocks are inspected only to explain rejection; they are not substitute inputs
for the complete-circuit correctness checks.
"""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import stim
from clifford_gadget import Gadget, atoms_from_text, bind_faults, compile_gadgets
from compile_terminal_gadget import compile_bundle, physical_text
from cultivation_study import CORPUS, corpus, validate, variant_text
from soft_cultivation_study import clifford_proxy


def inspect_pair(circuit, forward, inverse):
    fragment = circuit[forward[0] : inverse[1]]
    clean, _ = bind_faults(physical_text(fragment), 0, 0)
    result: dict[str, Any]
    try:
        width, _, _, atoms = atoms_from_text(clean)
        gadgets = [a for a in compile_gadgets(width, atoms) if isinstance(a, Gadget)]
        result = dict(recognized=len(gadgets) == 1)
    except ValueError as error:
        result = dict(recognized=False, reason=str(error))
    middle = circuit[forward[1] : inverse[0]]
    result.update(
        data_qubits=sum(len(op.targets_copy()) for op in circuit[slice(*forward)]),
        middle_operations=dict(Counter(op.name for op in middle)),
    )
    return result


def audit(shots):
    manifest = corpus()
    results = []
    for entry in manifest["files"]:
        if not entry["file"].endswith(".stim"):
            continue
        source = (CORPUS / entry["file"]).read_text()
        physical = variant_text(source, 0.001)
        circuit = stim.Circuit(clifford_proxy(physical)).flattened()
        groups: list[list[int]] = []
        for i, op in enumerate(circuit):
            if op.name in {"S", "S_DAG"}:
                if groups and groups[-1][1] == i:
                    groups[-1][1] = i + 1
                else:
                    groups.append([i, i + 1])
        with tempfile.TemporaryDirectory() as directory:
            recognition: dict[str, Any]
            try:
                compile_bundle(physical, Path(directory))
                recognition = dict(accepted=True)
            except ValueError as error:
                recognition = dict(accepted=False, reason=str(error))
        result = dict(
            file=entry["file"],
            sha256=entry["sha256"],
            physical_variant="author S/S_DAG replaced with T/T_DAG",
            noise_probability=0.001,
            qubits=circuit.num_qubits,
            complete_circuit_recognition=recognition,
            feedback_instruction_count=sum(
                any(t.is_measurement_record_target for t in op.targets_copy())
                for op in circuit
                if op.name in {"CX", "CY", "CZ"}
            ),
            penultimate_t_pair=inspect_pair(circuit, *groups[-4:-2]),
            final_t_pair=inspect_pair(circuit, *groups[-2:]),
            ordinary_clifft_validation=validate(source, shots),
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    return dict(repository=manifest["repository"], revision=manifest["revision"], results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("shots must be positive")
    args.output.write_text(json.dumps(audit(args.shots), indent=2) + "\n")


if __name__ == "__main__":
    main()
