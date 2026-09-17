"""Reproduce the bounded corpus gadget and terminal-code coverage screen."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import stim
from fold_contraction import compose
from msc_gadgets import contract_ancillas, read_gates, regions, terms


def terminal_code(text, data):
    """Certify the actual terminal CSS checks, not the preceding noisy boundary."""
    index = {q: k for k, q in enumerate(data)}
    last = [line for line in text.splitlines() if line.startswith("MPP ")][-1]
    checks = []
    xs: list[int] = []
    zs: list[int] = []
    for word in last.split()[1:]:
        axis = word[0]
        factors = word.split("*")
        if axis not in "XZ" or any(p[0] != axis or int(p[1:]) not in index for p in factors):
            raise ValueError("terminal checks are not CSS on the gadget data")
        mask = sum(1 << index[int(p[1:])] for p in factors)
        (xs if axis == "X" else zs).append(mask)
        checks.append(
            stim.PauliString("".join(axis if mask >> q & 1 else "_" for q in range(len(data))))
        )
    if len(checks) != len(data) - 1:
        raise ValueError("not a single logical qubit")
    # Stim independently rejects inconsistent, anticommuting or redundant checks.
    stim.Tableau.from_stabilizers(checks, allow_underconstrained=True)
    span = {0}
    for mask in xs:
        span |= {b ^ mask for b in span}
    if len(span) != 1 << len(xs) or any((b & z).bit_count() % 2 for b in span for z in zs):
        raise ValueError("invalid CSS span")
    logical = (1 << len(data)) - 1
    if logical in span or any(z.bit_count() % 2 for z in zs):
        raise ValueError("all-data X is not a logical representative")
    return [span, {b ^ logical for b in span}], len(checks)


def logical_matrix(op, basis):
    result = np.zeros((2, 2), dtype=complex)
    for column, support in enumerate(basis):
        for b in support:
            output, phase = op.column(b)
            for row, target in enumerate(basis):
                if output in target:
                    result[row, column] += np.exp(1j * np.pi * phase / 4) / len(support)
    return result


def ideal_boundary(region, basis, check):
    result = []
    for outcome in (0, 1):
        operators = terms(region, outcome)
        records = {q: operators[0].linear[region.wires.index(q)] // 4 for q in region.ancillas}
        reduced = contract_ancillas(region, operators, records)
        if len(reduced) != 2:
            raise ValueError("ideal branches do not share an ancilla record")
        correction = reduced[0]
        if correction.flips or correction.edges or any(c not in (0, 4) for c in correction.linear):
            raise ValueError("ideal boundary correction is not Z type")
        matrices = [logical_matrix(compose(correction, op), basis) for op in reduced]
        # Unit norm in the code subspace proves each corrected branch has no
        # leakage; merely computing V-dagger K V would not establish that.
        leakage = max(float(np.max(np.abs(m.conj().T @ m - np.eye(2)))) for m in matrices)
        actual = sum(matrices) / 2
        expected = (np.eye(2) + (-1) ** outcome * check) / 2
        result.append(
            {
                "true_root_outcome": outcome,
                "nonzero_ancilla_records": [q for q, bit in records.items() if bit],
                "data_z_correction": [
                    q for q, c in zip(region.data, correction.linear, strict=True) if c
                ],
                "code_preservation_error": leakage,
                "logical_projector_error": float(np.max(np.abs(actual - expected))),
            }
        )
    return result


def describe(path):
    text = path.read_text()
    found = regions(text)
    terminal = found[-1]
    basis, rank = terminal_code(text, terminal.data)
    op = terms(terminal, 0)[1]
    matrix = logical_matrix(op, basis)
    total = sum(g.name in {"T", "T_DAG"} for g in read_gates(text))
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "t_count": total,
        "certified_region_t_count": sum(r.t_count for r in found),
        "uncovered_t_lines": [
            g.line
            for g in read_gates(text)
            if g.name in {"T", "T_DAG"} and not any(r.start <= g.line <= r.end for r in found)
        ],
        "regions": [
            {
                "lines": [r.start, r.end],
                "physical_wires": len(r.wires),
                "data_wires": len(r.data),
                "ancillas": len(r.ancillas),
                "t_count": r.t_count,
                "coherent_terms": 2,
                "single_pauli_histories_in_test": 3 * len(r.wires) * (len(r.gates) + 1),
            }
            for r in found
        ],
        "terminal_code_rank": rank,
        "terminal_css_basis_entries_per_logical_state": len(basis[0]),
        "terminal_check_logical_matrix": [[[v.real, v.imag] for v in row] for row in matrix],
        "terminal_check_code_preservation_error": float(
            np.max(np.abs(matrix.conj().T @ matrix - np.eye(2)))
        ),
        "ideal_final_cultivation_boundary": ideal_boundary(found[-2], basis, matrix),
        "full_circuit_sampler_supported": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = Path(__file__).parent / "fixtures" / "msc"
    result = [describe(p) for p in sorted(fixtures.glob("*.stim"))]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
