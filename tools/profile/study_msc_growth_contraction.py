"""Compare sparse original-MSC growth contraction with full noisy histories.

State extraction and fault-bound gadget construction here are offline bridges.
Only GrowthContraction.contract is the new fixed code/ancilla calculation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fold_check import PhaseMonomial
from fold_contraction import ROOTS
from msc_boundaries import compile_boundaries, pauli
from msc_gadgets import terms
from msc_growth_contraction import FactoredTerm, GrowthContraction
from msc_instruments import InstrumentPlan
from msc_protocol import evaluate
from study_msc_instruments import amplitudes, copy_state
from study_msc_protocol import load

PAULIS = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def eigenstate(axis, sign):
    if axis == "Z":
        return np.array([1 - sign, sign], dtype=complex)
    return np.array([1, (-1) ** sign * (1j if axis == "Y" else 1)], dtype=complex) / np.sqrt(2)


def gadget_payload(program, boundary, contraction, history, outcomes):
    """Use the already-validated offline gadget binder to supply two terms."""
    region = list(program.regions.values())[0]
    records = program.outputs(outcomes, history)["records"]
    signs, frame = boundary.evaluate(records, history)
    ancestors = {}
    for p, sign in zip(boundary.checks[6:], signs[6:], strict=True):
        support = [q for q in range(program.width) if p[q]]
        if len(support) != 1:
            raise ValueError("source ancilla is not a single-wire state")
        q = support[0]
        ancestors[q] = eigenstate("_XYZ"[p[q]], sign)
    faults = program.faults(history)
    local = []
    for line, inserted in faults.items():
        if not region.start <= line <= region.end:
            continue
        offset = sum(g.line < line for g in region.gates)
        for axis, q in inserted:
            if q in region.wires:
                local.append((offset, axis, q))
            else:
                ancestors[q] = PAULIS[axis] @ ancestors[q]
    measured = next(g for g in region.gates if g.name == "MX")
    event = program.measurement_at[measured.line, 0]
    reset = next(g for g in region.gates if g.name == "RX")
    hidden = program.measurement_at[reset.line, 0]
    reset_flip = (
        sum(
            axis in {"Y", "Z"} and q == measured.targets[0]
            for line, inserted in faults.items()
            if measured.line < line < reset.line
            for axis, q in inserted
        )
        % 2
    )
    if outcomes[hidden] != outcomes[event] ^ reset_flip:
        raise ValueError("impossible gadget reset outcome")
    operators = terms(region, outcomes[event], local)
    mapping = {q: k for k, q in enumerate(region.wires)}
    result = []
    for op in operators:
        if op.edges:
            raise ValueError("gadget has coupled phase terms")
        ancillas = []
        for q in contraction.ancillas:
            value = ancestors[q].copy()
            if q in mapping:
                j = mapping[q]
                flip = (op.flips >> j) & 1
                value = np.array([value[0], ROOTS[op.linear[j]] * value[1]])
                if flip:
                    value = value[::-1].copy()
            ancillas.append(value)
        data = PhaseMonomial(
            7,
            sum(((op.flips >> mapping[q]) & 1) << j for j, q in enumerate(contraction.data)),
            op.global_phase,
            [op.linear[mapping[q]] for q in contraction.data],
        )
        result.append(FactoredTerm(data, np.array(ancillas)))
    return frame, result


def logical_bloch(state, data, norm):
    state = copy_state(state)
    x = pauli(state.width, "X", data)
    z = pauli(state.width, "Z", data)
    result = []
    for p in (x, 1j * x * z, z):
        sign = p.sign.real
        p.sign = 1
        result.append(sign * state.expectation(p) / norm)
    return result


def observe_history(program, boundaries, instrument, contraction, case):
    history = dict(case["faults"])
    outcomes = list(map(int, case["outcomes"]))
    source, growth = boundaries
    frame, payload = gadget_payload(program, source, contraction, history, outcomes)
    pending = {}
    result = {}

    def observe(op, state, norm):
        if op.line == source.end + 1:
            pending["norm"] = norm
            pending["logical"] = amplitudes(1, logical_bloch(state, source.data, norm))
        if op.line == growth.end + 1:
            bound = instrument.bind(outcomes, history)
            output = contraction.contract(pending["logical"], frame, payload, bound)
            predicted = float(np.vdot(output, output).real)
            expected = norm / pending["norm"]
            bloch = logical_bloch(state, growth.data, norm)
            x, y, z = bloch
            density = expected * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2
            relative_error = abs(predicted - expected) / expected
            density_error = (
                float(np.max(np.abs(np.outer(output, output.conj()) - density))) / expected
            )
            if relative_error > 1e-10 or density_error > 1e-10:
                raise AssertionError((case["name"], relative_error, density_error, output, bloch))
            result.update(
                {
                    "name": case["name"],
                    "gadget_and_growth_probability": expected,
                    "probability_relative_error": relative_error,
                    "normalized_density_error": density_error,
                    "input_frame": list(frame),
                    "input_syndrome": [bound.input_signs[k] for k in contraction.data_rows],
                    "logical_signs": list(bound.logical_signs),
                }
            )

    evaluate(program, history, outcomes, before_instruction=observe)
    if not result:
        raise AssertionError("growth contraction was not observed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    program = load(5)
    boundaries = compile_boundaries(program, load(3))
    instrument = InstrumentPlan(boundaries[-1])
    contraction = GrowthContraction(boundaries[0], instrument)
    source = Path(__file__).parent / "research" / "msc_protocol_data.json"
    cases = json.loads(source.read_text())["circuits"][1]["cases"][: args.limit]
    results = []
    for case in cases:
        results.append(observe_history(program, boundaries, instrument, contraction, case))
        print(case["name"], "passed", flush=True)
    args.output.write_text(
        json.dumps(
            {
                "data_wires": list(contraction.data),
                "source_entries": sum(map(len, contraction.source_basis)),
                "syndrome_sectors": 1 << len(contraction.data_rows),
                "decoder_bytes": contraction.decoder.nbytes,
                "ancilla_projectors": len(contraction.ancilla_rows),
                "unprojected_spectators": [
                    contraction.ancillas[q] for q in contraction.unprojected
                ],
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )
