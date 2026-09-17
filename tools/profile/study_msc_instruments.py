"""Validate weighted boundary instruments against complete coherent histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import stim
from clifford_branches import BranchState
from msc_boundaries import compile_boundaries
from msc_instruments import InstrumentPlan
from msc_protocol import Program, evaluate, project
from msc_reference import materialize
from study_msc_boundaries import fault_basis_histories
from study_msc_protocol import load


def copy_state(state):
    result = BranchState(state.width)
    result.terms = [term.copy() for term in state.terms]
    return result


def contract_reference(plan, bound, state):
    """Slow input-projector oracle, retaining interference between all terms.

    This is not the proposed hot contraction: CH state projections and overlaps
    deliberately provide an independent check of the compiled specification.
    """
    if not bound.reachable:
        return 0.0, None
    state = copy_state(state)
    for p, sign in zip(plan.input_paulis, bound.input_signs, strict=True):
        expectations = [term.tableau.peek_observable_expectation(p) for term in state.terms]
        if all(e != 0 for e in expectations):
            state.terms = [
                term for term, e in zip(state.terms, expectations, strict=True) if e == (-1) ** sign
            ]
        else:
            project(state, [("_XYZ"[p[q]], q) for q in range(len(p)) if p[q]], sign)
        if not state.terms:
            return 0.0, None
    norm = state.expectation()
    if norm <= 0:
        return 0.0, None
    bloch = [
        (-1) ** sign * state.expectation(p) / norm
        for p, sign in zip(plan.logical_paulis, bound.logical_signs, strict=True)
    ]
    return bound.probability_scale * norm, bloch


def amplitudes(probability, bloch):
    """Recover a pure logical ket up to its physically irrelevant common phase."""
    x, y, z = bloch
    if abs(x * x + y * y + z * z - 1) > 1e-8:
        raise ValueError("logical density matrix is not pure")
    if z >= 0:
        a = np.sqrt((1 + z) / 2)
        ket = np.array([a, (x + 1j * y) / (2 * a)])
    else:
        b = np.sqrt((1 - z) / 2)
        ket = np.array([(x - 1j * y) / (2 * b), b])
    return np.sqrt(probability) * ket


def observe_history(program, boundaries, plans, case):
    history = dict(case["faults"])
    outcomes = list(map(int, case["outcomes"]))
    pending = {}
    results = []

    def observe(op, state, norm):
        for boundary, plan in zip(boundaries, plans, strict=True):
            if op.line == boundary.instructions[0].line:
                pending[boundary.start] = (copy_state(state), norm)
            if op.line != boundary.end + 1:
                continue
            incoming, input_norm = pending.pop(boundary.start)
            bound = plan.bind(outcomes, history)
            predicted, bloch = contract_reference(plan, bound, incoming)
            outgoing = copy_state(state)
            actual = [
                p.sign.real * outgoing.expectation(plan.positive(p.copy())) / norm
                for p in plan.output_paulis
            ]
            probability_error = abs(predicted - norm) / norm
            bloch_error = float(np.max(np.abs(np.array(bloch) - actual)))
            if probability_error > 1e-8 or bloch_error > 1e-8:
                raise AssertionError(
                    (case["name"], boundary.start, probability_error, bloch, actual)
                )
            ket = amplitudes(predicted / input_norm, bloch)
            # Density-matrix comparison also checks the relative phase recovered
            # from Y; matching only X and Z would miss its conjugation.
            x, y, z = actual
            rho = norm / input_norm * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2
            density_error = float(np.max(np.abs(np.outer(ket, ket.conj()) - rho)))
            if density_error > 1e-8:
                raise AssertionError("weighted amplitude contraction differs")
            results.append(
                {
                    "interval_lines": [boundary.start, boundary.end],
                    "conditional_probability": norm / input_norm,
                    "probability_relative_error": probability_error,
                    "logical_bloch_error": bloch_error,
                    "weighted_density_error": density_error,
                    "logical_bloch": bloch,
                }
            )

    evaluate(program, history, outcomes, before_instruction=observe)
    if len(results) != len(plans) or pending:
        raise AssertionError("boundary observations missing")
    return results


def validate_fault_rows(boundary, plan):
    """Check the full input/constraint/logical flow basis independently in Stim."""
    instructions = [
        op for op in boundary.instructions if op.name not in {"DETECTOR", "OBSERVABLE_INCLUDE"}
    ]
    text = "\n".join(
        op.name
        + ("(" + ",".join(map(str, op.args)) + ")" if op.args else "")
        + " "
        + " ".join(op.targets)
        for op in instructions
    )
    local_program = Program(text)
    source_to_local_line = {op.line: k + 1 for k, op in enumerate(instructions)}
    site_map = {
        k: local_program.site_at[source_to_local_line[site.line], site.offset]
        for k, site in enumerate(plan.program.sites)
        if boundary.start <= site.line <= boundary.end
    }
    flows = plan.input_flows + plan.constraint_flows + plan.logical_flows
    rows = [plan.row(flow) for flow in flows]
    count = 0
    for history in fault_basis_histories(boundary):
        local_history = {site_map[k]: choice for k, choice in history.items()}
        circuit = stim.Circuit(materialize(local_program, local_history))
        fault_bits = plan.bits.encode(plan.program, history)
        signed = []
        for flow, row in zip(flows, rows, strict=True):
            sign = (row.faults & fault_bits).bit_count() & 1
            signed.append(
                stim.Flow(
                    input=flow.input_copy(),
                    output=flow.output_copy() * (-1 if sign else 1),
                    measurements=flow.measurements_copy(),
                )
            )
        if not circuit.has_all_flows(signed):
            raise AssertionError(("instrument fault-flow mismatch", history))
        count += 1
        if count % 512 == 0:
            print(f"interval {boundary.start}:{boundary.end}: {count} fault histories", flush=True)
    return {"fault_basis_histories": count, "signed_flows_per_history": len(flows)}


def summary(plan):
    def row(r):
        return {"outcomes": hex(r.records), "faults": hex(r.faults), "constant": r.constant}

    return {
        "events_including_resets": len(plan.events),
        "input_projector_rank": len(plan.input_paulis),
        "single_wire_input_projectors": sum(p.weight == 1 for p in plan.input_paulis),
        "multi_wire_input_projectors": sum(p.weight > 1 for p in plan.input_paulis),
        "record_constraints": len(plan.constraints),
        "probability_scale_power": plan.random_power,
        "input_projectors": [
            {"pauli": str(p), **row(r)}
            for p, r in zip(plan.input_paulis, plan.input_rows, strict=True)
        ],
        "constraints": [row(r) for r in plan.constraints],
        "logical_pullbacks": [
            {"axis": axis, "pauli": str(p), **row(r)}
            for axis, p, r in zip("XYZ", plan.logical_paulis, plan.logical_rows, strict=True)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source = Path(__file__).parent / "research" / "msc_protocol_data.json"
    results = []
    for circuit in json.loads(source.read_text())["circuits"]:
        program = load(circuit["distance"])
        boundaries = compile_boundaries(program, load(3))
        plans = [InstrumentPlan(b) for b in boundaries]
        certificates = [validate_fault_rows(b, p) for b, p in zip(boundaries, plans, strict=True)]
        print(f"d{circuit['distance']} fault basis passed", flush=True)
        cases = []
        for k, case in enumerate(circuit["cases"][: args.limit]):
            cases.append(
                {
                    "name": case["name"],
                    "boundaries": observe_history(program, boundaries, plans, case),
                }
            )
            print(f"d{circuit['distance']} {k + 1}: {case['name']} passed", flush=True)
        results.append(
            {
                "distance": circuit["distance"],
                "instruments": [
                    {**summary(p), **v} for p, v in zip(plans, certificates, strict=True)
                ],
                "cases": cases,
            }
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
