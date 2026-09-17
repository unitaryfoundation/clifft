"""Full-history checks of the coherent final cultivation and terminal block.

Initial-state extraction and gadget fault binding remain offline reference
bridges. TerminalContraction supplies the fixed sparse code contraction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fold_check import PhaseMonomial
from fold_contraction import ROOTS, compose
from msc_boundaries import compile_boundaries
from msc_gadgets import terms
from msc_growth_contraction import FactoredTerm, GrowthContraction
from msc_instruments import InstrumentPlan
from msc_protocol import evaluate
from msc_terminal_contraction import TerminalContraction
from study_msc_growth_contraction import PAULIS, eigenstate, gadget_payload, logical_bloch
from study_msc_instruments import amplitudes
from study_msc_protocol import load


def terminal_payload(program, boundary, plan, history, outcomes):
    """Bind the final two gadgets and intervening ancilla measurements offline."""
    records = program.outputs(outcomes, history)["records"]
    signs, frame = boundary.evaluate(records, history)
    ancillas = np.zeros((len(plan.ancillas), 2), dtype=complex)
    for p, sign in zip(
        boundary.checks[boundary.code_count :], signs[boundary.code_count :], strict=True
    ):
        support = [q for q in plan.ancillas if p[q]]
        if len(support) != 1:
            raise ValueError("terminal input spectator is not a single-wire state")
        q = support[0]
        ancillas[plan.ancillas.index(q)] = eigenstate("_XYZ"[p[q]], sign)
    payload = [FactoredTerm(PhaseMonomial(plan.width), ancillas, 1)]
    faults = program.faults(history)
    regions = {r.start: r for r in list(program.regions.values())[-2:]}
    skip_until = boundary.end
    for op in program.instructions:
        if op.line <= skip_until or op.line >= plan.measurement.line:
            continue
        if op.line in regions:
            region = regions[op.line]
            local = []
            for line, inserted in faults.items():
                if not region.start <= line <= region.end:
                    continue
                offset = sum(g.line < line for g in region.gates)
                for axis, q in inserted:
                    if q in region.wires:
                        local.append((offset, axis, q))
                    else:
                        j = plan.ancillas.index(q)
                        for term in payload:
                            term.ancillas[j] = PAULIS[axis] @ term.ancillas[j]
            measured = next(g for g in region.gates if g.name in {"MX", "MPP_Y"})
            event = program.measurement_at[measured.line, 0]
            reset = next((g for g in region.gates if g.name == "RX"), None)
            if reset is not None:
                flip = (
                    sum(
                        axis in {"Y", "Z"} and q == measured.targets[0]
                        for line, inserted in faults.items()
                        if measured.line < line < reset.line
                        for axis, q in inserted
                    )
                    % 2
                )
                if outcomes[program.measurement_at[reset.line, 0]] != outcomes[event] ^ flip:
                    raise ValueError("impossible terminal gadget reset outcome")
            expanded = []
            mapping = {q: k for k, q in enumerate(region.wires)}
            for branch in terms(region, outcomes[event], local):
                if branch.edges:
                    raise ValueError("terminal gadget has coupled phases")
                data_op = PhaseMonomial(
                    plan.width,
                    sum(((branch.flips >> mapping[q]) & 1) << j for j, q in enumerate(plan.data)),
                    branch.global_phase,
                    [branch.linear[mapping[q]] for q in plan.data],
                )
                for term in payload:
                    spectators = term.ancillas.copy()
                    for j, q in enumerate(plan.ancillas):
                        if q in mapping:
                            k = mapping[q]
                            spectators[j, 1] *= ROOTS[branch.linear[k]]
                            if branch.flips >> k & 1:
                                spectators[j] = spectators[j, ::-1].copy()
                    expanded.append(
                        FactoredTerm(compose(data_op, term.data), spectators, term.weight / 2)
                    )
            payload = expanded
            skip_until = region.end
        elif op.name == "MX":
            for offset, word in enumerate(op.targets):
                j = plan.ancillas.index(int(word))
                event = program.measurement_at[op.line, offset]
                target = eigenstate("X", outcomes[event])
                for term in payload:
                    term.weight *= np.vdot(target, term.ancillas[j])
                    term.ancillas[j] = target.copy()
        elif op.name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
            for axis, q in faults.get(op.line, []):
                for term in payload:
                    if q in plan.data:
                        term.data.append(axis, (plan.data.index(q),))
                    else:
                        j = plan.ancillas.index(q)
                        term.ancillas[j] = PAULIS[axis] @ term.ancillas[j]
        elif op.name not in {
            "TICK",
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "SHIFT_COORDS",
            "QUBIT_COORDS",
        }:
            raise ValueError("unsupported operation between terminal gadgets")
    if len(payload) != 4:
        raise ValueError("terminal block did not bind two two-term gadgets")
    return frame, payload


def observe_history(program, boundary, plan, case, *, growth=None):
    history = dict(case["faults"])
    outcomes = list(map(int, case["outcomes"]))
    frame, payload = terminal_payload(program, boundary, plan, history, outcomes)
    syndrome = [outcomes[k] for k in plan.events]
    if growth is not None:
        growth_boundary, instrument, contraction = growth
        growth_frame, growth_payload = gadget_payload(
            program, growth_boundary, contraction, history, outcomes
        )
    pending = {}
    result = {}

    def observe(op, state, norm):
        if growth is not None and op.line == growth_boundary.end + 1:
            pending["first_norm"] = norm
            incoming = amplitudes(1, logical_bloch(state, growth_boundary.data, norm))
            pending["grown"] = contraction.contract(
                incoming, growth_frame, growth_payload, instrument.bind(outcomes, history)
            )
        if op.line == boundary.end + 1:
            pending["norm"] = norm
            pending["logical"] = amplitudes(1, logical_bloch(state, plan.data, norm))
        if op.line == plan.measurement.line + 1:
            output = plan.contract(pending["logical"], frame, payload, syndrome)
            probability = float(np.vdot(output, output).real)
            expected = norm / pending["norm"]
            x, y, z = logical_bloch(state, plan.data, norm)
            density = expected * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2
            error = float(np.max(np.abs(np.outer(output, output.conj()) - density))) / expected
            probability_error = abs(probability - expected) / expected
            if max(error, probability_error) > 1e-10:
                raise AssertionError((case["name"], error, probability_error, output, density))
            result.update(
                {
                    "name": case["name"],
                    "terminal_probability": expected,
                    "probability_relative_error": probability_error,
                    "normalized_density_error": error,
                    "input_frame": list(frame),
                    "final_syndrome": syndrome,
                    "coherent_terms": len(payload),
                }
            )
            if growth is not None:
                chained = plan.contract(pending["grown"], frame, payload, syndrome)
                combined = norm / pending["first_norm"]
                expected_density = density * combined / expected
                chain_error = (
                    float(np.max(np.abs(np.outer(chained, chained.conj()) - expected_density)))
                    / combined
                )
                chain_probability_error = (
                    abs(float(np.vdot(chained, chained).real) - combined) / combined
                )
                if max(chain_error, chain_probability_error) > 1e-10:
                    raise AssertionError(
                        ("growth to terminal handoff differs", case["name"], chain_error)
                    )
                result.update(
                    {
                        "chained_probability": combined,
                        "chained_probability_relative_error": chain_probability_error,
                        "chained_normalized_density_error": chain_error,
                    }
                )

    actual = evaluate(program, history, outcomes, before_instruction=observe)
    if not result or actual["records"] != list(map(int, case["records"])):
        raise AssertionError("terminal observation missing or record contract changed")
    return result


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
        boundary = boundaries[-1]
        growth = None
        if len(boundaries) == 2:
            instrument = InstrumentPlan(boundary)
            growth = (boundaries[0], instrument, GrowthContraction(boundaries[0], instrument))
        plan = TerminalContraction(program, boundary)
        cases = []
        for case in circuit["cases"][: args.limit]:
            cases.append(observe_history(program, boundary, plan, case, growth=growth))
            print(circuit["distance"], case["name"], "passed", flush=True)
        results.append(
            {
                "distance": circuit["distance"],
                "data_wires": list(plan.data),
                "basis_entries": [len(row) for row in plan.supports],
                "syndrome_sectors": 1 << len(plan.duals),
                "decoder_entries": len(plan.decoder),
                "cases": cases,
            }
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
