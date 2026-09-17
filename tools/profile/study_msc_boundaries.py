"""Validate static MSC boundary signs against full histories and signed flows."""

import argparse
import json
from pathlib import Path

import stim
from msc_boundaries import compile_boundaries, pauli
from msc_protocol import evaluate
from study_msc_protocol import load


def observe_history(program, plans, case):
    history = dict(case["faults"])
    outcomes = list(map(int, case["outcomes"]))
    records = program.outputs(outcomes, history)["records"]
    entries = []

    def observe(op, state, norm):
        for plan in plans:
            if op.line != plan.end + 1:
                continue
            signs, (x, z) = plan.evaluate(records, history)
            comparisons = 0
            for check, sign in zip(plan.checks, signs, strict=True):
                for term in state.terms:
                    if term.tableau.peek_observable_expectation(check) != (-1) ** sign:
                        raise AssertionError(f"boundary sign mismatch at line {op.line}: {check}")
                    comparisons += 1
            for check, sign in zip(
                plan.checks[: plan.code_count], signs[: plan.code_count], strict=True
            ):
                flipped = (
                    sum(
                        ((z >> j) & 1) * int(check[q] in (1, 2))
                        + ((x >> j) & 1) * int(check[q] in (2, 3))
                        for j, q in enumerate(plan.data)
                    )
                    % 2
                )
                if flipped != sign:
                    raise AssertionError("compiled correction does not restore the positive code")
            entries.append(
                {
                    "entry_line": op.line,
                    "check_signs": "".join(map(str, signs)),
                    "data_syndrome": "".join(map(str, signs[: plan.code_count])),
                    "frame_x": x,
                    "frame_z": z,
                    "term_stabilizer_comparisons": comparisons,
                }
            )

    actual = evaluate(program, history, outcomes, before_instruction=observe)
    if len(entries) != len(plans):
        raise AssertionError("not all planned boundaries were reached")
    if actual["records"] != list(map(int, case["records"])):
        raise AssertionError("observing boundaries changed the record contract")
    return entries


def noisy_interval(plan, history):
    """Independent Stim circuit with deterministic Paulis and inverted readouts."""
    faults = plan.program.faults(history)
    circuit = stim.Circuit()
    quantum = {"R", "RX", "M", "MX", "MPP", "CX", "CZ", "H", "S", "S_DAG", "X", "Y", "Z"}
    for op in plan.instructions:
        if op.name in quantum:
            targets = list(op.targets)
            if op.name in {"M", "MX", "MPP"}:
                for offset, word in enumerate(targets):
                    if plan.program.site_at.get((op.line, offset)) in history:
                        targets[offset] = "!" + word
            circuit += stim.Circuit(op.name + " " + " ".join(targets))
        else:
            for axis, q in faults.get(op.line, []):
                circuit.append(axis, [q])
    return circuit


def fault_basis_histories(plan):
    yield {}
    for k, site in enumerate(plan.program.sites):
        if not plan.start <= site.line <= plan.end:
            continue
        if site.readout:
            yield {k: "flip"}
            continue
        choices = set()
        for offset in range(len(site.wires)):
            for axis in "XZ":
                choice = "I" * offset + axis + "I" * (len(site.wires) - offset - 1)
                if choice in site.choices:
                    choices.add(choice)
        if not choices:
            choices.update(site.choices)
        for choice in sorted(choices):
            yield {k: choice}


def validate_fault_rows(plan):
    certificates = []
    for check in plan.checks:
        if any(check[q] for q in plan.isolated):
            continue
        row, records = plan.flow_row(check)
        certificates.append((stim.PauliString(plan.program.width), check, row, records))
    for axis, (row, records) in plan.logical.items():
        incoming = pauli(plan.program.width, axis, list(plan.program.regions.values())[0].data)
        certificates.append((incoming, pauli(plan.program.width, axis, plan.data), row, records))
    for check, (row, records) in zip(plan.input_checks, plan.input_rows, strict=True):
        certificates.append((check, stim.PauliString(plan.program.width), row, records))
    count = 0
    for history in fault_basis_histories(plan):
        bits = plan.bits.encode(plan.program, history)
        flows = [
            stim.Flow(
                input=inp, output=(-out if row.evaluate(0, bits) else out), measurements=records
            )
            for inp, out, row, records in certificates
        ]
        if not noisy_interval(plan, history).has_all_flows(flows):
            raise AssertionError(f"signed fault-flow mismatch: {history}")
        count += 1
    return {"fault_basis_histories": count, "signed_flows_per_history": len(certificates)}


def summary(plan):
    used = 0
    rows = plan.rows + [r for r, _ in plan.logical.values()] + [r for r, _ in plan.input_rows]
    for row in rows:
        used |= row.faults
    return {
        "interval_lines": [plan.start, plan.end],
        "data_wires": list(plan.data),
        "independent_output_stabilizers": len(plan.checks),
        "data_stabilizers": plan.code_count,
        "isolated_spectators": plan.isolated,
        "local_visible_records": len(plan.local_records),
        "used_fault_bits": used.bit_count(),
        "logical_flows": {
            axis: {"local_record_indices": records, "constant": row.constant}
            for axis, (row, records) in plan.logical.items()
        },
        "measured_input_stabilizers": len(plan.input_rows),
        "output_rows": [
            {
                "pauli": str(check),
                "records": hex(row.records),
                "faults": hex(row.faults),
                "constant": row.constant,
            }
            for check, row in zip(plan.checks, plan.rows, strict=True)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = Path(__file__).parent / "research" / "msc_protocol_data.json"
    result = []
    for circuit in json.loads(fixture.read_text())["circuits"]:
        program = load(circuit["distance"])
        plans = compile_boundaries(program, load(3))
        boundaries = []
        for plan in plans:
            boundaries.append({**summary(plan), **validate_fault_rows(plan)})
            print(
                f"d{circuit['distance']} interval {plan.start}:{plan.end}: fault basis passed",
                flush=True,
            )
        cases = []
        for k, case in enumerate(circuit["cases"]):
            cases.append(
                {"name": case["name"], "boundaries": observe_history(program, plans, case)}
            )
            if k % 12 == 0:
                print(
                    f"d{circuit['distance']}: {k + 1}/{len(circuit['cases'])} full histories",
                    flush=True,
                )
        result.append({"distance": circuit["distance"], "boundaries": boundaries, "cases": cases})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
