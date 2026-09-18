"""Validate direct injection and complete fixed-history logical-block weights.

Gadget payload binding uses compiled parity polynomials and fixed schedules.
The coefficient path never obtains an initial or intermediate CH state.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import stim
from msc_boundaries import compile_boundaries
from msc_growth_contraction import GrowthContraction
from msc_injection import InjectionPlan
from msc_instruments import InstrumentPlan
from msc_protocol import evaluate
from msc_reference import materialize
from msc_static_gadgets import PayloadPlan
from msc_terminal_contraction import TerminalContraction
from study_msc_boundaries import fault_basis_histories
from study_msc_growth_contraction import logical_bloch
from study_msc_protocol import load


class FixedHistoryPlan:
    def __init__(self, program):
        self.program = program
        self.boundaries = compile_boundaries(program, load(3))
        self.injection = InjectionPlan(program, self.boundaries[0])
        self.terminal = TerminalContraction(program, self.boundaries[-1])
        self.growth = None
        if len(self.boundaries) == 2:
            self.instrument = InstrumentPlan(self.boundaries[-1])
            self.growth = GrowthContraction(self.boundaries[0], self.instrument)
            self.growth_payload = PayloadPlan(
                program, self.boundaries[0], self.growth, terminal=False
            )
        self.terminal_payload = PayloadPlan(
            program, self.boundaries[-1], self.terminal, terminal=True
        )

    def coefficients(self, history, outcomes):
        initial = self.injection.evaluate(outcomes, history)
        logical = initial
        if not np.any(initial):
            return initial, initial.copy()
        if self.growth is not None:
            frame, payload = self.growth_payload.bind(history, outcomes)
            logical = self.growth.contract(
                logical, frame, payload, self.instrument.bind(outcomes, history)
            )
        frame, payload = self.terminal_payload.bind(history, outcomes)
        syndrome = [outcomes[k] for k in self.terminal.events]
        return initial, self.terminal.contract(logical, frame, payload, syndrome)


def compare(plan, case):
    history = dict(case["faults"])
    outcomes = list(map(int, case["outcomes"]))
    initial, final = plan.coefficients(history, outcomes)
    result = {"name": case["name"]}

    def check(state, norm, output, data, label):
        predicted = float(np.vdot(output, output).real)
        x, y, z = logical_bloch(state, data, norm)
        density = norm * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2
        probability_error = abs(predicted - norm) / norm
        density_error = float(np.max(np.abs(np.outer(output, output.conj()) - density))) / norm
        if max(probability_error, density_error) > 1e-10:
            raise AssertionError((case["name"], label, probability_error, density_error))
        result[label] = {
            "probability": norm,
            "probability_relative_error": probability_error,
            "normalized_density_error": density_error,
        }

    def observe(op, state, norm):
        if op.line == plan.boundaries[0].end + 1:
            check(state, norm, initial, plan.boundaries[0].data, "injection")
        if op.line == plan.terminal.measurement.line + 1:
            check(state, norm, final, plan.terminal.data, "complete")

    actual = evaluate(plan.program, history, outcomes, before_instruction=observe)
    outputs = plan.program.outputs(outcomes, history)
    if outputs != {k: actual[k] for k in outputs} or outputs["records"] != list(
        map(int, case["records"])
    ):
        raise AssertionError("complete fixed-history record contract differs")
    if "injection" not in result or "complete" not in result:
        raise AssertionError("missing initial or final comparison")
    return result


def validate_fault_rows(plan):
    injection = plan.injection
    rows = [injection.instrument.row(flow) for flow in injection.certificates]
    prefix = SimpleNamespace(program=plan.program, start=0, end=injection.end)
    count = 0
    for history in fault_basis_histories(prefix):
        virtual_history = {injection.site_map[k]: value for k, value in history.items()}
        circuit = stim.Circuit("R " + " ".join(map(str, range(injection.virtual.width))))
        circuit += stim.Circuit(materialize(injection.virtual, virtual_history))
        bits = injection.instrument.bits.encode(injection.virtual, virtual_history)
        flows = [
            stim.Flow(
                input=flow.input_copy(),
                output=flow.output_copy() * (-1 if (row.faults & bits).bit_count() & 1 else 1),
                measurements=flow.measurements_copy(),
            )
            for flow, row in zip(injection.certificates, rows, strict=True)
        ]
        if not circuit.has_all_flows(flows):
            raise AssertionError(("injection fault certificate differs", history))
        count += 1
        if count % 512 == 0:
            print("prefix", plan.program.width, count, "fault histories passed", flush=True)
    return {"fault_basis_histories": count, "signed_flows_per_history": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-fault-basis", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).parent / "research" / "msc_protocol_data.json"
    results = []
    for circuit in json.loads(source.read_text())["circuits"]:
        plan = FixedHistoryPlan(load(circuit["distance"]))
        certificate = {} if args.skip_fault_basis else validate_fault_rows(plan)
        cases = []
        for case in circuit["cases"][: args.limit]:
            cases.append(compare(plan, case))
            print(circuit["distance"], case["name"], "passed", flush=True)
        results.append(
            {
                "distance": circuit["distance"],
                "injection_events": len(plan.injection.event_map),
                "record_constraints": len(plan.injection.constraints),
                "free_outcomes": plan.injection.free_outcomes,
                "choi_random_power": plan.injection.random_power,
                "choi_stabilizers": list(map(str, plan.injection.joint_paulis)),
                **certificate,
                "cases": cases,
            }
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
