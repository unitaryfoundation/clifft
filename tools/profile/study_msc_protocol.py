"""Bounded complete-trajectory checks for the original MSC controls."""

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

from msc_protocol import Program, evaluate
from msc_reference import classical_reference, materialize, reference

FIXTURES = Path(__file__).parent / "fixtures" / "msc"


def load(distance):
    return Program((FIXTURES / f"msc_d{distance}_inject_cultivate_p1e-3.stim").read_text())


def targeted_histories(program):
    yield "ideal", {}
    final = list(program.regions.values())[-1]
    last_noise = max(s.line for s in program.sites if not s.readout and s.line < final.start)
    logical = {
        k: "Z"
        for k, s in enumerate(program.sites)
        if s.line == last_noise and len(s.wires) == 1 and s.wires[0] in final.data
    }
    if len(logical) != len(final.data):
        raise ValueError("could not bind the complete legal logical-Z tail")
    yield "logical_z_tail", logical
    first = list(program.regions.values())[0]
    root = next(g for g in first.gates if g.name == "MX")
    reset = next(g for g in first.gates if g.name == "RX")
    for axis in "XYZ":
        erased = next(
            k
            for k, s in enumerate(program.sites)
            if root.line < s.line < reset.line and s.wires == root.targets and axis in s.choices
        )
        yield f"reset_erased_{axis}", {erased: axis}
    root_readout = program.site_at[root.line, 0]
    yield "root_readout_flip", {root_readout: "flip"}
    spectator = next(
        k
        for k, s in enumerate(program.sites)
        if first.start < s.line < first.end and len(s.wires) == 1 and s.wires[0] not in first.wires
    )
    yield "gadget_spectator_Y", {spectator: "Y"}
    pair = next(
        k
        for k, s in enumerate(program.sites)
        if first.start < s.line < first.end and len(s.wires) == 2
    )
    yield "gadget_pair_YZ", {pair: "YZ"}
    for record in sorted(set(program.controls.values())):
        event = program.measurements[program.records[record]]
        site = program.site_at[event.line, event.offset]
        yield f"feedforward_readout_{record}", {site: "flip"}


def clifft_reference(program, history, seed, executable, prefixes=False):
    with tempfile.TemporaryDirectory(prefix="msc-record-") as directory:
        path = Path(directory) / "physical.stim"
        path.write_text(materialize(program, history))
        args = [str(executable), str(path), str(seed)]
        if prefixes:
            args.append("--prefixes")
        return json.loads(subprocess.check_output(args, timeout=60))


def compare(program, history, expected):
    actual = evaluate(program, history, expected["outcomes"])
    classical = classical_reference(program, history, expected["outcomes"])
    for field in ("records", "detectors", "observables"):
        if actual[field] != classical[field]:
            raise AssertionError(f"Stim parity mismatch: {field}")
    result = {
        "faults": [[k, v] for k, v in sorted(history.items())],
        "outcomes": "".join(map(str, expected["outcomes"])),
        "records": "".join(map(str, actual["records"])),
        "detectors": "".join(map(str, actual["detectors"])),
        "observables": actual["observables"],
        "peak_terms": actual["peak_terms"],
        "gadget_exit_terms": actual["gadget_exit_terms"],
        "log_probability": math.log(actual["trajectory_probability"]),
    }
    if "probabilities" in expected:
        error = max(
            abs(a - b)
            for a, b in zip(actual["probabilities"], expected["probabilities"], strict=True)
        )
        if error > 3e-11:
            raise AssertionError(f"conditional probability mismatch: {error}")
        result["max_conditional_error"] = error
    if "log_probability" in expected:
        error = abs(result["log_probability"] - expected["log_probability"])
        if error > 3e-10:
            raise AssertionError(f"joint probability mismatch: {error}")
        result["log_probability_error"] = error
    for key in ("records", "detectors", "observables"):
        if key in expected and actual[key] != expected[key]:
            raise AssertionError(f"{key} mismatch")
    return result


def study(executable):
    circuits: list[dict] = []
    for distance in (3, 5):
        program = load(distance)
        cases = list(targeted_histories(program))
        cases += [(f"natural_{k}", program.history(7400 + k)) for k in range(24)]
        cases += [(f"stress_{k}", program.history(9600 + k, 10)) for k in range(12)]
        rows = []
        for k, (name, history) in enumerate(cases):
            seed = 25100 + k
            if distance == 3:
                oracle = reference(program, history, seed)
                row = compare(program, history, oracle)
                row["oracle"] = "Aer statevector"
            else:
                prefixes = name in {
                    "ideal",
                    "logical_z_tail",
                    "stress_0",
                } or name.startswith("feedforward_readout_")
                oracle = clifft_reference(program, history, seed, executable, prefixes)
                row = compare(program, history, oracle)
                row["oracle"] = "Clifft elementary gates"
                row["reference_active_width"] = oracle["peak_active_width"]
            row.update(name=name, seed=seed)
            if name == "logical_z_tail" and (
                "1" in row["detectors"] or row["observables"] != {"0": 1}
            ):
                raise AssertionError("legal logical error tail was not retained")
            rows.append(row)
            if k % 12 == 0:
                print(f"d{distance}: {k + 1}/{len(cases)} histories checked", flush=True)
        circuits.append(
            {
                "distance": distance,
                "physical_qubits": program.width,
                "physical_noise_sites_including_readout": len(program.sites),
                "record_count": len(program.records),
                "detector_count": len(program.detectors),
                "quantum_measurements_including_resets": len(program.measurements),
                "feedforward_operations": len(program.controls),
                "stim_parity_checks": len(rows),
                "cases": rows,
            }
        )
    return {"schema": 1, "circuits": circuits}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clifft-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(study(args.clifft_probe), indent=2) + "\n")
