"""Replay independently checked physical histories with the scheduling pass."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from folded_msc_family import make_circuit
from validate_folded_msc import bind_faults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent / "research"
    small = json.loads((root / "folded_msc_validation.json").read_text())
    large = json.loads((root / "folded_msc_f5_validation.json").read_text())
    report: dict = dict(
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("folded_msc_validation.json", "folded_msc_f5_validation.json")
        },
        cases=[],
    )
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "physical.stim"
        records_path = Path(directory) / "records.txt"
        for distance, cases in ((3, small["trajectories"]), (5, large["f5_histories"])):
            for index, case in enumerate(cases):
                circuit = make_circuit(
                    0.02 if distance == 3 else case["probability"], distance=distance
                )
                faults = {int(i): label for i, label in case["faults"].items()}
                physical = bind_faults(circuit, faults)
                records = case["records"] if distance == 3 else case["native"]["records"]
                expected = (
                    case["aer_log_probability"]
                    if distance == 3
                    else case["oracle"]["log_probability"]
                )
                source_path.write_text(physical.text())
                records_path.write_text(records)
                for mode in ("off", "budgeted", "unbounded"):
                    result = json.loads(
                        subprocess.check_output(
                            [
                                str(args.reference.resolve()),
                                str(source_path),
                                "0",
                                str(records_path),
                                mode,
                            ],
                            text=True,
                        )
                    )
                    error = abs(result["log_probability"] - expected)
                    if not result["reachable"] or error > 1e-10:
                        raise AssertionError(
                            f"scheduled physical replay mismatch: {distance} {index} {mode}"
                        )
                    report["cases"].append(
                        dict(
                            distance=distance,
                            case=index,
                            mode=mode,
                            faults=len(faults),
                            absolute_log_error=error,
                            peak_active_width=result["peak_active_width"],
                        )
                    )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"f{distance} history {index}: all modes agree", flush=True)


if __name__ == "__main__":
    main()
