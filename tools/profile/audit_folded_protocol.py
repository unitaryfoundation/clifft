"""Replay complete native folded-MSC attempts with their freshly sampled faults."""

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from compile_folded_protocol import compile_protocol
from folded_msc_family import make_circuit
from folded_msc_oracle import evaluate
from validate_folded_msc import aer_trajectory, bind_faults, native_replay


def validate_trace(circuit, metadata, sample, reference, *, oracle=True, aer_seed=None):
    faults = {}
    for key, sites in (
        ("prefix_faults", metadata["prefix_sites"]),
        ("suffix_faults", metadata["sites"]),
        ("earlier_faults", metadata.get("earlier_sites", [])),
        ("growth_faults", metadata.get("growth_sites", [])),
    ):
        for site_index, channel in sample.get(key, []):
            site = sites[site_index]
            faults[site["operation"]] = site["labels"][channel - 1]
    physical = bind_faults(circuit, faults)
    records = sample["records"]
    distance = metadata["distance"]
    coherent = evaluate(physical, records) if distance == 7 else None
    full = (
        dict(
            reachable=coherent["log_probability"] is not None,
            log_probability=coherent["log_probability"],
        )
        if coherent is not None
        else native_replay(physical, records, reference)
    )
    start = metadata.get(
        "prefix_stage", f"d{distance}_logical_check_1" if distance > 3 else "logical_check_1"
    )
    stop = next(i for i, op in enumerate(physical.operations) if op.stage == start)
    visible, hidden = metadata["prefix_visible"], metadata["prefix_hidden"]
    prefix = replace(
        physical,
        operations=physical.operations[:stop],
        measurements=physical.measurements[:visible],
    )
    prefix_records = records[:visible] + records[metadata["visible"] : metadata["visible"] + hidden]
    prior = native_replay(prefix, prefix_records, reference)
    if not full["reachable"] or not prior["reachable"]:
        raise AssertionError(f"unreachable native history with faults {faults}: {full}")
    error = abs(
        full["log_probability"] - prior["log_probability"] - sample["conditional_log_probability"]
    )
    if error > 1e-8:
        raise AssertionError(
            f"complete-attempt log probability differs by {error} with faults {faults}"
        )
    partners = metadata.get(
        "detector_partners", [metadata["visible"]] * len(metadata["postselect"])
    )
    detectors = "".join(
        str(int(records[i]) ^ (int(records[j]) if j < metadata["visible"] else 0))
        for i, j in zip(metadata["postselect"], partners, strict=True)
    )
    if sample["detectors"] != detectors or sample["observable"] != int(
        records[metadata["logical_slot"]]
    ):
        raise AssertionError("native output bits disagree with physical records")
    oracle_error = None
    if oracle or distance == 7:
        coherent = evaluate(physical, records) if coherent is None else coherent
        if coherent["log_probability"] is None:
            raise AssertionError(
                "independent coherent-Clifford oracle found an unreachable history"
            )
        oracle_error = (
            error if distance == 7 else abs(coherent["log_probability"] - full["log_probability"])
        )
        if oracle_error > 1e-8:
            raise AssertionError(f"independent oracle probability differs by {oracle_error}")
    aer_error = None
    if aer_seed is not None:
        aer_records, aer_probability = aer_trajectory(physical, aer_seed)
        coherent = evaluate(physical, aer_records)
        if coherent["log_probability"] is None:
            raise AssertionError("independent oracle rejected a dense-Aer trajectory")
        aer_error = abs(coherent["log_probability"] - aer_probability)
        if aer_error > 1e-8:
            raise AssertionError(f"dense Aer and independent oracle differ by {aer_error}")
    return dict(
        faults=len(faults),
        log_probability_error=error,
        oracle_error=oracle_error,
        full_reference="coherent-Clifford" if distance == 7 else "physical-Clifft",
        aer_reference_error=aer_error,
        accepted="1" not in detectors,
        observable=sample["observable"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", type=Path, default=Path("build-study/sample_folded_protocol"))
    parser.add_argument("--reference", type=Path, default=Path("build-study/replay_cultivation"))
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0, 0.001, 0.03])
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--oracle-cases", type=int, default=4)
    parser.add_argument("--schedule", choices=["off", "budgeted", "unbounded"], default="off")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-growth", action="store_true")
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    results = []
    with tempfile.TemporaryDirectory(prefix="folded-protocol-audit-") as tmp:
        for distance in args.distances:
            for probability in args.probabilities:
                circuit = make_circuit(probability, distance=distance)
                bundle = Path(tmp) / f"f{distance}-p{probability}"
                metadata = compile_protocol(
                    circuit, distance, bundle, native_growth=args.native_growth
                )
                output = subprocess.check_output(
                    [
                        str(args.sampler.resolve()),
                        str(bundle),
                        "1",
                        str(args.cases),
                        "9134",
                        "0",
                        args.schedule,
                        "0",
                    ],
                    text=True,
                )
                rows = [json.loads(line) for line in output.splitlines()]
                validations = [
                    validate_trace(
                        circuit,
                        metadata,
                        sample,
                        args.reference,
                        oracle=i < args.oracle_cases,
                        aer_seed=500 + i if distance == 3 and i < args.oracle_cases else None,
                    )
                    for i, sample in enumerate(rows[:-1])
                ]
                if probability == 0 and any(
                    not row["accepted"] or row["observable"] for row in validations
                ):
                    raise AssertionError(
                        "noiseless complete attempt was rejected or logically wrong"
                    )
                row = dict(
                    distance=distance,
                    native_growth=args.native_growth,
                    probability=probability,
                    circuit_sha256=metadata["circuit_sha256"],
                    schedule=args.schedule,
                    checks=validations,
                    samples=rows[:-1],
                    benchmark=rows[-1],
                )
                results.append(row)
                print(
                    f"f{distance} p={probability}: {len(validations)} complete histories validated",
                    flush=True,
                )
                args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
