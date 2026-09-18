"""End-to-end validation and many-shot timing of synthetic five-check sequences."""

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from msc_factor_sequence import Sequence
from msc_protocol import Program, evaluate
from msc_reference import classical_reference, reference


def decoded(sample):
    faults = [list(map(int, row)) for row in sample["faults"]]
    flips = list(map(int, sample["flips"]))
    outcomes = list(map(int, sample["outcomes"]))
    return (faults, flips), outcomes


def validate_sample(sequence, sample, directory, probe=None, coherent=False):
    history, outcomes = decoded(sample)
    actual = sequence.run(history=history, outcomes=outcomes)
    native = np.array([complex(*z) for z in sample["logical"]])
    logical = actual["logical"]
    error = abs(sample["probability"] / actual["probability"] - 1)
    density_error = float(
        np.max(abs(np.outer(native, native.conj()) - np.outer(logical, logical.conj())))
    )
    if error > 1e-10 or density_error > 1e-10 or tuple(sample["frame"]) != actual["frame"]:
        raise AssertionError("native weight or logical handoff differs")
    for key in ("records", "detectors"):
        if list(map(int, sample[key])) != actual[key]:
            raise AssertionError("native reported output differs")
    if sample["observables"] != actual["observables"]:
        raise AssertionError("native logical observable differs")
    program, bound = sequence.program_history(history)
    parity = classical_reference(program, bound, outcomes)
    for key in ("records", "detectors"):
        if parity[key] != actual[key]:
            raise AssertionError("Stim parity differs")
    if list(parity["observables"].values()) != actual["observables"]:
        raise AssertionError("Stim observable differs")
    result = dict(probability_relative_error=error, logical_density_error=density_error)
    if coherent:
        expected = evaluate(program, bound, outcomes)
        result["coherent_relative_error"] = abs(
            sample["probability"] / expected["trajectory_probability"] - 1
        )
        if result["coherent_relative_error"] > 1e-9:
            raise AssertionError("full coherent-stabilizer trajectory differs")
    if probe:
        path = directory / "fixed.stim"
        path.write_text(sequence.circuit(history))
        replay = json.loads(
            subprocess.check_output(
                [str(probe), str(path), "0", "--replay", sample["outcomes"]], text=True
            )
        )
        result["clifft_log_probability_error"] = abs(
            replay["log_probability"] - np.log(sample["probability"])
        )
        if result["clifft_log_probability_error"] > 1e-9:
            raise AssertionError("original elementary Clifft trajectory differs")
    return result


def study(native, baseline, probe, output, shots):
    cpu = min(os.sched_getaffinity(0))
    circuits: list[dict] = []
    data = dict(
        scope="synthetic noisy five-check CSS sequences; not authors MSC7",
        cpu=cpu,
        circuits=circuits,
    )
    with tempfile.TemporaryDirectory(prefix="msc-sequence-study-") as folder:
        directory = Path(folder)
        for distance in (3, 5, 7, 9):
            sequence = Sequence(distance)
            plan, circuit = directory / "plan.txt", directory / "circuit.stim"
            sequence.export(plan)
            circuit.write_text(sequence.circuit())
            stats = json.loads(
                subprocess.check_output([str(baseline), str(circuit), "0", "plan"], text=True)
            )
            case = dict(
                distance_label=distance,
                data_width=sequence.width,
                rounds=5,
                noise=sequence.noise,
                noise_sites=20 * sequence.width + sequence.record_count,
                records=sequence.record_count,
                detectors=4 * (sequence.stride - 1),
                non_clifford_gates=10 * sequence.width + 1,
                plan_bytes=plan.stat().st_size,
                clifft_plan=stats,
                validation=[],
                timings=[],
            )
            print("validating", distance, flush=True)
            for noise in (0.0, 0.001, 0.02):
                for axis in "XYZ":
                    trial = Sequence(distance, noise, axis)
                    trial.export(plan)
                    samples = subprocess.check_output(
                        [str(native), str(plan), "sample", "2", "9713"], text=True
                    )
                    for k, line in enumerate(samples.splitlines()):
                        sample = json.loads(line)
                        checks = validate_sample(
                            trial,
                            sample,
                            directory,
                            probe if distance <= 7 else None,
                            coherent=(noise == 0.02 and k == 0),
                        )
                        case["validation"].append(
                            dict(noise=noise, axis=axis, sample=sample, **checks)
                        )
            if distance == 3:
                case["aer_validation"] = []
                for axis in "XYZ":
                    trial = Sequence(distance, 0.02, axis)
                    for seed in (513, 812):
                        history = trial.history(seed)
                        aer = reference(Program(trial.circuit(history)), {}, seed)
                        result = trial.run(history=history, outcomes=aer["outcomes"])
                        error = abs(
                            result["probability"] / float(np.prod(aer["probabilities"])) - 1
                        )
                        if error > 1e-9:
                            raise AssertionError("original elementary Aer trajectory differs")
                        case["aer_validation"].append(
                            dict(axis=axis, seed=seed, relative_error=error)
                        )
            sequence.export(plan)
            native_shots = shots
            baseline_shots = shots if distance < 7 else max(1000, shots // 10)
            for repeat in range(3):
                print("timing", distance, repeat, flush=True)
                commands = [
                    (
                        "factor",
                        [str(native), str(plan), "bench", str(native_shots), str(773 + repeat)],
                    )
                ]
                if stats["peak_active_width"] <= 20:
                    commands.append(
                        (
                            "clifft",
                            [str(baseline), str(circuit), str(773 + repeat), str(baseline_shots)],
                        )
                    )
                for kind, command in commands:
                    timing = json.loads(
                        subprocess.check_output(["taskset", "-c", str(cpu), *command], text=True)
                    )
                    case["timings"].append(dict(kind=kind, repeat=repeat, **timing))
            case["median_microseconds"] = {
                kind: statistics.median(
                    t["microseconds_per_shot"] for t in case["timings"] if t["kind"] == kind
                )
                for kind in {t["kind"] for t in case["timings"]}
            }
            circuits.append(case)
            output.write_text(json.dumps(data, indent=2) + "\n")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.native, args.baseline, args.probe, args.output, args.shots)
