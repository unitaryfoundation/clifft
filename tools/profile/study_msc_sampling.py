"""Validate complete native MSC samples and time the original noisy circuits."""

import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from export_msc_sampler import export
from msc_protocol import evaluate
from msc_reference import classical_reference, materialize
from msc_sampling import Sampler
from study_msc_growth_contraction import logical_bloch
from study_msc_protocol import FIXTURES, load


def validate(plan, sample, probe=None):
    history = {k: plan.program.sites[k].choices[c] for k, c in sample["fault_choices"]}
    outcomes = sample["outcomes"]
    result = {}

    def observe(op, state, norm):
        if op.line == plan.terminal.measurement.line + 1:
            result["bloch"] = logical_bloch(state, plan.terminal.data, norm)

    actual = evaluate(plan.program, history, outcomes, before_instruction=observe)
    probability_error = abs(sample["probability"] / actual["trajectory_probability"] - 1)
    logical = np.asarray([complex(*z) for z in sample["logical"]])
    x, y, z = result.pop("bloch")
    density = np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]) / 2
    density_error = float(np.max(np.abs(np.outer(logical, logical.conj()) - density)))
    outputs = classical_reference(plan.program, history, outcomes)
    if (
        max(probability_error, density_error) > 1e-10
        or sample["records"] != outputs["records"]
        or sample["detectors"] != outputs["detectors"]
        or sample["observables"] != list(outputs["observables"].values())
    ):
        raise AssertionError((probability_error, density_error, outputs, sample))
    if probe:
        with tempfile.TemporaryDirectory(prefix="msc-native-replay-") as directory:
            path = Path(directory) / "physical.stim"
            path.write_text(materialize(plan.program, history))
            replay = json.loads(
                subprocess.check_output(
                    [str(probe), str(path), "0", "--replay", "".join(map(str, outcomes))],
                    text=True,
                    timeout=60,
                )
            )
            error = abs(math.expm1(math.log(sample["probability"]) - replay["log_probability"]))
            if error > 1e-10:
                raise AssertionError(("original-gate replay differs", error))
            result.update(
                elementary_probability_relative_error=error,
                elementary_peak_active_width=replay["peak_active_width"],
            )
    return {
        **sample,
        **result,
        "probability_relative_error": probability_error,
        "normalized_density_error": density_error,
    }


def study(native, baseline, probe, shots, repetitions, cases):
    result = []
    cpu = min(os.sched_getaffinity(0))
    for distance in (3, 5):
        plan = Sampler(load(distance))
        samples: list[dict] = []
        timings: list[dict] = []
        circuit = {"distance": distance, "cpu": cpu, "samples": samples, "timings": timings}
        with tempfile.TemporaryDirectory(prefix="msc-native-study-") as directory:
            path = Path(directory) / "plan.txt"
            original_sites = plan.program.sites
            for scale in (1, 20):
                plan.program.sites = [
                    replace(s, probability=min(1, scale * s.probability)) for s in original_sites
                ]
                export(plan, path)
                native_samples = subprocess.check_output(
                    [str(native), str(path), str(951 + scale), str(cases), "sample"], text=True
                )
                for k, line in enumerate(native_samples.splitlines()):
                    sample = validate(plan, json.loads(line), probe if k < 4 else None)
                    samples.append(
                        {
                            "noise_scale": scale,
                            **sample,
                            **{
                                key: "".join(map(str, sample[key]))
                                for key in ("outcomes", "records", "detectors")
                            },
                        }
                    )
                print(distance, "scale", scale, cases, "native samples passed", flush=True)
            plan.program.sites = original_sites
            export(plan, path)
            circuit["serialized_plan_bytes"] = path.stat().st_size
            for repeat in range(repetitions):
                commands = {
                    "blocks": [str(native), str(path), str(800 + repeat), str(shots), "benchmark"],
                    "clifft": [
                        str(baseline),
                        str(FIXTURES / f"msc_d{distance}_inject_cultivate_p1e-3.stim"),
                        str(800 + repeat),
                        str(shots),
                    ],
                }
                timing = {}
                for name, command in commands.items():
                    # Run contenders sequentially on the same available CPU.
                    timing[name] = json.loads(
                        subprocess.check_output(["taskset", "-c", str(cpu), *command], text=True)
                    )
                timings.append(timing)
                print(distance, "timing", repeat, timing, flush=True)
        result.append(circuit)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            study(args.native, args.baseline, args.probe, args.shots, args.repetitions, args.cases),
            indent=2,
        )
        + "\n"
    )
