"""Measure reconstructed fold-transversal protocols using ordinary Clifft."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from cultivation_study import profile
from folded_msc_family import make_circuit

import clifft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5), default=3)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--fault-shots", type=int)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0, 0.001, 0.003, 0.01])
    parser.add_argument("--profiler", type=Path, default=Path("build-study/profile_cultivation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots is None:
        args.shots = 262144 if args.distance == 3 else 8
    if args.fault_shots is None:
        args.fault_shots = 1000000 if args.distance == 3 else 0
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    source = make_circuit(0.001, distance=args.distance)
    diagnostic = profile(source.text(), args.profiler.resolve())
    report = dict(
        distance=args.distance,
        source_sha256=source.manifest()["sha256"],
        source_files_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("folded_msc.py", "folded_msc_family.py", "benchmark_folded_msc.py")
        },
        profiler_sha256=hashlib.sha256(args.profiler.read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree="uncommitted research generator; source hashes recorded",
        python=platform.python_version(),
        affinity=cpu,
        threads=1,
        cpu=next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        peak_active_width=max(max(row["before"], row["after"]) for row in diagnostic["actions"]),
        expensive_source_lines=diagnostic["expensive_source_lines"],
        cases=[],
        fixed_fault_cases=[],
    )
    for probability in args.probabilities:
        circuit = make_circuit(probability, distance=args.distance)
        for early in (False, True):
            start = time.perf_counter()
            mask = [1] * circuit.manifest()["num_detectors"] if early else None
            plan = clifft.compile(circuit.text(), postselection_mask=mask)
            compile_seconds = time.perf_counter() - start

            def sample(shots, seed):
                if early:
                    return clifft.sample_survivors(
                        plan, shots, seed=seed, threads=1, keep_records=True
                    )
                return clifft.sample(plan, shots, seed=seed, threads=1)

            sample(64 if args.distance == 3 else 1, 1)
            batches = []
            for repeat in range(3):
                start = time.perf_counter()
                result = sample(args.shots, 471 + repeat)
                elapsed = time.perf_counter() - start
                accepted = ~result.detectors.any(axis=1)
                batches.append(
                    dict(
                        seconds=elapsed,
                        accepted=int(accepted.sum()),
                        logical_failures=int(result.observables[accepted].sum()),
                    )
                )
            case = dict(
                probability=probability,
                early_rejection=early,
                shots_per_batch=args.shots,
                compile_seconds=compile_seconds,
                batches=batches,
                median_seconds_per_attempt=statistics.median(b["seconds"] for b in batches)
                / args.shots,
            )
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    plan = clifft.compile(
        source.text(), postselection_mask=[1] * source.manifest()["num_detectors"]
    )
    for k in (1, 2, 3) if args.fault_shots else ():
        start = time.perf_counter()
        result = clifft.sample_k_survivors(
            plan, args.fault_shots, k=k, seed=906 + k, threads=1, keep_records=True
        )
        case = dict(
            k=k,
            shots=args.fault_shots,
            seconds=time.perf_counter() - start,
            accepted=len(result.measurements),
            logical_failures=int(result.observables.sum()),
        )
        report["fixed_fault_cases"].append(case)
        print(json.dumps(case), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
