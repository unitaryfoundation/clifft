"""Benchmark fresh complete attempts and replay their physical fault histories."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import stim
from clifford_gadget import native_replay_text
from compile_terminal_gadget import compile_bundle, physical_text
from soft_cultivation_study import clifford_proxy, source_text, zero_noise


def materialize(source, metadata, shot):
    prefix_choices = dict(shot["prefix_faults"])
    suffix_choices = dict(shot["suffix_faults"])
    selected = {}
    readout = set(shot["prefix_readout"])
    for specs, choices in [
        (metadata["prefix_noise_sites"], prefix_choices),
        (metadata["suffix_noise_sites"], suffix_choices),
    ]:
        for index, spec in enumerate(specs):
            choice = choices.get(index, 0)
            if "record" in spec:
                if choice:
                    readout.add(spec["record"])
            elif choice:
                selected[spec["op"], spec["group"]] = spec["labels"][choice - 1]
    full, prefix = stim.Circuit(), stim.Circuit()
    record = 0
    for i, op in enumerate(stim.Circuit(clifford_proxy(source)).flattened()):
        output = stim.Circuit()
        name, targets = op.name, op.targets_copy()
        if name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
            size = 2 if name == "DEPOLARIZE2" else 1
            for start in range(0, len(targets), size):
                label = selected.get((i, start // size), "I" * size)
                for target, axis in zip(targets[start : start + size], label, strict=True):
                    if axis != "I":
                        output.append(axis, [target])
        elif name in {"MX", "M", "MPP"}:
            groups = op.target_groups() if name == "MPP" else [[t] for t in targets]
            for group in groups:
                if name == "MPP":
                    expanded: list[stim.GateTarget] = []
                    for target in group:
                        if expanded:
                            expanded.append(stim.target_combiner())
                        expanded.append(target)
                    group = expanded
                output.append(name, group, int(record in readout))
                record += 1
        else:
            output.append(op)
        full += output
        if i < metadata["begin"]:
            prefix += output
    return physical_text(full), physical_text(prefix)


def validate_shot(source, metadata, shot, native, directory):
    full, prefix = materialize(source, metadata, shot)
    records = list(map(int, shot["records"]))
    prefix_records = (
        records[: metadata["prefix_visible"]]
        + records[metadata["visible"] : metadata["visible"] + metadata["prefix_hidden"]]
    )
    logs = []
    for name, text, forced in [("full", full, records), ("prefix", prefix, prefix_records)]:
        path, record_path = directory / f"{name}.stim", directory / f"{name}.records"
        path.write_text(native_replay_text(text))
        record_path.write_text("".join(map(str, forced)))
        result = json.loads(
            subprocess.check_output([str(native), str(path), "0", str(record_path)], text=True)
        )
        if not result["reachable"]:
            raise AssertionError(f"native {name} replay declined the generated physical history")
        logs.append(result["log_probability"])
    error = abs(logs[0] - logs[1] - shot["conditional_log_probability"])
    if error > 1e-9:
        raise AssertionError(f"native complete-history log probability differs by {error}")
    converter = stim.Circuit(clifford_proxy(source)).compile_m2d_converter(
        skip_reference_sample=True
    )
    detectors, observables = converter.convert(
        measurements=np.array([records[: metadata["visible"]]], dtype=np.bool_),
        separate_observables=True,
    )
    if "".join(map(lambda b: str(int(b)), detectors[0])) != shot["detectors"]:
        raise AssertionError("raw detector records differ")
    if "".join(map(lambda b: str(int(b)), observables[0])) != shot["observables"]:
        raise AssertionError("raw observable records differ")
    return dict(
        absolute_log_error=error, full_log_probability=logs[0], prefix_log_probability=logs[1]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", type=Path, default=Path("build-study/sample_terminal_gadget"))
    parser.add_argument("--reference", type=Path, default=Path("build-study/replay_cultivation"))
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--traces", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    source, manifest = source_text()
    report = dict(
        provenance=manifest,
        python=platform.python_version(),
        stim=stim.__version__,
        affinity=cpu,
        threads=1,
        cpu=next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        native_binary_sha256=hashlib.sha256(args.sampler.read_bytes()).hexdigest(),
        reference_binary_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        cases=[],
    )
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for noise, text in [
            ("source", source),
            ("zero", zero_noise(source)),
            ("stress", source.replace("(0.0005)", "(0.005)")),
        ]:
            bundle = directory / noise
            start = time.perf_counter()
            metadata = compile_bundle(text, bundle)
            python_compile_seconds = time.perf_counter() - start
            output = subprocess.check_output(
                [str(args.sampler.resolve()), str(bundle), str(args.shots), str(args.traces), "71"],
                text=True,
            )
            rows = [json.loads(line) for line in output.splitlines()]
            samples = [row for row in rows if row["kind"] == "sample"]
            benchmark = rows[-1]
            for shot in samples:
                shot["validation"] = validate_shot(
                    text, metadata, shot, args.reference.resolve(), directory
                )
            compiled = statistics.median(benchmark["compiled_seconds"]) / args.shots
            ordinary = statistics.median(benchmark["ordinary_seconds"]) / args.shots
            report["cases"].append(
                dict(
                    noise=noise,
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                    python_compile_seconds=python_compile_seconds,
                    benchmark=benchmark,
                    compiled_seconds_per_shot=compiled,
                    ordinary_seconds_per_shot=ordinary,
                    speedup=ordinary / compiled,
                    samples=samples,
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                noise,
                "compiled",
                compiled,
                "ordinary",
                ordinary,
                "speedup",
                ordinary / compiled,
                flush=True,
            )


if __name__ == "__main__":
    main()
