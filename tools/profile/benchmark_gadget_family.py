"""Sweep complete geometric gadget attempts and validate generated histories."""

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
from benchmark_native_terminal import materialize, validate_shot
from clifford_gadget import atoms_from_text, compile_gadgets, evaluate
from compile_terminal_gadget import compile_bundle
from gadget_family import make_circuit
from soft_cultivation_study import clifford_proxy


def coherent_check(source, metadata, shot):
    """Alternate stabilizer-overlap evaluation; shares the gadget identity."""
    physical, _ = materialize(source, metadata, shot)
    width, _, _, atoms = atoms_from_text(physical)
    answer = evaluate(compile_gadgets(width, atoms), width, list(map(int, shot["records"])))
    # This family has no prefix measurements or resets, so its prefix norm is 1.
    error = abs(answer["log_probability"] - shot["conditional_log_probability"])
    if error > 1e-9:
        raise AssertionError("stabilizer-overlap reference disagrees with native sampling")
    converter = stim.Circuit(clifford_proxy(source)).compile_m2d_converter(
        skip_reference_sample=True
    )
    detectors, observables = converter.convert(
        measurements=np.array([list(map(int, shot["records"][: metadata["visible"]]))], bool),
        separate_observables=True,
    )
    for actual, expected in [
        (detectors[0], shot["detectors"]),
        (observables[0], shot["observables"]),
    ]:
        if "".join(str(int(b)) for b in actual) != expected:
            raise AssertionError("Stim output parity differs")
    return dict(absolute_log_error=error, peak_reference_terms=answer["peak_terms"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", type=Path, default=Path("build-study/sample_terminal_gadget"))
    parser.add_argument("--reference", type=Path, default=Path("build-study/replay_cultivation"))
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--traces", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots <= 0 or args.traces <= 0:
        parser.error("shots and traces must be positive")
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    report = dict(
        geometry_source="https://github.com/Strilanc/magic-state-cultivation/blob/"
        "871e68ff6df2f75190b1bfd6351459d1b5a037e3/src/cultiv/_construction/_color_code.py",
        construction="ideal logical input; geometric unflagged tree; raw Y/CSS output",
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
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        compiler=subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        generator_sha256=hashlib.sha256(
            Path(__file__).with_name("gadget_family.py").read_bytes()
        ).hexdigest(),
        cases=[],
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for distance in (3, 5, 7, 9):
            # Keep even the much faster small-circuit baseline batches long
            # enough to measure; the argument is a minimum batch size.
            shots = max(args.shots, {3: 65536, 5: 8192}.get(distance, 256))
            for probability in (0, 0.001, 0.01):
                source = make_circuit(distance, probability)
                bundle = directory / f"d{distance}-p{probability}"
                start = time.perf_counter()
                metadata = compile_bundle(source, bundle)
                preparation = time.perf_counter() - start
                output = subprocess.check_output(
                    [
                        str(args.sampler.resolve()),
                        str(bundle),
                        str(shots),
                        str(args.traces),
                        "812",
                        "24",
                    ],
                    text=True,
                )
                rows = [json.loads(line) for line in output.splitlines()]
                samples, benchmark = rows[:-1], rows[-1]
                method = (
                    "ordinary full physical replay"
                    if benchmark["baseline_enabled"]
                    else "coherent stabilizer overlaps"
                )
                for shot in samples:
                    if benchmark["baseline_enabled"]:
                        shot["validation"] = validate_shot(
                            source, metadata, shot, args.reference.resolve(), directory
                        )
                    else:
                        shot["validation"] = coherent_check(source, metadata, shot)
                plans = metadata["contractions"]
                compiled = statistics.median(benchmark["compiled_seconds"]) / shots
                ordinary = (
                    statistics.median(benchmark["ordinary_seconds"]) / shots
                    if benchmark["baseline_enabled"]
                    else None
                )
                report["cases"].append(
                    dict(
                        distance=distance,
                        data_qubits=len(metadata["data"]),
                        probability=probability,
                        circuit_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        python_compile_seconds=preparation,
                        bundle_bytes=sum(p.stat().st_size for p in bundle.iterdir()),
                        max_contraction_scope=max(p["peak_scope"] for p in plans),
                        contraction_numeric_payload_bytes=sum(
                            16 * (p["storage_complex_entries"] + (1 << p["peak_scope"]))
                            + 8
                            * (p["gather_entries"] + p["leaf_parity_entries"] + p["output_entries"])
                            for p in plans
                        ),
                        ordinary_coefficient_bytes=16 * (1 << benchmark["ordinary_peak_width"]),
                        benchmark=benchmark,
                        compiled_seconds_per_attempt=compiled,
                        ordinary_seconds_per_attempt=ordinary,
                        speedup=ordinary / compiled if ordinary is not None else None,
                        validation_method=method,
                        samples=samples,
                    )
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"d={distance} p={probability}: compiled={compiled:.6g}s "
                    f"ordinary={ordinary} scope={max(p['peak_scope'] for p in plans)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
