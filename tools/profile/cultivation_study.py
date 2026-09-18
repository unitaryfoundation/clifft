"""Validate and measure pinned physical cultivation circuits on ordinary Clifft."""

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import stim

import clifft

CORPUS = Path(__file__).parent / "fixtures/cliffordea"


def corpus():
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    for entry in manifest["files"]:
        data = (CORPUS / entry["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"source checksum mismatch: {entry['file']}")
    return manifest


def variant_text(source, probability, variant="T"):
    # Match the author's make_variant_text transformation on these pinned files.
    text = source.replace("0.001", str(probability))
    if variant == "T":
        text = re.sub(r"(?m)^([ \t]*)S(_DAG)?(?=[ \t\r\n]|$)", r"\1T\2", text)
    elif variant != "S":
        raise ValueError("unknown state variant")
    return text


def validate(source, shots):
    zero = clifft.compile(variant_text(source, 0))
    result = clifft.sample(zero, 256, seed=72, threads=1)
    if result.detectors.any() or result.observables.any():
        raise AssertionError("noiseless T cultivation is not deterministic")
    proxy = stim.Circuit(source)
    program = clifft.compile(source)
    observed = clifft.sample(program, shots, seed=190, threads=1)
    converter = proxy.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=observed.measurements.astype(bool), separate_observables=True
    )
    np.testing.assert_array_equal(detectors, observed.detectors)
    np.testing.assert_array_equal(observables, observed.observables)
    expected = proxy.compile_detector_sampler(seed=811).sample(shots, append_observables=True)
    actual = np.concatenate([detectors, observables], axis=1)
    # Test the acceptance event as well as individual output marginals.
    actual = np.column_stack([actual, ~detectors.any(axis=1)])
    expected = np.column_stack([expected, ~expected[:, : proxy.num_detectors].any(axis=1)])
    pooled = (actual.mean(axis=0) + expected.mean(axis=0)) / 2
    tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / shots) + 2 / shots
    differences = np.abs(actual.mean(axis=0) - expected.mean(axis=0))
    if np.any(differences > tolerance):
        raise AssertionError("S proxy disagrees with independent Stim sampling")
    return dict(
        noiseless_t_shots=256,
        stim_shots=shots,
        stim_max_output_marginal_difference=float(differences.max()),
        stim_parities_match=True,
    )


def profile(text, executable):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "circuit.stim"
        path.write_text(text)
        output = subprocess.check_output([str(executable), str(path)], text=True)
    rows: list[dict[str, Any]] = list(csv.DictReader(io.StringIO(output), delimiter="\t"))
    for row in rows:
        for key in ("action", "before", "after", "dense_passes", "detector", "noise_only"):
            row[key] = int(row[key])
        row["lines"] = [int(x) for x in row["lines"].split(",") if x]
        # Native TSV remains available when full expression diagnostics are needed.
        row.pop("description")
    by_width: Counter[int] = Counter()
    by_line: dict[int, float] = defaultdict(float)
    for row in rows:
        visits = row["dense_passes"] * (1 << row["before"])
        by_width[row["before"]] += visits
        for line in row["lines"]:
            by_line[line] += visits / len(row["lines"])
    source = text.splitlines()
    last_noise = max(
        (
            i
            for i, line in enumerate(source, 1)
            if re.match(r"(?:DEPOLARIZE[12]|[XYZ]_ERROR|M[XYZ]?|MPP)\(", line.strip())
        ),
        default=0,
    )
    return dict(
        # This is static unfused work, not measured CPU time or surviving-shot cost.
        predicted_coefficient_visits_by_width=dict(sorted(by_width.items())),
        expensive_source_lines=[
            dict(line=line, predicted_visits=visits, source=source[line - 1])
            for line, visits in sorted(by_line.items(), key=lambda item: -item[1])[:12]
        ],
        state_independent_detector_count=sum(row["noise_only"] == 1 for row in rows),
        detector_count=sum(row["detector"] >= 0 for row in rows),
        last_noise_source_line=last_noise,
        predicted_visits_after_last_noise=sum(
            v for line, v in by_line.items() if line > last_noise
        ),
        actions=rows,
    )


def benchmark(text, shots, repeats, diagnostic, fault_count=None):
    rows = []
    reference = clifft.compile(text)
    detectors = [a for a in diagnostic["actions"] if a["detector"] >= 0]
    detector_order = [a["detector"] for a in detectors]
    screen_indices = [a["detector"] for a in detectors if a["noise_only"] == 1]
    accumulated = 0
    work_at_detector = {}
    for action in diagnostic["actions"]:
        accumulated += action["dense_passes"] * (1 << action["before"])
        if action["detector"] >= 0:
            work_at_detector[action["detector"]] = accumulated
    for early in (False, True):
        start = time.perf_counter()
        plan = clifft.compile(
            text, postselection_mask=[1] * reference.num_detectors if early else None
        )
        compile_seconds = time.perf_counter() - start

        def sample(count, seed):
            function: Any
            kwargs = dict(seed=seed, threads=1, batch_size="auto")
            if early:
                kwargs["keep_records"] = True
            if fault_count is not None:
                kwargs["k"] = fault_count
                function = clifft.sample_k_survivors if early else clifft.sample_k
            else:
                function = clifft.sample_survivors if early else clifft.sample
            return function(plan, count, **kwargs)

        sample(64, 71)
        elapsed, passed, errors = [], [], []
        first_rejections: Counter[int] = Counter()
        screened = 0
        early_work = screened_work = 0
        for repeat in range(repeats):
            start = time.perf_counter()
            result = sample(shots, 170 + repeat)
            elapsed.append(time.perf_counter() - start)
            accepted = ~result.detectors.any(axis=1)
            passed.append(int(accepted.sum()))
            errors.append(int(result.observables[accepted].any(axis=1).sum()))
            if not early:
                screen = result.detectors[:, screen_indices].any(axis=1)
                screened += int(screen.sum())
                first = np.asarray(detector_order)[
                    result.detectors[:, detector_order].argmax(axis=1)
                ]
                first_rejections.update(first[~accepted].tolist())
                work = np.asarray([work_at_detector[int(d)] for d in first])
                work[accepted] = accumulated
                early_work += int(work.sum())
                screened_work += int(work[~screen].sum())
        median = statistics.median(elapsed)
        rows.append(
            dict(
                early_rejection=early,
                fault_count=fault_count,
                compile_seconds=compile_seconds,
                sample_seconds=elapsed,
                passed_shots=passed,
                logical_errors=errors,
                first_rejected_detector=dict(sorted(first_rejections.items())),
                potential_screen_rejected=screened if not early else None,
                predicted_early_coefficient_visits=early_work if not early else None,
                predicted_screen_then_early_coefficient_visits=(
                    screened_work if not early else None
                ),
                median_seconds_per_attempt=median / shots,
                attempts_per_second=shots / median,
                accepted_per_second=sum(passed) / sum(elapsed),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.shots <= 0 or args.repeats <= 0:
        parser.error("shots and repeats must be positive")
    manifest = corpus()
    result = dict(
        clifft_version=clifft.__version__,
        clifft_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        stim_version=stim.__version__,
        numpy_version=np.__version__,
        platform=platform.platform(),
        cpu=next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        runtime_isa=clifft.runtime_isa(),
        threads=1,
        batch_size="auto",
        shots=args.shots,
        repeats=args.repeats,
        provenance=manifest,
        circuits=[],
    )
    for entry in manifest["files"]:
        if not entry["file"].endswith(".stim"):
            continue
        source = (CORPUS / entry["file"]).read_text()
        text = variant_text(source, 0.001)
        plan = clifft.compile(text)
        diagnostic = profile(text, args.profiler.resolve())
        if (
            max(max(a["before"], a["after"]) for a in diagnostic["actions"])
            != plan.peak_active_width
        ):
            raise AssertionError("profiler and Python program disagree on peak width")
        row = dict(
            file=entry["file"],
            t_sha256=hashlib.sha256(text.encode()).hexdigest(),
            qubits=plan.num_qubits,
            measurements=plan.num_measurements,
            detectors=plan.num_detectors,
            noise_sites=len(plan.noise_site_probabilities),
            peak_active_width=plan.peak_active_width,
            coefficient_bytes=16 * (1 << plan.peak_active_width),
            validation=validate(source, max(args.shots, 32768)),
            profile=diagnostic,
            benchmarks=[],
        )
        for probability, fault_count in [(0, None), (0.001, None), (0.001, 3), (0.001, 5)]:
            for measurement in benchmark(
                variant_text(source, probability), args.shots, args.repeats, diagnostic, fault_count
            ):
                measurement["physical_noise"] = probability
                row["benchmarks"].append(measurement)
        result["circuits"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(entry["file"], "peak", plan.peak_active_width, "validated and measured", flush=True)


if __name__ == "__main__":
    main()
