"""Validate and profile the pinned, unverified SOFT physical T workload."""

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import stim
from cultivation_study import profile

import clifft

CORPUS = Path(__file__).parent / "fixtures/soft"


def source_text():
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    for entry in manifest["files"]:
        if hashlib.sha256((CORPUS / entry["file"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("pinned SOFT source checksum mismatch")
    return (CORPUS / manifest["files"][0]["file"]).read_text(), manifest


def zero_noise(text):
    # Preserve the physical gates and source lines, including indented repeats.
    return re.sub(
        r"(?m)^([ \t]*)(DEPOLARIZE[12]|[XYZ]_ERROR|M[XZY]?|MPP)\([^)]*\)",
        r"\1\2(0)",
        text,
    )


def clifford_proxy(text):
    return re.sub(r"(?m)^([ \t]*)T(_DAG)?(?=[ \t\r\n]|$)", r"\1S\2", text)


def fwht(values):
    result = values.copy()
    width = 1
    while width < len(result):
        blocks = result.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return result


def stabilizer_diagnostic(state, max_translations=32):
    """Numerical Pauli-stabilizer count, filtering X by magnitude symmetry.

    A Pauli stabilizer must preserve computational-basis probabilities under
    its XOR translation. Walsh autocorrelation filters those translations in
    O(k*2**k), before checking every Z phase for each remaining X. Large uniform
    states decline instead of scanning 4**k. This is a tolerance-based diagnostic,
    not an exact arithmetic certificate or a runtime recovery algorithm.
    """
    state = state / np.linalg.norm(state)
    width = len(state).bit_length() - 1
    probabilities = np.abs(state) ** 2
    correlation = fwht(fwht(probabilities) ** 2) / len(state)
    difference = 2 * (correlation[0] - correlation)
    tolerance = 1e-10 * correlation[0]
    candidates = np.flatnonzero(difference <= tolerance)
    row: dict[str, Any] = dict(
        width=width,
        magnitude_translation_candidates=len(candidates),
        magnitude_tolerance=float(tolerance),
        support_above_1e_minus_20=int((probabilities > 1e-20).sum()),
        numerical_nullity=None,
    )
    if len(candidates) > max_translations and width > 10:
        row["declined"] = "too many magnitude-preserving translations"
        return row
    indices = np.arange(len(state))
    count = 0
    for x in candidates:
        # The canonical Pauli's global i phase does not affect this magnitude.
        expectations = np.abs(fwht(np.conj(state) * state[indices ^ x]))
        count += int((np.abs(expectations - 1) < 1e-8).sum())
    if not count or count & (count - 1):
        raise AssertionError("numerical stabilizers do not form a power-of-two set")
    row["numerical_pauli_stabilizer_count"] = count
    row["numerical_nullity"] = width - (count.bit_length() - 1)
    return row


def snapshot(text, executable, seed):
    with tempfile.TemporaryDirectory() as directory:
        source, state = Path(directory) / "prefix.stim", Path(directory) / "state.bin"
        source.write_text(text)
        row = json.loads(
            subprocess.check_output([str(executable), str(source), str(state), str(seed)])
        )
        data = np.fromfile(state, dtype=np.float64).reshape(2, -1)
    row.pop("records")
    row.update(stabilizer_diagnostic(data[0] + 1j * data[1]))
    return row


def output_summary(result):
    passed = ~result.detectors.any(axis=1)
    return dict(
        all_zero_detectors=int(passed.sum()),
        observable_one=int(result.observables.sum()),
        observable_one_with_zero_detectors=int(result.observables[passed].sum()),
        fired_detector_counts={
            str(i): int(n) for i, n in enumerate(result.detectors.sum(axis=0)) if n
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.shots <= 0 or args.repeats <= 0:
        parser.error("shots and repeats must be positive")
    # Pin one available CPU; the selected affinity and ISA are part of the data.
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    source, manifest = source_text()
    zero = zero_noise(source)
    diagnostic = profile(source, args.profiler.resolve())
    report = dict(
        provenance=manifest,
        clifft_version=clifft.__version__,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        platform=platform.platform(),
        cpu=next(
            x.split(":", 1)[1].strip()
            for x in Path("/proc/cpuinfo").read_text().splitlines()
            if x.startswith("model name")
        ),
        cpu_affinity=cpu,
        isa=clifft.runtime_isa(),
        stim_version=stim.__version__,
        threads=1,
        batch_size=1,
        shots=args.shots,
        repeats=args.repeats,
        profile=diagnostic,
        benchmarks=[],
        snapshots=[],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    # The proxy is only an independent check of classical record declarations
    # and stochastic Clifford behavior. It is never used as a T-state oracle.
    proxy = stim.Circuit(clifford_proxy(zero))
    proxy.detector_error_model()
    converter = proxy.compile_m2d_converter(skip_reference_sample=True)
    reference_d, reference_o = converter.convert(
        measurements=proxy.reference_sample()[None, :], separate_observables=True
    )
    d, o = converter.convert(
        measurements=proxy.compile_sampler(seed=71).sample(4096), separate_observables=True
    )
    report["zero_s_proxy"] = dict(
        shots=4096, detector_ones=int(d.sum()), observable_ones=int(o.sum())
    )
    report["zero_s_proxy"]["raw_reference_one_detectors"] = np.flatnonzero(reference_d[0]).tolist()
    noisy_proxy = stim.Circuit(clifford_proxy(source))
    actual = clifft.sample(clifft.compile(str(noisy_proxy)), 32768, seed=190, threads=1)
    expected_d, expected_o = noisy_proxy.compile_detector_sampler(seed=811).sample(
        32768, separate_observables=True
    )
    expected_d ^= reference_d
    expected_o ^= reference_o
    actual_outputs = np.column_stack(
        [actual.detectors, actual.observables, ~actual.detectors.any(axis=1)]
    )
    expected_outputs = np.column_stack([expected_d, expected_o, ~expected_d.any(axis=1)])
    pooled = (actual_outputs.mean(axis=0) + expected_outputs.mean(axis=0)) / 2
    differences = np.abs(actual_outputs.mean(axis=0) - expected_outputs.mean(axis=0))
    tolerance = 7 * np.sqrt(2 * pooled * (1 - pooled) / 32768) + 2 / 32768
    if np.any(differences > tolerance):
        raise AssertionError("S proxy differs from Stim beyond smoke-test tolerance")
    report["noisy_s_proxy"] = dict(shots=32768, max_marginal_difference=float(differences.max()))
    for label, text in [("zero", zero), ("noisy", source)]:
        start = time.perf_counter()
        plan = clifft.compile(text)
        compile_seconds = time.perf_counter() - start
        report[label + "_plan"] = dict(
            compile_seconds=compile_seconds,
            qubits=plan.num_qubits,
            measurements=plan.num_measurements,
            detectors=plan.num_detectors,
            noise_sites=len(plan.noise_site_probabilities),
            peak_active_width=plan.peak_active_width,
            coefficient_bytes=16 * (1 << plan.peak_active_width),
            coefficient_and_measurement_scratch_bytes=24 * (1 << plan.peak_active_width),
        )
        for records in (True, False):

            def sample(shots, seed):
                if records:
                    return clifft.sample(plan, shots, seed=seed, threads=1, batch_size=1)
                return clifft.sample_survivors(
                    plan, shots, seed=seed, threads=1, batch_size=1, keep_records=False
                )

            sample(16, 11)
            elapsed, outputs = [], []
            for repeat in range(args.repeats):
                start = time.perf_counter()
                result = sample(args.shots, 71 + repeat)
                elapsed.append(time.perf_counter() - start)
                if records:
                    d, o = converter.convert(
                        measurements=result.measurements.astype(bool), separate_observables=True
                    )
                    np.testing.assert_array_equal(d, result.detectors)
                    np.testing.assert_array_equal(o, result.observables)
                    outputs.append(output_summary(result))
                else:
                    if result.discards or result.passed_shots != args.shots:
                        raise AssertionError("unpostselected aggregate call discarded attempts")
                    outputs.append(
                        dict(
                            total_shots=result.total_shots,
                            observable_one=int(result.observable_ones[0]),
                        )
                    )
            row = dict(
                noise=label,
                records=records,
                postselection=False,
                seconds=elapsed,
                attempts_per_second=args.shots / statistics.median(elapsed),
                outputs=outputs,
            )
            report["benchmarks"].append(row)
            save()
            print(label, "records", records, row["attempts_per_second"], flush=True)
    for label, text in [("zero", zero), ("noisy", source)]:
        for end in (400, 404, 471, 684, 686, 729, 777, 822):
            for seed in (71, 72):
                row = snapshot("\n".join(text.splitlines()[:end]), args.snapshot.resolve(), seed)
                row.update(noise=label, source_end_line=end, seed=seed)
                report["snapshots"].append(row)
                print("state", label, end, seed, row["width"], row["numerical_nullity"], flush=True)
                save()
    # Linux ru_maxrss is the process high-water mark, including Python and all
    # analysis arrays; it is intentionally not presented as worker memory.
    report["process_peak_rss_kib_including_analysis"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    save()


if __name__ == "__main__":
    main()
