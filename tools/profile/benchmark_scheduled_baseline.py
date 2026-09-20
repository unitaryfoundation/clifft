"""Compare default and explicitly scheduled Clifft on the research circuits."""

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

from folded_msc_family import make_circuit as folded_circuit

import clifft

MODES = ("off", "budgeted", "unbounded")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scheduled_compile(source, mode, mask=None):
    passes = clifft.default_hir_pass_manager()
    schedule = None
    if mode != "off":
        schedule = getattr(clifft, "ActiveWidthSchedulePass")(
            search_budget=None if mode == "unbounded" else 16.0
        )
        passes.add(schedule)
    start = time.perf_counter()
    plan = clifft.compile(source, hir_passes=passes, postselection_mask=mask)
    elapsed = time.perf_counter() - start
    stats = (
        {
            name: getattr(schedule, name)
            for name in (
                "incumbent_peak",
                "result_peak",
                "incumbent_dense_work",
                "result_dense_work",
                "applied",
                "swept_ops",
            )
        }
        if schedule is not None
        else None
    )
    return plan, elapsed, stats


def folded_cases(args):
    for distance in args.distances or (3, 5):
        shots = 262144 if distance == 3 else 8
        probabilities = (0, 0.001, 0.003, 0.01) if distance == 3 else (0, 0.001)
        for probability in probabilities:
            circuit = folded_circuit(probability, distance=distance)
            source = circuit.text()
            for early in (False, True):
                mask = [1] * circuit.manifest()["num_detectors"] if early else None
                for mode in args.modes:
                    plan, compile_seconds, stats = scheduled_compile(source, mode, mask)

                    def sample(count, seed):
                        if early:
                            return clifft.sample_survivors(
                                plan, count, seed=seed, threads=1, keep_records=True
                            )
                        return clifft.sample(plan, count, seed=seed, threads=1)

                    sample(64 if distance == 3 else 1, 1)
                    batches = []
                    for repeat in range(3):
                        start = time.perf_counter()
                        result = sample(shots, 471 + repeat)
                        elapsed = time.perf_counter() - start
                        accepted = ~result.detectors.any(axis=1)
                        batches.append(
                            dict(
                                seconds=elapsed,
                                accepted=int(accepted.sum()),
                                logical_failures=int(result.observables[accepted].sum()),
                            )
                        )
                    yield dict(
                        family="folded_msc",
                        distance=distance,
                        probability=probability,
                        circuit_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        early_rejection=early,
                        mode=mode,
                        compile_seconds=compile_seconds,
                        scheduler=stats,
                        peak_active_width=plan.peak_active_width,
                        coefficient_bytes=16 * (1 << plan.peak_active_width),
                        shots_per_batch=shots,
                        batches=batches,
                        median_seconds_per_attempt=statistics.median(b["seconds"] for b in batches)
                        / shots,
                    )


def synthetic_cases(args):
    from compile_terminal_gadget import compile_bundle
    from gadget_family import make_circuit

    with tempfile.TemporaryDirectory() as temporary:
        for distance in args.distances or (3, 5, 7, 9):
            shots = {3: 65536, 5: 8192, 7: 256, 9: 256}[distance]
            for probability in (0, 0.001, 0.01):
                source = make_circuit(distance, probability)
                bundle = Path(temporary) / f"d{distance}-p{probability}"
                start = time.perf_counter()
                metadata = compile_bundle(source, bundle)
                bundle_seconds = time.perf_counter() - start
                traces = None
                for mode in args.modes:
                    # Keep the gadget path fixed while changing only the full
                    # ordinary baseline. Its traced histories must stay identical.
                    output = subprocess.check_output(
                        [
                            str(args.sampler.resolve()),
                            str(bundle),
                            str(shots),
                            "8",
                            "812",
                            "24",
                            mode,
                        ],
                        text=True,
                    )
                    rows = [json.loads(line) for line in output.splitlines()]
                    if traces is not None and traces != rows[:-1]:
                        raise AssertionError("baseline mode changed gadget histories")
                    traces = rows[:-1]
                    benchmark = rows[-1]
                    ordinary = (
                        statistics.median(benchmark["ordinary_seconds"]) / shots
                        if benchmark["baseline_enabled"]
                        else None
                    )
                    gadget = statistics.median(benchmark["compiled_seconds"]) / shots
                    yield dict(
                        family="synthetic_gadget",
                        distance=distance,
                        probability=probability,
                        circuit_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        mode=mode,
                        early_rejection=False,
                        shots_per_batch=shots,
                        compile_seconds=benchmark["ordinary_compile_seconds"],
                        python_bundle_seconds=bundle_seconds,
                        peak_active_width=benchmark["ordinary_peak_width"],
                        coefficient_bytes=16 * (1 << benchmark["ordinary_peak_width"]),
                        median_seconds_per_attempt=ordinary,
                        gadget_seconds_per_attempt=gadget,
                        ordinary_over_gadget=ordinary / gadget if ordinary is not None else None,
                        benchmark=benchmark,
                        gadget_histories_sha256=hashlib.sha256(
                            json.dumps(traces, sort_keys=True).encode()
                        ).hexdigest(),
                        contraction_numeric_payload_bytes=sum(
                            16 * (p["storage_complex_entries"] + (1 << p["peak_scope"]))
                            + 8
                            * (p["gather_entries"] + p["leaf_parity_entries"] + p["output_entries"])
                            for p in metadata["contractions"]
                        ),
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("folded", "synthetic"), required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--distances", nargs="+", type=int)
    parser.add_argument("--sampler", type=Path)
    parser.add_argument("--build-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not hasattr(clifft, "ActiveWidthSchedulePass"):
        parser.error("loaded Clifft lacks the scheduling pass; check the Python environment")
    if args.family == "synthetic" and args.sampler is None:
        parser.error("synthetic measurements require --sampler")
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    root = args.build_source.resolve()
    report = dict(
        base_revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        merged_revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "MERGE_HEAD"], text=True
        ).strip(),
        build_tree_diff_sha256=hashlib.sha256(
            subprocess.check_output(
                ["git", "-C", str(root), "diff", "HEAD", "--", "src", "cmake", "CMakeLists.txt"]
            )
        ).hexdigest(),
        python=platform.python_version(),
        extension=str(clifft._clifft_core.__file__),
        extension_sha256=digest(clifft._clifft_core.__file__),
        sampler_sha256=digest(args.sampler) if args.sampler else None,
        driver_sha256=digest(__file__),
        affinity=cpu,
        threads=1,
        cpu=next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        schedule_options=dict(
            beam_width=8,
            noise_transparent=True,
            sink_neutral_rotations=True,
            budgeted_search_budget=16,
        ),
        memory_scope="coefficient array per lane; excludes scratch, plans and output storage",
        cases=[],
    )
    for case in folded_cases(args) if args.family == "folded" else synthetic_cases(args):
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in case.items() if k not in ("batches", "benchmark")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
