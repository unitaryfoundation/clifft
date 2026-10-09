"""Assess frozen whole-circuit inputs with isolated per-backend measurements."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import stim
from automatic_specialization import FaultModel, analyze, specialize_prefix
from conditional_phase_frontend import ConditionalPhase
from study_automatic_specialization import candidates, sample_parities
from study_shared_phase_specialization import features

import clifft
import clifft._clifft_core as core


def sources(args: Any) -> dict[str, str]:
    manifest = json.loads((args.benchmark_dir / "manifest.json").read_text())
    result = {}
    for entry in manifest["cases"]:
        source = (args.benchmark_dir / entry["file"]).read_text()
        if hashlib.sha256(source.encode()).hexdigest() != entry["sha256"]:
            raise ValueError("Benchmark circuit hash differs")
        result[entry["file"]] = source
    result.update({name: value[1] for name, value in candidates(args.merlin_checkout).items()})
    result["unsupported_rotation"] = "H 0\nR_PAULI(.137) X0\nT 0\nMY 0\n"
    return result


def worker(source: str, args: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": args.worker,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "process_peak_before_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    start = perf_counter()
    if args.worker == "ordinary":
        hir, info = analyze(source)
        result["plan"] = info
        if info["peak_width"] > args.max_width:
            return {**result, "status": "width_budget", "setup_seconds": perf_counter() - start}
        program = clifft.lower(hir)
        ready = perf_counter()
        sample = clifft.sample(program, shots=args.shots, seed=args.seed, threads=1)
        elapsed = perf_counter() - ready
        sample_parities(source, sample)
        arrays = sample.measurements
        result.update(
            status="sampled",
            setup_seconds=ready - start,
            seconds_per_shot=elapsed / args.shots,
            shots=args.shots,
            widths={info["peak_width"]: args.shots},
        )
    elif args.worker == "merlin":
        merlin = importlib.import_module("merlin")
        try:
            sampler = merlin.CircuitSampler(source, seed=args.seed)
            ready = perf_counter()
            sample = sampler.sample(args.shots)
            elapsed = perf_counter() - ready
        except (ValueError, RuntimeError, merlin.MeasurementError) as error:
            return {
                **result,
                "status": "unsupported",
                "reason": str(error),
                "setup_seconds": perf_counter() - start,
            }
        arrays = sample.measurements
        result.update(
            status="sampled",
            setup_seconds=ready - start,
            seconds_per_shot=elapsed / args.shots,
            shots=args.shots,
        )
    else:
        front = ConditionalPhase(source) if args.worker == "conditional" else None
        model = front.model if front else FaultModel(source)
        ready = perf_counter()
        result.update(setup_seconds=ready - start, noise_sites=len(model.sites))
        if front:
            result["first_region"] = (
                front.first.metadata() if front.first else {"rejection": front.rejection}
            )
        faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x8719)
        times, histories, records = [], [], []
        stages = dict.fromkeys(("draw", "rewrite", "analyze", "lower", "sample"), 0.0)
        widths: Counter[int] = Counter()
        routes: Counter[str] = Counter()
        carrier_stops: Counter[str] = Counter()
        failure_count = 0
        for _ in range(args.shots):
            start = perf_counter()
            history = model.draw(faults)
            drawn = perf_counter()
            seed = quantum.getrandbits(64)
            if front:
                branch = front.rewrite(history, seed)
                text = branch.source
                routes[branch.stop_reason] += 1
                if branch.carrier:
                    carrier_stops[branch.carrier["stop_reason"]] += 1
            else:
                text = model.render(history)
                if args.worker == "fault_prefix":
                    text, _ = specialize_prefix(text, seed)
            rewritten = perf_counter()
            hir, info = analyze(text)
            analyzed = perf_counter()
            widths[info["peak_width"]] += 1
            histories.append(history)
            if info["peak_width"] > args.max_width:
                # Keep every history in the assessment, including failures.
                # Never allocate the rejected dense state or redraw its faults.
                failure_count += 1
                lowered = ended = analyzed
            else:
                program = clifft.lower(hir)
                lowered = perf_counter()
                sample = clifft.sample(program, shots=1, seed=quantum.getrandbits(64), threads=1)
                ended = perf_counter()
                sample_parities(source, sample)
                records.append(sample.measurements[0].copy())
            for stage, duration in zip(
                stages,
                (
                    drawn - start,
                    rewritten - drawn,
                    analyzed - rewritten,
                    lowered - analyzed,
                    ended - lowered,
                ),
            ):
                stages[stage] += duration
            times.append(ended - start)
        result.update(
            status="sampled" if not failure_count else "partial_width_budget",
            shots=args.shots,
            completed_shots=len(records),
            width_failures=failure_count,
            widths=dict(widths),
            routes=dict(routes),
            carrier_stops=dict(carrier_stops),
            seconds_per_attempt=statistics.mean(times),
            stage_seconds=stages,
            maximum_seconds_per_attempt=max(times),
            history_sha256=hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
            fault_weights=dict(Counter(map(len, histories))),
        )
        if failure_count:
            return result
        result["seconds_per_shot"] = result["seconds_per_attempt"]
        arrays = np.stack(records)
    if arrays.shape != (args.shots, clifft.parse(source).num_measurements):
        raise AssertionError("Sampling changed the complete visible record contract")
    observed = features(source, arrays)
    result["record_sha256"] = hashlib.sha256(arrays.astype("u1").tobytes()).hexdigest()
    result["feature_means"] = observed.mean(axis=0).tolist()
    result["feature_variances"] = observed.var(axis=0, ddof=1).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=271019)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument(
        "--worker", choices=("conditional", "ordinary", "fault_only", "fault_prefix", "merlin")
    )
    args = parser.parse_args()
    if args.shots < 32 or not 0 <= args.max_width <= 16:
        parser.error("Require at least 32 shots and a width budget at most sixteen")
    panel = sources(args)
    if args.worker:
        row = worker(panel[args.cases[0]], args)
        row["process_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        print(json.dumps(row), flush=True)
        return
    selected = args.cases or [
        "quadcycle-ideal.stim",
        "quadcycle-noisy.stim",
        "steane-D.stim",
        "steane-E.stim",
        "bt27_scored",
        "bt27_direct_x",
        "cultivation_d3",
        "cultivation_d5",
        "15to1_scored",
        "noncommuting_control",
        "unsupported_rotation",
    ]
    result: dict[str, Any] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "stim": stim.__version__,
            "extension": core.__file__,
            "extension_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            "merlin_revision": subprocess.check_output(
                ["git", "-C", str(args.merlin_checkout), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "cases": {},
        "memory_scope": (
            "Fresh process per case/backend; Linux ru_maxrss includes Python, imports, "
            "input generation, planning, and sampling."
        ),
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "conditional_phase_frontend.py",
                "automatic_specialization.py",
                "deferred_phase_specialization.py",
                "regional_phase_specialization.py",
                "shared_phase_specialization.py",
                "one_core_phase_specialization.py",
            )
        },
    }
    for name in selected:
        rows = {}
        for backend in ("conditional", "ordinary", "fault_only", "fault_prefix", "merlin"):
            command = [
                sys.executable,
                str(Path(__file__)),
                "--benchmark-dir",
                str(args.benchmark_dir),
                "--merlin-checkout",
                str(args.merlin_checkout),
                "--output",
                str(args.output),
                "--shots",
                str(args.shots),
                "--max-width",
                str(args.max_width),
                "--seed",
                str(args.seed),
                "--cases",
                name,
                "--worker",
                backend,
            ]
            child = subprocess.run(
                command, capture_output=True, text=True, timeout=180, check=False
            )
            if child.returncode:
                raise RuntimeError(f"{name}/{backend} failed: {child.stderr}")
            rows[backend] = json.loads(child.stdout)
            print(
                name,
                backend,
                rows[backend]["status"],
                rows[backend].get("seconds_per_shot"),
                flush=True,
            )
        candidate = rows["conditional"]
        for backend, row in rows.items():
            if (
                backend == "conditional"
                or row["status"] != "sampled"
                or candidate["status"] != "sampled"
            ):
                continue
            a, b = np.array(candidate["feature_means"]), np.array(row["feature_means"])
            variance = (
                np.array(candidate["feature_variances"]) + np.array(row["feature_variances"])
            ) / args.shots
            score = float(
                np.max(
                    np.maximum(0, abs(a - b) - 4 / args.shots)
                    / np.maximum(np.sqrt(variance), 1e-12)
                )
            )
            row["moment_comparison"] = {
                "maximum_score": score,
                "passed": score <= 7,
                "scope": "Finite sampling bug check, not a distribution or rare-event certificate",
            }
            if score > 7:
                raise AssertionError(f"{name}/{backend} record moments differ")
        result["cases"][name] = rows
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
