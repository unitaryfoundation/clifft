"""Measure shared preparation optimization with fresh complete trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from compiled_prefix_reuse import ReusablePhase, compile_stages
from conditional_phase_frontend import ConditionalPhase
from study_automatic_specialization import sample_parities, stress_histories
from study_factory_frontend import sources
from validate_automatic_specialization import visible_probability

import clifft
import clifft._clifft_core as core


def study(source: str, args: Any) -> dict[str, Any]:
    start = perf_counter()
    front = ReusablePhase(source, args.exporter)
    setup = perf_counter() - start
    result: dict[str, Any] = {
        "setup_seconds": setup,
        "reuse": front.reuse_info,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    modes = ("fresh", "reuse_full", "reuse", "skip_only")
    times: dict[str, list[float]] = {m: [] for m in modes}
    stages: dict[str, dict[str, float]] = {m: defaultdict(float) for m in modes}
    widths: dict[str, Counter[int]] = {m: Counter() for m in modes}
    counts: dict[str, Counter[int]] = {m: Counter() for m in modes}
    samples: dict[str, list[Any]] = {m: [] for m in modes}
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x8719)
    histories = []
    max_probability_error = max_relative_error = 0.0
    probability_checks = 0
    unchanged_checks = 0
    for index in range(args.shots):
        start = perf_counter()
        history = front.model.draw(faults)
        draw = perf_counter() - start
        histories.append(history)
        seed, sample_seed = quantum.getrandbits(64), quantum.getrandbits(64)
        programs = {}
        emitted_sources = {}
        # Alternate the order to reduce systematic warm-cache timing bias.
        order = modes[index % len(modes) :] + modes[: index % len(modes)]
        for mode in order:
            begin = perf_counter()
            if mode in {"fresh", "skip_only"}:
                branch = ConditionalPhase.rewrite(front, history, seed)
            else:
                branch = front.rewrite(history, seed)
            rewritten = perf_counter()
            phase = mode in {"fresh", "reuse_full"} or (mode == "reuse" and front.phase_required)
            hir, info = compile_stages(branch.source, phase=phase)
            analyzed = perf_counter()
            width = info["width"]
            widths[mode][width] += 1
            counts[mode][hir.num_t_gates] += 1
            for stage, value in info["seconds"].items():
                stages[mode][stage] += value
            if width <= args.max_width:
                program = clifft.lower(hir)
                lowered = perf_counter()
                sample = clifft.sample(program, shots=1, seed=sample_seed, threads=1)
                end = perf_counter()
                programs[mode] = program
                samples[mode].append(sample.measurements[0].copy())
                sample_parities(source, sample)
                stages[mode]["lower"] += lowered - analyzed
                stages[mode]["sample"] += end - lowered
            else:
                end = analyzed
            stages[mode]["draw"] += draw
            stages[mode]["rewrite"] += rewritten - begin
            times[mode].append(end - begin + draw)
            emitted_sources[mode] = branch.source
        # Both modes use the same sampled preparation branch. Selected replay
        # checks validate probabilities, not bitwise equality of quantum RNG.
        if not front.reuse_info["eligible"]:
            assert emitted_sources["fresh"] == emitted_sources["reuse"]
            assert emitted_sources["fresh"] == emitted_sources["reuse_full"]
            if "fresh" in programs and "reuse" in programs:
                np.testing.assert_array_equal(samples["fresh"][-1], samples["reuse"][-1])
            unchanged_checks += 1
        elif index < 16 and "fresh" in programs and "reuse" in programs:
            for mode in ("fresh", "reuse"):
                record = list(map(int, samples[mode][-1]))
                a, b = [visible_probability(programs[m], record) for m in ("fresh", "reuse")]
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-300)
                max_probability_error = max(max_probability_error, abs(a - b))
                if b:
                    max_relative_error = max(max_relative_error, abs(a - b) / b)
                probability_checks += 1
    result["modes"] = {
        m: {
            "seconds_per_attempt": statistics.mean(times[m]),
            "maximum_seconds_per_attempt": max(times[m]),
            "completed_shots": len(samples[m]),
            "widths": dict(widths[m]),
            "t_counts": dict(counts[m]),
            "stage_seconds": dict(stages[m]),
            "width_failures": args.shots - len(samples[m]),
            "record_sha256": hashlib.sha256(
                np.asarray(samples[m], dtype="u1").tobytes()
            ).hexdigest(),
        }
        for m in modes
    }
    stress = []
    for history in stress_histories(front.model):
        branch = front.rewrite(history, 419)
        hir, info = compile_stages(branch.source, phase=front.phase_required)
        stress.append({"history": history, "width": info["width"], "t_count": hir.num_t_gates})
    result.update(
        histories_sha256=hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
        fault_weights=dict(Counter(map(len, histories))),
        stress=stress,
        conditional_record_checks=probability_checks,
        unchanged_fallback_checks=unchanged_checks,
        maximum_probability_error=max_probability_error,
        maximum_relative_error=max_relative_error,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=271019)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.shots < 32 or not 0 <= args.max_width <= 16:
        parser.error("Require at least 32 shots and a width budget at most sixteen")
    panel = sources(args)
    selected = args.cases or [
        "quadcycle-noisy.stim",
        "steane-D.stim",
        "steane-E.stim",
        "bt27_scored",
        "bt27_direct_x",
        "cultivation_d3",
        "cultivation_d5",
        "noncommuting_control",
        "unsupported_rotation",
    ]
    result: dict[str, Any] = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("compiled_prefix_reuse.py"),
                Path(__file__).with_name("export_optimized_prefix.cpp"),
                Path(__file__).with_name("conditional_phase_frontend.py"),
            ]
        },
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "cases": {},
    }
    for name in selected:
        print(name, "starting", flush=True)
        result["cases"][name] = study(panel[name], args)
        print(
            name,
            {
                m: (r["seconds_per_attempt"], r["widths"])
                for m, r in result["cases"][name]["modes"].items()
            },
            flush=True,
        )
        result["process_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result["exporter_peak_bytes"] = (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
