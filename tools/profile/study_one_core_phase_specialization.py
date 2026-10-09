"""Measure a one-rotation bridge and its attempted second phase reduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from automatic_specialization import analyze
from one_core_phase_specialization import OneCorePhase
from study_automatic_specialization import candidates, sample_parities, stress_histories
from study_regional_phase_specialization import raw_info
from study_shared_phase_specialization import features

import clifft


def small_cases() -> dict[str, str]:
    return {
        "coherent_cancellation": "H 0\nT 0\nH 0\nH 0\nT_DAG 0\n",
        "measured_cancellation": (
            "H 0 1\nT 0\nCX 1 0\nM 1\nR 1\nT_DAG 0\nMX 0\n"
            "DETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        ),
        "old_record_feedback": (
            "H 2\nM(.13) 2\nH 0 1\nT 0\nM 1\nCX rec[-2] 0\n"
            "T_DAG 0\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-1]\n"
        ),
        "new_record_feedback": "H 0 1\nT 0\nM 1\nCX rec[-1] 0\nT_DAG 0\nMX 0\n",
        "product_measurement": ("H 0 1 2\nT 0\nMPP(.17) !Z1*X2\nCZ rec[-1] 0\nT_DAG 0\nMY 0\n"),
        "hidden_reset": "H 0\nT 0\nCX 0 1\nR 1\nT_DAG 0\nH 0\nM 0\n",
        "erase_by_measurement": "H 0\nT 0\nM 0\nT_DAG 0\nH 0\nM 0\n",
        "noncommuting_measurement": "H 0 1\nT 0\nM 1\nCX rec[-1] 0\nMX 0\nT 0\nM 0\n",
        "overlapping_reset": "H 0\nT 0\nH 0\nR 0\nH 0\nT 0\nMX 0\n",
        "non_diagonal_entry": "H 0\nT 0\nH 0\nT_DAG 0\nMX 0\n",
        "imaginary_representative": "H 0\nS 0\nT 0\nH 0\nT_DAG 0\nM 0\n",
        "equivalent_diagonal_entry": (
            "H 0\nCX 0 1\nH 2\nT 2\nH 2\nCX 2 0\nCX 2 1\nH 2\nT_DAG 2\nMX 2\nM 0 1\n"
        ),
        "negative_angle": "H 0\nT_DAG 0\nH 0\nS 0\nS_DAG 0\nH 0\nT 0\nMY 0\n",
        "retained_magic": "H 0\nT 0\nH 0\nS 0\nM 1\n",
        "unsupported_rotation": "H 0\nT 0\nH 0\nR_PAULI(.137) Z0\nMX 0\n",
        "noise_mixture": (
            "H 0 1\nY_ERROR(.07) 0\nT 0\nM(.13) !1\nCX rec[-1] 0\n"
            "Z_ERROR(.17) 0\nT_DAG 0\nMX(.19) 0\n"
        ),
        "categorical": (
            "H 0 1\nT 0\nM(.13) 1\nCX rec[-1] 0\n"
            "PAULI_CHANNEL_2(.01,.02,.03,.01,.02,.01,.02,.01,.01,.01,.01,.01,.01,.01,.01) 0 1\n"
            "T_DAG 0\nMX 0\nM 1\n"
        ),
        "synthesis_expansion": (
            "H 0 1 2 3 4\nT 0\nM 4\nCX rec[-1] 0\nCX 1 0\nCX 2 0\nCX 3 0\nT 1\nMX 0\n"
        ),
    }


def panel(checkout: Path) -> dict[str, str]:
    known = candidates(checkout)
    result = {
        name: known[name][1]
        for name in ("cultivation_d3", "cultivation_d5", "15to1_scored", "15to1_direct_x")
    }
    result.update(
        {
            name: small_cases()[name]
            for name in ("measured_cancellation", "equivalent_diagonal_entry")
        }
    )
    return result


def study(source: str, args: Any) -> dict[str, Any]:
    _, ordinary = analyze(source)
    started = perf_counter()
    core = OneCorePhase(source)
    setup = perf_counter() - started
    stress = []
    for history in stress_histories(core.model):
        for seed in (17, 419):
            branch = core.rewrite(history, seed, attempt_second=not args.carrier_only)
            stress.append(
                {
                    "history": history,
                    "seed": seed,
                    "first": raw_info(branch.first_source)[1],
                    "transported": raw_info(branch.transported_source)[1],
                    "candidate": raw_info(branch.candidate_source)[1],
                    "selected": raw_info(branch.source)[1],
                    "optimized_first": analyze(branch.first_source)[1],
                    "optimized_transported": analyze(branch.transported_source)[1],
                    "optimized_candidate": analyze(branch.candidate_source)[1],
                    "metadata": branch.metadata,
                }
            )
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x8719)
    stages = dict.fromkeys(("draw", "rewrite", "analyze_and_lower", "sample"), 0.0)
    times, histories, records = [], [], []
    reasons: Counter[str] = Counter()
    widths: Counter[int] = Counter()
    raw_widths: Counter[int] = Counter()
    final_t: Counter[int] = Counter()
    candidate_widths: Counter[int] = Counter()
    second_t: Counter[int] = Counter()
    supports: Counter[int] = Counter()
    decisions: Counter[int] = Counter()
    accepted = second = max_tableau = 0
    for _ in range(args.shots):
        start = perf_counter()
        history = core.model.draw(faults)
        drawn = perf_counter()
        branch = core.rewrite(
            history, quantum.getrandbits(64), attempt_second=not args.carrier_only
        )
        rewritten = perf_counter()
        _, raw = raw_info(branch.source)
        hir, info = analyze(branch.source)
        if info["peak_width"] > args.max_width:
            raise ValueError("Selected continuation exceeds the execution budget")
        program = clifft.lower(hir)
        lowered = perf_counter()
        sample = clifft.sample(program, shots=1, seed=quantum.getrandbits(64), threads=1)
        end = perf_counter()
        for stage, duration in zip(
            stages, (drawn - start, rewritten - drawn, lowered - rewritten, end - lowered)
        ):
            stages[stage] += duration
        times.append(end - start)
        histories.append(history)
        records.append(sample.measurements[0].copy())
        sample_parities(source, sample)
        widths[info["peak_width"]] += 1
        raw_widths[raw["peak_width"]] += 1
        final_t[info["output_t"]] += 1
        metadata = branch.metadata
        reasons[metadata["stop_reason"]] += 1
        decisions[len(metadata["random_decisions"])] += 1
        max_tableau = max(max_tableau, metadata["tableau_qubits"])
        if "second_region" in metadata:
            second += 1
            accepted += metadata["accepted_second_region"]
            candidate_widths[metadata["candidate_cost"]["width"]] += 1
            second_t[metadata["second_region"]["residual_t"]] += 1
            supports[metadata["second_region"]["support_variables"]] += 1
    result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "ordinary": ordinary,
        "first_region": core.first.metadata(),
        "setup_seconds": setup,
        "stress": stress,
        "fresh": {
            "shots": args.shots,
            "mean_seconds_per_shot": statistics.mean(times),
            "maximum_seconds_per_shot": max(times),
            "stage_seconds": stages,
            "selected_widths": dict(widths),
            "selected_raw_widths": dict(raw_widths),
            "optimized_t_counts": dict(final_t),
            "candidate_widths": dict(candidate_widths),
            "second_region_t_counts": dict(second_t),
            "second_region_support_variables": dict(supports),
            "second_regions_attempted": second,
            "second_regions_accepted": accepted,
            "stop_reasons": dict(reasons),
            "random_decision_counts": dict(decisions),
            "maximum_tableau_qubits": max_tableau,
            "fault_counts": dict(sorted(Counter(map(len, histories)).items())),
            "history_sha256": hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
            "record_sha256": hashlib.sha256(np.stack(records).tobytes()).hexdigest(),
        },
    }
    merlin = importlib.import_module("merlin")
    try:
        start = perf_counter()
        sampler = merlin.CircuitSampler(source, seed=args.seed + 29)
        initialized = perf_counter()
        reference = sampler.sample(args.shots)
        elapsed = perf_counter() - initialized
        a, b = features(source, np.stack(records)), features(source, reference.measurements)
        variance = a.var(axis=0, ddof=1) / args.shots + b.var(axis=0, ddof=1) / args.shots
        delta = np.maximum(0, abs(a.mean(axis=0) - b.mean(axis=0)) - 4 / args.shots)
        score = float(max(delta / np.maximum(np.sqrt(variance), 1e-12)))
        if score > 7:
            raise AssertionError("Fresh record/parity moments differ from Merlin")
        result["merlin"] = {
            "status": "sampled",
            "setup_seconds": initialized - start,
            "seconds_per_shot": elapsed / args.shots,
            "moment_features": a.shape[1],
            "maximum_score": score,
            "scope": "Finite fresh-noise bug check, not full distribution or rare-event validation",
        }
    except (ValueError, RuntimeError, merlin.MeasurementError) as error:
        result["merlin"] = {"status": "unsupported", "reason": str(error)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=714917)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--carrier-only", action="store_true")
    args = parser.parse_args()
    if args.shots < 32 or not 0 <= args.max_width <= 16:
        parser.error("Require at least 32 shots and an execution width budget at most sixteen")
    sources = panel(args.merlin_checkout)
    selected = args.cases or list(sources)
    if set(selected) - sources.keys():
        parser.error("Unknown case")
    result: dict[str, Any] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "affinity": sorted(os.sched_getaffinity(0)),
        "final_compilation": "Existing HIR optimization of the complete conditional source",
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "one_core_phase_specialization.py",
                "regional_phase_specialization.py",
                "shared_phase_specialization.py",
                "automatic_specialization.py",
                "study_automatic_specialization.py",
                "study_regional_phase_specialization.py",
                "study_shared_phase_specialization.py",
            )
        },
        "cases": {},
    }
    for name in selected:
        print(name, "starting", flush=True)
        result["cases"][name] = study(sources[name], args)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(name, result["cases"][name]["fresh"], flush=True)


if __name__ == "__main__":
    main()
