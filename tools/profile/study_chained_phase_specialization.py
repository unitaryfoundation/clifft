"""Measure chained reductions across measurement-dependent Clifford boundaries."""

from __future__ import annotations

import argparse
import hashlib
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
from chained_phase_specialization import ChainedPhase
from study_automatic_specialization import candidates, sample_parities, stress_histories
from study_regional_phase_specialization import raw_info

import clifft


def measured_chain(rounds: int = 2, *, noisy: bool = False) -> str:
    lines = ["H 0 1"]
    for index in range(rounds):
        probability = (0.07 if index == 0 else 0.11) if noisy else 0
        lines += [f"MPP({probability}) Z0*Z1", "CX rec[-1] 1", "T 0"]
        if noisy and index == 0:
            lines += ["Y_ERROR(.13) 1"]
        lines += ["T_DAG 1", "H 0", "S 1", "CZ rec[-1] 0"]
        if noisy and index == 0:
            lines += ["Z_ERROR(.09) 0"]
    lines += ["MX 0", "MY 1", "DETECTOR rec[-1] rec[-2]", "OBSERVABLE_INCLUDE(0) rec[-3]"]
    return "\n".join(lines) + "\n"


def parity_phase(support: list[int], inverse: bool = False) -> list[str]:
    target = support[-1]
    compute = [f"CX {q} {target}" for q in support[:-1]]
    return compute + [f"{'T_DAG' if inverse else 'T'} {target}"] + compute[::-1]


def branch_dependent_entry() -> str:
    lines = ["H 0", "M 0", "H 1 2"]
    # The parity expansion of 4*a*b*c gives a conditional CZ when a is
    # measured. A following H makes the next support depend on that outcome.
    for support, inverse in (
        ([0], False),
        ([1], False),
        ([2], False),
        ([0, 1], True),
        ([0, 2], True),
        ([1, 2], True),
        ([0, 1, 2], False),
    ):
        lines += parity_phase(support, inverse)
    lines += ["H 1", "M 0", "CZ rec[-1] 2", "T 1", "H 1", "M 1 2"]
    return "\n".join(lines) + "\n"


def echo_region(qubits: int, seed: int, probability: float) -> list[str]:
    rng = random.Random(seed)
    supports = [rng.sample(range(qubits), min(3, qubits)) for _ in range(qubits)]
    lines = []
    for _ in range(8):
        for support in supports:
            target = support[-1]
            compute = [f"CX {q} {target}" for q in support[:-1]]
            lines += (
                compute + [f"T {target}", f"DEPOLARIZE1({probability}) {target}"] + compute[::-1]
            )
    return lines


def echo_chain(
    qubits: int, *, tail_magic: int = 0, seed: int = 179, probability: float = 0.001
) -> str:
    lines = [f"H {q}\nS {q}\nDEPOLARIZE1({probability}) {q}" for q in range(qubits)]
    lines += echo_region(qubits, seed, probability)
    lines += [f"H {q}" for q in range(qubits)]
    lines += [f"M({probability}) 0", "H 0", "CX rec[-1] 1", f"DEPOLARIZE2({probability}) 0 1"]
    lines += echo_region(qubits, seed + 1, probability)
    lines += [f"H {q}" for q in range(qubits)]
    for q in range(tail_magic):
        lines += [f"T {q}", f"DEPOLARIZE1({probability}) {q}"]
    lines += [f"MX({probability}) {q}" for q in range(qubits)]
    lines += ["DETECTOR rec[-1] rec[-2]", "OBSERVABLE_INCLUDE(0) rec[-1]"]
    return "\n".join(lines) + "\n"


def small_cases() -> dict[str, str]:
    result = {
        "two_measured": measured_chain(),
        "three_measured": measured_chain(3),
        "noisy_measured": measured_chain(noisy=True),
        "branch_dependent": branch_dependent_entry(),
        "surviving_magic": "H 0 1\nT 0\nH 0\nMPP Z0*Z1\nT 0\nT_DAG 1\nH 0\nM 0 1\n",
        "reset_after_magic": "H 0\nT 0\nR 0\nH 0\nT 0\nT_DAG 0\nH 0\nM 0\n",
        "clifford_input": "H 0\nM 0\n",
        "unsupported_later_rotation": "H 0\nT 0\nT_DAG 0\nH 0\nR_PAULI(.137) X0\nH 0\nM 0\n",
    }
    for seed in range(4):
        source = echo_chain(3, tail_magic=seed % 2, seed=191 + seed, probability=0.03)
        if seed % 2:
            source = source.replace("H 0\nCX rec[-1] 1", "RY 0\nCZ rec[-1] 1")
        result[f"general_{seed}"] = source
    return result


def study(source: str, args: Any) -> dict[str, Any]:
    _, ordinary = analyze(source)
    started = perf_counter()
    chain = ChainedPhase(source, max_regions=args.max_regions)
    setup = perf_counter() - started
    once = ChainedPhase(source, max_regions=1)
    stress: list[dict[str, Any]] = []
    for index, history in enumerate(stress_histories(chain.model)):
        one = once.sample_history(history, random.Random(args.seed + index))
        many = chain.sample_history(history, random.Random(args.seed + index))
        _, one_info = raw_info(one.source)
        _, many_info = raw_info(many.source)
        stress.append(
            {
                "history": history,
                "one_region": one_info,
                "chained": many_info,
                "steps": many.steps,
                "stop_reason": many.stop_reason,
            }
        )
    result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "ordinary": ordinary,
        "initial_analysis_seconds": setup,
        "stress": stress,
    }
    if max(row["chained"]["peak_width"] for row in stress) > args.max_width:
        result["status"] = "width_budget"
        return result
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 192834)
    stages = dict.fromkeys(("draw", "rewrite", "trace_and_lower", "sample"), 0.0)
    times, histories, arrays = [], [], []
    step_counts: Counter[int] = Counter()
    stops: Counter[str] = Counter()
    widths: Counter[int] = Counter()
    later_bases: set[str] = set()
    later_analysis = 0.0
    for _ in range(args.shots):
        start = perf_counter()
        history = chain.model.draw(faults)
        drawn = perf_counter()
        rewritten = chain.sample_history(history, quantum)
        composed = perf_counter()
        hir, info = raw_info(rewritten.source)
        if info["peak_width"] > args.max_width:
            raise ValueError("Fresh chained circuit exceeds the execution width budget")
        program = clifft.lower(hir)
        lowered = perf_counter()
        sampled = clifft.sample(program, shots=1, seed=quantum.getrandbits(64), threads=1)
        ended = perf_counter()
        for key, seconds in zip(
            stages, (drawn - start, composed - drawn, lowered - composed, ended - lowered)
        ):
            stages[key] += seconds
        times.append(ended - start)
        later_analysis += sum(step["analysis_seconds"] for step in rewritten.steps[1:])
        later_bases.update(step["entry_basis_sha256"] for step in rewritten.steps[1:])
        step_counts[len(rewritten.steps)] += 1
        stops[rewritten.stop_reason] += 1
        widths[info["peak_width"]] += 1
        sample_parities(source, sampled)
        arrays.append(sampled.measurements[0].copy())
        histories.append(history)
    result.update(
        status="sampled",
        shots=args.shots,
        mean_seconds_per_shot=statistics.mean(times),
        stage_seconds=stages,
        later_analysis_seconds=later_analysis,
        unique_later_entry_bases=len(later_bases),
        steps=dict(step_counts),
        stops=dict(stops),
        widths=dict(widths),
        unique_histories=len(set(histories)),
        fault_counts=dict(sorted(Counter(map(len, histories)).items())),
        history_sha256=hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
        record_sha256=hashlib.sha256(np.stack(arrays).tobytes()).hexdigest(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=351923)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.shots < 1 or not 0 <= args.max_width <= 16 or not 1 <= args.max_regions <= 32:
        parser.error("Require positive shots, width at most sixteen, and one to thirty-two regions")
    known = candidates(args.merlin_checkout)
    sources = {
        name: source for name, source in small_cases().items() if not name.startswith("general_")
    }
    for n, tail in ((6, 0), (24, 0), (48, 0), (48, 2)):
        sources[f"echo_{n}_tail_{tail}"] = echo_chain(n, tail_magic=tail)
    sources["bt27_scored"] = known["bt27_scored"][1]
    sources["15to1_scored"] = known["15to1_scored"][1]
    selected = args.cases or list(sources)
    if set(selected) - sources.keys():
        parser.error("Unknown case")
    result: dict[str, Any] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "chained_phase_specialization.py",
                "regional_phase_specialization.py",
                "shared_phase_specialization.py",
            )
        },
        "cases": {},
    }
    for name in selected:
        row = study(sources[name], args)
        result["cases"][name] = row
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            name,
            row["status"],
            "ordinary",
            row["ordinary"]["peak_width"],
            "one region",
            max(x["one_region"]["peak_width"] for x in row["stress"]),
            "chained",
            max(x["chained"]["peak_width"] for x in row["stress"]),
            "steps",
            row.get("steps"),
            flush=True,
        )


if __name__ == "__main__":
    main()
