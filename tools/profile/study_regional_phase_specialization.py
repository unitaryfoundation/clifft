"""Study quantum-state exits and complete-circuit width after continuation."""

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
from regional_phase_specialization import RegionalPhase
from study_automatic_specialization import candidates, sample_parities, stress_histories

import clifft


def small_cases() -> dict[str, str]:
    result = {
        "constant_offset": "X 0\nT 0\nT 1\nH 0 1\nT 1\nCX 0 1\nM 0 1\n",
        "new_tail_wire": "H 0\nT 0\nH 0\nCX 0 2\nT 2\nM 0\nMX 2\n",
        "coherence": "H 0\nS 0\nT 0\nT_DAG 0\nH 0\nT 0\nMX 0\n",
        "record_correlation": "H 0\nCX 0 1\nM 0\nT 1\nT_DAG 1\nH 1\nT 1\nMX 1\n",
        "prefix_measurement_state": (
            "H 2\nM(.11) !2\nH 0 1\nCX 0 1\nS 1\nT 0\nT_DAG 1\nH 0\nT 1\nCX 0 2\nMY 1\nMX 0\nM 2\n"
        ),
        "noise_mixture": (
            "H 0 1\nY_ERROR(.13) 0\nMPP Z0*Z1\nT 0\n"
            "X_ERROR(.21) 1\nT_DAG 1\nCX rec[-1] 0\n"
            "H 0\nT 0\nZ_ERROR(.09) 1\nM(.17) !0\n"
            "CX rec[-1] 1\nH 1\nT 1\nMX 1\n"
            "DETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]\n"
        ),
    }
    for seed in range(8):
        rng = random.Random(89171 + seed)
        lines = ["H 0 1 2"]
        for _ in range(5):
            a, b = rng.sample(range(3), 2)
            lines += [
                f"{rng.choice(('H', 'S', 'S_DAG'))} {a}",
                f"{rng.choice(('CX', 'CZ'))} {a} {b}",
                f"DEPOLARIZE2(.04) {a} {b}",
            ]
        lines += ["MPP(.07) !Y0*Z1", "CX rec[-1] 2"]
        if seed % 3 == 0:
            lines += ["R 1", "H 1"]
        for _ in range(6):
            a, b = rng.sample(range(3), 2)
            lines += [
                f"{rng.choice(('T', 'T_DAG'))} {a}",
                f"DEPOLARIZE1(.08) {a}",
                f"{rng.choice(('CX', 'CZ'))} {a} {b}",
            ]
        if seed % 2:
            lines += ["MPP(.03) X0*Y1", "CZ rec[-1] 2"]
        lines += [
            "H 0",
            "T 0",
            "CX 0 2",
            "M(.05) !1",
            "CX rec[-1] 2",
            "RY 1",
            "T 1",
            "H 1",
            "CZ 1 2",
            "DEPOLARIZE2(.02) 1 2",
            "MX 0",
            "MY 1",
            "M 2",
            "DETECTOR rec[-1] rec[-3]",
            "OBSERVABLE_INCLUDE(0) rec[-2] rec[-4]",
        ]
        result[f"general_{seed}"] = "\n".join(lines) + "\n"
    return result


def parity_echo(qubits: int, tail_magic: int) -> str:
    """Construct cancellation by parity algebra, with a tunable hard continuation."""
    rng = random.Random(9381 + qubits)
    lines = []
    for q in range(qubits):
        lines += [f"H {q}", f"S {q}", f"DEPOLARIZE1(.001) {q}"]
    supports = [rng.sample(range(qubits), 3) for _ in range(qubits)]
    # A parity phase accumulated eight times is identity. Interleaved Pauli
    # faults change the Clifford correction, including correlated CZ terms.
    for _ in range(8):
        for a, b, q in supports:
            lines += [
                f"CX {a} {q}",
                f"DEPOLARIZE2(.001) {a} {q}",
                f"CX {b} {q}",
                f"T {q}",
                f"DEPOLARIZE1(.001) {q}",
                f"CX {b} {q}",
                f"CX {a} {q}",
            ]
    for q in range(qubits):
        lines.append(f"H {q}")
    for q in range(tail_magic):
        lines += [f"T {q}", f"DEPOLARIZE1(.001) {q}"]
    lines += ["M(.001) 0", "CX rec[-1] 1", "H 1", "T 1"]
    lines += [f"MX(.001) {q}" for q in range(1, qubits)]
    lines += ["DETECTOR rec[-1] rec[-2]", "OBSERVABLE_INCLUDE(0) rec[-1]"]
    return "\n".join(lines) + "\n"


def panel(checkout: Path) -> dict[str, str]:
    known = candidates(checkout)
    result = {
        name: known[name][1]
        for name in ("bt27_scored", "bt27_direct_x", "cultivation_d3", "noncommuting_control")
    }
    result.update(
        {name: small_cases()[name] for name in ("prefix_measurement_state", "noise_mixture")}
    )
    for qubits, tail in ((6, 2), (24, 2), (48, 2), (24, 24)):
        result[f"parity_echo_{qubits}_tail_{tail}"] = parity_echo(qubits, tail)
    return result


def raw_info(source: str) -> tuple[Any, dict[str, int]]:
    hir = clifft.trace(clifft.parse(source))
    return hir, {"t_count": hir.num_t_gates, "peak_width": clifft.active_width_trace(hir)["peak"]}


def study(source: str, args: Any) -> dict[str, Any]:
    _, ordinary = analyze(source)
    started = perf_counter()
    region = RegionalPhase(source)
    setup = perf_counter() - started
    stress: list[dict[str, Any]] = []
    for history in stress_histories(region.model):
        prefix, _ = region.split_history(history)
        ideal_bits = region.shared.controls(()) >> 1
        controls = region.shared.controls(prefix, ideal_bits)
        _, entry = raw_info(region.shared.render_state(controls))
        _, complete = raw_info(region.compose(history, ideal_bits))
        stress.append({"history": history, "region": entry, "complete": complete})
    result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "ordinary": ordinary,
        "region": {**region.metadata(), "analysis_seconds": setup},
        "stress": stress,
    }
    if max(row["complete"]["peak_width"] for row in stress) > args.max_width:
        result["status"] = "width_budget"
        return result
    times, arrays, histories = [], [], []
    widths: Counter[int] = Counter()
    stages = dict.fromkeys(("draw", "compose", "trace_and_lower", "sample"), 0.0)
    rng = random.Random(args.seed)
    for _ in range(args.shots):
        start = perf_counter()
        history = region.model.draw(rng)
        drawn = perf_counter()
        composed = region.compose(history)
        rendered = perf_counter()
        hir, info = raw_info(composed)
        if info["peak_width"] > args.max_width:
            raise ValueError("A fresh continuation exceeds the execution width budget")
        program = clifft.lower(hir)
        lowered = perf_counter()
        sample = clifft.sample(program, shots=1, seed=rng.getrandbits(64), threads=1)
        end = perf_counter()
        for stage, duration in zip(
            stages, (drawn - start, rendered - drawn, lowered - rendered, end - lowered)
        ):
            stages[stage] += duration
        times.append(end - start)
        widths[info["peak_width"]] += 1
        sample_parities(source, sample)
        arrays.append(sample.measurements[0].copy())
        histories.append(history)
    result.update(
        status="sampled",
        shots=args.shots,
        mean_seconds_per_shot=statistics.mean(times),
        stage_seconds=stages,
        widths=dict(widths),
        unique_histories=len(set(histories)),
        fault_counts=dict(sorted(Counter(map(len, histories)).items())),
        record_sha256=hashlib.sha256(np.stack(arrays).tobytes()).hexdigest(),
        history_sha256=hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=739122)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.shots < 1 or not 0 <= args.max_width <= 16:
        parser.error("Require positive shots and a width budget between zero and sixteen")
    sources = panel(args.merlin_checkout)
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
                "shared_phase_specialization.py",
                "regional_phase_specialization.py",
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
            "ordinary width",
            row["ordinary"]["peak_width"],
            "region width",
            max(x["region"]["peak_width"] for x in row["stress"]),
            "complete width",
            max(x["complete"]["peak_width"] for x in row["stress"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
