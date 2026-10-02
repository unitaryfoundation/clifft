"""Benchmark the native phase pass through normal Clifft compilation and sampling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import clifft


def _manager(phase: Any | None) -> Any:
    manager = clifft.HirPassManager()
    manager.add(clifft.PeepholeFusionPass())
    if phase is not None:
        manager.add(phase)
    manager.add(clifft.StatevectorSqueezePass())
    manager.add(clifft.ActiveWidthSchedulePass())
    return manager


def _pair(source: str, mask: list[int]) -> tuple[list[Any], dict[str, Any]]:
    programs = []
    times = []
    phase = clifft.PhasePolynomialPass()
    for manager in (_manager(None), _manager(phase)):
        start = time.perf_counter()
        program = clifft.compile(
            source, postselection_mask=mask, hir_passes=manager, normalize_syndromes=False
        )
        times.append(time.perf_counter() - start)
        programs.append(program)
    for field in ("num_qubits", "num_measurements", "num_detectors", "num_observables"):
        assert getattr(programs[0], field) == getattr(programs[1], field), field
    np.testing.assert_array_equal(
        programs[0].noise_site_probabilities, programs[1].noise_site_probabilities
    )
    return programs, dict(
        applied=phase.applied,
        blocks_examined=phase.blocks_examined,
        blocks_reduced=phase.blocks_reduced,
        oversized_blocks=phase.oversized_blocks,
        expansion_rejections=phase.expansion_rejections,
        input_t=phase.input_t_count,
        output_t=phase.output_t_count,
        before_phase_peak=phase.incumbent_peak,
        after_phase_peak=phase.result_peak,
        baseline_peak=programs[0].peak_active_width,
        native_peak=programs[1].peak_active_width,
        baseline_actions=programs[0].num_actions,
        native_actions=programs[1].num_actions,
        compile_seconds=times,
        noise_sites=len(programs[0].noise_site_probabilities),
    )


def _timings(programs: list[Any], target: float, repeats: int) -> dict[str, Any]:
    counts = []
    for program in programs:
        start = time.perf_counter()
        clifft.sample_survivors(program, shots=1, seed=20261001, threads=1, batch_size=1)
        elapsed = time.perf_counter() - start
        shots = min(200000, max(1, math.ceil(target / max(elapsed, 1e-9))))
        start = time.perf_counter()
        clifft.sample_survivors(program, shots=shots, seed=20261001, threads=1, batch_size=1)
        elapsed = time.perf_counter() - start
        counts.append(min(200000, max(1, math.ceil(shots * target / max(elapsed, 1e-9)))))
    times: list[list[float]] = [[], []]
    outcomes: list[list[dict[str, Any]]] = [[], []]
    for repeat in range(repeats):
        for arm in [0, 1] if repeat % 2 == 0 else [1, 0]:
            start = time.perf_counter()
            result = clifft.sample_survivors(
                programs[arm], shots=counts[arm], seed=20261001 + repeat, threads=1, batch_size=1
            )
            times[arm].append(time.perf_counter() - start)
            outcomes[arm].append(
                dict(
                    total=result.total_shots,
                    passed=result.passed_shots,
                    observable_ones=result.observable_ones.tolist(),
                )
            )
    seconds_per_shot = [statistics.median(values) / count for values, count in zip(times, counts)]
    return dict(
        shots=counts,
        timings_seconds=times,
        seconds_per_shot=seconds_per_shot,
        speedup=seconds_per_shot[0] / seconds_per_shot[1],
        counts=outcomes,
    )


def _noisy(source: str, kind: str, n: int) -> str:
    if kind == "post_layer":
        lines = source.splitlines()
        last_t = max(j for j, line in enumerate(lines) if line.startswith(("T ", "T_DAG ")))
        lines.insert(last_t + 1, "DEPOLARIZE1(0.001) " + " ".join(map(str, range(n))))
        return "\n".join(lines)
    lines = []
    for line in source.splitlines():
        words = line.split()
        if not words or words[0] not in ("H", "S", "S_DAG", "T", "T_DAG", "Z", "CX", "CZ", "SWAP"):
            lines.append(line)
            continue
        gate = words[0]
        arity = 2 if gate in ("CX", "CZ", "SWAP") else 1
        for index in range(1, len(words), arity):
            targets = words[index : index + arity]
            lines.append(gate + " " + " ".join(targets))
            if kind == "gate_z":
                lines.append("Z_ERROR(0.001) " + " ".join(targets))
            else:
                lines.append(f"DEPOLARIZE{arity}(0.001) " + " ".join(targets))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--existing-bench", type=Path)
    parser.add_argument("--sample-limit", type=int, default=64)
    parser.add_argument("--target-seconds", type=float, default=0.03)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = list(csv.DictReader((args.corpus / "circuits.csv").open()))
    rows = []
    start = time.perf_counter()
    for item in inputs:
        _, stats = _pair(
            (args.corpus / item["circuit"]).read_text(), [1] * int(item["postselection_detectors"])
        )
        rows.append(dict(item, **stats))
        if len(rows) % 100 == 0:
            print(
                f"native scan {len(rows)}/{len(inputs)} in {time.perf_counter() - start:.1f}s",
                flush=True,
            )
    (args.output / "scan.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("scan done", flush=True)
    summary = dict(
        version=clifft.__version__,
        isa=clifft.runtime_isa(),
        platform=platform.platform(),
        baseline="PeepholeFusionPass, StatevectorSqueezePass, ActiveWidthSchedulePass",
        native=(
            "PeepholeFusionPass, PhasePolynomialPass, "
            "StatevectorSqueezePass, ActiveWidthSchedulePass"
        ),
        threads=1,
        batch_size=1,
        circuits=len(rows),
        applied=sum(row["applied"] for row in rows),
        lower_peak=sum(row["native_peak"] < row["baseline_peak"] for row in rows),
        higher_peak=sum(row["native_peak"] > row["baseline_peak"] for row in rows),
        baseline_peak_histogram=dict(sorted(Counter(row["baseline_peak"] for row in rows).items())),
        native_peak_histogram=dict(sorted(Counter(row["native_peak"] for row in rows).items())),
        input_t=sum(row["input_t"] for row in rows),
        output_t=sum(row["output_t"] for row in rows),
        median_compile_ms=[
            1000 * statistics.median(row["compile_seconds"][arm] for row in rows) for arm in (0, 1)
        ],
        maximum_native_compile_seconds=max(row["compile_seconds"][1] for row in rows),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    selected = random.Random(20261001).sample(rows, min(args.sample_limit, len(rows)))
    for number in (5, 262, 273, 300, 689, 1037):
        row = rows[number - 1]
        if row not in selected:
            selected.append(row)
    sampled = []
    with (args.output / "sampling.jsonl").open("w") as stream:
        for row in selected:
            programs, stats = _pair(
                (args.corpus / row["circuit"]).read_text(),
                [1] * int(row["postselection_detectors"]),
            )
            result = dict(
                circuit=row["circuit"],
                **stats,
                **_timings(programs, args.target_seconds, args.repeats),
            )
            sampled.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            print(
                f"native sample {len(sampled)}/{len(selected)} {row['circuit']} "
                f"speedup={result['speedup']:.1f}",
                flush=True,
            )
    summary.update(
        sampled=len(sampled),
        target_seconds=args.target_seconds,
        repeats=args.repeats,
        median_sampled_speedup=statistics.median(row["speedup"] for row in sampled),
        equal_shots_sampled_speedup=sum(row["seconds_per_shot"][0] for row in sampled)
        / sum(row["seconds_per_shot"][1] for row in sampled),
        slower_over_ten_percent=sum(row["speedup"] < 1 / 1.1 for row in sampled),
    )
    noisy = []
    for number in (5, 262, 300, 689, 1037):
        row = rows[number - 1]
        source = (args.corpus / row["circuit"]).read_text()
        n = clifft.parse(source).num_qubits
        for kind in ("post_layer", "gate_z", "gate_depolarizing"):
            programs, stats = _pair(
                _noisy(source, kind, n), [1] * int(row["postselection_detectors"])
            )
            result = dict(
                circuit=row["circuit"],
                noise_model=kind,
                **stats,
                **_timings(programs, args.target_seconds, args.repeats),
            )
            noisy.append(result)
            print(
                f"native noise {number} {kind} {stats['baseline_peak']} -> {stats['native_peak']} "
                f"speedup={result['speedup']:.2f}",
                flush=True,
            )
    (args.output / "noise.json").write_text(json.dumps(noisy, indent=2) + "\n")
    existing = []
    if args.existing_bench:
        manifest = json.loads((args.existing_bench / "manifests/workloads.v1.json").read_text())
        for workload in manifest["workloads"]:
            source = (args.existing_bench / "manifests" / workload["artifact"]["path"]).read_text()
            count = (
                workload["expected_metadata"]["num_detectors"]
                if workload["semantics"]["postselect_all_detectors"]
                else 0
            )
            programs, stats = _pair(source, [1] * count)
            result = dict(
                workload=workload["id"],
                **stats,
                **_timings(programs, args.target_seconds, args.repeats),
            )
            existing.append(result)
            print(
                f"native existing {workload['id']} "
                f"{stats['baseline_peak']} -> {stats['native_peak']} "
                f"speedup={result['speedup']:.2f}",
                flush=True,
            )
    (args.output / "existing.json").write_text(json.dumps(existing, indent=2) + "\n")
    summary["noise_models"] = " ".join(
        (
            "Post-T layer: depolarizing on all qubits.",
            "Gate Z: independent dephasing on each target after each unitary.",
            "Gate depolarizing: 1q or 2q depolarizing after each unitary. Measurements are ideal.",
        )
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
