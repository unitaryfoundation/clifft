"""Compare static, sampled-fault, and sampled-prefix algebraic optimization.

The panel is selected here; the specialization implementation receives only
raw circuit text. Every fresh shot recompiles, with no history cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from automatic_specialization import FaultModel, History, analyze, specialize_prefix
from profile_external_feedback import MERLIN_REVISION

import clifft

ROOT = Path(__file__).resolve().parents[2]


def measurement_control(*, noise: bool, feedback: bool = False) -> str:
    lines = ["H 0 1"]
    if noise:
        lines.append("PAULI_CHANNEL_1(0.013,0.017,0.023) 0")
    lines.append("MPP(0.031) Z0*Z1" if noise else "MPP Z0*Z1")
    lines += ["T 0", "T_DAG 1"]
    if feedback:
        lines.append("CX rec[-1] 0")
    lines += [
        "H 0 1",
        "M 0 1",
        "DETECTOR rec[-1] rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-3]",
    ]
    return "\n".join(lines) + "\n"


def candidates(checkout: Path) -> dict[str, tuple[str, str]]:
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MERLIN_REVISION or subprocess.check_output(
        ["git", "-C", str(checkout), "diff", "HEAD", "--", "benchmarks", "experiments/cultivation"],
        text=True,
    ):
        raise ValueError("Expected an unchanged pinned Merlin generator")
    sys.path.insert(0, str(checkout.resolve()))
    distillation = importlib.import_module("benchmarks.protocols.distillation")
    cultivation = importlib.import_module("benchmarks.protocols.cultivation")
    switching = importlib.import_module("benchmarks.protocols.code_switching")
    from study_circuit_noise import synthetic_bt_noise

    panel = {}
    for protocol, logical in (("15to1", (0,)), ("bh", (0, 1))):
        for scored in (False, True):
            source = distillation.build_distillation_case(
                protocol, p_phys=0.001, noisy_clifford=True, target_scoring=scored
            ).circuit
            if not scored:
                # An unmeasured logical state would not test residual sampling work.
                for index, qubit in enumerate(logical):
                    source += f"MX {qubit}\nOBSERVABLE_INCLUDE({index}) rec[-1]\n"
            name = protocol + ("_scored" if scored else "_direct_x")
            panel[name] = ("distillation", source)
    for distance in (3, 5):
        panel[f"cultivation_d{distance}"] = (
            "cultivation_holdout" if distance == 5 else "cultivation",
            cultivation.build_cultivation_case(distance).circuit,
        )
    for size in (27, 81):
        protocol = switching.build_code_switching_case(
            f"bt{size}", p_phys=0, target_scoring=False
        ).circuit
        scored_source = switching.build_code_switching_case(
            f"bt{size}", p_phys=0, target_scoring=True
        ).circuit
        if not scored_source.startswith(protocol):
            raise AssertionError("Verification tail is not an extension")
        noisy = synthetic_bt_noise(protocol, {"prep", "phase", "decoder"}, 0.001)
        tail = scored_source[len(protocol) :]
        direct_readout = (
            "\n".join(
                line for line in tail.splitlines() if line.startswith(("MX ", "OBSERVABLE_INCLUDE"))
            )
            + "\n"
        )
        panel[f"bt{size}_scored"] = ("existing_bt_reference", noisy + tail)
        panel[f"bt{size}_direct_x"] = ("existing_bt_reference", noisy + direct_readout)
    panel["measured_coset"] = ("measurement_control", measurement_control(noise=True))
    panel["measured_feedback"] = (
        "measurement_control",
        measurement_control(noise=True, feedback=True),
    )
    rng = random.Random(271828)
    lines = ["H 0 1 2 3"]
    for _ in range(6):
        for q in range(4):
            lines += [f"T {q}", f"H {q}", f"DEPOLARIZE1(0.001) {q}"]
        a, b = rng.sample(range(4), 2)
        lines += [f"CX {a} {b}", f"DEPOLARIZE2(0.001) {a} {b}"]
    panel["noncommuting_control"] = ("negative_control", "\n".join(lines + ["M 0 1 2 3"]))
    return panel


def stress_histories(model: FaultModel) -> list[History]:
    if not model.sites:
        return [()]
    positions = sorted({0, len(model.sites) // 3, 2 * len(model.sites) // 3, len(model.sites) - 1})
    result: list[History] = [()]
    for site in positions:
        outcomes = len(model.sites[site].replacements)
        for outcome in sorted({1, (outcomes - 1) // 2 or 1, outcomes - 1}):
            result.append(((site, outcome),))
    rng = random.Random(7351)
    for weight in (2, 5, 12):
        result.append(
            tuple(
                (site, rng.randrange(1, len(model.sites[site].replacements)))
                for site in sorted(
                    rng.sample(range(len(model.sites)), min(weight, len(model.sites)))
                )
            )
        )
    return list(dict.fromkeys(result))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "t_counts": dict(sorted(Counter(r["output_t"] for r in rows).items())),
        "widths": dict(sorted(Counter(r["peak_width"] for r in rows).items())),
        "any_capped": any(r["phase"]["blocks_capped"] for r in rows),
    }


def sample_parities(source: str, sample: Any) -> None:
    parsed = clifft.parse(source)
    detectors = []
    observables = np.zeros_like(sample.observables)
    for node in parsed.nodes:
        if node.gate.name not in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        value = np.zeros(len(sample.measurements), dtype="u1")
        for target in node.targets:
            if not target.is_rec:
                raise AssertionError("Expected a record parity")
            value ^= sample.measurements[:, target.value]
        if node.gate.name == "DETECTOR":
            detectors.append(value)
        else:
            observables[:, int(node.arg)] ^= value
    expected = np.column_stack(detectors) if detectors else np.empty((len(sample.measurements), 0))
    np.testing.assert_array_equal(expected, sample.detectors)
    np.testing.assert_array_equal(observables, sample.observables)


def fresh_shots(
    model: FaultModel, *, mode: str, shots: int, seed: int, max_width: int
) -> dict[str, Any]:
    faults, measurements = random.Random(seed), random.Random(seed ^ 0x19449)
    rows, histories, records = [], [], []
    totals = dict.fromkeys(("fault_draw", "render", "prefix", "analyze", "lower", "sample"), 0.0)
    elapsed = 0.0
    for _ in range(shots):
        started = perf_counter()
        history = model.draw(faults)
        drawn = perf_counter()
        text = model.render(history)
        rendered = perf_counter()
        branch = None
        if mode == "fault_and_prefix":
            text, branch = specialize_prefix(text, measurements.getrandbits(64))
        specialized = perf_counter()
        hir, info = analyze(text)
        inspected = perf_counter()
        if info["peak_width"] > max_width:
            return {"status": "width_budget", "first_case": info, "history": history}
        program = clifft.lower(hir)
        lowered = perf_counter()
        sample = clifft.sample(program, shots=1, seed=measurements.getrandbits(64), threads=1)
        finished = perf_counter()
        elapsed += finished - started
        for field, duration in (
            ("fault_draw", drawn - started),
            ("render", rendered - drawn),
            ("prefix", specialized - rendered),
            ("analyze", inspected - specialized),
            ("lower", lowered - inspected),
            ("sample", finished - lowered),
        ):
            totals[field] += duration
        # Output parity checks and report construction are excluded from timing.
        sample_parities(model.source, sample)
        if branch and branch["status"] == "specialized":
            np.testing.assert_array_equal(
                sample.measurements[0, : branch["records"]], branch["outcomes"]
            )
        rows.append(info)
        histories.append(history)
        records.append(sample.measurements[0].tolist())
    return {
        "status": "sampled",
        "shots": shots,
        "seconds_per_shot": elapsed / shots,
        "stage_seconds": dict(totals),
        "summary": summarize(rows),
        "fault_counts": dict(sorted(Counter(map(len, histories)).items())),
        "distinct_histories": len(set(histories)),
        "history_sha256": hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
        "record_sha256": hashlib.sha256(json.dumps(records).encode()).hexdigest(),
        "output_parities": "passed",
    }


def study_case(name: str, source: str, args: Any) -> dict[str, Any]:
    started = perf_counter()
    model = FaultModel(source)
    model_seconds = perf_counter() - started
    result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "qubits": model.num_qubits,
        "records": model.num_records,
        "noise_sites": len(model.sites),
        "expected_faults": sum(1 - s.probabilities[0] for s in model.sites),
        "model_seconds": model_seconds,
    }
    for label, text in (("static_noisy", source), ("static_ideal", model.render(()))):
        started = perf_counter()
        hir, info = analyze(text)
        if info["peak_width"] <= args.max_width:
            program = clifft.lower(hir)
            info["compile_seconds"] = perf_counter() - started
            clifft.sample(program, shots=1, seed=107, threads=1)
            batch_times = {}
            for count in sorted({1, args.shots, args.batch_shots}):
                times = []
                for repeat in range(3):
                    start_sample = perf_counter()
                    sample = clifft.sample(program, shots=count, seed=109 + repeat, threads=1)
                    times.append(perf_counter() - start_sample)
                batch_times[str(count)] = statistics.median(times)
            sample_parities(source, sample)
            info["batch_seconds"] = batch_times
            info["sample_seconds_per_shot"] = batch_times[str(args.batch_shots)] / args.batch_shots
            info["batch_shots"] = args.batch_shots
            info["compile_and_batch_seconds"] = {
                count: info["compile_seconds"] + value for count, value in batch_times.items()
            }
        else:
            info["execution"] = "width_budget"
        result[label] = info
    stress: list[dict[str, Any]] = []
    for history in stress_histories(model):
        text = model.render(history)
        _, fixed = analyze(text)
        conditional, branch = specialize_prefix(text, 8317)
        _, measured = analyze(conditional)
        stress.append(
            {"history": history, "fault": fixed, "fault_and_prefix": measured, "prefix": branch}
        )
    result["stress"] = stress
    result["stress_summary"] = {
        mode: summarize([r[mode] for r in stress]) for mode in ("fault", "fault_and_prefix")
    }
    result["fresh"] = {
        mode: fresh_shots(
            model, mode=mode, shots=args.shots, seed=args.seed, max_width=args.max_width
        )
        for mode in ("fault", "fault_and_prefix")
    }
    # This compares the same complete source where Merlin accepts its readouts.
    merlin = importlib.import_module("merlin")

    try:
        before = perf_counter()
        sampler = merlin.CircuitSampler(source, seed=args.seed + 19)
        setup = perf_counter() - before
        before = perf_counter()
        merlin_sample = sampler.sample(args.shots)
        duration = perf_counter() - before
        result["merlin"] = {
            "status": "sampled",
            "setup_seconds": setup,
            "seconds_per_shot": duration / args.shots,
            "records": list(merlin_sample.measurements.shape),
        }
    except (ValueError, RuntimeError, merlin.MeasurementError) as error:
        result["merlin"] = {"status": "unsupported", "reason": str(error)}
    print(
        json.dumps(
            {
                "case": name,
                "stress": result["stress_summary"],
                "fresh": {
                    mode: {k: v for k, v in info.items() if k in {"status", "seconds_per_shot"}}
                    for mode, info in result["fresh"].items()
                },
            }
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--batch-shots", type=int, default=1024)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=791929)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.shots < 1 or args.batch_shots < 1 or not 0 <= args.max_width <= 20:
        parser.error("Shot counts must be positive and the width budget must be in [0, 20]")
    panel = candidates(args.merlin_checkout)
    selected = args.cases or list(panel)
    if set(selected) - panel.keys():
        parser.error("Unknown panel case")
    result: dict[str, Any] = {
        "study": "existing algebra with sampled faults and initial-prefix outcomes",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "merlin_revision": MERLIN_REVISION,
        "versions": {p: importlib.metadata.version(p) for p in ("clifft", "stim", "qiskit-aer")},
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("automatic_specialization.py"))
        },
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": {},
    }
    for name in selected:
        family, source = panel[name]
        result["cases"][name] = {"family": family, **study_case(name, source, args)}
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
