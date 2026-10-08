"""Predeclared equivalence margins for three complete BT27 sampling rates.

Exact binomial intervals and a union bound cover all six underlying rates.
Containment of each difference interval establishes agreement to its declared
absolute tolerance, not equality of the complete output distributions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import queue
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from study_bt27_scored_sampling import converter_for, output_summary, sources

beta = importlib.import_module("scipy.stats").beta

METRICS = {"acceptance": 0.003, "undetected_per_shot": 0.001, "undetected_given_acceptance": 0.03}
ALPHA = 0.05
INTERVAL_ALPHA = ALPHA / (2 * len(METRICS))


def interval(k: Any, n: Any) -> tuple[Any, Any]:
    k, n = np.asarray(k), np.asarray(n)
    low = np.where(k == 0, 0.0, beta.ppf(INTERVAL_ALPHA / 2, k, n - k + 1))
    high = np.where(k == n, 1.0, beta.ppf(1 - INTERVAL_ALPHA / 2, k + 1, n - k))
    return low, high


def counts(row: dict[str, Any]) -> dict[str, tuple[int, int]]:
    accepted = row["zero_syndrome_shots"]
    errors = row["zero_syndrome_shots_with_logical_events"]
    return {
        "acceptance": (accepted, row["shots"]),
        "undetected_per_shot": (errors, row["shots"]),
        "undetected_given_acceptance": (errors, accepted),
    }


def planning(shots: int) -> dict[str, Any]:
    # Pilot counts choose the budget only. They are not pooled into the test.
    acceptance, joint = 262 / 8192, 26 / 8192
    rng = np.random.default_rng(817331)
    categories = [joint, acceptance - joint, 1 - acceptance]
    a, b = (rng.multinomial(shots, categories, size=10000) for _ in range(2))
    passed = np.ones(len(a), dtype=bool)
    powers = {}
    for name, margin in METRICS.items():
        if name == "acceptance":
            ka, kb = a[:, :2].sum(axis=1), b[:, :2].sum(axis=1)
            na = nb = shots
        else:
            ka, kb = a[:, 0], b[:, 0]
            na = a[:, :2].sum(axis=1) if name.endswith("acceptance") else shots
            nb = b[:, :2].sum(axis=1) if name.endswith("acceptance") else shots
        la, ua = interval(ka, na)
        lb, ub = interval(kb, nb)
        ok = ((la - ub) > -margin) & ((ua - lb) < margin)
        passed &= ok
        powers[name] = float(ok.mean())
    return {
        "pilot_acceptance": acceptance,
        "pilot_undetected_per_shot": joint,
        "hypothesis_for_power_only": "Both simulators have the pooled pilot rates",
        "simulated_trials": len(a),
        "seed": 817331,
        "estimated_power_by_metric": powers,
        "estimated_power_all_metrics": float(passed.mean()),
    }


def assess(workers: dict[str, Any]) -> dict[str, Any]:
    totals = {}
    for backend in ("prototype", "merlin"):
        rows = [r["outputs"] for k, r in workers.items() if k.startswith(backend + "_")]
        totals[backend] = {
            field: sum(row[field] for row in rows)
            for field in (
                "shots",
                "zero_syndrome_shots",
                "zero_syndrome_shots_with_logical_events",
            )
        }
    a, b = counts(totals["prototype"]), counts(totals["merlin"])
    result = {}
    for name, margin in METRICS.items():
        la, ua = map(float, interval(*a[name]))
        lb, ub = map(float, interval(*b[name]))
        difference = [la - ub, ua - lb]
        result[name] = {
            "prototype_count_and_denominator": a[name],
            "merlin_count_and_denominator": b[name],
            "prototype_rate": a[name][0] / a[name][1] if a[name][1] else None,
            "merlin_rate": b[name][0] / b[name][1] if b[name][1] else None,
            "prototype_interval": [la, ua],
            "merlin_interval": [lb, ub],
            "difference_interval": difference,
            "absolute_margin": margin,
            "equivalent_within_margin": difference[0] > -margin and difference[1] < margin,
        }
    return {
        "totals": totals,
        "metrics": result,
        "all_margins_met": all(r["equivalent_within_margin"] for r in result.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--chunk-shots", type=int, default=32768)
    args = parser.parse_args()
    if min(args.jobs, args.chunks, args.chunk_shots) < 1:
        parser.error("Positive jobs, chunks, and chunk sizes required")
    _, _, source = sources(args.merlin_checkout, "gate_noise")
    converter = converter_for(source)
    protocol = {
        "model": "gate_noise",
        "shots_per_backend": args.chunks * args.chunk_shots,
        "chunks_per_backend": args.chunks,
        "shots_per_chunk": args.chunk_shots,
        "absolute_equivalence_margins": METRICS,
        "family_confidence_at_least": 1 - ALPHA,
        "individual_interval_confidence": 1 - INTERVAL_ALPHA,
        "interval_method": "Two-sided Clopper-Pearson, Bonferroni over six rates",
        "decision": "All three difference intervals strictly inside their declared margins",
        "stopping": "Fixed budget; no interim tests or extension based on outcomes",
        "prototype_seeds": [872001 + i * 101 for i in range(args.chunks)],
        "merlin_seeds": [937901 + i * 103 for i in range(args.chunks)],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "scope": "Three specified rates at p=0.001, not total-distribution equivalence",
        "reference": "https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm",
        "power_planning": planning(args.chunks * args.chunk_shots),
    }
    document: dict[str, Any] = {"protocol": protocol, "workers": {}}
    if args.output.exists():
        document = json.loads(args.output.read_text())
        if document["protocol"] != protocol:
            raise ValueError("Existing declared protocol differs; do not change it after sampling")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    if args.plan_only:
        print(json.dumps(protocol, indent=2))
        return
    cpus: queue.Queue[int] = queue.Queue()
    available = sorted(os.sched_getaffinity(0))
    selected = (available[1:] or available)[: args.jobs]
    for cpu in selected:
        cpus.put(cpu)
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="bt27-rate-equivalence-") as temp:

        def run(backend: str, i: int) -> tuple[str, dict[str, Any]]:
            from types import SimpleNamespace

            name = f"{backend}_{i}"
            path = Path(temp) / (name + ".npz")
            seed = protocol[backend + "_seeds"][i]
            cpu = cpus.get()
            try:
                command = [
                    "taskset",
                    "-c",
                    str(cpu),
                    sys.executable,
                    str(Path(__file__).with_name("study_bt27_scored_sampling.py")),
                    "--binary",
                    str(args.binary.resolve()),
                    "--merlin-checkout",
                    str(args.merlin_checkout.resolve()),
                    "--worker",
                    backend,
                    "--model",
                    "gate_noise",
                    "--shots",
                    str(args.chunk_shots),
                    "--seed",
                    str(seed),
                    "--samples",
                    str(path),
                ]
                row = json.loads(subprocess.check_output(command, text=True))
                with np.load(path) as arrays:
                    sample = SimpleNamespace(**{key: arrays[key] for key in arrays.files})
                detectors, observables = converter.convert(
                    measurements=sample.measurements.astype(bool), separate_observables=True
                )
                np.testing.assert_array_equal(sample.detectors, detectors)
                np.testing.assert_array_equal(sample.observables, observables)
                row["outputs"] = output_summary(sample, converter)
                if row["source_sha256"] != protocol["source_sha256"]:
                    raise AssertionError("Stochastic source changed")
                row["seed"], row["cpu"] = seed, cpu
                path.unlink()
                return name, row
            finally:
                cpus.put(cpu)

        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            futures = [
                pool.submit(run, backend, i)
                for i in range(args.chunks)
                for backend in ("prototype", "merlin")
                if f"{backend}_{i}" not in document["workers"]
            ]
            for future in as_completed(futures):
                name, row = future.result()
                document["workers"][name] = row
                args.output.write_text(json.dumps(document, indent=2) + "\n")
                print(f"Completed {name}: {args.chunk_shots} fresh shots", flush=True)
    document["assessment"] = assess(document["workers"])
    document["last_invocation_seconds"] = perf_counter() - started
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document["assessment"], indent=2))


if __name__ == "__main__":
    main()
