"""Compare representative joint statistics against the established trajectory API."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from experiment import ROOT, module, python_model, run


def feature_counts(arrays, seed=341):
    m, h = arrays["measurements"], arrays["heralds"]
    symbols = np.where(h, 2, m)
    # Loss placeholders are excluded from symbol parity; their correlations with
    # loss counts and final status are tested separately.
    bits = np.where(h, 0, m)
    counts: dict[str, Counter[Any]] = {}
    for q in range(symbols.shape[1]):
        counts[f"symbol:{q}"] = Counter(map(int, symbols[:, q]))
    for key in ("detectors", "observables", "status"):
        for q in range(arrays[key].shape[1]):
            counts[f"{key}:{q}"] = Counter(map(int, arrays[key][:, q]))
    rng = np.random.default_rng(seed)
    for i in range(24):
        selected = rng.choice(bits.shape[1], min(bits.shape[1], 2 + i % 7), replace=False)
        parity = np.bitwise_xor.reduce(bits[:, selected], axis=1)
        loss_count = np.minimum(h[:, selected].sum(axis=1), 2)
        counts[f"joint:{i}"] = Counter(
            map(tuple, np.column_stack([parity, loss_count, symbols[:, 0]]).tolist())
        )
    counts["herald_count"] = Counter(map(int, np.minimum(h.sum(axis=1), 4)))
    if arrays["detectors"].shape[1]:
        rejected = arrays["detectors"].any(axis=1)
        logical = (
            arrays["observables"][:, 0]
            if arrays["observables"].shape[1]
            else np.zeros(len(m), dtype=int)
        )
        counts["accept-logical-herald"] = Counter(
            map(tuple, np.column_stack([~rejected, logical, h.any(axis=1)]).tolist())
        )
    return counts


def compare_features(left, right, nl, nr):
    max_z = 0
    failures = []
    for name in left:
        for value in left[name].keys() | right[name].keys():
            a, b = left[name].get(value, 0) / nl, right[name].get(value, 0) / nr
            p = (a * nl + b * nr) / (nl + nr)
            sigma = math.sqrt(p * (1 - p) * (1 / nl + 1 / nr))
            z = abs(a - b) / sigma if sigma else 0
            max_z = max(max_z, z)
            if abs(a - b) > 7 * sigma + 2 / min(nl, nr):
                failures.append((name, value, a, b, z))
    return max_z, failures


def main():
    import clifft
    from clifft import noncomp

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {2})
    neutral = module(
        "neutral_compare", ROOT / "docs/guide/scripts/neutral_atom_leakage_tutorial.py"
    )
    cases = [
        ("surface_d5_loss0.001", 4096),
        ("cultivation_d3_loss0", 20000),
        ("cultivation_d3_loss0.001", 16000),
        ("cultivation_d5_loss0", 20000),
        ("cultivation_d5_loss0.001", 4096),
        ("cultivation_d3_depol_partner", 16000),
        ("neutral_two_row_matched", 12000),
        ("neutral_two_row_exact", 12000),
        ("neutral_two_row_ldu_matched", 12000),
        ("neutral_two_row_ldu_exact", 4096),
    ]
    summaries = []
    for i, (name, shots) in enumerate(cases):
        directory = args.output / name
        case = args.workloads / name
        text = (case / "circuit.stim").read_text()
        spec = json.loads((case / "model.json").read_text())
        info, arrays = run(
            args.binary, directory / "hybrid", text, spec, shots, capacity=20, seed=900 + i
        )
        program = noncomp.compile(
            text, python_model(spec), final_status=True, execution="trajectory"
        )
        start = time.perf_counter()
        sample = program.sample(shots, seed=1900 + i, threads=1, collect_stats=True)
        baseline_time = time.perf_counter() - start
        reference = {
            k: np.asarray(getattr(sample, "final_status" if k == "status" else k))
            for k in ("measurements", "heralds", "detectors", "observables", "status")
        }
        np.savez_compressed(directory / "reference.npz", **reference)
        left, right = feature_counts(arrays), feature_counts(reference)
        if name.startswith("neutral"):
            circuit_name = name.removeprefix("neutral_").rsplit("_", 1)[0]
            for features, values in [(left, arrays), (right, reference)]:
                symbols = np.where(values["heralds"], 2, values["measurements"])
                raw = Counter("".join(map(str, row)) for row in symbols.tolist())
                decoded = neutral.decode_counts(circuit_name, raw)
                features["decoded-with-reject"] = Counter(decoded)
                features["decoded-with-reject"]["rejected"] = shots - sum(decoded.values())
        maximum, failures = compare_features(left, right, shots, shots)
        assert sample.stats is not None
        summary = {
            "name": name,
            "reference_version": clifft.version(),
            "shots_per_backend": shots,
            "max_z": maximum,
            "failures": failures,
            "hybrid_us": info["us_per_shot"],
            "baseline_us": baseline_time * 1e6 / shots,
            "hybrid_peak_width": info["peak_width"],
            "baseline_peak_width": sample.stats.peak_active_width,
            "continuations": sample.stats.continuation_compilations,
        }
        for key, features in [("hybrid", left), ("reference", right)]:
            if "decoded-with-reject" in features:
                summary[key + "_decoded"] = dict(features["decoded-with-reject"])
        summaries.append(summary)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        assert not failures, name


if __name__ == "__main__":
    main()
