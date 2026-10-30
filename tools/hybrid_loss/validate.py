"""Independent distribution checks for the experimental online-coordinate executor."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from experiment import DEFAULT_MODEL, ROOT, run


def check_distribution(records, expected, label):
    counts = Counter(map(tuple, records.tolist()))
    shots = len(records)
    max_z = 0.0
    tvd = 0.0
    for key in counts.keys() | expected.keys():
        p = expected.get(key, 0.0)
        observed = counts.get(key, 0) / shots
        if p < 1e-12:
            assert counts.get(key, 0) == 0, (label, "impossible outcome", key, counts[key])
        sigma = math.sqrt(max(0, p * (1 - p)) / shots)
        delta = abs(observed - p)
        assert delta <= 7 * sigma + 2 / shots, (label, key, observed, p)
        if sigma > 0:
            max_z = max(max_z, delta / sigma)
        tvd += delta / 2
    return {"case": label, "shots": shots, "max_z": max_z, "tvd": tvd}


def run_validation(binary, output):
    import stim
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    output.mkdir(parents=True, exist_ok=True)
    results = []
    rng = np.random.default_rng(713)
    aer = AerSimulator(method="statevector", max_parallel_threads=1)
    # Independent exact unitary probabilities exercise inverse-frame signs and
    # active-coordinate promotion/compaction across several physical bases.
    for case in range(40):
        n = 1 + case % 5
        qc = QuantumCircuit(n)
        lines = []
        names = ["H", "S", "S_DAG", "T", "T_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG"]
        methods = [qc.h, qc.s, qc.sdg, qc.t, qc.tdg, qc.x, qc.y, qc.z, qc.sx, qc.sxdg]
        for layer in range(45):
            if n > 1 and rng.random() < 0.35:
                a, b = map(int, rng.choice(n, 2, replace=False))
                name = rng.choice(["CX", "CZ", "SWAP"])
                getattr(qc, name.lower())(a, b)
                lines.append(f"{name} {a} {b}")
            elif rng.random() < 0.15:
                q = int(rng.integers(n))
                angle = float(rng.uniform(-1, 1))
                axis = rng.choice(["X", "Y", "Z"])
                getattr(qc, "r" + axis.lower())(angle * np.pi, q)
                lines.append(f"R_{axis}({angle}) {q}")
            else:
                index = int(rng.integers(len(names)))
                q = int(rng.integers(n))
                methods[index](q)
                lines.append(f"{names[index]} {q}")
        qc.save_statevector()
        state = np.asarray(aer.run(qc).result().get_statevector())
        expected = {
            tuple((i >> q) & 1 for q in range(n)): float(abs(v) ** 2) for i, v in enumerate(state)
        }
        lines.append("M " + " ".join(map(str, range(n))))
        _, sampled = run(
            binary, output / f"aer-{case}", "\n".join(lines), DEFAULT_MODEL, 30000, seed=case + 100
        )
        results.append(check_distribution(sampled["measurements"], expected, f"aer-{case}"))
    print("Aer unitary checks passed", flush=True)

    # Stim supplies an independent stabilizer reference with intermediate collapse,
    # qubit reuse, noisy gates, and classical feedback.
    for case in range(20):
        n = 5
        lines = []
        for layer in range(25):
            a, b = map(int, rng.choice(n, 2, replace=False))
            lines.extend([f"H {a}", f"S {b}", f"CX {a} {b}", f"DEPOLARIZE2(0.03) {a} {b}"])
            if layer in (6, 13):
                lines.extend([f"MR {a}", f"CX rec[-1] {b}"])
        lines.append("M 0 1 2 3 4")
        text = "\n".join(lines)
        reference = stim.Circuit(text).compile_sampler(seed=case).sample(100000)
        _, sampled = run(
            binary, output / f"stim-{case}", text, DEFAULT_MODEL, 100000, seed=case + 200
        )
        # Two independent samples; use pooled frequencies and two-sample variance.
        counts = Counter(map(tuple, reference.astype(int).tolist()))
        expected = {k: v / len(reference) for k, v in counts.items()}
        observed = Counter(map(tuple, sampled["measurements"].tolist()))
        maximum = 0.0
        for key in counts.keys() | observed.keys():
            left, right = counts.get(key, 0) / 100000, observed.get(key, 0) / 100000
            p = (left + right) / 2
            sigma = math.sqrt(2 * p * (1 - p) / 100000)
            assert abs(left - right) < 7 * sigma + 2e-5, (case, key, left, right)
            if sigma:
                maximum = max(maximum, abs(left - right) / sigma)
        results.append({"case": f"stim-{case}", "max_z": maximum, "shots": 100000})
    print("Stim noisy feedback checks passed", flush=True)

    sys.path.insert(0, str(ROOT / "tests/python"))
    import utils_noncomp_enumerator as exact

    classifier = [[1, 0, 1, 0, 0.5], [0, 1, 0, 1, 0.5]]
    matrix = np.zeros((5, 5))
    matrix[2, 0] = 0.08
    matrix[3, 1] = 0.37
    matrix[1, 2] = 0.4
    matrix[0, 3] = 0.2
    matrix[4, 1] = 0.1
    matrix[0, 0] = 0.03
    for case in range(24):
        n = 2 + case % 2
        lines = ["H 0", "CX 0 1", "T 0", "H 1", "S 1", "T_DAG 1"]
        for layer in range(5):
            a, b = map(int, rng.choice(n, 2, replace=False))
            lines += [f"CX {a} {b}", f"T {b}", f"H {a}"]
        lines += ["LEVEL_TRANSITION[decay] 0", "LEAKAGE(0.17) 1", "CX 0 1", "T 1", "H 0"]
        if case % 3 == 0:
            lines += ["M 0", "R 0", "H 0", "CX 0 1", "T_DAG 1"]
        if case % 3 == 1:
            lines += ["LOSS(0.23) 1", "CX 0 1", "H 0 1"]
        lines += ["LEVEL_TRANSITION[decay] 0 1", "H 0 1", "M " + " ".join(map(str, range(n)))]
        text = "\n".join(lines)
        model: dict[str, Any] = {
            "initial_state": [1, 0, 0, 0, 0],
            "transitions": {"decay": matrix.tolist()},
            "classifier": classifier,
            "damping": "neglect" if case % 2 else "exact",
        }
        reference = exact.enumerate_exact(
            text,
            initial=model["initial_state"],
            transitions=model["transitions"],
            classifier=classifier,
            damping=model["damping"],
        )
        assert reference.dropped_mass < 1e-8
        _, sampled = run(binary, output / f"kraus-{case}", text, model, 60000, seed=case + 300)
        results.append(
            check_distribution(sampled["measurements"], reference.record_probs, f"kraus-{case}")
        )
        for q, distribution in enumerate(reference.noncomp_level_probs):
            for level, value in distribution.items():
                observed = (sampled["status"][:, q] == level - 1).mean()
                assert (
                    abs(observed - value) < 7 * math.sqrt(value * (1 - value) / 60000) + 2 / 60000
                )
    print("Exact Kraus joint-distribution checks passed", flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {3})
    results = run_validation(args.binary, args.output)
    print(f"Passed {len(results)} independent distribution checks")
