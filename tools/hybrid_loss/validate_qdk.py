"""Direct ternary-record checks on the explicitly matched loss subset."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from compare import compare_features
from experiment import DEFAULT_MODEL, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {3})
    os.environ["RAYON_NUM_THREADS"] = "1"
    from qdk import Result, stim
    from qdk.simulation import run_qir

    cases = [
        ("lost_control", "H 0\nLOSS(0.25) 0\nCX 0 1\nM 0 1"),
        ("bell_trace", "H 0\nCX 0 1\nLOSS(0.2) 0\nMX 0 1"),
        ("lost_target", "H 0\nLOSS(0.25) 1\nCX 0 1\nMX 0\nM 1"),
        ("reset", "X 0\nLOSS(0.25) 0\nMR 0\nM 0"),
        ("noncliff_before_loss", "H 0\nT 0\nCX 0 1\nLOSS(0.2) 0\nT_DAG 1\nMX 0 1"),
        ("noncliff_after_loss", "H 0 1\nT 0\nCX 0 1\nLOSS(0.2) 1\nT 0\nCX 0 1\nMX 0 1"),
    ]
    summaries = []
    for i, (name, text) in enumerate(cases):
        _, arrays = run(args.binary, args.output / name, text, DEFAULT_MODEL, 20000, seed=40 + i)
        symbols = np.where(arrays["heralds"], 2, arrays["measurements"])
        qir, noise = stim.compile(text.replace("LOSS(", "LOSS_ERROR("))
        result = run_qir(qir, shots=20000, seed=140 + i, noise=noise, type="stabilizer")
        reference = [
            [2 if x == Result.Loss else int(x == Result.One) for x in row] for row in result
        ]
        z, failures = compare_features(
            {"joint": Counter(map(tuple, symbols.tolist()))},
            {"joint": Counter(map(tuple, reference))},
            20000,
            20000,
        )
        assert not failures, failures
        summaries.append(
            {"case": name, "shots_per_backend": 20000, "max_z": z, "failures": failures}
        )
        print(summaries[-1], flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
