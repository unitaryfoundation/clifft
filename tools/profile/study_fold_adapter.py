"""Reproduce parsed-region validation without compiling a per-circuit executable."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from clifford_branches import logical_tail_history, materialize, paired_hook_histories, run_history
from fold_adapter import Adapter
from fold_blocks import f7_diagnostic_histories, syndrome_sector_histories
from fold_cultivation import logical
from fold_growth import growth_fault_histories, reconstruction_with_growth
from fold_plan_data import Writer
from fold_schedule import measurement_chunks


def held_out():
    r = reconstruction_with_growth(7, "dual_rotate")
    rng = random.Random(97103)
    for stage in r.stages:
        if not stage.name.startswith("syndrome"):
            continue
        chunks = measurement_chunks(stage.operations)
        rng.shuffle(chunks)
        for chunk in chunks:
            indices = [k for k, op in enumerate(chunk) if op.name == "CX"]
            gates = [chunk[k] for k in indices]
            rng.shuffle(gates)
            for k, gate in zip(indices, gates, strict=True):
                chunk[k] = gate
        stage.operations = [op for chunk in chunks for op in chunk]
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    circuit, plan = args.output / "held.stim", args.output / "held.bin"
    text = held_out().text(0.001)
    # The adapter gets only circuit text; stage names and generator objects do not cross it.
    text = "\n".join(line for line in text.splitlines() if not line.startswith("#")) + "\n"
    circuit.write_text(text)
    adapter = Adapter(args.worker)
    protocol, metadata = adapter.prepare(circuit, plan)
    r = protocol.reconstruction
    cases = [("ideal", [s.operations for s in r.stages])]
    cases += [(f"natural_{seed}", materialize(r, 0.001, seed)[0]) for seed in range(1000, 1064)]
    cases += [(f"stress_{seed}", materialize(r, 0.01, seed)[0]) for seed in range(2000, 2032)]
    cases += paired_hook_histories(r) + [("logical_tail", logical_tail_history(r))]
    cases += syndrome_sector_histories(protocol) + f7_diagnostic_histories(protocol)
    cases += growth_fault_histories(r)
    rows = []
    for name, stages in cases:
        result = protocol.evaluate(protocol.encode(stages))
        reference, _ = run_history(r, stages)
        p = reference.expectation()
        xyz = [reference.expectation(logical(7, a)) / p for a in "XYZ"] if p > 1e-12 else None
        p_error = abs(result.acceptance - p)
        xyz_error = (
            max(abs(a - b) for a, b in zip(result.logical_xyz, xyz, strict=True))
            if result.logical_xyz is not None and xyz is not None
            else 0
        )
        if p_error > 2e-12 or xyz_error > 2e-12 or (xyz is None) != (result.logical_xyz is None):
            raise ValueError(f"reference mismatch: {name}")
        rows.append(
            {
                "name": name,
                **vars(result),
                "reference_acceptance": p,
                "reference_xyz": xyz,
                "probability_error": p_error,
                "probe_error": xyz_error,
            }
        )
    fixture_path = args.output / "fixtures.bin"
    fixture_metadata = Writer(protocol).write(fixture_path, cases)
    native = json.loads(subprocess.check_output([str(args.worker), "--check", str(fixture_path)]))
    result = {
        "planning": metadata,
        "fixture_plan": fixture_metadata,
        "native": native,
        "circuit_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "histories": rows,
    }
    if args.catalog:
        result["previous_catalog"] = json.loads(
            subprocess.check_output(
                [str(args.catalog), str(circuit), "0", "19331", "1", "1", "ordinary"]
            )
        )
    timings = []
    for _ in range(args.trials):
        start = time.perf_counter()
        run = adapter.run(plan, circuit, shots=args.shots)
        run["process_wall_seconds"] = time.perf_counter() - start
        for key in ("probabilities", "qubits", "measurements", "detectors", "exp_vals"):
            run.pop(key, None)
        timings.append(run)
    result["timings"] = timings
    result["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "fold_adapter.py",
            "fold_plan_data.py",
            "fold_plan_worker.cpp",
            "fold_blocks.py",
            "fold_blocks_native.h",
            "fold_recognition.h",
            "study_fold_adapter.py",
        )
    }
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "histories": len(rows),
                "native": native,
                "planning": metadata,
                "max_probability_error": max(row["probability_error"] for row in rows),
                "max_probe_error": max(row["probe_error"] for row in rows),
                "timings_us": [row["sample_ms"] * 1000 / args.shots for row in timings],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
