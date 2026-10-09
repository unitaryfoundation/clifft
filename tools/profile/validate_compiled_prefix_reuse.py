"""Check reused preparation states and complete conditional instruments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from automatic_specialization import History, exact_record_probabilities
from compiled_prefix_reuse import ReusablePhase, compile_stages, optimize_preparation
from study_automatic_specialization import sample_parities
from validate_automatic_specialization import all_histories, remap
from validate_conditional_phase_frontend import leaves
from validate_regional_phase_specialization import cq_blocks

import clifft


def check(front: ReusablePhase, history: History) -> dict[str, Any]:
    expected = cq_blocks(front.model.render(history))
    observed = np.zeros_like(expected)
    records = np.zeros(len(expected))
    branches = leaves(front, history)
    for branch, weight in branches:
        observed += weight * cq_blocks(branch.source)
        hir, _ = compile_stages(branch.source, phase=front.phase_required)
        program = clifft.lower(hir)
        records += weight * np.asarray(exact_record_probabilities(program, front.model.num_records))
        sample_parities(front.model.source, clifft.sample(program, shots=3, seed=919, threads=1))
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        records, np.trace(expected, axis1=1, axis2=2).real, atol=1e-10, rtol=0
    )
    return {
        "branches": len(branches),
        "instrument_error": float(np.max(abs(observed - expected))),
        "record_error": float(np.max(abs(records - np.trace(expected, axis1=1, axis2=2).real))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preparations = []
    for seed in range(24):
        rng = random.Random(seed + 719)
        lines = ["H 0 1 2"]
        for _ in range(12):
            a, b = rng.sample(range(3), 2)
            lines += [
                f"{rng.choice(['H', 'S', 'S_DAG', 'X', 'Y', 'Z', 'T', 'T_DAG'])} {a}",
                f"CX {a} {b}",
            ]
        source = "\n".join(lines) + "\n"
        optimized, info = optimize_preparation(source, args.exporter)
        expected, actual = cq_blocks(source), cq_blocks(optimized)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        preparations.append({**info, "state_error": float(np.max(abs(actual - expected)))})
    originals = {
        "ccz": "H 0 1 2\nCCZ 0 1 2\nH 0\nMY 1\nR 2\nCX 0 2\n",
        "negative_complex": "H 0 1 2\nS 1\nT_DAG 0\nCX 1 2\nT 2\nH 0\nMPP Y0*Z2\n",
        "observed_prefix": (
            "H 0 1 2\nM(.13) 2\nCX rec[-1] 0\nT 0 1\nCX 0 1\n"
            "E(.17) X0 Y1\nELSE_CORRELATED_ERROR(.23) Z1\nH 0\n"
            "M 0\nCZ rec[-1] 1\nMY 1\nDETECTOR rec[-2] rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-3]\n"
        ),
        "later_magic": "H 0 1\nT 0 1\nH 0\nT_DAG 0\nM 1\nCZ rec[-1] 0\nMY 0\n",
        "carrier_fallback": "H 0 1\nT 0\nM 1\nCX rec[-1] 0\nT_DAG 0\nMX 0\n",
        "initial_fallback": "H 0\nR_PAULI(.137) Z0\nMY 0\n",
        "clifford_fallback": "H 0\nT 0\nT_DAG 0\nMX 0\n",
    }
    sources = originals | {name + "_renamed": remap(s) for name, s in originals.items()}
    rows = []
    for name, source in sources.items():
        front = ReusablePhase(source, args.exporter)
        histories = all_histories(front.model)
        checks = [check(front, h) for h, _ in histories]
        rows.append(
            {
                "case": name,
                "reuse": front.reuse_info,
                "phase_required": front.phase_required,
                "histories": len(histories),
                "checks": checks,
                "mixture_error_bound": sum(
                    w * c["instrument_error"] for (_, w), c in zip(histories, checks)
                ),
            }
        )
        print(name, "passed", front.reuse_info["eligible"], flush=True)
    assert all(
        r["reuse"]["eligible"]
        for r in rows
        if r["case"].split("_renamed")[0]
        in {"ccz", "negative_complex", "observed_prefix", "later_magic"}
    )
    assert all(r["phase_required"] for r in rows if r["case"].startswith("later_magic"))
    result = {
        "preparations": preparations,
        "instruments": rows,
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("compiled_prefix_reuse.py"),
                Path(__file__).with_name("export_optimized_prefix.cpp"),
            ]
        },
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
