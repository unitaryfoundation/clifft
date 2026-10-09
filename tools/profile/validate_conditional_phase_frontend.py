"""Check automatic routing against complete small quantum instruments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from automatic_specialization import History, analyze, exact_record_probabilities
from conditional_phase_frontend import ConditionalPhase, ConditionalResult
from one_core_phase_specialization import NeedDecision
from shared_phase_specialization import SharedPhase
from study_automatic_specialization import sample_parities
from study_chained_phase_specialization import small_cases as chain_cases
from study_one_core_phase_specialization import small_cases as carrier_cases
from validate_automatic_specialization import all_histories, remap
from validate_chained_phase_specialization import NeedPrefix
from validate_regional_phase_specialization import cq_blocks
from validate_shared_phase_specialization import prefix_values

import clifft


def leaves(front: ConditionalPhase, history: History) -> list[tuple[ConditionalResult, float]]:
    pending: list[tuple[list[int], tuple[int, ...], float]] = [([], (), 1.0)]
    result = []
    expansions = 0
    while pending:
        prefixes, decisions, weight = pending.pop()

        def choose(shared: SharedPhase, stage: int) -> int:
            if stage == len(prefixes):
                raise NeedPrefix(shared)
            return prefixes[stage]

        try:
            branch = front.rewrite(history, 197, choose_prefix=choose, decisions=decisions)
        except NeedPrefix as request:
            values = prefix_values(request.shared)
            pending.extend((prefixes + [v], decisions, weight / len(values)) for v in values)
        except NeedDecision:
            pending.extend((prefixes, decisions + (v,), weight / 2) for v in (0, 1))
        else:
            result.append((branch, weight))
        expansions += 1
        if expansions > 1024:
            raise ValueError("Exact frontend enumeration exceeds its branch budget")
    if abs(sum(weight for _, weight in result) - 1) > 1e-12:
        raise AssertionError("Routing lost trajectory probability")
    return result


def compare(front: ConditionalPhase, history: History) -> dict[str, Any]:
    expected = cq_blocks(front.model.render(history))
    observed = np.zeros_like(expected)
    records = np.zeros(len(expected))
    branches = leaves(front, history)
    routes = set()
    for branch, weight in branches:
        observed += weight * cq_blocks(branch.source)
        hir, _ = analyze(branch.source)
        program = clifft.lower(hir)
        records += weight * np.asarray(exact_record_probabilities(program, front.model.num_records))
        sample_parities(front.model.source, clifft.sample(program, shots=3, seed=975, threads=1))
        routes.add(branch.stop_reason)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-10)
    expected_records = np.trace(expected, axis1=1, axis2=2).real
    np.testing.assert_allclose(records, expected_records, rtol=0, atol=1e-10)
    return {
        "branches": len(branches),
        "maximum_instrument_error": float(np.max(abs(observed - expected))),
        "maximum_record_error": float(np.max(abs(records - expected_records))),
        "routes": sorted(routes),
        "maximum_regions": max(len(b.steps) for b, _ in branches),
    }


def validate_rng() -> list[dict[str, Any]]:
    rows = []
    for name, source in (
        ("chain", chain_cases()["three_measured"]),
        ("carrier", carrier_cases()["new_record_feedback"]),
    ):
        front = ConditionalPhase(source)
        expected = np.trace(cq_blocks(source), axis1=1, axis2=2).real
        counts = np.zeros(len(expected))
        rng = random.Random(29183)
        shots = 2048
        for _ in range(shots):
            branch = front.rewrite((), rng.getrandbits(64))
            hir, _ = analyze(branch.source)
            record = clifft.sample(
                clifft.lower(hir), shots=1, seed=rng.getrandbits(64), threads=1
            ).measurements[0]
            counts[sum(int(bit) << i for i, bit in enumerate(record))] += 1
        error = abs(counts / shots - expected)
        bound = 7 * np.sqrt(expected * (1 - expected) / shots) + 2 / shots
        if np.any(error > bound):
            raise AssertionError("Fresh frontend outcomes differ from the complete Aer law")
        rows.append({"case": name, "shots": shots, "maximum_frequency_error": float(max(error))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    originals = (
        chain_cases()
        | carrier_cases()
        | {
            "two_rotations": "H 0 1\nT 0 1\nH 0\nT 0\nMX 0 1\n",
            "unsupported_initial": "H 0\nR_PAULI(.137) X0\nT 0\nMY 0\n",
            "correlated_movable": (
                "H 0 1 2\nT 0\nM(.13) 2\nE(.2) X0 Y1\n"
                "ELSE_CORRELATED_ERROR(.3) Z1\nT_DAG 0\nMX 0 1\n"
            ),
            "correlated_blocked": (
                "H 0 1 2\nT 0\nM 2\nE(.2) X0\nELSE_CORRELATED_ERROR(.3) Y2 Z1\nT_DAG 0\nMX 0 1\n"
            ),
            "correlated_prefix": (
                "H 0 1\nE(.2) X0 Y1\nELSE_CORRELATED_ERROR(.3) Z0\n"
                "M(.13) 1\nCX rec[-1] 0\nT 0\nH 0\nT_DAG 0\nMX 0\n"
            ),
        }
    )
    sources = originals | {name + "_renamed": remap(s) for name, s in originals.items()}
    rows = []
    for name, source in sources.items():
        front = ConditionalPhase(source)
        histories: list[History] = [()]
        if front.model.sites:
            histories.append(
                tuple((i, len(s.replacements) - 1) for i, s in enumerate(front.model.sites))
            )
        rows.append({"case": name, "checks": [compare(front, h) for h in histories]})
        print(name, "passed", flush=True)
    mixtures = []
    for name in ("correlated_movable", "correlated_blocked", "correlated_prefix"):
        front = ConditionalPhase(originals[name])
        weighted_histories = all_histories(front.model)
        checks = [compare(front, h) for h, _ in weighted_histories]
        mixtures.append(
            {
                "case": name,
                "histories": len(weighted_histories),
                "probability_mass": sum(p for _, p in weighted_histories),
                "weighted_instrument_error_bound": sum(
                    p * row["maximum_instrument_error"]
                    for (_, p), row in zip(weighted_histories, checks)
                ),
            }
        )
    moved = ConditionalPhase(originals["correlated_movable"]).first
    blocked = ConditionalPhase(originals["correlated_blocked"]).first
    assert moved is not None and blocked is not None
    assert int(moved.audit["moved_operations"]) > 0 and blocked.audit["moved_operations"] == 0
    # The complete original state, including earlier sampled records, must
    # remain valid when the bounded chain stops before the next region.
    budget = compare(ConditionalPhase(chain_cases()["three_measured"], max_regions=1), ())
    result = {
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "conditional_phase_frontend.py",
                "automatic_specialization.py",
                "deferred_phase_specialization.py",
                "regional_phase_specialization.py",
                "shared_phase_specialization.py",
                "one_core_phase_specialization.py",
            )
        },
        "cases": rows,
        "mixtures": mixtures,
        "region_budget": budget,
        "rng": validate_rng(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
