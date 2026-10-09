"""Check the complete record/state law of one-rotation transport and re-entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import stim
from automatic_specialization import History, analyze, exact_record_probabilities
from one_core_phase_specialization import CoreResult, NeedDecision, OneCorePhase, representative
from study_automatic_specialization import sample_parities
from study_one_core_phase_specialization import small_cases
from validate_automatic_specialization import all_histories, remap
from validate_regional_phase_specialization import cq_blocks
from validate_shared_phase_specialization import prefix_values

import clifft


def leaves(core: OneCorePhase, history: History) -> list[tuple[CoreResult, float]]:
    values = prefix_values(core.first.shared)
    result = []
    for prefix in values:
        pending: list[tuple[int, ...]] = [()]
        expansions = 0
        while pending:
            choices = pending.pop()
            try:
                branch = core.rewrite(history, 197, prefix_bits=prefix, decisions=choices)
            except NeedDecision:
                pending += [choices + (0,), choices + (1,)]
                expansions += 1
                if expansions > 256:
                    raise ValueError("Exhaustive bridge branches exceed the validation budget")
            else:
                probability = branch.metadata["decision_probability"] / len(values)
                result.append((branch, probability))
    if abs(sum(weight for _, weight in result) - 1) > 1e-12:
        raise AssertionError("Conditional instrument lost probability mass")
    return result


def compare(core: OneCorePhase, history: History) -> dict[str, Any]:
    expected = cq_blocks(core.model.render(history))
    candidates, transported, selected = (np.zeros_like(expected) for _ in range(3))
    records = np.zeros(len(expected))
    optimized_records = np.zeros(len(expected))
    branches = leaves(core, history)
    for branch, probability in branches:
        transported += probability * cq_blocks(branch.transported_source)
        candidates += probability * cq_blocks(branch.candidate_source)
        selected += probability * cq_blocks(branch.source)
        program = clifft.lower(clifft.trace(clifft.parse(branch.source)))
        records += probability * np.asarray(
            exact_record_probabilities(program, core.model.num_records)
        )
        sample_parities(core.model.source, clifft.sample(program, shots=3, seed=175, threads=1))
        hir, _ = analyze(branch.source)
        optimized = clifft.lower(hir)
        optimized_records += probability * np.asarray(
            exact_record_probabilities(optimized, core.model.num_records)
        )
        sample_parities(core.model.source, clifft.sample(optimized, shots=3, seed=175, threads=1))
        if "candidate_cost" in branch.metadata:
            a = branch.metadata["candidate_cost"]
            b = branch.metadata["transported_cost"]
            preferred = (a["width"], a["t_count"]) < (b["width"], b["t_count"])
            if preferred != branch.metadata["accepted_second_region"]:
                raise AssertionError("Selection failed the bounded no-regression rule")
    for observed in (transported, candidates, selected):
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-10)
    expected_records = np.trace(expected, axis1=1, axis2=2).real
    np.testing.assert_allclose(records, expected_records, rtol=0, atol=1e-10)
    np.testing.assert_allclose(optimized_records, expected_records, rtol=0, atol=1e-10)
    return {
        "branches": len(branches),
        "transport_instrument_error": float(np.max(abs(transported - expected))),
        "candidate_instrument_error": float(np.max(abs(candidates - expected))),
        "selected_instrument_error": float(np.max(abs(selected - expected))),
        "record_probability_error": float(np.max(abs(records - expected_records))),
        "optimized_record_probability_error": float(
            np.max(abs(optimized_records - expected_records))
        ),
        "stop_reasons": dict(Counter(b.metadata["stop_reason"] for b, _ in branches)),
        "second_regions_accepted": sum(
            b.metadata.get("accepted_second_region", False) for b, _ in branches
        ),
        "branch_records": [
            {
                "weight": probability,
                "records": branch.metadata["sampled_records"],
                "random_decisions": branch.metadata["random_decisions"],
                "rotation_erased": branch.metadata["rotation_erased"],
                "representative_changes": branch.metadata["representative_changes"],
            }
            for branch, probability in branches
        ],
    }


def validate_small() -> dict[str, Any]:
    original = small_cases()
    sources = original | {name + "_renamed": remap(source) for name, source in original.items()}
    rows = []
    for index, (name, source) in enumerate(sources.items()):
        core = OneCorePhase(source)
        histories: list[History] = [()]
        rng = random.Random(39171 + index)
        if core.model.sites:
            histories.append(
                tuple(
                    (site, rng.randrange(1, len(core.model.sites[site].replacements)))
                    for site in range(len(core.model.sites))
                )
            )
        rows.append(
            {"case": name, "histories": histories, "checks": [compare(core, h) for h in histories]}
        )
        print(name, "complete conditional instrument passed", flush=True)
    return {"cases": rows}


def validate_mixtures() -> list[dict[str, Any]]:
    rows = []
    for name in ("noise_mixture", "categorical"):
        core = OneCorePhase(small_cases()[name])
        histories = all_histories(core.model)
        checks = [compare(core, history) for history, _ in histories]
        rows.append(
            {
                "case": name,
                "histories": len(histories),
                "branches": sum(row["branches"] for row in checks),
                "probability_mass": sum(probability for _, probability in histories),
                "weighted_instrument_error_bound": sum(
                    probability * row["selected_instrument_error"]
                    for (_, probability), row in zip(histories, checks)
                ),
                "maximum_record_probability_error": max(
                    row["record_probability_error"] for row in checks
                ),
            }
        )
        print(name, "complete noise mixture passed", flush=True)
    return rows


def validate_guards() -> dict[str, Any]:
    cases = small_cases()
    reasons = {}
    for name in (
        "noncommuting_measurement",
        "overlapping_reset",
        "non_diagonal_entry",
        "imaginary_representative",
        "unsupported_rotation",
    ):
        reason = {
            branch.metadata["stop_reason"] for branch, _ in leaves(OneCorePhase(cases[name]), ())
        }
        expected = "non_diagonal_entry" if name == "imaginary_representative" else name
        expected = "unsupported_bridge" if name == "unsupported_rotation" else expected
        if reason != {expected}:
            raise AssertionError("A conservative bridge boundary changed")
        reasons[name] = list(reason)
    simulator = stim.TableauSimulator()
    simulator.do(stim.Circuit("H 0\nS 0"))
    if representative(stim.PauliString("X"), simulator, zero_x=1) is not None:
        raise AssertionError("An imaginary Pauli product was used as a phase rotation")
    coherent = OneCorePhase(cases["coherent_cancellation"]).rewrite((), 917)
    damaged = "\n".join(
        line for line in coherent.transported_source.splitlines() if not line.startswith("T ")
    )
    error = float(np.max(abs(cq_blocks(damaged) - cq_blocks(coherent.source))))
    if error < 0.1:
        raise AssertionError("The quantum interface oracle missed a dropped carried phase")
    for source in ("H 0 1\nT 0 1\nH 0\nT 0\n", "H 0\nM 0\n"):
        try:
            OneCorePhase(source)
        except ValueError:
            pass
        else:
            raise AssertionError("An input outside the one-rotation contract was accepted")
    expansion = leaves(OneCorePhase(cases["synthesis_expansion"]), ())
    if any(branch.metadata["accepted_second_region"] for branch, _ in expansion):
        raise AssertionError("An expanded second synthesis displaced the smaller continuation")
    if {tuple(branch.metadata["sampled_records"]) for branch, _ in expansion} != {(0,), (1,)}:
        raise AssertionError("Rejection lost an already sampled branch")
    return {
        "fallbacks": reasons,
        "imaginary_product_rejected": True,
        "dropped_phase_error": error,
        "rejected_expansion_branches": len(expansion),
    }


def validate_rng(shots: int) -> dict[str, Any]:
    source = small_cases()["new_record_feedback"]
    expected = np.trace(cq_blocks(source), axis1=1, axis2=2).real
    core = OneCorePhase(source)
    quantum = random.Random(18973)
    counts = np.zeros(len(expected))
    hashes: list[str] = []
    for index in range(shots):
        branch = core.rewrite((), quantum.getrandbits(64))
        hir, _ = analyze(branch.source)
        program = clifft.lower(hir)
        record = clifft.sample(
            program, shots=1, seed=quantum.getrandbits(64), threads=1
        ).measurements[0]
        counts[sum(int(bit) << i for i, bit in enumerate(record))] += 1
        if index < 8:
            hashes.append(hashlib.sha256(branch.source.encode()).hexdigest())
    error = abs(counts / shots - expected)
    bound = 7 * np.sqrt(expected * (1 - expected) / shots) + 2 / shots
    if np.any(error > bound):
        raise AssertionError("Fresh host outcomes disagree with the complete Aer record law")
    quantum = random.Random(18973)
    for expected_hash in hashes:
        branch = core.rewrite((), quantum.getrandbits(64))
        quantum.getrandbits(64)
        if hashlib.sha256(branch.source.encode()).hexdigest() != expected_hash:
            raise AssertionError("Explicit seeds failed to reproduce the host branches")
    return {"shots": shots, "maximum_frequency_error": float(max(error)), "source_hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=2048)
    args = parser.parse_args()
    if args.shots < 32:
        parser.error("Require at least 32 RNG-check shots")
    result = {
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "one_core_phase_specialization.py",
                "study_one_core_phase_specialization.py",
                "regional_phase_specialization.py",
                "shared_phase_specialization.py",
                "automatic_specialization.py",
                "validate_automatic_specialization.py",
                "validate_regional_phase_specialization.py",
                "validate_shared_phase_specialization.py",
            )
        },
        "small": validate_small(),
        "mixtures": validate_mixtures(),
        "guards": validate_guards(),
        "rng": validate_rng(args.shots),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
