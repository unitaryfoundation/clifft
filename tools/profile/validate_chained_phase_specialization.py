"""Enumerate conditional chains and compare their complete record/state laws."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from automatic_specialization import History, exact_record_probabilities
from chained_phase_specialization import ChainedPhase, ChainResult
from shared_phase_specialization import SharedPhase
from study_automatic_specialization import candidates, sample_parities, stress_histories
from study_chained_phase_specialization import measured_chain, small_cases
from validate_automatic_specialization import all_histories, remap
from validate_regional_phase_specialization import cq_blocks
from validate_shared_phase_specialization import prefix_values

import clifft


class NeedPrefix(Exception):
    def __init__(self, shared: SharedPhase):
        self.shared = shared


def leaves(chain: ChainedPhase, history: History) -> list[tuple[ChainResult, float]]:
    pending: list[tuple[list[int], float]] = [([], 1.0)]
    result = []
    expansions = 0
    while pending:
        choices, weight = pending.pop()

        def choose(shared: SharedPhase, stage: int) -> int:
            if stage == len(choices):
                raise NeedPrefix(shared)
            return choices[stage]

        try:
            branch = chain.rewrite(history, choose)
        except NeedPrefix as request:
            values = prefix_values(request.shared)
            pending.extend((choices + [value], weight / len(values)) for value in values)
            expansions += len(values)
            if expansions > 512:
                raise ValueError("Conditional chain enumeration exceeds its branch budget")
        else:
            result.append((branch, weight))
    if abs(sum(weight for _, weight in result) - 1) > 1e-12:
        raise AssertionError("Conditional chain lost branch probability")
    return result


def compare(chain: ChainedPhase, history: History) -> dict[str, Any]:
    reference = cq_blocks(chain.model.render(history))
    expected = np.trace(reference, axis1=1, axis2=2).real
    observed = np.zeros_like(reference)
    records = np.zeros_like(expected)
    branches = leaves(chain, history)
    for branch, weight in branches:
        observed += weight * cq_blocks(branch.source)
        program = clifft.lower(clifft.trace(clifft.parse(branch.source)))
        records += weight * np.asarray(exact_record_probabilities(program, chain.model.num_records))
        sample_parities(chain.model.source, clifft.sample(program, shots=3, seed=953, threads=1))
    np.testing.assert_allclose(observed, reference, rtol=0, atol=1e-10)
    np.testing.assert_allclose(records, expected, rtol=0, atol=1e-10)
    return {
        "branches": len(branches),
        "maximum_record_state_error": float(np.max(abs(observed - reference))),
        "maximum_record_probability_error": float(np.max(abs(records - expected))),
        "step_counts": dict(Counter(len(branch.steps) for branch, _ in branches)),
        "stop_reasons": dict(Counter(branch.stop_reason for branch, _ in branches)),
        "branch_paths": [
            {
                "probability": weight,
                "residual_t": [step["residual_t"] for step in branch.steps],
                "prefix_records": [step["prefix_records"] for step in branch.steps],
                "record_values": [step["record_values"] for step in branch.steps],
            }
            for branch, weight in branches
        ],
    }


def validate_small() -> dict[str, Any]:
    originals = small_cases()
    sources = originals | {name + "_renamed": remap(source) for name, source in originals.items()}
    result = []
    for index, (name, source) in enumerate(sources.items()):
        chain = ChainedPhase(source)
        histories: list[History] = [()]
        rng = random.Random(93851 + index)
        if chain.model.sites:
            histories.append(
                tuple(
                    (site, rng.randrange(1, len(chain.model.sites[site].replacements)))
                    for site in sorted(
                        rng.sample(range(len(chain.model.sites)), min(5, len(chain.model.sites)))
                    )
                )
            )
            histories.append(chain.model.draw(rng))
        histories = list(dict.fromkeys(histories))
        checks = [compare(chain, history) for history in histories]
        result.append({"case": name, "histories": histories, "checks": checks})
        print(name, "joint record/state law passed", flush=True)
    return {"cases": result}


def validate_noise() -> dict[str, Any]:
    chain = ChainedPhase(measured_chain(noisy=True))
    histories = all_histories(chain.model)
    checks = [compare(chain, history) for history, _ in histories]
    return {
        "histories": len(histories),
        "branches": sum(row["branches"] for row in checks),
        "probability_mass": sum(probability for _, probability in histories),
        "weighted_record_state_error_bound": sum(
            probability * row["maximum_record_state_error"]
            for (_, probability), row in zip(histories, checks)
        ),
        "maximum_record_probability_error": max(
            row["maximum_record_probability_error"] for row in checks
        ),
    }


def validate_guards(panel: dict[str, tuple[str, str]]) -> dict[str, Any]:
    budget_checks = []
    for maximum in (1, 2):
        chain = ChainedPhase(measured_chain(3), max_regions=maximum)
        row = compare(chain, ())
        if set(row["stop_reasons"]) != {"region_budget"} or set(row["step_counts"]) != {maximum}:
            raise AssertionError("A region limit did not retain the complete continuation")
        budget_checks.append({"max_regions": maximum, **row})
    variants = leaves(ChainedPhase(small_cases()["branch_dependent"]), ())
    mass = {True: 0.0, False: 0.0}
    bases = set()
    for branch, probability in variants:
        if len(branch.steps) != 2:
            raise AssertionError("Branch-dependent case did not reach its second region")
        mass[branch.steps[1]["clifford_exit"]] += probability
        bases.add(branch.steps[1]["entry_basis_sha256"])
    if mass != {True: 0.5, False: 0.5} or len(bases) != 2:
        raise AssertionError("Second analysis ignored its measurement-dependent input state")

    def corrupt_old_record(shared: SharedPhase, stage: int) -> int:
        values = prefix_values(shared)
        return values[0] ^ 1 if stage else next(value for value in values if value & 1)

    try:
        ChainedPhase(small_cases()["branch_dependent"]).rewrite((), corrupt_old_record)
    except AssertionError as error:
        if "already observed" not in str(error):
            raise
    else:
        raise AssertionError("A changed old record was accepted")
    capped_source = "T 0\nH " + " ".join(map(str, range(129))) + "\nT 0\nM 0\n"
    capped = ChainedPhase(capped_source).rewrite((), lambda _shared, _stage: 0)
    once = ChainedPhase(capped_source, max_regions=1).rewrite((), lambda _shared, _stage: 0)
    if capped.stop_reason != "unsupported_next_region" or capped.source != once.source:
        raise AssertionError("A later analysis budget failure changed the continuation")
    unsupported = leaves(ChainedPhase(small_cases()["unsupported_later_rotation"]), ())
    if any(branch.stop_reason != "unsupported_next_region" for branch, _ in unsupported):
        raise AssertionError("Absence of T gates incorrectly certified a non-Clifford rotation")
    fault_branches = leaves(ChainedPhase(measured_chain(noisy=True)), ((0, 1),))
    for branch, _ in fault_branches:
        if any(
            (step["record_values"] & 1) != (branch.steps[0]["record_values"] & 1)
            for step in branch.steps
        ):
            raise AssertionError("A readout fault was lost across a region boundary")
    # These protocol examples have a non-Clifford first exit. The chain must
    # return the already-validated single-region source without a second attempt.
    protocol_checks = []
    for name in ("bt27_scored", "15to1_scored"):
        chain = ChainedPhase(panel[name][1])
        assert chain.first is not None
        count = 0
        for history in stress_histories(chain.model):
            for seed in (101, 383):
                sampler = chain.first.shared.prefix.compile_sampler(seed=seed)
                value = int.from_bytes(sampler.sample(1, bit_packed=True)[0].tobytes(), "little")

                def only_first(_: SharedPhase, stage: int) -> int:
                    if stage:
                        raise AssertionError("An unsupported non-Clifford entry was analyzed")
                    return value

                result = chain.rewrite(history, only_first)
                if result.stop_reason != "nonclifford_exit" or len(result.steps) != 1:
                    raise AssertionError("Expected a conservative non-Clifford fallback")
                if result.source != chain.first.compose(history, value):
                    raise AssertionError("Fallback changed the single-region computation")
                count += 1
        protocol_checks.append({"case": name, "identical_conditional_sources": count})
    return {
        "budget_checks": budget_checks,
        "second_region_clifford_probability": mass[True],
        "second_region_nonclifford_probability": mass[False],
        "distinct_second_entry_bases": len(bases),
        "changed_old_record_rejected": True,
        "unsupported_next_region_reason": capped.rejection,
        "non_t_rotation_retained": True,
        "protocol_fallbacks": protocol_checks,
    }


def validate_rng(shots: int = 2048) -> dict[str, Any]:
    chain = ChainedPhase(measured_chain(3))
    density = cq_blocks(chain.model.source)
    expected = np.trace(density, axis1=1, axis2=2).real
    counts = np.zeros(len(expected))
    rng = random.Random(389113)
    hashes: list[str] = []
    for _ in range(shots):
        result = chain.sample_history((), rng)
        program = clifft.lower(clifft.trace(clifft.parse(result.source)))
        sample = clifft.sample(program, shots=1, seed=rng.getrandbits(64), threads=1)
        value = sum(int(bit) << i for i, bit in enumerate(sample.measurements[0]))
        counts[value] += 1
        if len(hashes) < 8:
            hashes.append(hashlib.sha256(result.source.encode()).hexdigest())
    observed = counts / shots
    tolerance = 7 * np.sqrt(expected * (1 - expected) / shots) + 2 / shots
    if np.any(abs(observed - expected) > tolerance):
        raise AssertionError(
            "Fresh chain sampling disagrees with the complete reference record law"
        )
    repeated = random.Random(389113)
    for expected_hash in hashes:
        result = chain.sample_history((), repeated)
        if hashlib.sha256(result.source.encode()).hexdigest() != expected_hash:
            raise AssertionError("Explicit chain seeds are not reproducible")
        repeated.getrandbits(64)
    return {
        "shots": shots,
        "expected_record_probabilities": expected.tolist(),
        "observed_record_probabilities": observed.tolist(),
        "maximum_absolute_error": float(np.max(abs(observed - expected))),
        "reproducible_initial_sources": len(hashes),
        "scope": "Detect repeated seeds or resampled old outcomes; not a rare-error-rate study",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = candidates(args.merlin_checkout)
    result = {
        "small": validate_small(),
        "noise_mixture": validate_noise(),
        "guards": validate_guards(panel),
        "fresh_rng": validate_rng(),
    }
    result["source_hashes"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            Path(__file__).name,
            "chained_phase_specialization.py",
            "study_chained_phase_specialization.py",
            "regional_phase_specialization.py",
            "shared_phase_specialization.py",
        )
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("all chained validations passed", flush=True)


if __name__ == "__main__":
    main()
