"""Check measurement deferral as an instrument and as a dependency permutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import stim
from automatic_specialization import (
    ANNOTATIONS,
    MEASUREMENTS,
    History,
    exact_record_probabilities,
    record_count,
)
from deferred_phase_specialization import DeferredPhase
from regional_phase_specialization import RegionalPhase
from shared_phase_specialization import SharedPhase
from study_automatic_specialization import sample_parities, stress_histories
from study_deferred_phase_specialization import bridge_probe, panel
from validate_automatic_specialization import all_histories, remap
from validate_regional_phase_specialization import cq_blocks
from validate_shared_phase_specialization import exact_law, prefix_law, prefix_values

import clifft


def dependency_certificate(original: str, scheduled: str) -> dict[str, int]:
    """Check unchanged wire order and classical record semantics independently.

    Operations on distinct tensor factors commute, including local instruments
    and independent noise. Absolute record targets must retain their identities.
    """

    def projections(source: str) -> tuple[dict[int, list[Any]], list[Any]]:
        wires: dict[int, list[Any]] = defaultdict(list)
        records = []
        available = 0
        for node in clifft.parse(source).nodes:
            gate = node.gate.name
            # Both M(p) and explicit READOUT_NOISE parse to a noiseless M plus
            # a flip node; the former may retain an immaterial zero argument.
            args = () if gate in MEASUREMENTS else tuple(node.args)
            signature = (gate, args, tuple(map(repr, node.targets)))
            if any(t.is_rec and not 0 <= t.value < available for t in node.targets):
                raise AssertionError("Scheduling placed a record use before its producer")
            if gate in MEASUREMENTS | ANNOTATIONS | {"READOUT_NOISE"}:
                records.append(signature)
            if gate not in ANNOTATIONS | {"MPAD", "READOUT_NOISE"}:
                for q in {t.value for t in node.targets if not t.is_rec}:
                    wires[q].append(signature)
            available += record_count(node)
        return dict(wires), records

    a, b = projections(original), projections(scheduled)
    if a != b:
        raise AssertionError("Scheduling changed a wire order or classical record dependency")
    return {"wire_projections": len(a[0]), "record_and_annotation_operations": len(a[1])}


def small_cases() -> dict[str, str]:
    result = {
        "disjoint": "H 0 1\nT 0\nMX 1\nT_DAG 0\nH 0\nT 0\nM 0\n",
        "entangled": "H 0\nCX 0 1\nT 0\nMY 1\nT_DAG 0\nH 0\nM 0\nCX rec[-2] 1\n",
        "old_record": (
            "H 2\nM(.13) 2\nH 0 1\nCX 0 1\nT 0\nMX 1\n"
            "CZ rec[-2] 0\nT_DAG 0\nH 0\nM 0\n"
            "DETECTOR rec[-3] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
        ),
        "new_record_barrier": "H 0 1\nT 0\nMX 1\nCX rec[-1] 0\nT_DAG 0\nM 0\n",
        "shared_wire_barrier": "H 0 1\nT 0\nMX 1\nCX 0 1\nT_DAG 0\nM 0\n",
        "noise_barrier": "H 0 1\nT 0\nMX 1\nDEPOLARIZE2(.1) 0 1\nT_DAG 0\nM 0\n",
        "reset_barrier": "H 0 1\nT 0\nMX 1\nR 1\nT_DAG 0\nM 0\n",
        "product_measurement": (
            "H 0 2\nCX 0 1\nCX 0 2\nT 2\nMPP(.17) !X0*Y1\nT_DAG 2\nMY 2\nCX rec[-1] 0\n"
        ),
        "repeat_measurement": "H 0 1\nCX 0 1\nT 0\nMY 1\nM 1\nT_DAG 0\nMX 0\n",
        "padding": "H 0\nT 0\nMPAD 1\nT_DAG 0\nH 0\nM 0\nCZ rec[-2] 0\n",
        "noise_mixture": (
            "H 0\nX_ERROR(.07) 0\nCX 0 1\nT 0\nMX(.13) !1\n"
            "Z_ERROR(.17) 0\nT_DAG 0\nH 0\nM(.19) 0\nCX rec[-2] 1\n"
        ),
        "categorical": (
            "H 0 1 2\nT 0\nMY(.13) !2\n"
            "PAULI_CHANNEL_2(.01,.02,.03,.01,.02,.01,.02,.01,.01,.01,.01,.01,.01,.01,.01) 0 1\n"
            "T_DAG 0\nCX 0 1\nH 0\nM 0\n"
        ),
    }
    for seed in range(4):
        rng = random.Random(9731 + seed)
        lines = ["H 0 1 2"]
        for _ in range(5):
            a, b = rng.sample(range(3), 2)
            lines += [f"{rng.choice(('H', 'S', 'S_DAG'))} {a}", f"CX {a} {b}"]
        lines += [
            "T 0",
            "MY(.07) 1",
            "PAULI_CHANNEL_1(.03,.05,.07) 0",
            "T_DAG 0",
            "T 2",
            "MPP X1*Z2",
            "T 0",
            "H 0",
            "MX 0",
            "CZ rec[-2] 2",
        ]
        result[f"general_{seed}"] = "\n".join(lines) + "\n"
    return result


def compare(deferred: DeferredPhase, history: History) -> dict[str, Any]:
    original = deferred.model.render(history)
    scheduled = deferred.region.model.render(deferred.map_history(history))
    dependency_certificate(original, scheduled)
    reference = cq_blocks(original)
    direct = cq_blocks(scheduled)
    np.testing.assert_allclose(direct, reference, rtol=0, atol=1e-10)
    values = prefix_values(deferred.region.shared)
    mapped, _ = deferred.region.split_history(deferred.map_history(history))
    prefix_law(deferred.region.shared, mapped)
    observed = np.zeros_like(reference)
    records = np.zeros(len(reference))
    for value in values:
        text = deferred.compose(history, value)
        observed += cq_blocks(text) / len(values)
        program = clifft.lower(clifft.trace(clifft.parse(text)))
        records += np.asarray(
            exact_record_probabilities(program, deferred.model.num_records)
        ) / len(values)
        sample_parities(deferred.model.source, clifft.sample(program, shots=3, seed=719, threads=1))
    expected_records = np.trace(reference, axis1=1, axis2=2).real
    np.testing.assert_allclose(observed, reference, rtol=0, atol=1e-10)
    np.testing.assert_allclose(records, expected_records, rtol=0, atol=1e-10)
    return {
        "branches": len(values),
        "scheduling_instrument_error": float(np.max(abs(direct - reference))),
        "complete_instrument_error": float(np.max(abs(observed - reference))),
        "record_probability_error": float(np.max(abs(records - expected_records))),
    }


def validate_small() -> dict[str, Any]:
    originals = small_cases()
    sources = originals | {name + "_renamed": remap(source) for name, source in originals.items()}
    rows = []
    for index, (name, source) in enumerate(sources.items()):
        deferred = DeferredPhase(source)
        certificate = dependency_certificate(source, deferred.scheduled_source)
        histories: list[History] = [()]
        rng = random.Random(41351 + index)
        if deferred.model.sites:
            histories.append(
                tuple(
                    (site, rng.randrange(1, len(deferred.model.sites[site].replacements)))
                    for site in range(len(deferred.model.sites))
                )
            )
        rows.append(
            {
                "case": name,
                "histories": histories,
                "certificate": certificate,
                "scheduling": deferred.audit,
                "checks": [compare(deferred, history) for history in histories],
            }
        )
        print(name, "complete instrument passed", flush=True)
    return {"cases": rows}


def validate_mixtures() -> list[dict[str, Any]]:
    result = []
    for name in ("noise_mixture", "categorical"):
        deferred = DeferredPhase(small_cases()[name])
        histories = all_histories(deferred.model)
        checks = [compare(deferred, history) for history, _ in histories]
        result.append(
            {
                "case": name,
                "histories": len(histories),
                "probability_mass": sum(probability for _, probability in histories),
                "weighted_instrument_error_bound": sum(
                    probability * row["complete_instrument_error"]
                    for (_, probability), row in zip(histories, checks)
                ),
                "maximum_record_probability_error": max(
                    row["record_probability_error"] for row in checks
                ),
            }
        )
        print(name, "complete categorical mixture passed", flush=True)
    return result


def validate_protocols(sources: dict[str, str]) -> list[dict[str, Any]]:
    results = []
    for name, source in sources.items():
        deferred = DeferredPhase(source)
        certificate = dependency_certificate(source, deferred.scheduled_source)
        row: dict[str, Any] = {
            "case": name,
            "certificate": certificate,
            "scheduling": deferred.audit,
        }
        if "scored" in name:
            terminal = SharedPhase(source)
            if terminal.prefix != deferred.region.shared.prefix:
                raise AssertionError("The compared preparations differ")
            if terminal.nonclifford or deferred.region.shared.nonclifford:
                raise AssertionError("Expected a Clifford scored reduction")
            hashes = []
            histories = stress_histories(deferred.model)
            for history in histories:
                mapped = deferred.map_history(history)
                dependency_certificate(
                    deferred.model.render(history), deferred.region.model.render(mapped)
                )
                for _ in range(2):
                    ideal = terminal.controls(()) >> 1
                    expected = stim.Circuit(
                        terminal.render(terminal.controls(history, ideal), None)
                    )
                    observed = stim.Circuit(deferred.compose(history, ideal))
                    m = deferred.model.num_records
                    a = exact_law(expected, list(range(m)), m)
                    b = exact_law(observed, list(range(m)), m)
                    if a != b:
                        raise AssertionError("Complete scored law differs from terminal reduction")
                    hashes.append(hashlib.sha256(str(b).encode()).hexdigest())
            row.update(
                histories=histories,
                conditional_laws=len(hashes),
                law_hashes=hashes,
                scope=(
                    "Compared to the prior shared-phase algorithm; "
                    "not an independent large-state proof"
                ),
            )
        results.append(row)
        print(name, "protocol certificate passed", flush=True)
    return results


def validate_guards() -> dict[str, Any]:
    cases = small_cases()
    for name in ("new_record_barrier", "shared_wire_barrier", "noise_barrier", "reset_barrier"):
        deferred = DeferredPhase(cases[name])
        if deferred.audit["moved_operations"]:
            raise AssertionError("Dependent work crossed a measurement")
    try:
        dependency_certificate("H 0\nCX 0 1\nM 0\n", "H 0\nM 0\nCX 0 1\n")
    except AssertionError:
        pass
    else:
        raise AssertionError("Certificate missed a noncommuting reorder")
    deferred = DeferredPhase(cases["noise_mixture"])
    for history in (((0, 0),), ((0, 1), (0, 1)), ((len(deferred.model.sites), 1),)):
        try:
            deferred.map_history(history)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid original fault history was accepted")
    probes = []
    for source, survives in (
        ("H 0\nT 0\nRX 1\nH 1\nT 0\n", True),
        ("H 0\nT 0\nR 0\nH 0\nT 0\n", False),
        ("H 0\nT 0\nCX 0 1\nRX 1\nT 0\n", False),
        ("H 0\nT 0\nCX 0 1\nMX 1\nCZ rec[-1] 0\nRX 1\nT 0\n", True),
    ):
        row = bridge_probe(RegionalPhase(source), (), 417)
        if (row["first_logical_loss"] is None) != survives:
            raise AssertionError("Bell probe misclassified loss of its encoded logical qubit")
        probes.append(row)
    return {
        "dependent_boundaries": 4,
        "detected_bad_schedule": True,
        "invalid_histories": 3,
        "bell_probe_controls": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "deferred_phase_specialization.py",
                "study_deferred_phase_specialization.py",
                "regional_phase_specialization.py",
                "shared_phase_specialization.py",
                "validate_regional_phase_specialization.py",
                "validate_shared_phase_specialization.py",
                "automatic_specialization.py",
                "validate_automatic_specialization.py",
            )
        },
        "small": validate_small(),
        "mixtures": validate_mixtures(),
        "guards": validate_guards(),
        "protocols": validate_protocols(panel(args.merlin_checkout)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
