"""Exact affine output laws for fixed-history complete scored BT27 circuits.

Both candidate and reference HIRs are reduced by Clifft's existing optimizer,
then interpreted independently by Stim. This checks complete output laws for
the tested histories conditional on those compiler rewrites; it is not an
independent proof of the shared optimizer or the unreduced non-Clifford input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import numpy as np
import stim
from profile_bt27_fault_specialization import ROOT
from study_bt27_scored_sampling import ScoredSampler, converter_for, sources
from study_circuit_noise import History, Model


def canonical(rows: list[int], columns: int) -> tuple[int, ...]:
    basis: dict[int, int] = {}
    mask = (1 << columns) - 1
    for row in rows:
        for pivot in sorted(basis, reverse=True):
            if row >> pivot & 1:
                row ^= basis[pivot]
        if not row & mask:
            if row:
                raise AssertionError("Inconsistent affine output constraints")
            continue
        pivot = (row & mask).bit_length() - 1
        for other in basis:
            if basis[other] >> pivot & 1:
                basis[other] ^= row
        basis[pivot] = row
    return tuple(basis[pivot] for pivot in sorted(basis, reverse=True))


def output_maps(circuit: stim.Circuit, order: list[int]) -> list[int]:
    detectors: list[int] = []
    observables = [0] * circuit.num_observables
    measured = 0
    for instruction in circuit:
        gate = instruction.name
        if gate in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            mask = 0
            for target in instruction.targets_copy():
                if not target.is_measurement_record_target:
                    raise AssertionError("Non-record output target")
                mask ^= 1 << order[measured + target.value]
            if gate == "DETECTOR":
                detectors.append(mask)
            else:
                observables[int(instruction.gate_args_copy()[0])] ^= mask
        else:
            measured += stim.Circuit(str(instruction)).num_measurements
    if measured != len(order):
        raise AssertionError("Exported measurement map has the wrong length")
    return detectors + observables


def exact_law(
    circuit: stim.Circuit, order: list[int], visible: int, flips: int = 0
) -> tuple[int, ...]:
    if len(set(order)) != len(order) or set(order) != set(range(len(order))):
        raise AssertionError("Expected every visible and hidden slot exactly once")
    total = len(order)
    # Initial resets fix the input. Final resets trace out the final quantum
    # state, leaving all deterministic relations among measurement records.
    boundary = stim.Circuit("R " + " ".join(map(str, range(circuit.num_qubits))))
    flows = (boundary + circuit + boundary).flow_generators()
    rows = []
    for flow in flows:
        a, b = flow.input_copy(), flow.output_copy()
        if a.weight or any(b[q] in (1, 2) for q in range(len(b))):
            raise AssertionError("Reset boundaries did not erase quantum input and output")
        # Every remaining output Z has eigenvalue +1 after the final resets.
        # Substitute it even when a generator mixes Z with record terms; do
        # not assume Stim chose generators separating quantum and record bits.
        sign = a.sign * b.sign
        if sign not in (1, -1):
            raise AssertionError("Non-real classical flow")
        row = int(sign == -1) << total
        for index in flow.measurements_copy():
            row ^= 1 << order[index]
        rows.append(row)
    reduced = canonical(rows, total)
    visible_mask = (1 << visible) - 1
    hidden_mask = ((1 << total) - 1) ^ visible_mask
    # Hidden columns have higher pivots. Eliminating them and retaining only
    # rows without hidden terms takes the exact marginal over reset outcomes.
    rows = []
    maps = output_maps(circuit, order)
    columns = visible + len(maps)
    for row in reduced:
        if row & hidden_mask:
            continue
        mask = row & visible_mask
        sign_bit = (row >> total) ^ ((mask & flips).bit_count() & 1)
        rows.append(mask | (sign_bit << columns))
    for i, mask in enumerate(maps):
        if mask & ~visible_mask:
            raise AssertionError("Declared output depends on a hidden record")
        rows.append(mask | (1 << (visible + i)))
    return canonical(rows, columns)


def exported(path: Path, visible: int, flips: int = 0) -> tuple[tuple[int, ...], list[int]]:
    text = path.read_text()
    order = [int(line.split()[2]) for line in text.splitlines() if line.startswith("# RECORD ")]
    circuit = stim.Circuit(text)
    return exact_law(circuit, order, visible, flips), output_maps(circuit, order)


def require_support(sample: Any, law: tuple[int, ...]) -> None:
    values = np.column_stack((sample.measurements, sample.detectors, sample.observables))
    columns = values.shape[1]
    for row in law:
        indices = [i for i in range(columns) if row >> i & 1]
        parity = np.bitwise_xor.reduce(values[:, indices], axis=1, initial=0)
        if np.any(parity != (row >> columns)):
            raise AssertionError("Sample lies outside the exact joint output support")


def validate_exporter(binary: Path) -> dict[str, Any]:
    cases = [
        "H 0\nM 0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1] rec[-2]\n",
        "RX 0\nM 0\nRX 0\nMX !0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "H 0\nM 0\nH 1\nCZ rec[-1] 1\nMX 1\n",
        "H 0\nM 0\nS 1\nCX rec[-1] 1\nS_DAG 1\nM 1\nMRX 0\nMX 0\n",
    ]
    rng = random.Random(581119)
    for _ in range(32):
        lines, measured = [], 0
        for _ in range(96):
            gate = rng.choice(("H", "S", "X", "Y", "Z", "CX", "M", "MX", "R", "RX"))
            a, b = rng.sample(range(5), 2)
            lines.append(f"{gate} {a} {b}" if gate == "CX" else f"{gate} {a}")
            if gate in ("M", "MX"):
                measured += 1
                lines.append(f"C{rng.choice('XZ')} rec[-{rng.randint(1, measured)}] {b}")
        lines += ["M 0 1 2 3 4", "OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]"]
        cases.append("\n".join(lines))
    with tempfile.TemporaryDirectory(prefix="bt27-export-check-") as temp:
        original, converted = Path(temp) / "original.stim", Path(temp) / "converted.stim"
        for text in cases:
            original.write_text(text)
            subprocess.run(
                [str(binary.resolve()), "--export-source", str(original), str(converted)],
                check=True,
            )
            circuit = stim.Circuit(text)
            n = circuit.num_measurements
            expected = exact_law(circuit, list(range(n)), n)
            actual, _ = exported(converted, n)
            if actual != expected:
                raise AssertionError("HIR exporter disagrees with independent Stim circuit")
    # Equal individual bit means do not establish the joint law. These two
    # circuits have unbiased bits but different deterministic parity relations.
    correlated = stim.Circuit("H 0\nM 0\nM 0")
    independent = stim.Circuit("H 0 1\nM 0 1")
    if exact_law(correlated, [0, 1], 2) == exact_law(independent, [0, 1], 2):
        raise AssertionError("Correlation negative control was not detected")
    if canonical([0b101, 0b110], 2) != canonical([0b011, 0b101], 2):
        raise AssertionError("Canonicalization depends on the generating basis")
    return {"independent_clifford_circuits": len(cases), "correlation_negative_detected": True}


def histories(model: Model, baseline: dict[str, Any]) -> dict[str, tuple[str, History]]:
    result = {
        name: (row["group"], tuple(tuple(x) for x in row["history"]))
        for name, row in baseline["fixed_validation"]["cases"].items()
    }
    seen = {history for _, history in result.values()}

    def add(name: str, group: str, history: History) -> None:
        if history not in seen:
            result[name] = group, history
            seen.add(history)

    groups: dict[tuple[str, str], list[int]] = {}
    positions: dict[str, list[int]] = {}
    for i, site in enumerate(model.sites):
        groups.setdefault((site.gate, site.position), []).append(i)
        positions.setdefault(site.position, []).append(i)
    for (gate, position), indices in groups.items():
        for site_id in (indices[0], indices[-1]):
            for outcome in range(len(model.sites[site_id].outcomes)):
                add(f"edge_{site_id}_{outcome}", f"edge:{gate}:{position}", ((site_id, outcome),))
    rng = random.Random(1108291)
    regions = list(positions.values())
    for i in range(64):
        selected = {rng.choice(regions[i % 3]), rng.choice(regions[i // 3 % 3])}
        if len(selected) == 2:
            add(
                f"pair_{i}",
                "paired_regions",
                tuple((j, rng.randrange(len(model.sites[j].outcomes))) for j in sorted(selected)),
            )
    for i in range(64):
        add(f"new_random_{i}", "fresh_p001_history", model.draw(rng))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = perf_counter()
    model, tail, source = sources(args.merlin_checkout, "gate_noise")
    baseline_path = ROOT / "research/conditional_clifford/bt27-scored-sampling.json"
    baseline = json.loads(baseline_path.read_text())
    if (
        hashlib.sha256(source.encode()).hexdigest()
        != baseline["stochastic_models"]["gate_noise"]["prototype"]["source_sha256"]
    ):
        raise AssertionError("Baseline source no longer matches")
    cases = histories(model, baseline)
    if args.smoke:
        readout = next(name for name, (group, _) in cases.items() if group.startswith("READOUT"))
        keys = ("identity", "decoder_x_before_scoring", "dense_4170_0", readout)
        cases = {name: row for name, row in cases.items() if name in keys}
    checks = validate_exporter(args.binary)
    merlin = importlib.import_module("merlin")
    converter = converter_for(source)
    source_maps = output_maps(
        stim.Circuit(
            "\n".join(line for line in source.splitlines() if not line.startswith(("T ", "T_DAG ")))
        ),
        list(range(135)),
    )
    rows = {}
    record_negatives = 0
    with ScoredSampler(args.binary, model, tail, export_clifford=True) as sampler:
        for name, (group, history) in cases.items():
            correction = sampler.controls.evaluate(history)
            sample, telemetry = sampler.fixed(history, args.shots, 511913, check=True)
            prefix = telemetry["name"]
            shared, shared_maps = exported(
                sampler.directory / f"{prefix}.shared.stim", 135, correction.records
            )
            fresh, fresh_maps = exported(sampler.directory / f"{prefix}.fresh.stim", 135)
            if shared != fresh or shared_maps != source_maps or fresh_maps != source_maps:
                raise AssertionError(f"Exact joint output law mismatch: {name}")
            require_support(sample, fresh)
            independent = merlin.CircuitSampler(model.render(history) + tail, seed=981137).sample(
                args.shots
            )
            # Fixed-fault Merlin references can differ. Restore the original
            # zero-reference parities directly from its measurement records.
            d, o = converter.convert(
                measurements=independent.measurements.astype(bool), separate_observables=True
            )
            from types import SimpleNamespace

            require_support(
                SimpleNamespace(measurements=independent.measurements, detectors=d, observables=o),
                fresh,
            )
            uncorrected, _ = exported(sampler.directory / f"{prefix}.shared.stim", 135)
            record_negatives += uncorrected != fresh
            rows[name] = {
                "group": group,
                "history_sha256": hashlib.sha256(json.dumps(history).encode()).hexdigest(),
                "history": history if name not in baseline["fixed_validation"]["cases"] else None,
                "fault_count": len(history),
                "constraint_rank": len(fresh),
                "random_dimension": 216 - len(fresh),
                "joint_law_sha256": hashlib.sha256(str(fresh).encode()).hexdigest(),
                "raw_boundary_matches": telemetry["checked"],
                "complete_output_laws_equal": True,
            }
            if name == "decoder_x_before_scoring":
                evaluate = sampler.controls.evaluate
                with patch.object(
                    sampler.controls,
                    "evaluate",
                    lambda h: replace(evaluate(h), final_x=0, final_z=0),
                ):
                    _, wrong_row = sampler.fixed(history, 1, 511913)
                wrong, _ = exported(sampler.directory / f"{wrong_row['name']}.shared.stim", 135)
                if wrong == fresh:
                    raise AssertionError("Omitted decoder correction was not detected")
                rows[name]["omitted_decoder_pauli_negative_detected"] = True
            if len(rows) % 16 == 0:
                print(
                    f"Exact laws and sample support checked for {len(rows)}/{len(cases)} cases",
                    flush=True,
                )
    if not args.smoke and record_negatives == 0:
        raise AssertionError("No omitted-record-restoration negative was detected")
    document = {
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "stim_version": stim.__version__,
        "exporter_checks": checks,
        "shots_per_history_per_sampler_for_support": args.shots,
        "cases_checked": len(rows),
        "unique_histories": len({history for _, history in cases.values()}),
        "groups": dict(Counter(group for group, _ in cases.values())),
        "random_dimension_counts": dict(Counter(row["random_dimension"] for row in rows.values())),
        "omitted_record_restoration_negative_cases": record_negatives,
        "elapsed_seconds": perf_counter() - started,
        "scope": __doc__,
        "limitations": (
            "Exact equality of affine laws after shared compiler rewrites. Support checks do not "
            "prove uniform execution or cover untested histories. Statistical rates are separate."
        ),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: value for key, value in document.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
