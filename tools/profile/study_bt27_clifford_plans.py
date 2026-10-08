"""Audit fixed-plan reuse and construction costs for direct scored Clifford circuits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import stim
from bt27_scored_clifford import ScoredClifford
from profile_bt27_fault_specialization import ROOT
from study_bt27_direct_clifford import extension_hash, restore
from study_bt27_scored_sampling import converter_for, output_summary, sources
from validate_bt27_scored_equivalence import exact_law, histories, output_maps

import clifft


def compact_converter(source: str) -> tuple[Any, dict[str, Any]]:
    circuit = stim.Circuit(
        "\n".join(line for line in source.splitlines() if not line.startswith(("T ", "T_DAG ")))
    )
    width = circuit.num_measurements
    maps = output_maps(circuit, list(range(width)))
    lines = ["M " + " ".join(map(str, range(width)))]
    for i, mask in enumerate(maps):
        gate = (
            "DETECTOR"
            if i < circuit.num_detectors
            else f"OBSERVABLE_INCLUDE({i - circuit.num_detectors})"
        )
        lines.append(
            gate + " " + " ".join(f"rec[{j - width}]" for j in range(width) if mask >> j & 1)
        )
    text = "\n".join(lines)
    converter = stim.Circuit(text).compile_m2d_converter(skip_reference_sample=True)
    original = converter_for(source)
    # A zero row and every basis vector determine this linear output map.
    inputs = np.vstack((np.zeros((1, width), dtype=bool), np.eye(width, dtype=bool)))
    np.testing.assert_array_equal(
        original.convert(measurements=inputs, append_observables=True),
        converter.convert(measurements=inputs, append_observables=True),
    )
    return converter, {
        "basis_rows_checked": len(inputs),
        "output_bits": len(maps),
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scope": "Record parities only with skip_reference_sample; no quantum reference evolution",
    }


def plan_audit(reducer: ScoredClifford) -> dict[str, Any]:
    baseline = json.loads(
        (ROOT / "research/conditional_clifford/bt27-scored-sampling.json").read_text()
    )
    reference = json.loads(
        (ROOT / "research/conditional_clifford/bt27-scored-exact-equivalence.json").read_text()
    )
    rows = {}
    base_kinds: list[tuple[str, int]] = []
    for name, (_, history) in histories(reducer.model, baseline).items():
        correction, shift = reducer.evaluate(history)
        source = reducer.render(correction)
        hir = clifft.trace(clifft.parse(source))
        program = clifft.lower(hir, expected_detectors=[0] * 72, expected_observables=[0] * 9)
        assert hir.num_t_gates == program.peak_active_width == 0
        kinds = []
        random_records = []
        hidden_random = 0
        for i in range(program.num_actions):
            text = program.inspect_action(i)
            gate = text.split()[0]
            if gate not in ("MEASURE_DORMANT", "RECORD_CLASSICAL"):
                if gate not in ("WRITE_DETECTOR", "WRITE_OBSERVABLE"):
                    raise AssertionError(f"Unexpected Clifford plan action: {text}")
                continue
            match = re.search(r"\brecord=r(\d+)\b", text)
            if match is None:
                raise AssertionError("Changed diagnostic record format")
            record = int(match[1])
            if record < 135:
                kinds.append((gate, record))
                if gate == "MEASURE_DORMANT":
                    random_records.append(record)
            elif gate == "MEASURE_DORMANT":
                hidden_random += 1
        if name == "identity":
            base_kinds = kinds
        if len(kinds) != 135 or not base_kinds:
            raise AssertionError("Expected identity first and every visible record exactly once")
        law = exact_law(stim.Circuit(source), list(range(135)), 135, correction.records)
        digest = hashlib.sha256(str(law).encode()).hexdigest()
        if digest != reference["cases"][name]["joint_law_sha256"]:
            raise AssertionError("Direct law differs from the retained reference")
        dimension = 216 - len(law)
        if dimension != len(random_records):
            raise AssertionError("Visible random actions and exact-law dimension differ")
        changed = [
            record
            for (a, record), (b, target) in zip(base_kinds, kinds)
            if (a, record) != (b, target)
        ]
        rows[name] = {
            "fault_count": len(history),
            "logical_shift": shift,
            "num_actions": program.num_actions,
            "visible_random_bits": dimension,
            "hidden_random_actions": hidden_random,
            "random_visible_records": random_records,
            "changed_measurement_kinds_from_ideal": changed,
            "measurement_kind_sha256": hashlib.sha256(str(kinds).encode()).hexdigest(),
            "joint_law_sha256": digest,
        }
    witness = rows["decoder_x_before_scoring"]
    assert rows["identity"]["visible_random_bits"] == 30
    assert witness["visible_random_bits"] == 34
    assert witness["changed_measurement_kinds_from_ideal"] == [130, 131, 133, 134]
    return {
        "cases": rows,
        "dimension_counts": dict(Counter(row["visible_random_bits"] for row in rows.values())),
        "distinct_measurement_kind_patterns": len(
            {row["measurement_kind_sha256"] for row in rows.values()}
        ),
        "conclusion": (
            "One fixed affine record map plus fault-dependent sign flips cannot change "
            "its random dimension"
        ),
        "limits": (
            "Action shapes are an obstruction diagnostic, not a proof that equal shapes "
            "have compatible expressions"
        ),
    }


def timed_host(reducer: ScoredClifford, converter: Any, shots: int) -> tuple[Any, dict[str, Any]]:
    stages: defaultdict[str, float] = defaultdict(float)
    rng, measurements = random.Random(918173), random.Random(591823)
    arrays = {
        field: np.empty((shots, width), dtype="u1")
        for field, width in (("measurements", 135), ("detectors", 72), ("observables", 9))
    }
    started = perf_counter()
    for i in range(shots):
        start = perf_counter()
        history = reducer.model.draw(rng)
        stages["draw"] += perf_counter() - start
        start = perf_counter()
        correction, _ = reducer.evaluate(history)
        stages["controls"] += perf_counter() - start
        start = perf_counter()
        text = reducer.render(correction)
        stages["source"] += perf_counter() - start
        start = perf_counter()
        circuit = clifft.parse(text)
        stages["parse"] += perf_counter() - start
        start = perf_counter()
        hir = clifft.trace(circuit)
        stages["trace"] += perf_counter() - start
        start = perf_counter()
        program = clifft.lower(hir, expected_detectors=[0] * 72, expected_observables=[0] * 9)
        stages["lower"] += perf_counter() - start
        assert hir.num_t_gates == program.peak_active_width == 0
        start = perf_counter()
        sample = clifft.sample(
            program, shots=1, seed=measurements.getrandbits(64), threads=1, batch_size=1
        )
        stages["sample"] += perf_counter() - start
        start = perf_counter()
        restored = restore(sample, correction.records, converter)
        for field in arrays:
            arrays[field][i] = getattr(restored, field)[0]
        stages["restore_check_collect"] += perf_counter() - start
    return SimpleNamespace(**arrays), {
        "shots": shots,
        "seconds_per_shot": (perf_counter() - started) / shots,
        "stage_seconds_per_shot": {key: value / shots for key, value in stages.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    if args.shots < 16:
        parser.error("At least 16 shots required")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    start = perf_counter()
    model, tail, source = sources(args.merlin_checkout, "gate_noise")
    reducer = ScoredClifford(model, tail)
    compact, checks = compact_converter(source)
    setup = perf_counter() - start
    audit = plan_audit(reducer)
    print(f"Plan audit passed: {len(audit['cases'])} histories", flush=True)
    old, old_timing = timed_host(reducer, converter_for(source), args.shots)
    print("Full-converter timing complete", flush=True)
    new, new_timing = timed_host(reducer, compact, args.shots)
    for field in ("measurements", "detectors", "observables"):
        np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
    document = {
        "initial_state": "All qubits zero at circuit entry; all preparation operations remain live",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "clifft_extension_sha256": extension_hash("clifft._clifft_core"),
        "stim_version": stim.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "seeds": {"faults": 918173, "measurements": 591823},
        "setup_seconds": setup,
        "plan_audit": audit,
        "compact_parity_map": checks,
        "full_converter_host": old_timing,
        "compact_converter_host": new_timing,
        "paired_outputs_identical": True,
        "outputs": output_summary(new, compact),
        "limits": (
            "Offline host only; no reusable conditional plan implemented. Single-run timings "
            "include assertions and sampling-call setup. Equal diagnostic shapes do not "
            "certify plan compatibility."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(
        json.dumps(
            {
                "full_ms": old_timing["seconds_per_shot"] * 1000,
                "compact_ms": new_timing["seconds_per_shot"] * 1000,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
