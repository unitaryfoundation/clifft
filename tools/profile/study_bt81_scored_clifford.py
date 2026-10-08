"""Exercise the scored Clifford reduction on the larger BT81 preparation and decoder."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import stim
from analyze_bt27_phase_corrections import region
from bt27_scored_clifford import ScoredClifford
from profile_external_feedback import MERLIN_REVISION
from study_bt27_clifford_plans import compact_converter
from study_bt27_direct_clifford import extension_hash, verify_derivatives
from study_bt27_scored_sampling import compare, output_summary, sources
from study_circuit_noise import History, Model, synthetic_bt_noise
from validate_bt27_scored_equivalence import exact_law, require_support

import clifft


def restored(sample: Any, flips: int, converter: Any, *, check_raw: bool = True) -> Any:
    records = sample.measurements.astype(bool)
    if check_raw:
        d, o = converter.convert(measurements=records, separate_observables=True)
        np.testing.assert_array_equal(sample.detectors, d)
        np.testing.assert_array_equal(sample.observables, o)
    mask = np.array([flips >> q & 1 for q in range(records.shape[1])], dtype=bool)
    records = records ^ mask
    d, o = converter.convert(measurements=records, separate_observables=True)
    return SimpleNamespace(measurements=records, detectors=d, observables=o)


def program_for(text: str) -> Any:
    hir = clifft.trace(clifft.parse(text))
    if hir.num_t_gates:
        raise AssertionError("Direct reduction retained T gates")
    program = clifft.lower(
        hir,
        expected_detectors=[0] * hir.num_detectors,
        expected_observables=[0] * hir.num_observables,
    )
    if program.peak_active_width:
        raise AssertionError("Direct Clifford circuit requires active coordinates")
    return program


def fixed_histories(model: Model) -> dict[str, History]:
    groups: dict[tuple[str, str], list[int]] = {}
    for i, site in enumerate(model.sites):
        groups.setdefault((site.gate, site.position), []).append(i)
    result: dict[str, History] = {"identity": ()}
    for indices in groups.values():
        for i in (indices[0], indices[-1]):
            result[f"single_{i}"] = ((i, i % len(model.sites[i].outcomes)),)
    for case in range(2):
        result[f"dense_{case}"] = tuple(
            (i, (i * case) % len(site.outcomes)) for i, site in enumerate(model.sites)
        )
    rng = random.Random(816119)
    for i in range(16):
        result[f"sampled_{i}"] = model.draw(rng)
    return result


def validate_fixed(
    reducer: ScoredClifford, tail: str, converter: Any, shots: int
) -> dict[str, Any]:
    merlin = importlib.import_module("merlin")
    rows = {}
    for name, history in fixed_histories(reducer.model).items():
        correction, shift = reducer.evaluate(history)
        text = reducer.render(correction)
        circuit = stim.Circuit(text)
        width = circuit.num_measurements
        law = exact_law(circuit, list(range(width)), width, correction.records)
        actual = restored(
            clifft.sample(program_for(text), shots=shots, seed=810517, threads=1, batch_size=1),
            correction.records,
            converter,
        )
        records = circuit.compile_sampler(seed=771193).sample(shots)
        direct = restored(
            SimpleNamespace(measurements=records), correction.records, converter, check_raw=False
        )
        other = merlin.CircuitSampler(reducer.model.render(history) + tail, seed=811921).sample(
            shots
        )
        # Merlin may normalize a fixed faulty source against its own reference.
        reference = restored(other, 0, converter, check_raw=False)
        for sample in (actual, direct, reference):
            require_support(sample, law)
        if name == "identity" and (np.any(actual.detectors) or np.any(actual.observables)):
            raise AssertionError("Ideal scored BT81 has nonzero declared outputs")
        rows[name] = {
            "history": history if not name.startswith("dense_") else None,
            "history_sha256": hashlib.sha256(json.dumps(history).encode()).hexdigest(),
            "fault_count": len(history),
            "logical_shift": shift,
            "random_dimension": width + circuit.num_detectors + circuit.num_observables - len(law),
            "joint_law_sha256": hashlib.sha256(str(law).encode()).hexdigest(),
            "clifft_stim_check": compare(actual, direct, converter),
            "clifft_merlin_check": compare(actual, reference, converter),
        }
        if len(rows) % 8 == 0:
            print(f"BT81 fixed histories checked: {len(rows)}", flush=True)
    return {
        "cases": rows,
        "shots_per_history_per_backend": shots,
        "backends": ["direct_clifft", "direct_stim", "original_location_merlin"],
        "t_count": 0,
        "peak_active_width": 0,
        "scope": (
            "Exact laws of direct circuits; sample support and distribution checks against "
            "original-location Merlin, not exact law equality of two independent reductions"
        ),
    }


def fresh_samples(
    reducer: ScoredClifford, source: str, converter: Any, shots: int
) -> dict[str, Any]:
    merlin = importlib.import_module("merlin")
    width = reducer.model.num_measurements + len(reducer.logical)
    shape = stim.Circuit(reducer.render(reducer.evaluate(())[0]))
    arrays = {
        field: np.empty((shots, size), dtype=bool)
        for field, size in (
            ("measurements", width),
            ("detectors", shape.num_detectors),
            ("observables", shape.num_observables),
        )
    }
    rng, measurement_rng = random.Random(815231), random.Random(814019)
    fault_counts: Counter[int] = Counter()
    start = perf_counter()
    for i in range(shots):
        history = reducer.model.draw(rng)
        fault_counts[len(history)] += 1
        correction, _ = reducer.evaluate(history)
        program = program_for(reducer.render(correction))
        sample = clifft.sample(
            program, shots=1, seed=measurement_rng.getrandbits(64), threads=1, batch_size=1
        )
        sample = restored(sample, correction.records, converter)
        for field in arrays:
            arrays[field][i] = getattr(sample, field)[0]
    elapsed = perf_counter() - start
    actual = SimpleNamespace(**arrays)
    sampler = merlin.CircuitSampler(source, seed=819313)
    start = perf_counter()
    other = sampler.sample(shots)
    merlin_elapsed = perf_counter() - start
    return {
        "shots": shots,
        "fault_counts": dict(fault_counts),
        "prototype_seconds_per_shot": elapsed / shots,
        "merlin_seconds_per_shot": merlin_elapsed / shots,
        "prototype_outputs": output_summary(actual, converter),
        "merlin_outputs": output_summary(other, converter),
        "comparison": compare(actual, other, converter),
        "scope": (
            "Fresh full physical noise; finite bug check, not a predeclared rate-equivalence test"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--fixed-shots", type=int, default=128)
    args = parser.parse_args()
    if min(args.shots, args.fixed_shots) < 16:
        parser.error("At least 16 shots required")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = perf_counter()
    # Reuse the pinned-checkout verification and generator import setup.
    sources(args.merlin_checkout, "ideal")
    build = importlib.import_module("benchmarks.protocols.code_switching").build_code_switching_case
    ideal = build("bt81", p_phys=0, target_scoring=False).circuit
    scored = build("bt81", p_phys=0, target_scoring=True).circuit
    if not scored.startswith(ideal):
        raise AssertionError("Expected scoring appended to the protocol")
    tail = scored[len(ideal) :]
    model = Model(synthetic_bt_noise(ideal, {"prep", "phase", "decoder"}, 0.001))
    source = model.render(None) + tail
    reducer = ScoredClifford(model, tail)
    converter, map_checks = compact_converter(source)
    setup = perf_counter() - started
    _, phase, suffix, _ = region(model.render(()))
    invalid = "\n".join([f"RX {q}" for q in range(model.num_qubits)] + phase + suffix)
    try:
        ScoredClifford(Model(invalid), tail)
    except ValueError as error:
        if "do not cancel" not in str(error):
            raise
    else:
        raise AssertionError("Unconstrained preparation was incorrectly accepted")
    document = {
        "merlin_revision": MERLIN_REVISION,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "initial_state": "All qubits zero at circuit entry with the full preparation executed",
        "versions": {"stim": stim.__version__, "numpy": np.__version__},
        "extensions": {
            name: extension_hash(name) for name in ("clifft._clifft_core", "merlin._core")
        },
        "seeds": {
            "fixed_histories": 816119,
            "fixed_clifft": 810517,
            "fixed_stim": 771193,
            "fixed_merlin": 811921,
            "fresh_faults": 815231,
            "fresh_measurements": 814019,
            "fresh_merlin": 819313,
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "num_qubits": model.num_qubits,
        "noise_sites": len(model.sites),
        "noise_site_types": dict(Counter(site.gate for site in model.sites)),
        "setup_seconds": setup,
        "preparation_identity": reducer.certificate,
        "logical_qubits": reducer.logical,
        "decoder_row_weights": [row.bit_count() for row in reducer.rows],
        "candidate_cz_pairs": len(reducer.pairs),
        "scoring_response_payload_bytes": sum(
            (value.bit_length() + 7) // 8 for value in reducer.first + list(reducer.mixed.values())
        ),
        "compact_parity_map": map_checks,
        "unconstrained_preparation_negative_detected": True,
        "derivative_checks": verify_derivatives(reducer),
    }
    print("BT81 preparation and all derivative certificates passed", flush=True)
    document["fixed_validation"] = validate_fixed(reducer, tail, converter, args.fixed_shots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    document["stochastic_validation"] = fresh_samples(reducer, source, converter, args.shots)
    document["elapsed_seconds"] = perf_counter() - started
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print("BT81 fresh noisy sampling checks passed", flush=True)


if __name__ == "__main__":
    main()
