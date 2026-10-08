"""Validate direct scored BT27 Clifford construction without a HIR optimizer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import stim
from analyze_bt27_phase_corrections import region
from bt27_circuit_corrections import bits
from bt27_scored_clifford import ScoredClifford, derivative_tables, phase_response
from profile_bt27_fault_specialization import ROOT
from profile_external_feedback import MERLIN_REVISION
from study_bt27_scored_sampling import compare, converter_for, output_summary, sources
from study_circuit_noise import Model
from validate_bt27_phase_corrections import append_gates, parity_rotation
from validate_bt27_scored_equivalence import exact_law, histories, require_support

import clifft


def packed_gates(packed: int, width: int, pairs: list[tuple[int, ...]]) -> list[str]:
    gates = [f"Z {q}" for q in bits(packed & ((1 << width) - 1))]
    gates += [f"CZ {pairs[i][0]} {pairs[i][1]}" for i in bits(packed >> width)]
    return gates


def verify_derivatives(reducer: ScoredClifford) -> dict[str, Any]:
    table = np.zeros(1 << len(reducer.logical), dtype="u1")
    values = np.arange(len(table))
    for triple in reducer.triples:
        term = np.ones(len(table), dtype="u1")
        for i in triple:
            term &= ((values >> i) & 1).astype("u1")
        table ^= term
    pairs = list(combinations(range(reducer.width), 2))
    x = [0] + [1 << q for q in range(reducer.width)] + [(1 << a) | (1 << b) for a, b in pairs]
    y = np.array(
        [
            sum(((row & value).bit_count() & 1) << i for i, row in enumerate(reducer.rows))
            for value in x
        ]
    )
    pair_indices = {pair: i for i, pair in enumerate(reducer.pairs)}
    negative_shifts = 0
    for shift in range(len(table)):
        packed = reducer.derivative(shift)
        z = [(packed >> q) & 1 for q in range(reducer.width)]
        cz = packed >> reducer.width
        observed = np.array(
            [0]
            + z
            + [
                z[a] ^ z[b] ^ ((cz >> pair_indices[a, b]) & 1 if (a, b) in pair_indices else 0)
                for a, b in pairs
            ],
            dtype="u1",
        )
        expected = table[y ^ shift] ^ table[y] ^ table[shift]
        np.testing.assert_array_equal(observed, expected)
        incomplete = phase_response(shift, reducer.first, {})
        negative_shifts += incomplete != packed
    if not negative_shifts:
        raise AssertionError("Omitted mixed shift terms were not detected")
    return {
        "logical_shifts": len(table),
        "physical_basis_points_per_shift": len(x),
        "exact_evaluations": len(table) * len(x),
        "scope": "Degree-two Boolean polynomials are determined by weights zero, one, and two",
        "omitted_mixed_terms_detected_shifts": negative_shifts,
    }


def verify_aer() -> dict[str, Any]:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    rng = random.Random(194201)
    circuits = []
    negative_index = 0
    for case in range(65):
        width = 4
        decoder = [f"CX {a} {b}" for a, b in (rng.sample(range(width), 2) for _ in range(10))]
        triples = [tuple(sorted(rng.sample(range(width), 3))) for _ in range(2)]
        delta = [f"{rng.choice(('S', 'S_DAG', 'Z'))} {q}" for q in range(width)]
        delta += [f"CZ {a} {b}" for a, b in (rng.sample(range(width), 2) for _ in range(3))]
        a, b, c = (rng.randrange(1 << width) for _ in range(3))
        if case == 64:
            decoder, triples, delta, a, b, c = [], [(0, 1, 2)], [], 0, 3, 0
        rows = [1 << q for q in range(width)]
        for line in decoder:
            _, control, target = line.split()
            rows[int(target)] ^= rows[int(control)]
        score = [
            gate
            for triple in triples
            for degree in (1, 2, 3)
            for support in combinations(triple, degree)
            for gate in parity_rotation(list(support), degree == 2)
        ]
        inverse = [
            line.replace("T_DAG", "t").replace("T ", "T_DAG ").replace("t", "T")
            for line in reversed(score)
        ]
        physical = decoder + inverse + list(reversed(decoder))
        phase_x = [f"X {q}" for q in bits(a)]
        final = [f"Z {q}" for q in bits(c)] + [f"X {q}" for q in bits(b)]
        shift = sum((((row & a).bit_count() & 1) ^ (b >> i & 1)) << i for i, row in enumerate(rows))
        pairs, first, mixed = derivative_tables(rows, triples, width, [])
        response = phase_response(shift, first, mixed)
        initial = QuantumCircuit(width * 2)
        for q in range(width):
            initial.h(q)
            initial.cx(q, q + width)
        left, right = initial.copy(), initial.copy()
        append_gates(left, physical + delta + phase_x + decoder + final + score)
        append_gates(
            right, delta + packed_gates(response, width, pairs) + phase_x + decoder + final
        )
        circuits += [left, right]
        if case == 64:
            wrong = initial.copy()
            append_gates(
                wrong, packed_gates(phase_response(shift, first, {}), width, pairs) + final
            )
            negative_index = len(circuits)
            circuits.append(wrong)
    for circuit in circuits:
        circuit.save_statevector()
    result = AerSimulator(method="statevector", max_parallel_threads=1).run(circuits).result()
    maximum_error = 0.0
    for i in range(65):
        left_state, right_state = (np.asarray(result.get_statevector(2 * i + j)) for j in (0, 1))
        overlap = np.vdot(right_state, left_state)
        aligned = right_state * overlap / abs(overlap)
        maximum_error = max(maximum_error, float(np.max(np.abs(left_state - aligned))))
        np.testing.assert_allclose(left_state, aligned, atol=1e-12, rtol=0)
    negative_overlap = float(
        abs(
            np.vdot(
                result.get_statevector(negative_index - 2), result.get_statevector(negative_index)
            )
        )
    )
    if negative_overlap > 1e-12:
        raise AssertionError("Aer did not detect the omitted mixed-fault Z term")
    return {
        "choi_pairs": 65,
        "maximum_amplitude_error_up_to_global_phase": maximum_error,
        "omitted_mixed_term_choi_overlap": negative_overlap,
    }


def restore(sample: Any, flips: int, converter: Any) -> Any:
    d, o = converter.convert(
        measurements=sample.measurements.astype(bool), separate_observables=True
    )
    np.testing.assert_array_equal(sample.detectors, d)
    np.testing.assert_array_equal(sample.observables, o)
    records = sample.measurements ^ np.array([flips >> q & 1 for q in range(135)], dtype="u1")
    d, o = converter.convert(measurements=records.astype(bool), separate_observables=True)
    return SimpleNamespace(measurements=records, detectors=d, observables=o)


def clifft_sample(text: str, shots: int, seed: int) -> Any:
    hir = clifft.trace(clifft.parse(text))
    if hir.num_t_gates:
        raise AssertionError("Direct construction contains non-Clifford gates")
    executable = clifft.lower(hir, expected_detectors=[0] * 72, expected_observables=[0] * 9)
    if executable.peak_active_width != 0:
        raise AssertionError("Direct Clifford execution has nonzero active width")
    return clifft.sample(executable, shots=shots, seed=seed, threads=1, batch_size=1)


def validate_fixed(reducer: ScoredClifford, source: str, shots: int, smoke: bool) -> dict[str, Any]:
    baseline = json.loads(
        (ROOT / "research/conditional_clifford/bt27-scored-sampling.json").read_text()
    )
    reference = json.loads(
        (ROOT / "research/conditional_clifford/bt27-scored-exact-equivalence.json").read_text()
    )
    if hashlib.sha256(source.encode()).hexdigest() != reference["source_sha256"]:
        raise AssertionError("Noise model differs from the validated source")
    cases = histories(reducer.model, baseline)
    if smoke:
        cases = {
            name: cases[name]
            for name in (
                "identity",
                "decoder_x_before_scoring",
                "dense_4170_0",
                "dense_4170_1",
                "pair_0",
            )
        }
    converter = converter_for(source)
    rows = {}
    for name, (group, history) in cases.items():
        correction, shift = reducer.evaluate(history)
        text = reducer.render(correction)
        circuit = stim.Circuit(text)
        law = exact_law(circuit, list(range(135)), 135, correction.records)
        digest = hashlib.sha256(str(law).encode()).hexdigest()
        if digest != reference["cases"][name]["joint_law_sha256"]:
            raise AssertionError(f"Direct Clifford law differs from the baseline: {name}")
        actual = restore(clifft_sample(text, shots, 837291), correction.records, converter)
        records = circuit.compile_sampler(seed=511121).sample(shots)
        d, o = converter.convert(measurements=records, separate_observables=True)
        independent = restore(
            SimpleNamespace(measurements=records, detectors=d, observables=o),
            correction.records,
            converter,
        )
        require_support(actual, law)
        require_support(independent, law)
        validation = compare(actual, independent, converter)
        rows[name] = {
            "group": group,
            "fault_count": len(history),
            "logical_shift": shift,
            "correction_gates": len(reducer.gates(correction)),
            "exact_law_sha256": digest,
            "sampling_check": validation,
        }
        if name == "decoder_x_before_scoring":
            original = reducer.controls.evaluate(history)
            if (
                original.x
                | original.s
                | original.z
                | original.cz
                | original.records
                | original.final_z
                or original.final_x != 1
            ):
                raise AssertionError("Expected the pinned isolated decoder X0 fault")
            wrong = ScoredClifford.gates(reducer, original)
            wrong_text = "\n".join(
                reducer.prefix + wrong + reducer.suffix + ["X 0"] + reducer.readout
            )
            wrong_law = exact_law(stim.Circuit(wrong_text), list(range(135)), 135)
            if wrong_law == law:
                raise AssertionError("Omitting the final-X derivative was not detected")
            rows[name]["omitted_decoder_shift_negative_detected"] = True
        if len(rows) % 32 == 0:
            print(
                f"Direct laws and Clifft/Stim samples checked: {len(rows)}/{len(cases)}",
                flush=True,
            )
    return {
        "cases": rows,
        "shots_per_history_per_backend": shots,
        "optimizer_invocations": 0,
        "t_count": 0,
        "peak_active_width": 0,
    }


def stochastic(args: Any) -> dict[str, Any]:
    result = {}
    merlin = importlib.import_module("merlin")
    for model_name in ("ideal", "original", "gate_noise"):
        setup_start = perf_counter()
        model, tail, source = sources(args.merlin_checkout, model_name)
        reducer = ScoredClifford(model, tail)
        setup_seconds = perf_counter() - setup_start
        converter = converter_for(source)
        fault_rng, measurement_rng = random.Random(660701), random.Random(73319)
        arrays = {
            field: np.empty((args.shots, n), dtype="u1")
            for field, n in (("measurements", 135), ("detectors", 72), ("observables", 9))
        }
        fault_counts: Counter[int] = Counter()
        stages: defaultdict[str, float] = defaultdict(float)
        started = perf_counter()
        for i in range(args.shots):
            history = model.draw(fault_rng) if model.sites else ()
            fault_counts[len(history)] += 1
            start = perf_counter()
            correction, _ = reducer.evaluate(history)
            stages["controls_seconds"] += perf_counter() - start
            start = perf_counter()
            text = reducer.render(correction)
            stages["source_seconds"] += perf_counter() - start
            start = perf_counter()
            sample = clifft_sample(text, 1, measurement_rng.getrandbits(64))
            stages["ordinary_clifft_seconds"] += perf_counter() - start
            start = perf_counter()
            sample = restore(sample, correction.records, converter)
            for field in arrays:
                arrays[field][i] = getattr(sample, field)[0]
            stages["restoration_seconds"] += perf_counter() - start
        elapsed = perf_counter() - started
        actual = SimpleNamespace(**arrays)
        merlin_start = perf_counter()
        other = merlin.CircuitSampler(source, seed=111917)
        merlin_setup = perf_counter() - merlin_start
        merlin_start = perf_counter()
        expected = other.sample(args.shots)
        merlin_elapsed = perf_counter() - merlin_start
        validation = compare(actual, expected, converter)
        if model_name == "ideal" and (np.any(actual.detectors) or np.any(actual.observables)):
            raise AssertionError("Ideal direct circuit produced nonzero scoring outputs")
        result[model_name] = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "shots": args.shots,
            "prototype_setup_seconds": setup_seconds,
            "prototype_seconds_per_shot": elapsed / args.shots,
            "merlin_setup_seconds": merlin_setup,
            "merlin_seconds_per_shot": merlin_elapsed / args.shots,
            "total_stage_seconds": dict(stages),
            "fault_counts": dict(fault_counts),
            "prototype_outputs": output_summary(actual, converter),
            "merlin_outputs": output_summary(expected, converter),
            "validation": validation,
        }
        print(
            f"Fresh {model_name}: direct {1000 * elapsed / args.shots:.3f} ms/shot, "
            f"Merlin {1000 * merlin_elapsed / args.shots:.3f} ms/shot",
            flush=True,
        )
    return result


def extension_hash(name: str) -> str:
    module = importlib.import_module(name)
    assert module.__file__ is not None
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--fixed-shots", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.shots, args.fixed_shots) < 16:
        parser.error("At least 16 shots required")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = perf_counter()
    model, tail, source = sources(args.merlin_checkout, "gate_noise")
    start = perf_counter()
    reducer = ScoredClifford(model, tail)
    setup_seconds = perf_counter() - start
    _, physical, suffix, _ = region(model.render(()))
    unconstrained = "\n".join([f"RX {q}" for q in range(model.num_qubits)] + physical + suffix)
    try:
        ScoredClifford(Model(unconstrained), tail)
    except ValueError as error:
        if "do not cancel" not in str(error):
            raise
    else:
        raise AssertionError("Preparation identity accepted unconstrained computational support")
    document = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "merlin_revision": MERLIN_REVISION,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("merlin-sim", "qiskit-aer", "stim", "numpy")
        },
        "extension_sha256": {
            name: extension_hash(name) for name in ("clifft._clifft_core", "merlin._core")
        },
        "baseline_sha256": {
            name: hashlib.sha256(
                (ROOT / "research/conditional_clifford" / name).read_bytes()
            ).hexdigest()
            for name in ("bt27-scored-sampling.json", "bt27-scored-exact-equivalence.json")
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "seeds": {
            "fixed_clifft": 837291,
            "fixed_stim": 511121,
            "fresh_faults": 660701,
            "fresh_measurements": 73319,
            "fresh_merlin": 111917,
            "aer_cases": 194201,
        },
        "control_setup_seconds": setup_seconds,
        "preparation_identity": reducer.certificate,
        "unconstrained_preparation_negative_detected": True,
        "candidate_cz_pairs": len(reducer.pairs),
        "first_order_responses": len(reducer.first),
        "mixed_responses": len(reducer.mixed),
        "scoring_response_payload_bytes": sum(
            (value.bit_length() + 7) // 8 for value in reducer.first + list(reducer.mixed.values())
        ),
        "derivative_validation": verify_derivatives(reducer),
        "aer_validation": verify_aer(),
        "fixed_validation": validate_fixed(reducer, source, args.fixed_shots, args.smoke),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    document["stochastic_validation"] = stochastic(args)
    document["elapsed_seconds"] = perf_counter() - started
    document["limits"] = (
        "Offline research construction for supported BT27 Pauli/readout models with ideal scoring. "
        "No production executor change or history cache. Direct circuits use no optimizer. "
        "Fixed-case laws match the retained compiler-derived references; fresh statistical checks "
        "here are bug checks, not a repeated equivalence-margin study. Local timing is one run."
    )
    args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
