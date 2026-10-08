"""Validate offline reduced-core reuse with faults throughout BT27.

The sampled histories retain live measurement outcomes. Physical Pauli faults
and readout flips become a boundary Clifford, record flips, and a final Pauli.
Only the existing native boundary diagnostic plans or executes programs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import random
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import stim
from analyze_bt27_phase_corrections import region
from bt27_circuit_corrections import Controls, Correction, bits
from profile_bt27_fault_specialization import FIXTURE, ROOT, source
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from study_bt27_boundary_reuse import plan_stats, read_samples, summarize, write_output
from study_circuit_noise import History, Model, synthetic_bt_noise
from validate_bt27_fault_specialization import compare, probes
from validate_bt27_phase_corrections import append_gates, parity_rotation


def validate_phase_operators() -> dict[str, Any]:
    rng = random.Random(15191)
    pairs = []
    fault_counts = []
    for _ in range(64):
        gates = [
            gate
            for _ in range(8)
            for gate in parity_rotation(
                rng.sample(range(4), rng.randint(1, 4)), rng.choice([False, True])
            )
        ]
        model = Model(
            "\n".join(line for gate in gates for line in (gate, "DEPOLARIZE1(0.1) 0 1 2 3"))
        )
        controls = Controls(model, (0, len(gates)))
        history = model.draw(rng)
        fault_counts.append(len(history))
        initial = QuantumCircuit(8)
        for q in range(4):
            initial.h(q)
            initial.cx(q, q + 4)
        left, right = initial.copy(), initial.copy()
        append_gates(left, model.render(history).splitlines())
        append_gates(right, gates + controls.gates(controls.evaluate(history)))
        pairs.append((left, right))
    # Two flipped T signs add two S_DAG terms, producing a required Z. A
    # control that only XORs their S bits would incorrectly erase that Z.
    model = Model("X_ERROR(0.1) 0\nT 0\nT 0\n")
    controls = Controls(model, (0, 2))
    correction = controls.evaluate(((0, 0),))
    if correction.z != 1 or correction.s:
        raise AssertionError("Expected the carry between two S corrections")
    initial = QuantumCircuit(2)
    initial.h(0)
    initial.cx(0, 1)
    left, correct, incomplete = initial.copy(), initial.copy(), initial.copy()
    append_gates(left, model.render(((0, 0),)).splitlines())
    append_gates(correct, model.render(()).splitlines() + controls.gates(correction))
    append_gates(
        incomplete, model.render(()).splitlines() + controls.gates(replace(correction, z=0))
    )
    pairs.append((left, correct))
    circuits = [circuit for pair in pairs for circuit in pair] + [incomplete]
    for circuit in circuits:
        circuit.save_statevector()
    results = AerSimulator(method="statevector", max_parallel_threads=1).run(circuits).result()
    maximum_error = 0.0
    for i in range(len(pairs)):
        a = np.asarray(results.get_statevector(2 * i))
        b = np.asarray(results.get_statevector(2 * i + 1))
        overlap = np.vdot(b, a)
        aligned = b * overlap / abs(overlap) if abs(overlap) else b
        maximum_error = max(maximum_error, float(np.max(np.abs(a - aligned))))
        np.testing.assert_allclose(a, aligned, atol=1e-12, rtol=0)
    negative_overlap = float(
        abs(
            np.vdot(
                results.get_statevector(len(circuits) - 3),
                results.get_statevector(len(circuits) - 1),
            )
        )
    )
    if negative_overlap > 1e-12:
        raise AssertionError("The omitted-carry negative control was not detected")
    return {
        "choi_operator_pairs": len(pairs),
        "random_case_fault_count_range": [min(fault_counts), max(fault_counts)],
        "maximum_amplitude_error_up_to_global_phase": maximum_error,
        "omitted_s_carry_negative_control_overlap": negative_overlap,
    }


def restored(samples: Any, correction: Correction, probe_list: list[str], converter: Any) -> Any:
    before_det, before_obs = converter.convert(
        measurements=samples.measurements.astype(bool), separate_observables=True
    )
    np.testing.assert_array_equal(samples.detectors, before_det)
    np.testing.assert_array_equal(samples.observables, before_obs)
    flips = np.array(
        [correction.records >> q & 1 for q in range(samples.measurements.shape[1])], dtype="u1"
    )
    records = samples.measurements ^ flips
    detectors, observables = converter.convert(
        measurements=records.astype(bool), separate_observables=True
    )
    signs = []
    for probe in probe_list:
        parity = 0
        for target in probe.split("*"):
            axis, q = target[0], int(target[1:])
            if axis in "XY":
                parity ^= correction.final_z >> q & 1
            if axis in "YZ":
                parity ^= correction.final_x >> q & 1
        signs.append(1 - 2 * parity)
    return SimpleNamespace(
        measurements=records,
        detectors=detectors,
        observables=observables,
        exp_vals=samples.exp_vals * np.array(signs),
    )


def validate_clifford_flows(
    model: Model, controls: Controls, cases: dict[str, tuple[str, History]]
) -> dict[str, Any]:
    # Retain instruction positions and replace T gates by identities. This
    # isolates the record/feedback and output-frame algebra from phase reduction.
    clifford = copy.copy(model)
    clifford.entries = [
        replace(entry, gate="I") if entry.gate in ("T", "T_DAG") else entry
        for entry in model.entries
    ]
    linear = Controls(clifford, controls.phase_bounds)
    end = linear.phase_bounds[1]
    ideal = clifford.render(()).splitlines()
    checked = 0
    record_negative = frame_negative = 0
    for name, (_, history) in cases.items():
        correction = linear.evaluate(history)
        original = stim.Circuit(clifford.render(history))
        relocated = stim.Circuit("\n".join(ideal[:end] + linear.gates(correction) + ideal[end:]))
        expected = {str(flow) for flow in original.flow_generators()}
        actual: set[str] = set()
        missing_records: set[str] = set()
        missing_frame: set[str] = set()
        for flow in relocated.flow_generators():
            output = flow.output_copy()
            record_parity = sum(correction.records >> q & 1 for q in flow.measurements_copy()) % 2
            frame_parity = 0
            for q in range(len(output)):
                if output[q] in (1, 2):
                    frame_parity ^= correction.final_z >> q & 1
                if output[q] in (2, 3):
                    frame_parity ^= correction.final_x >> q & 1
            for result, parity in (
                (actual, record_parity ^ frame_parity),
                (missing_records, frame_parity),
                (missing_frame, record_parity),
            ):
                result.add(
                    str(
                        stim.Flow(
                            input=flow.input_copy(),
                            output=output * (-1 if parity else 1),
                            measurements=flow.measurements_copy(),
                        )
                    )
                )
            checked += 1
        if actual != expected:
            raise AssertionError(f"Clifford stabilizer-flow reconstruction failed: {name}")
        record_negative += missing_records != expected
        frame_negative += missing_frame != expected
    if not record_negative or not frame_negative:
        raise AssertionError("Record and output-frame omissions must each fail the flow oracle")
    return {
        "stim_version": stim.__version__,
        "patterns": len(cases),
        "signed_flow_generators_checked": checked,
        "omitted_record_restoration_negative_cases_detected": record_negative,
        "omitted_final_pauli_negative_cases_detected": frame_negative,
        "scope": "Exact signed flow-generator sets on a T-to-I Clifford control",
    }


def histories(model: Model, study: dict[str, Any]) -> dict[str, tuple[str, History]]:
    result: dict[str, tuple[str, History]] = {"identity": ("identity", ())}
    for i, row in enumerate(study["models"]["bt27_all_gate_noise"]["fixed"]):
        history = tuple((site, outcome) for site, outcome in row["history"])
        if history:
            result[f"previous_{i}"] = "previous_full_noise", history
    groups: dict[tuple[str, str], list[int]] = {}
    for index, noise_site in enumerate(model.sites):
        groups.setdefault((noise_site.gate, noise_site.position), []).append(index)
    for group, indices in groups.items():
        site = indices[len(indices) // 2]
        for outcome in range(len(model.sites[site].outcomes)):
            result[f"single_{site}_{outcome}"] = ":".join(group), ((site, outcome),)
    rng = random.Random(615197)
    for i in range(64):
        result[f"sampled_{i}"] = "sampled_p001", model.draw(rng)
    high_noise = Model(synthetic_bt_noise(source(), {"prep", "phase", "decoder"}, 0.005))
    for i in range(16):
        result[f"higher_{i}"] = "sampled_p005", high_noise.draw(rng)
    for weight in (40, 128, 512, len(model.sites)):
        for i in range(2):
            history = tuple(
                (site, rng.randrange(len(model.sites[site].outcomes)))
                for site in sorted(rng.sample(range(len(model.sites)), weight))
            )
            result[f"dense_{weight}_{i}"] = "dense_stress", history
    return result


def reuse_statistics(model: Model, controls: Controls) -> dict[str, Any]:
    rng = random.Random(20261009)
    sampled = [model.draw(rng) for _ in range(10000)]
    start = perf_counter()
    summaries = [controls.evaluate(history) for history in sampled]
    elapsed = perf_counter() - start
    return {
        "draws": len(sampled),
        "unique_complete_histories": len(set(sampled)),
        "unique_full_correction_summaries": len(set(summaries)),
        "unique_boundary_cliffords": len({(c.x, c.s, c.z, c.cz) for c in summaries}),
        "total_control_evaluation_seconds": elapsed,
        "mean_control_evaluation_seconds": elapsed / len(sampled),
        "limitations": (
            "Algebraic summaries are physical corrections, not minimized actions on the "
            "reachable state. Counts do not establish executable equivalence classes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--study",
        type=Path,
        default=ROOT / "research/conditional_clifford/circuit-noise-study.json",
    )
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--without-probes", action="store_true")
    args = parser.parse_args()
    if not 16 <= args.shots <= 65536:
        parser.error("shots must be between 16 and 65536")
    started = perf_counter()
    study = json.loads(args.study.read_text())
    reference = study["models"]["bt27_all_gate_noise"]["reference"]
    if len(reference["detectors"]) != 72 or any(reference["detectors"]) or reference["observables"]:
        raise ValueError("Expected BT27's ideal zero reference")
    model = Model(synthetic_bt_noise(source(), {"prep", "phase", "decoder"}, 0.001))
    if (
        hashlib.sha256(model.render(None).encode()).hexdigest()
        != study["models"]["bt27_all_gate_noise"]["source_sha256"]
    ):
        raise ValueError("The noise model changed relative to the source study")
    start = perf_counter()
    controls = Controls(model)
    setup_seconds = perf_counter() - start
    aer = validate_phase_operators()
    print(f"Aer operator checks: {aer}", flush=True)
    reuse = reuse_statistics(model, controls)
    print(f"Precomputed controls: {controls.metadata()}; reuse: {reuse}", flush=True)
    cases = histories(model, study)
    clifford_validation = validate_clifford_flows(model, controls, cases)
    print(f"Stim flow checks: {clifford_validation}", flush=True)
    prefix, body, suffix, _ = region(model.render(()))
    probe_list = [] if args.without_probes else probes()
    probe_text = "".join(f"EXP_VAL {probe}\n" for probe in probe_list)
    converter = stim.Circuit(
        "\n".join(
            line for line in model.render(()).splitlines() if not line.startswith(("T ", "T_DAG "))
        )
    ).compile_m2d_converter(skip_reference_sample=True)
    merlin = importlib.import_module("merlin")
    results: dict[str, Any] = {}
    corrections, formula_times = {}, {}
    with tempfile.TemporaryDirectory(prefix="bt27-circuit-reuse-") as temp:
        directory = Path(temp)
        (directory / "prefix.stim").write_text("\n".join(prefix + body) + "\n")
        (directory / "full.stim").write_text(model.render(()) + probe_text)
        (directory / "names.txt").write_text("\n".join(cases) + "\n")
        for i, (key, (_, history)) in enumerate(cases.items()):
            start = perf_counter()
            correction = controls.evaluate(history)
            gates = controls.gates(correction)
            formula_times[key] = perf_counter() - start
            corrections[key] = correction
            (directory / f"{i}.correction").write_text("\n".join(gates + ["I 134"]) + "\n")
            (directory / f"{i}.moved").write_text(
                "\n".join(prefix + body + gates + suffix) + "\n" + probe_text
            )
            (directory / f"{i}.original").write_text(model.render(history) + probe_text)
        with subprocess.Popen(
            [str(args.binary.resolve()), str(directory), str(args.shots)],
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            assert process.stdout is not None
            try:
                setup = json.loads(next(process.stdout))
                print(f"Shared prefix: {setup}", flush=True)
                for i, line in enumerate(process.stdout):
                    row = json.loads(line)
                    key = row["name"]
                    if key != list(cases)[i]:
                        raise AssertionError("Native case order changed")
                    correction = corrections[key]
                    row["correction_formula_seconds"] = formula_times[key]
                    row["group"], row["history"] = cases[key]
                    row["fault_count"] = len(cases[key][1])
                    row["correction"] = asdict(correction)
                    row["correction_gate_counts"] = dict(
                        Counter(gate.split()[0] for gate in controls.gates(correction))
                    )
                    row["flipped_records"] = bits(correction.records)
                    shared = read_samples(
                        directory / f"{i}.shared.bin", args.shots, len(probe_list)
                    )
                    start = perf_counter()
                    shared = restored(shared, correction, probe_list, converter)
                    row["output_restoration_seconds"] = perf_counter() - start
                    fresh = read_samples(directory / f"{i}.fresh.bin", args.shots, len(probe_list))
                    row["fresh_validation"] = compare(shared, fresh, converter, reference)
                    independent = merlin.CircuitSampler(
                        (directory / f"{i}.original").read_text(), seed=715591
                    ).sample(args.shots)
                    row["merlin_validation"] = compare(shared, independent, converter, reference)
                    row["shared_plan"] = plan_stats(directory / f"{i}.plan", row["prefix_actions"])
                    row["fresh_plan"] = plan_stats(directory / f"{i}.fresh.plan", 0)
                    results[key] = row
                    if len(results) % 16 == 0:
                        print(f"Validated {len(results)} circuit-noise patterns", flush=True)
                if process.wait() != 0:
                    raise RuntimeError("Native boundary diagnostic failed")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    if results.keys() != cases.keys():
        raise AssertionError("Incomplete full-circuit sweep")
    summary = summarize(results)
    summary["median_output_restoration_seconds"] = statistics.median(
        row["output_restoration_seconds"] for row in results.values()
    )
    core = importlib.import_module("merlin._core")
    assert core.__file__ is not None
    output = {
        "study_sha256": hashlib.sha256(args.study.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "merlin_version": importlib.metadata.version("merlin-sim"),
        "merlin_extension_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "shots_per_pattern_per_sampler": args.shots,
        "pauli_probes": len(probe_list),
        "control_setup_seconds": setup_seconds,
        "control_structure": controls.metadata(),
        "aer_operator_validation": aer,
        "stim_clifford_flow_validation": clifford_validation,
        "history_reuse": reuse,
        "setup": setup,
        "summary": summary,
        "limitations": (
            "Offline physical control evaluation, output restoration, composition and planning; "
            "no shared executor state. Finite full-circuit probes, not tomography. "
            "Stratified/stress cases are not probability-weighted estimates. Local timings "
            "include probe instrumentation and may overlap independent validation."
        ),
        "elapsed_seconds": perf_counter() - started,
        "results": results,
    }
    write_output(args.output, output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
