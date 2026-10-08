"""Complete BT27 target scoring with fresh categorical noise on every shot.

This research host constructs each variant offline before calling the existing
executor. It returns full record, detector, and logical-observable arrays. The
ideal scoring tail is kept identical to the pinned Merlin generator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import stim
from analyze_bt27_phase_corrections import region
from bt27_circuit_corrections import Controls, bits
from profile_bt27_fault_specialization import ROOT, optimize, source
from profile_external_feedback import MERLIN_REVISION
from study_circuit_noise import History, Model, synthetic_bt_noise
from validate_bt27_fault_specialization import parity_features

import clifft


def sources(checkout: Path, model_name: str) -> tuple[Model, str, str]:
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MERLIN_REVISION:
        raise ValueError("Unexpected Merlin generator revision")
    if subprocess.check_output(
        ["git", "-C", str(checkout), "diff", "HEAD", "--", "benchmarks"], text=True
    ):
        raise ValueError("Merlin generator has uncommitted changes")
    sys.path.insert(0, str(checkout.resolve()))
    build = importlib.import_module("benchmarks.protocols.code_switching").build_code_switching_case
    protocol = build("bt27", p_phys=0, target_scoring=False).circuit
    scored = build("bt27", p_phys=0, target_scoring=True).circuit
    if not scored.startswith(protocol) or Model.signature(protocol) != Model.signature(source()):
        raise ValueError("BT27 fixture and generator no longer agree")
    tail = scored[len(protocol) :]
    if model_name == "original":
        protocol = build("bt27", p_phys=0.001, target_scoring=False).circuit
        exact = build("bt27", p_phys=0.001, target_scoring=True).circuit
        if protocol + tail != exact:
            raise AssertionError("Original scored source differs from Merlin")
    elif model_name == "gate_noise":
        protocol = synthetic_bt_noise(protocol, {"prep", "phase", "decoder"}, 0.001)
    elif model_name != "ideal":
        raise ValueError("Unknown noise model")
    return Model(protocol), tail, protocol + tail


class ScoredSampler:
    def __init__(self, binary: Path, model: Model, tail: str):
        self.model, self.tail = model, tail
        self.controls = Controls(model)
        self.prefix, self.body, self.suffix, _ = region(model.render(()))
        self.temp = tempfile.TemporaryDirectory(prefix="bt27-scored-")
        self.directory = Path(self.temp.name)
        (self.directory / "prefix.stim").write_text("\n".join(self.prefix + self.body) + "\n")
        (self.directory / "protocol.stim").write_text(model.render(()))
        (self.directory / "scored.stim").write_text(model.render(()) + tail)
        self.errors = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            [str(binary.resolve()), str(self.directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            text=True,
        )
        self.counter = 0
        try:
            self.setup = self.response()
        except BaseException:
            self.close()
            raise

    def response(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            self.errors.seek(0)
            raise RuntimeError("Native scored sampler failed: " + self.errors.read())
        result: dict[str, Any] = json.loads(line)
        return result

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.errors.close()
        self.temp.cleanup()

    def __enter__(self) -> ScoredSampler:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def fixed(
        self, history: History, shots: int, seed: int, *, check: bool = False
    ) -> tuple[Any, dict[str, Any]]:
        started = perf_counter()
        correction = self.controls.evaluate(history)
        gates = self.controls.gates(correction)
        final = [f"Z {q}" for q in bits(correction.final_z)]
        final += [f"X {q}" for q in bits(correction.final_x)]
        flips = "".join(str(correction.records >> q & 1) for q in range(126)) + "0" * 9
        control_seconds = perf_counter() - started
        name = f"r_{self.counter}"
        self.counter += 1
        if check:
            moved = "\n".join(self.prefix + self.body + gates + self.suffix + final) + "\n"
            (self.directory / f"{name}.moved").write_text(moved + self.tail)
        request = f"{name} {shots} {seed} {int(check)} {len(gates)} {len(final)} {flips}\n"
        request += "".join(gate + "\n" for gate in gates + final)
        assert self.process.stdin is not None
        self.process.stdin.write(request)
        self.process.stdin.flush()
        row = self.response()
        if row["name"] != name or row["shots"] != shots or row["checked"] != check:
            raise AssertionError("Scored sampler response does not match its request")
        arrays = {}
        for field, columns in (("measurements", 135), ("detectors", 72), ("observables", 9)):
            values = np.frombuffer(row.pop(field).encode(), dtype="u1") - ord("0")
            if values.size != shots * columns or np.any(values > 1):
                raise AssertionError("Malformed scored sample array")
            arrays[field] = values.reshape(shots, columns)
        row["control_seconds"] = control_seconds
        return SimpleNamespace(**arrays), row

    def sample(self, shots: int, seed: int) -> tuple[Any, dict[str, Any]]:
        fault_rng, measurement_rng = random.Random(seed), random.Random(seed ^ 0x61F99A)
        arrays = {
            field: np.empty((shots, columns), dtype="u1")
            for field, columns in (("measurements", 135), ("detectors", 72), ("observables", 9))
        }
        rows, histories = [], []
        started = perf_counter()
        for i in range(shots):
            history = self.model.draw(fault_rng) if self.model.sites else ()
            sample, row = self.fixed(history, 1, measurement_rng.getrandbits(64))
            for field in arrays:
                arrays[field][i] = getattr(sample, field)[0]
            rows.append(row)
            histories.append(history)
        elapsed = perf_counter() - started
        return SimpleNamespace(**arrays), {
            "sample_seconds": elapsed,
            "seconds_per_shot": elapsed / shots,
            "unique_histories": len(set(histories)),
            "fault_counts": dict(sorted(Counter(map(len, histories)).items())),
            "history_sha256": hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
            "widths": dict(Counter(row["peak_width"] for row in rows)),
            "t_counts": dict(Counter(row["t_count"] for row in rows)),
            "native_peak_rss_kib": max(row["peak_rss_kib"] for row in rows),
            "total_stage_seconds": {
                field: sum(row[field] for row in rows)
                for field in (
                    "control_seconds",
                    "compose_seconds",
                    "plan_seconds",
                    "prepare_seconds",
                    "sample_seconds",
                    "restore_seconds",
                )
            },
        }


def converter_for(text: str) -> Any:
    # This evaluates record parities only; removing T gates does not change
    # record numbering, and no Stim reference sample is requested.
    clifford = "\n".join(
        line for line in text.splitlines() if not line.startswith(("T ", "T_DAG "))
    )
    return stim.Circuit(clifford).compile_m2d_converter(skip_reference_sample=True)


def features(sample: Any, converter: Any, *, check_reference: bool) -> np.ndarray:
    records = sample.measurements.astype(bool)
    detectors, observables = converter.convert(measurements=records, separate_observables=True)
    if check_reference:
        np.testing.assert_array_equal(sample.detectors, detectors)
        np.testing.assert_array_equal(sample.observables, observables)
    r, d, o = (1.0 - 2.0 * value for value in (records, detectors, observables))
    # Every nonempty parity determines the full nine-bit logical distribution;
    # detector correlations additionally test its association with syndromes.
    logical_parities = np.ones((len(records), 512))
    for mask in range(1, 512):
        low = mask & -mask
        logical_parities[:, mask] = logical_parities[:, mask ^ low] * o[:, low.bit_length() - 1]
    accepted = ~np.any(detectors, axis=1)
    return np.column_stack(
        (
            r,
            d,
            parity_features(records, 193, 32),
            parity_features(detectors, 751, 32),
            logical_parities[:, 1:],
            (d[:, :, None] * o[:, None, :]).reshape(len(records), -1),
            accepted,
            accepted[:, None] * logical_parities[:, 1:],
        )
    )


def compare(
    left: Any, right: Any, converter: Any, *, right_is_fixed_merlin: bool = False
) -> dict[str, Any]:
    a = features(left, converter, check_reference=True)
    b = features(right, converter, check_reference=not right_is_fixed_merlin)
    variance = a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b)
    difference = np.maximum(0, np.abs(a.mean(axis=0) - b.mean(axis=0)) - 2 / len(a) - 2 / len(b))
    scores = difference / np.maximum(np.sqrt(variance), 1e-12)
    if scores.max() > 6:
        raise AssertionError(f"Scored joint-output mismatch: {scores.max():.3f} sigma")
    return {"features": a.shape[1], "maximum_score": float(scores.max())}


def output_summary(sample: Any, converter: Any) -> dict[str, Any]:
    detectors, observables = converter.convert(
        measurements=sample.measurements.astype(bool), separate_observables=True
    )
    accepted = ~np.any(detectors, axis=1)
    logical = observables @ (1 << np.arange(9))
    return {
        "shots": len(logical),
        "zero_syndrome_shots": int(accepted.sum()),
        "shots_with_logical_events": int(np.count_nonzero(logical)),
        "zero_syndrome_shots_with_logical_events": int(np.count_nonzero(logical[accepted])),
        "logical_pattern_counts": dict(sorted(Counter(map(int, logical)).items())),
        "records_sha256": hashlib.sha256(sample.measurements.tobytes()).hexdigest(),
    }


def worker(args: Any) -> None:
    started = perf_counter()
    model, tail, text = sources(args.merlin_checkout, args.model)
    if args.worker == "prototype":
        with ScoredSampler(args.binary, model, tail) as sampler:
            setup = perf_counter() - started
            sample, telemetry = sampler.sample(args.shots, args.seed)
            telemetry["native_setup"] = sampler.setup
    else:
        merlin = importlib.import_module("merlin")
        reference_sampler = merlin.CircuitSampler(text, seed=args.seed)
        setup = perf_counter() - started
        start = perf_counter()
        sample = reference_sampler.sample(args.shots)
        elapsed = perf_counter() - start
        telemetry = {"sample_seconds": elapsed, "seconds_per_shot": elapsed / args.shots}
    # Materialization is part of sample time; disk interchange is separate.
    telemetry["setup_seconds"] = setup
    # VmHWM starts with this executable image, avoiding a pre-exec parent peak.
    telemetry["python_process_peak_rss_kib"] = next(
        int(line.split()[1])
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmHWM:")
    )
    telemetry["source_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    np.savez(
        args.samples,
        measurements=sample.measurements,
        detectors=sample.detectors,
        observables=sample.observables,
    )
    print(json.dumps(telemetry), flush=True)


def decoder_boundary_control(sample: Any, model: Model, tail: str) -> dict[str, Any]:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    logical = {q: i for i, q in enumerate((0, 1, 2, 27, 28, 29, 54, 55, 56))}
    scoring = QuantumCircuit(9)
    for node in clifft.parse(tail).nodes:
        gate = node.gate.name
        if gate in ("MX", "OBSERVABLE_INCLUDE"):
            continue
        targets = [logical[target.value] for target in node.targets]
        if gate == "CX":
            scoring.cx(*targets)
        elif gate == "T":
            scoring.t(targets)
        elif gate == "T_DAG":
            scoring.tdg(targets)
        else:
            raise AssertionError("Unexpected logical scoring gate")
    circuit = QuantumCircuit(9)
    circuit.h(range(9))
    circuit.compose(scoring.inverse(), inplace=True)
    circuit.x(0)
    circuit.compose(scoring, inplace=True)
    circuit.h(range(9))
    circuit.save_statevector()
    state = AerSimulator(method="statevector").run(circuit).result().get_statevector()
    probabilities = np.abs(np.asarray(state)) ** 2
    patterns = sample.observables @ (1 << np.arange(9))
    empirical = np.bincount(patterns, minlength=512) / len(patterns)
    deviation = np.maximum(0, np.abs(empirical - probabilities) - 2 / len(patterns))
    stderr = np.sqrt(probabilities * (1 - probabilities) / len(patterns))
    score = float(np.max(deviation / np.maximum(stderr, 1e-12)))
    if score > 6 or np.any(sample.detectors):
        raise AssertionError("Isolated decoder X disagrees with the logical Aer oracle")

    # Deliberately commute X through CCZ as if it were only a readout change.
    # X after target scoring cannot change the subsequent X-basis measurement.
    wrong_text = model.render(()) + tail.replace("MX ", "X 0\nMX ", 1)
    wrong_hir, info = optimize(wrong_text)
    if info["peak_width"] > 16:
        raise AssertionError("Negative control exceeds the execution budget")
    wrong = clifft.sample(
        clifft.lower(wrong_hir, expected_detectors=[0] * 72, expected_observables=[0] * 9),
        shots=len(patterns),
        seed=91518,
        threads=1,
        batch_size=1,
    )
    events = int(np.count_nonzero(patterns))
    if np.any(wrong.observables) or events < len(patterns) / 2:
        raise AssertionError("Decoder-boundary negative control did not distinguish the error")
    return {
        "aer_maximum_score": score,
        "aer_logical_event_probability": float(1 - probabilities[0]),
        "correct_logical_event_shots": events,
        "incorrectly_commuted_logical_event_shots": 0,
        "shots": len(patterns),
    }


def validate_fixed(args: Any) -> dict[str, Any]:
    from study_bt27_circuit_reuse import histories

    merlin = importlib.import_module("merlin")
    model, tail, text = sources(args.merlin_checkout, "gate_noise")
    converter = converter_for(text)
    cases = histories(
        model,
        json.loads((ROOT / "research/conditional_clifford/circuit-noise-study.json").read_text()),
    )
    # A small optional smoke run retains identity, a readout fault, an internal
    # phase fault, a decoder fault, and the two all-sites stress cases.
    if args.smoke:
        keys = [next(iter(cases))]
        keys += [
            next(k for k, (group, _) in cases.items() if group == group_name)
            for group_name in (
                "READOUT_NOISE:before_first_t",
                "DEPOLARIZE2:between_t_gates",
                "DEPOLARIZE2:after_last_t",
            )
        ]
        keys += [k for k in cases if k.startswith("dense_4170_")]
        cases = {k: cases[k] for k in keys}
    cases["decoder_x_before_scoring"] = ("isolated_logical_x", ((3336, 0),))
    rows = {}
    ideal, ideal_info = optimize(model.render(()) + tail)
    if ideal_info["peak_width"] > 16:
        raise AssertionError("Ideal scored reference exceeds execution budget")
    reference = clifft.compute_reference_syndrome(ideal)
    if any(reference["detectors"]) or any(reference["observables"]):
        raise AssertionError("Expected the ideal zero scoring reference")
    with ScoredSampler(args.binary, model, tail) as sampler:
        isolated = sampler.controls.evaluate(cases["decoder_x_before_scoring"][1])
        if (
            isolated.x,
            isolated.s,
            isolated.z,
            isolated.cz,
            isolated.records,
            isolated.final_x,
            isolated.final_z,
        ) != (0, 0, 0, 0, 0, 1, 0):
            raise AssertionError("Pinned decoder fault no longer isolates physical X0")
        for name, (group, history) in cases.items():
            sample, row = sampler.fixed(history, args.fixed_shots, 381917, check=True)
            physical = model.render(history) + tail
            fresh_hir, fresh_info = optimize(physical)
            if fresh_info["peak_width"] > 16:
                raise AssertionError("Fixed scored oracle exceeds execution budget")
            fresh = clifft.sample(
                clifft.lower(
                    fresh_hir,
                    expected_detectors=reference["detectors"],
                    expected_observables=reference["observables"],
                ),
                shots=args.fixed_shots,
                seed=281313,
                threads=1,
                batch_size=1,
            )
            independent = merlin.CircuitSampler(physical, seed=153191).sample(args.fixed_shots)
            rows[name] = {
                "group": group,
                "history": history,
                "fault_count": len(history),
                "peak_width": row["peak_width"],
                "t_count": row["t_count"],
                "raw_boundary_matches": row["checked"],
                "fresh_peak_width": fresh_info["peak_width"],
                "fresh_t_count": fresh_info["t_count"],
                "fresh_validation": compare(sample, fresh, converter),
                "merlin_validation": compare(
                    sample, independent, converter, right_is_fixed_merlin=True
                ),
            }
            if name == "decoder_x_before_scoring":
                rows[name]["boundary_negative_control"] = decoder_boundary_control(
                    sample, model, tail
                )
            if len(rows) % 16 == 0:
                print(f"Validated {len(rows)} scored fixed histories", flush=True)
    return {"shots_per_case_per_sampler": args.fixed_shots, "cases": rows}


def inspect_stochastic_sources(checkout: Path) -> dict[str, Any]:
    result = {}
    for name in ("ideal", "original", "gate_noise"):
        _, _, text = sources(checkout, name)
        _, result[name] = optimize(text)
    return result


def write_output(path: Path, document: dict[str, Any]) -> None:
    metadata = {key: value for key, value in document.items() if key != "fixed_validation"}
    fixed = document["fixed_validation"]
    # Dense histories must remain reproducible without expanding past the
    # repository's research-artifact size budget.
    rows = ",\n".join(
        f"      {json.dumps(key)}: {json.dumps(value)}" for key, value in fixed["cases"].items()
    )
    text = json.dumps(metadata, indent=2).removesuffix("\n}")
    path.write_text(
        text
        + ',\n  "fixed_validation": {\n    "shots_per_case_per_sampler": '
        + str(fixed["shots_per_case_per_sampler"])
        + ',\n    "cases": {\n'
        + rows
        + "\n    }\n  }\n}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--fixed-shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", choices=("prototype", "merlin"))
    parser.add_argument("--model", choices=("ideal", "original", "gate_noise"))
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()
    if args.shots < 16 or not 16 <= args.fixed_shots <= 65536:
        parser.error("At least 16 shots are required; fixed cases are limited to 65536")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.worker:
        if args.model is None or args.samples is None:
            parser.error("Workers require a model and a sample-output path")
        worker(args)
        return
    if args.output is None:
        parser.error("--output is required")
    started = perf_counter()
    core = importlib.import_module("merlin._core")
    assert core.__file__ is not None
    document: dict[str, Any] = {
        "merlin_revision": MERLIN_REVISION,
        "merlin_version": importlib.metadata.version("merlin-sim"),
        "merlin_extension_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "stochastic_prototype_seed": args.seed,
        "stochastic_merlin_seed": args.seed + 1,
        "fixed_seeds": {"prototype": 381917, "fresh_clifft": 281313, "merlin": 153191},
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "shots_per_stochastic_sampler": args.shots,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "stochastic_source_compilation": inspect_stochastic_sources(args.merlin_checkout),
        "fixed_validation": validate_fixed(args),
        "stochastic_models": {},
        "limitations": (
            "Research host constructs every variant before ordinary execution; no production API. "
            "Single local timing run per model, finite distribution checks, no postselection. "
            "Scoring is ideal and identical to Merlin. RSS fields are separate process high-water "
            "marks, not additive simultaneous unique resident memory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bt27-scored-validation-") as temp:
        for model_name in ("ideal", "original", "gate_noise"):
            _, _, text = sources(args.merlin_checkout, model_name)
            converter = converter_for(text)
            outputs, telemetry = {}, {}
            for backend in ("prototype", "merlin"):
                print(f"Sampling complete {model_name} with {backend}", flush=True)
                sample_path = Path(temp) / f"{model_name}-{backend}.npz"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--binary",
                    str(args.binary.resolve()),
                    "--merlin-checkout",
                    str(args.merlin_checkout.resolve()),
                    "--worker",
                    backend,
                    "--model",
                    model_name,
                    "--shots",
                    str(args.shots),
                    "--samples",
                    str(sample_path),
                    "--seed",
                    str(args.seed + (1 if backend == "merlin" else 0)),
                ]
                telemetry[backend] = json.loads(subprocess.check_output(command, text=True))
                with np.load(sample_path) as data:
                    outputs[backend] = SimpleNamespace(
                        **{key: data[key].copy() for key in data.files}
                    )
                telemetry[backend]["outputs"] = output_summary(outputs[backend], converter)
                if model_name == "ideal" and (
                    np.any(outputs[backend].detectors) or np.any(outputs[backend].observables)
                ):
                    raise AssertionError("Ideal scored circuit produced detector or logical events")
            if telemetry["prototype"]["source_sha256"] != telemetry["merlin"]["source_sha256"]:
                raise AssertionError("Sampler source mismatch")
            telemetry["validation"] = compare(outputs["prototype"], outputs["merlin"], converter)
            telemetry["sample_time_ratio"] = (
                telemetry["prototype"]["sample_seconds"] / telemetry["merlin"]["sample_seconds"]
            )
            document["stochastic_models"][model_name] = telemetry
            document["elapsed_seconds"] = perf_counter() - started
            write_output(args.output, document)
            print(
                f"{model_name}: {telemetry['validation']}; "
                f"time ratio {telemetry['sample_time_ratio']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
