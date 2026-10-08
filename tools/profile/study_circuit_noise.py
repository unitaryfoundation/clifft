"""Broaden fault specialization to gate and readout noise across full circuits.

Synthetic BT gate noise is a diagnostic model without idle noise or a hardware
schedule. Existing cultivation noise is retained or uniformly rescaled. Wide
programs are inspected without allocating their dense execution state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import stim
from analyze_bt27_phase_corrections import region
from profile_bt27_fault_specialization import FIXTURE, optimize, source
from profile_external_feedback import MERLIN_REVISION
from validate_bt27_fault_specialization import compare, probes

import clifft

History = tuple[tuple[int, int], ...]
CHANNELS = {"DEPOLARIZE1": 1, "DEPOLARIZE2": 2, "X_ERROR": 1, "Y_ERROR": 1, "Z_ERROR": 1}
MEASUREMENTS = {"M", "MX", "MY", "MPP"}


@dataclass
class Entry:
    gate: str
    targets: list[str]
    args: list[float]
    sites: tuple[int, ...] = ()
    readout_site: int | None = None

    def text(self, *, flip: bool = False, probability: float | None = None) -> str:
        targets = self.targets.copy()
        if flip:
            targets[0] = targets[0][1:] if targets[0].startswith("!") else "!" + targets[0]
        args = [probability] if probability is not None else self.args
        arg_text = "(" + ",".join(format(x, ".17g") for x in args) + ")" if args else ""
        separator = "*" if self.gate == "MPP" else " "
        return self.gate + arg_text + " " + separator.join(targets)


@dataclass(frozen=True)
class Site:
    gate: str
    probability: float
    targets: tuple[str, ...]
    outcomes: tuple[str, ...]
    position: str


class Model:
    def __init__(self, text: str):
        circuit = clifft.parse(text)
        self.num_qubits = circuit.num_qubits
        self.entries: list[Entry] = []
        self.sites: list[Site] = []
        records = 0
        phase_indices = [i for i, n in enumerate(circuit.nodes) if n.gate.name in ("T", "T_DAG")]
        for index, node in enumerate(circuit.nodes):
            gate = node.gate.name
            if node.tag:
                raise ValueError("Tagged instructions are outside this diagnostic")
            position = (
                "no_t_gates"
                if not phase_indices
                else "before_first_t"
                if index < phase_indices[0]
                else "after_last_t"
                if index > phase_indices[-1]
                else "between_t_gates"
            )
            if gate == "READOUT_NOISE":
                if self.entries[-1].gate not in MEASUREMENTS or len(node.args) != 1:
                    raise ValueError("Expected readout noise immediately after a measurement")
                if node.targets[0].value != records - 1:
                    raise ValueError("Expected readout noise on the latest record")
                self.entries[-1].readout_site = len(self.sites)
                self.sites.append(Site(gate, node.arg, (), ("flip",), position))
                continue
            targets = [f"rec[{t.value - records}]" if t.is_rec else repr(t) for t in node.targets]
            args = [] if gate in MEASUREMENTS else list(node.args)
            entry = Entry(gate, targets, args)
            if gate in CHANNELS:
                arity = CHANNELS[gate]
                if len(targets) % arity or len(args) != 1:
                    raise ValueError("Malformed independent Pauli channel")
                ids = []
                outcomes = (
                    tuple("".join(x) for x in product("IXYZ", repeat=arity) if set(x) != {"I"})
                    if gate.startswith("DEPOLARIZE")
                    else (gate[0],)
                )
                for first in range(0, len(targets), arity):
                    ids.append(len(self.sites))
                    self.sites.append(
                        Site(
                            gate,
                            node.arg,
                            tuple(targets[first : first + arity]),
                            outcomes,
                            position,
                        )
                    )
                entry.sites = tuple(ids)
            elif "ERROR" in gate or "CHANNEL" in gate or gate.startswith("DEPOLARIZE"):
                raise ValueError(f"Unsupported noise channel {gate}")
            self.entries.append(entry)
            if gate in MEASUREMENTS:
                records += 1 if gate == "MPP" else len(targets)
        if records != circuit.num_measurements:
            raise AssertionError("Measurement count changed in model extraction")
        self.num_measurements = records
        # Reconstruction must retain feedback offsets, inverted targets, and
        # every noise instruction before any histories are specialized.
        original = self.signature(text)
        if self.signature(self.render(None)) != original:
            raise AssertionError("Noise-model reconstruction changed parsed instructions")

    @staticmethod
    def signature(text: str) -> list[tuple[Any, ...]]:
        return [
            (node.gate.name, tuple(node.args), tuple(map(repr, node.targets)))
            for node in clifft.parse(text).nodes
        ]

    def render(self, history: History | None) -> str:
        fixed = dict(history or ())
        lines = []
        for entry in self.entries:
            if entry.sites:
                if history is None:
                    lines.append(entry.text())
                else:
                    for index in entry.sites:
                        if index in fixed:
                            site = self.sites[index]
                            outcome = site.outcomes[fixed[index]]
                            lines += [f"{p} {q}" for p, q in zip(outcome, site.targets) if p != "I"]
            elif entry.readout_site is not None:
                probability = (
                    self.sites[entry.readout_site].probability if history is None else None
                )
                lines.append(entry.text(flip=entry.readout_site in fixed, probability=probability))
            else:
                lines.append(entry.text())
        return "\n".join(lines) + "\n"

    def draw(self, rng: random.Random) -> History:
        probabilities = {site.probability for site in self.sites}
        if len(probabilities) != 1:
            raise ValueError("This sparse history sampler requires uniform site probabilities")
        p = next(iter(probabilities))
        if p == 0:
            return ()
        if not 0 < p < 1:
            raise ValueError("Sparse history sampling requires probability below one")
        result = []
        index = 0
        while index < len(self.sites):
            index += int(math.log1p(-rng.random()) / math.log1p(-p))
            if index < len(self.sites):
                result.append((index, rng.randrange(len(self.sites[index].outcomes))))
                index += 1
        return tuple(result)


def synthetic_bt_noise(text: str, regions: set[str], p: float) -> str:
    prefix, body, _, _ = region(text)
    lines = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip() or line.lstrip().startswith("#"):
            lines.append(line)
            continue
        where = (
            "prep"
            if index < len(prefix)
            else "phase"
            if index < len(prefix) + len(body)
            else "decoder"
        )
        if where not in regions:
            lines.append(line)
            continue
        gate, *targets = line.split()
        if gate in ("M", "MX"):
            lines.append(f"{gate}({p}) " + " ".join(targets))
        else:
            lines.append(line)
            if gate == "CX":
                channel = "DEPOLARIZE1" if targets[0].startswith("rec") else "DEPOLARIZE2"
                noisy_targets = targets[1:] if channel == "DEPOLARIZE1" else targets
                lines.append(f"{channel}({p}) " + " ".join(noisy_targets))
            elif gate in ("T", "T_DAG", "RX"):
                channel = "Z_ERROR" if gate == "RX" else "DEPOLARIZE1"
                lines.append(f"{channel}({p}) " + " ".join(targets))
            elif gate != "DETECTOR":
                raise ValueError(f"Unspecified synthetic noise policy for {gate}")
    return "\n".join(lines) + "\n"


def history_summary(model: Model, count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    histories = [model.draw(rng) for _ in range(count)]
    log_p0 = sum(math.log1p(-site.probability) for site in model.sites)
    p0 = math.exp(log_p0)
    collision = math.exp(
        sum(
            math.log((1 - s.probability) ** 2 + s.probability**2 / len(s.outcomes))
            for s in model.sites
        )
    )
    return {
        "noise_sites": len(model.sites),
        "site_types": dict(Counter(s.gate for s in model.sites)),
        "site_positions": dict(Counter(s.position for s in model.sites)),
        "expected_faults": sum(s.probability for s in model.sites),
        "probability_zero_faults": p0,
        "probability_at_most_one_fault": p0
        * (1 + sum(s.probability / (1 - s.probability) for s in model.sites)),
        "independent_pair_exact_history_collision_probability": collision,
        "draws": count,
        "unique_histories": len(set(histories)),
        "repeat_fraction_unbounded_history_cache": (count - len(set(histories))) / count,
        "empirical_fault_counts": dict(sorted(Counter(map(len, histories)).items())),
    }


def compare_observables(c: Any, m: Any, converter: Any) -> float:
    _, left = converter.convert(measurements=c.measurements.astype(bool), separate_observables=True)
    _, right = converter.convert(
        measurements=m.measurements.astype(bool), separate_observables=True
    )
    if left.shape[1] == 0:
        return 0.0
    a, b = 1.0 - 2.0 * left, 1.0 - 2.0 * right
    variance = a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b)
    score = np.maximum(
        0, np.abs(a.mean(axis=0) - b.mean(axis=0)) - 2 / len(a) - 2 / len(b)
    ) / np.maximum(np.sqrt(variance), 1e-12)
    if score.max() > 6:
        raise AssertionError("Independent observable mismatch")
    return float(score.max())


def study_model(name: str, model: Model, args: Any, merlin: Any) -> dict[str, Any]:
    ideal, ideal_info = optimize(model.render(()))
    _, noisy_info = optimize(model.render(None))
    output: dict[str, Any] = {
        "source_sha256": hashlib.sha256(model.render(None).encode()).hexdigest(),
        "num_qubits": model.num_qubits,
        "num_measurements": model.num_measurements,
        "parsed_instruction_roundtrip": "passed",
        "history_statistics": history_summary(model, args.history_draws, args.seed),
        "ideal": ideal_info,
        "unfixed_noise": noisy_info,
        "fixed": [],
    }
    can_reference = ideal_info["peak_width"] <= args.max_width
    reference = clifft.compute_reference_syndrome(ideal) if can_reference else None
    output["reference"] = reference
    # Removing T gates changes quantum evolution, but not this converter's
    # record-parity function because its own reference sampling is disabled.
    converter = stim.Circuit(
        "\n".join(
            line for line in model.render(()).splitlines() if not line.startswith(("T ", "T_DAG "))
        )
    ).compile_m2d_converter(skip_reference_sample=True)
    probe_text = "".join(f"EXP_VAL {p}\n" for p in probes()) if name.startswith("bt27") else ""
    rng = random.Random(args.seed + 1)
    histories = [model.draw(rng) for _ in range(args.fixed_draws)]
    for history in dict.fromkeys([()] + histories):
        text = model.render(history)
        hir, info = optimize(text)
        row = {"history": history, "fault_count": len(history), **info}
        row["execution"] = "skipped_width_budget"
        if info["peak_width"] <= args.max_width and reference is not None:
            if probe_text:
                hir, probe_info = optimize(text + probe_text)
                if probe_info["peak_width"] > args.max_width:
                    raise AssertionError("Probes exceeded the execution budget")
            program = clifft.lower(
                hir,
                expected_detectors=reference["detectors"],
                expected_observables=reference["observables"],
            )
            c = clifft.sample(program, shots=args.shots, seed=8017, threads=1, batch_size=1)
            m = merlin.CircuitSampler(text + probe_text, seed=53177).sample(args.shots)
            row["validation"] = compare(c, m, converter, reference)
            row["validation"]["maximum_observable_score"] = compare_observables(c, m, converter)
            if name == "target_qec_clifford_control":
                samples = stim.Circuit(text).compile_sampler(seed=42899).sample(args.shots)
                oracle = SimpleNamespace(measurements=samples, exp_vals=np.empty((args.shots, 0)))
                row["stim_validation"] = compare(c, oracle, converter, reference)
                row["stim_validation"]["maximum_observable_score"] = compare_observables(
                    c, oracle, converter
                )
            row["execution"] = "validated_against_merlin"
        output["fixed"].append(row)
    output["fixed_summary"] = {
        "draws": len(histories),
        "unique_draws": len(set(histories)),
        "compiled_patterns_including_identity": len(output["fixed"]),
        "widths": dict(Counter(r["peak_width"] for r in output["fixed"])),
        "t_counts": dict(Counter(r["t_count"] for r in output["fixed"])),
        "executed_patterns": sum(
            r["execution"] == "validated_against_merlin" for r in output["fixed"]
        ),
        "median_optimizer_seconds": statistics.median(
            r["optimizer_seconds"] for r in output["fixed"]
        ),
    }
    if noisy_info["peak_width"] <= args.max_width and reference is not None:
        text = model.render(None) + probe_text
        program = clifft.compile(
            text,
            expected_detectors=reference["detectors"],
            expected_observables=reference["observables"],
        )
        if program.peak_active_width > args.max_width:
            raise AssertionError("Unfixed probes exceeded execution budget")
        c = clifft.sample(program, shots=args.shots, seed=31993, threads=1, batch_size=1)
        m = merlin.CircuitSampler(text, seed=91733).sample(args.shots)
        output["unfixed_validation"] = compare(c, m, converter, reference)
        output["unfixed_validation"]["maximum_observable_score"] = compare_observables(
            c, m, converter
        )
        if name == "target_qec_clifford_control":
            samples = stim.Circuit(text).compile_sampler(seed=37199).sample(args.shots)
            oracle = SimpleNamespace(measurements=samples, exp_vals=np.empty((args.shots, 0)))
            output["unfixed_stim_validation"] = compare(c, oracle, converter, reference)
            output["unfixed_stim_validation"]["maximum_observable_score"] = compare_observables(
                c, oracle, converter
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-draws", type=int, default=16)
    parser.add_argument("--history-draws", type=int, default=10000)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--max-width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    if (
        args.fixed_draws < 1
        or args.history_draws < 1
        or args.shots < 16
        or not 0 <= args.max_width <= 16
    ):
        parser.error(
            "positive draw counts, at least 16 shots, and width budget at most 16 required"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(args.merlin_checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MERLIN_REVISION:
        raise ValueError("Unexpected Merlin generator revision")
    sys.path.insert(0, str(args.merlin_checkout.resolve()))
    generator = importlib.import_module("benchmarks.protocols.code_switching")
    merlin = importlib.import_module("merlin")
    sources = {"bt27_one_layer": source(noise="DEPOLARIZE1(0.001)")}
    for selected in ("prep", "phase", "decoder", "all"):
        regions = {"prep", "phase", "decoder"} if selected == "all" else {selected}
        sources[f"bt27_{selected}_gate_noise"] = synthetic_bt_noise(source(), regions, 0.001)
    cultivation = (FIXTURE.parent / "cultivation_d5.stim").read_text()
    sources["cultivation_d5_p001"] = cultivation.replace("(0.005)", "(0.001)")
    sources["cultivation_d5_p005"] = cultivation
    sources["target_qec_clifford_control"] = (FIXTURE.parent / "target_qec.stim").read_text()
    large = generator.build_code_switching_case("bt81", p_phys=0, target_scoring=False).circuit
    sources["bt81_all_gate_noise"] = synthetic_bt_noise(large, {"prep", "phase", "decoder"}, 0.001)
    started = perf_counter()
    results: dict[str, Any] = {}
    document = {
        "merlin_generator_revision": revision,
        "shots_per_pattern_per_simulator": args.shots,
        "history_draws_per_model": args.history_draws,
        "fixed_draws_per_model": args.fixed_draws,
        "seed": args.seed,
        "limitations": (
            "Synthetic BT gate noise omits idle errors and scheduling; finite sampled "
            "histories and probes, not a logical-error-rate study. Wide cases are inspected "
            "only. Exact-history repetition is not a measure of equivalent-region reuse."
        ),
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, text in sources.items():
        print(f"Studying {name}", flush=True)
        results[name] = study_model(name, Model(text), args, merlin)
        print(json.dumps(results[name]["fixed_summary"]), flush=True)
        document["elapsed_seconds"] = perf_counter() - started
        args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
