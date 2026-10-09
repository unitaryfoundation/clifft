"""Diagnose realistic region boundaries and measure a dependency-based extension."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import stim
from automatic_specialization import (
    ANNOTATIONS,
    MEASUREMENTS,
    FaultModel,
    History,
    analyze,
    instruction_text,
    record_count,
)
from deferred_phase_specialization import DeferredPhase
from regional_phase_specialization import RegionalPhase
from shared_phase_specialization import SharedPhase, bits
from study_automatic_specialization import candidates, sample_parities, stress_histories
from study_regional_phase_specialization import raw_info
from study_shared_phase_specialization import features

import clifft


def all_gate_noise(source: str, probability: float) -> str:
    """A diagnostic with heterogeneous Pauli noise, including the scoring tail."""
    ideal = FaultModel(source).render(())
    lines = []
    records = 0
    for node in clifft.parse(ideal).nodes:
        gate = node.gate.name
        line = instruction_text(node, records)
        qs = [t.value for t in node.targets if not t.is_rec]
        if gate in MEASUREMENTS:
            name, targets = line.split(" ", 1)
            line = f"{name}({probability:.17g}) {targets}"
        lines.append(line)
        if gate not in ANNOTATIONS | MEASUREMENTS:
            if len(qs) == 2:
                lines.append(f"DEPOLARIZE2({probability:.17g}) {qs[0]} {qs[1]}")
            elif len(qs) == 1:
                args = ",".join(f"{probability * weight:.17g}" for weight in (0.2, 0.3, 0.5))
                lines.append(f"PAULI_CHANNEL_1({args}) {qs[0]}")
            else:
                raise ValueError("Unexpected gate arity in all-gate diagnostic")
        records += record_count(node)
    return "\n".join(lines) + "\n"


def panel(checkout: Path) -> dict[str, str]:
    known = candidates(checkout)
    result = {
        name: known[name][1]
        for name in (
            "bt27_scored",
            "bt27_direct_x",
            "15to1_scored",
            "cultivation_d3",
            "cultivation_d5",
            "bt81_scored",
        )
    }
    for probability in (0.0001, 0.001, 0.01):
        result[f"bt27_scored_all_gates_{probability}"] = all_gate_noise(
            result["bt27_scored"], probability
        )
    result["bt27_direct_x_all_gates_0.001"] = all_gate_noise(result["bt27_direct_x"], 0.001)
    return result


def phase_summary(shared: SharedPhase) -> dict[str, Any]:
    nonclifford = {
        term: value
        for term, value in shared.nominal.items()
        if (term.bit_count() == 1 and value % 2)
        or (term.bit_count() == 2 and value % 4)
        or term.bit_count() >= 3
    }
    return {
        "terms": [[bits(term), value] for term, value in sorted(nonclifford.items())],
        "participating_support_coordinates": len(
            set(q for term in nonclifford for q in bits(term))
        ),
        "metadata": shared.metadata(),
    }


def logical_witnesses(simulator: stim.TableauSimulator, reference: int) -> list[str | None]:
    """Find reference X/Z correlations supported only on retained physical wires."""
    rows = simulator.canonical_stabilizers()
    basis: dict[int, tuple[int, int]] = {}
    for i, row in enumerate(rows):
        mask = sum(
            int(row[q] in (1, 2)) << (2 * (q - reference))
            | int(row[q] in (2, 3)) << (2 * (q - reference) + 1)
            for q in range(reference, len(row))
        )
        choice = 1 << i
        while mask:
            pivot = mask.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = mask, choice
                break
            other, combination = basis[pivot]
            mask ^= other
            choice ^= combination
    witnesses: list[str | None] = []
    for target in (1, 2):
        remaining, choice = target, 0
        while remaining:
            pivot = remaining.bit_length() - 1
            if pivot not in basis:
                break
            other, combination = basis[pivot]
            remaining ^= other
            choice ^= combination
        if remaining:
            witnesses.append(None)
        else:
            pauli = stim.PauliString(len(rows))
            for i in bits(choice):
                pauli *= rows[i]
            if simulator.peek_observable_expectation(pauli) != 1:
                raise AssertionError("Logical witness is not a stabilizer")
            if any(pauli[q] for q in range(reference + 1, len(pauli))):
                raise AssertionError("Logical witness depends on discarded reset information")
            witnesses.append(str(pauli))
    return witnesses


def bridge_probe(region: RegionalPhase, history: History, seed: int) -> dict[str, Any]:
    """Probe whether a one-T encoding preserves its logical qubit through a bridge.

    Replace its known |+> input at T with half of a Bell pair. Stabilizer
    evolution then detects loss of that logical qubit without dense simulation.
    Resets swap into fresh environment wires, so no hidden outcome is selected.
    This diagnoses the instrument, not the outcome law for the magic input.
    """
    prefix, tail = region.split_history(history)
    ideal = int.from_bytes(
        region.shared.prefix.compile_sampler(seed=seed).sample(1, bit_packed=True)[0].tobytes(),
        "little",
    )
    state = region.shared.render_state(region.shared.controls(prefix, ideal))
    lines = state.splitlines()
    locations = [i for i, line in enumerate(lines) if line.startswith(("T ", "T_DAG "))]
    if len(locations) != 1 or region.shared.prefix_records:
        raise ValueError("Bell probe requires one T and an unmeasured preparation")
    at = locations[0]
    q = int(lines[at].split()[1])
    reference = region.model.num_qubits
    simulator = stim.TableauSimulator(seed=seed)
    simulator.set_num_qubits(reference + 1)
    simulator.do(stim.Circuit("\n".join(lines[:at])))
    if simulator.peek_x(q) != 1:
        raise AssertionError("The injected logical input is not an independent |+> state")
    simulator.cx(q, reference)
    simulator.do(stim.Circuit("\n".join(lines[at + 1 :])))

    def reference_bloch() -> list[int]:
        return [
            simulator.peek_x(reference),
            simulator.peek_y(reference),
            simulator.peek_z(reference),
        ]

    entry = reference_bloch()
    entry_witnesses = logical_witnesses(simulator, reference)
    environments = 0
    counts: Counter[str] = Counter()
    branch_events: Counter[str] = Counter()
    first_loss = None
    next_gate = "END"
    for line in tail:
        if not line:
            continue
        gate = line.split()[0].split("(")[0]
        if gate in {"T", "T_DAG"}:
            next_gate = gate
            break
        if gate in {"M", "MX", "MY"}:
            q = int(line.split()[1].lstrip("!"))
            expectation = {
                "M": simulator.peek_z,
                "MX": simulator.peek_x,
                "MY": simulator.peek_y,
            }[gate](q)
            branch_events[("random_" if expectation == 0 else "determinate_") + gate] += 1
        elif gate == "MPP":
            raise ValueError("Extend the branch diagnostic before probing product measurements")
        if gate in {"R", "RX", "RY"}:
            q = int(line.split()[1])
            simulator.swap(q, reference + 1 + environments)
            environments += 1
            if gate != "R":
                simulator.h(q)
            if gate == "RY":
                simulator.s(q)
        else:
            simulator.do(stim.Circuit(line))
        if gate not in ANNOTATIONS:
            counts[gate] += 1
        if first_loss is None and gate in {"R", "RX", "RY", "M", "MX", "MY"}:
            if any(witness is None for witness in logical_witnesses(simulator, reference)):
                first_loss = {"gate": gate, "instruction": line}
    return {
        "history": history,
        "seed": seed,
        "entry_reference_bloch": entry,
        "next_region_reference_bloch": reference_bloch(),
        "entry_logical_witnesses": entry_witnesses,
        "next_region_logical_witnesses": logical_witnesses(simulator, reference),
        "purified_resets": environments,
        "first_logical_loss": first_loss,
        "bridge_gate_counts": dict(counts),
        "bridge_measurement_predictability": dict(branch_events),
        "next_gate": next_gate,
        "sampled_visible_records": list(map(int, simulator.current_measurement_record())),
        "scope": (
            "Conditional Bell-probe instrument with resets purified; "
            "not probabilities for the original non-Clifford input"
        ),
    }


def study(source: str, args: Any) -> dict[str, Any]:
    _, ordinary = analyze(source)
    before = RegionalPhase(source)
    started = perf_counter()
    deferred = DeferredPhase(source)
    setup = perf_counter() - started
    stress = []
    for history in stress_histories(deferred.model):
        ideal = before.shared.controls(()) >> 1
        stress.append(
            {
                "history": history,
                "before": raw_info(before.compose(history, ideal))[1],
                "after": raw_info(deferred.compose(history, ideal))[1],
            }
        )
    result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "ordinary": ordinary,
        "before": {**before.metadata(), "phase": phase_summary(before.shared)},
        "after": {**deferred.metadata(), "phase": phase_summary(deferred.region.shared)},
        "setup_seconds": setup,
        "stress": stress,
    }
    if before.boundary_gate in {"R", "RX", "RY"}:
        result["bridge_probes"] = [
            bridge_probe(before, history, seed)
            for history in stress_histories(before.model)
            for seed in (41, 317)
        ]
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x31817)
    started = perf_counter()
    sampler = deferred.region.shared.prefix.compile_sampler(seed=quantum.getrandbits(64))
    result["prefix_sampler_setup_seconds"] = perf_counter() - started
    result["setup_seconds"] += result["prefix_sampler_setup_seconds"]
    stages = dict.fromkeys(("draw", "prefix_and_compose", "trace_and_lower", "sample"), 0.0)
    times, histories, records = [], [], []
    widths: Counter[int] = Counter()
    for _ in range(args.shots):
        start = perf_counter()
        history = deferred.model.draw(faults)
        drawn = perf_counter()
        ideal = int.from_bytes(sampler.sample(1, bit_packed=True)[0].tobytes(), "little")
        composed = deferred.compose(history, ideal)
        rendered = perf_counter()
        hir, info = raw_info(composed)
        if info["peak_width"] > args.max_width:
            result["fresh"] = {"status": "width_budget", "history": history, **info}
            return result
        program = clifft.lower(hir)
        lowered = perf_counter()
        sample = clifft.sample(program, shots=1, seed=quantum.getrandbits(64), threads=1)
        end = perf_counter()
        for stage, duration in zip(
            stages, (drawn - start, rendered - drawn, lowered - rendered, end - lowered)
        ):
            stages[stage] += duration
        times.append(end - start)
        histories.append(history)
        widths[info["peak_width"]] += 1
        sample_parities(source, sample)
        records.append(sample.measurements[0].copy())
    result["fresh"] = {
        "status": "sampled",
        "shots": args.shots,
        "mean_seconds_per_shot": statistics.mean(times),
        "maximum_seconds_per_shot": max(times),
        "stage_seconds": stages,
        "widths": dict(widths),
        "fault_counts": dict(sorted(Counter(map(len, histories)).items())),
        "unique_histories": len(set(histories)),
        "history_sha256": hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
        "record_sha256": hashlib.sha256(np.stack(records).tobytes()).hexdigest(),
    }
    merlin = importlib.import_module("merlin")
    try:
        start = perf_counter()
        reference = merlin.CircuitSampler(source, seed=args.seed + 51)
        initialized = perf_counter()
        sample = reference.sample(args.shots)
        elapsed = perf_counter() - initialized
        a, b = features(source, np.stack(records)), features(source, sample.measurements)
        variance = a.var(axis=0, ddof=1) / args.shots + b.var(axis=0, ddof=1) / args.shots
        delta = np.maximum(0, abs(a.mean(axis=0) - b.mean(axis=0)) - 4 / args.shots)
        score = float(max(delta / np.maximum(np.sqrt(variance), 1e-12)))
        if score > 7:
            raise AssertionError("Fresh output moment check differs from Merlin")
        result["merlin"] = {
            "status": "sampled",
            "setup_seconds": initialized - start,
            "seconds_per_shot": elapsed / args.shots,
            "moment_features": a.shape[1],
            "maximum_score": score,
            "scope": "Small fresh-noise bug check, not rare-event or distributional equivalence",
        }
    except (ValueError, RuntimeError, merlin.MeasurementError) as error:
        result["merlin"] = {"status": "unsupported", "reason": str(error)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=713815)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.shots < 32 or not 0 <= args.max_width <= 16:
        parser.error("Require at least 32 shots and a width budget between zero and sixteen")
    sources = panel(args.merlin_checkout)
    selected = args.cases or list(sources)
    if set(selected) - sources.keys():
        parser.error("Unknown case")
    result: dict[str, Any] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "deferred_phase_specialization.py",
                "regional_phase_specialization.py",
                "shared_phase_specialization.py",
                "automatic_specialization.py",
                "study_automatic_specialization.py",
                "study_regional_phase_specialization.py",
                "study_shared_phase_specialization.py",
            )
        },
        "cases": {},
    }
    for name in selected:
        print(name, "starting", flush=True)
        result["cases"][name] = study(sources[name], args)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(name, result["cases"][name]["fresh"], flush=True)


if __name__ == "__main__":
    main()
