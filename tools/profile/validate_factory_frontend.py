"""Apply automatic discovery to exported factories and independent D/E queries."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import stim
from automatic_specialization import History, analyze
from conditional_phase_frontend import ConditionalPhase
from study_automatic_specialization import sample_parities
from validate_automatic_specialization import visible_probability
from validate_shared_phase_specialization import normalized_lines, prefix_law

import clifft


def load_manifest(directory: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["initial_state"] != "all_zero":
        raise ValueError("Unsupported benchmark contract")
    for entry in manifest["cases"]:
        path = directory / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Benchmark input hash differs")
        parsed = clifft.parse(path.read_text())
        if (parsed.num_qubits, parsed.num_measurements) != (
            entry["physical_qubits"],
            entry["measurement_records"],
        ):
            raise ValueError("Benchmark physical interface differs")
    return manifest


def references(checkout: Path) -> tuple[Any, Any, dict[str, str]]:
    hashes = {}
    names = (
        "validate_steane_continuation.py",
        "validate_theory_ccz_faults.py",
        "study_quadcycle_factory.py",
        "validate_phase_continuation.py",
    )
    for name in names:
        relative = "tools/profile/" + name
        content = (checkout / relative).read_bytes()
        expected = subprocess.check_output(
            ["git", "-C", str(checkout), "show", "1bcbf960:" + relative]
        )
        if content != expected:
            raise ValueError("External reference differs from its pinned benchmark revision")
        hashes[name] = hashlib.sha256(content).hexdigest()
    # Only the independent reference modules come from the benchmark checkout.
    # Their imports of the candidate's generic helpers resolve in this worktree.
    sys.path.append(str(checkout / "tools/profile"))
    return (
        importlib.import_module("validate_steane_continuation"),
        importlib.import_module("validate_phase_continuation"),
        hashes,
    )


def compile_source(source: str) -> tuple[Any, int]:
    hir, info = analyze(source)
    if info["peak_width"] > 12:
        raise ValueError("Validation exceeds the dense execution width budget")
    return clifft.lower(hir), info["peak_width"]


def automatic(source: str) -> tuple[Any, int]:
    front = ConditionalPhase(source)
    branch = front.rewrite((), 7319)
    # These references describe unconditional records. Their initial encoded
    # preparations are unitary, so no outcome-conditioned comparison is allowed.
    if branch.carrier or any(step["prefix_records"] for step in branch.steps):
        raise ValueError("Reference queries require an unsampled visible prefix")
    return compile_source(branch.source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-limit", type=int, default=0)
    args = parser.parse_args()
    manifest = load_manifest(args.benchmark_dir)
    ref, channels, reference_hashes = references(args.reference_checkout)
    result: dict[str, Any] = {
        "manifest_sha256": hashlib.sha256(
            (args.benchmark_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "reference_revision": "1bcbf960",
        "reference_hashes": reference_hashes,
        "channel": channels.channel_control(),
        "three_qubit_channel": channels.three_qubit_control(),
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                Path(__file__).name,
                "conditional_phase_frontend.py",
                "automatic_specialization.py",
                "deferred_phase_specialization.py",
                "regional_phase_specialization.py",
                "shared_phase_specialization.py",
                "one_core_phase_specialization.py",
            )
        },
    }
    counts: dict[str, int] = {}
    max_relative = max_absolute = 0.0

    def check(program: Any, record: list[int], expected: Any, category: str) -> None:
        nonlocal max_relative, max_absolute
        value = float(expected)
        actual = visible_probability(program, record)
        delta = abs(actual - value)
        max_absolute = max(max_absolute, delta)
        if value:
            max_relative = max(max_relative, delta / value)
        if not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-300):
            raise AssertionError(f"{category}: {actual} differs from {value}")
        counts[category] = counts.get(category, 0) + 1

    for entry in manifest["cases"]:
        if "record_probability_queries" not in entry:
            continue
        program, _ = automatic((args.benchmark_dir / entry["file"]).read_text())
        for query in entry["record_probability_queries"]:
            check(program, query["record"], Fraction(query["probability"]), "manifest")
    print("All exported exact queries passed", flush=True)
    workload = ref.Workload(ref.load_css(args.merlin_checkout))
    data_probes = "\n".join(workload.checks + workload.z_checks) + "\n"
    extraction_rows = []
    for protocol in ("D", "E"):
        extraction = ref.Extraction(workload, protocol)
        singles = [
            ((i, outcome),)
            for i in range(81, len(extraction.model.sites))
            for outcome in range(1, len(extraction.model.sites[i].replacements))
        ]
        cases: list[tuple[str, History, History]] = [
            ("ideal", (), ()),
            ("matching_data_x", (), ((0, 1), (27, 1))),
            ("ancilla_z_before", (), ((81, 3),)),
            ("ancilla_z_after_first_coupling", (), ((extraction.n, 3),)),
            ("readout_flip", (), ((extraction.n + 81, 1),)),
        ]
        cases += [("single", (), h) for h in singles[: args.tail_limit]]
        region_rng, tail_rng, sample_rng = (
            random.Random(3605),
            random.Random(3606),
            random.Random(3610),
        )
        for count in (1, 2, 3):
            for _ in range(6):
                region = tuple(
                    (i, region_rng.randrange(1, len(workload.events[i])))
                    for i in sorted(region_rng.sample(range(135), count))
                )
                tail = tuple(
                    (i, tail_rng.randrange(1, len(extraction.model.sites[i].replacements)))
                    for i in sorted(tail_rng.sample(range(len(extraction.model.sites)), count))
                )
                cases.append(("mixed", region, tail))
        widths = set()
        witnesses = {}
        for index, (name, region, tail) in enumerate(cases):
            expected = extraction.reference(region, tail)
            original = workload.shared.model.render(region)
            raw_source = original + extraction.model.render(tail) + data_probes
            raw, width = automatic(raw_source)
            widths.add(width)
            coarse, width = automatic(original + extraction.coarse_source(tail) + data_probes)
            widths.add(width)
            for syndrome, weight in expected["law"].items():
                record = expected["anc_syndrome"] + ref.bits(syndrome, 24) + expected["data_z"]
                check(coarse, record, weight, "complete_coarse_law")
            samples = (
                clifft.sample(raw, shots=2, seed=7301, threads=1).measurements.astype(int).tolist()
            )
            samples += [extraction.draw_record(expected, sample_rng) for _ in range(2)]
            invalid = samples[0].copy()
            invalid[0] ^= 1
            for record in samples + [invalid]:
                check(raw, record, extraction.record_probability(record, expected), "raw_record")
            if name != "single":
                for hadamard in (False, True):
                    scored, width = automatic(
                        raw_source + extraction.score_source(expected["x"], hadamard)
                    )
                    widths.add(width)
                    samples = (
                        clifft.sample(scored, shots=2, seed=7302, threads=1)
                        .measurements.astype(int)
                        .tolist()
                    )
                    samples += [
                        extraction.draw_score(expected, sample_rng, hadamard) for _ in range(2)
                    ]
                    for record in samples:
                        check(
                            scored,
                            record,
                            extraction.score_probability(record, expected, hadamard),
                            "state_probe",
                        )
            if name not in {"single", "mixed"}:
                witnesses[name] = str(expected["acceptance"])
            if index % 10 == 0:
                print(protocol, index + 1, "/", len(cases), "histories passed", flush=True)
        assert witnesses == {
            "ideal": "1",
            "matching_data_x": "0" if protocol == "D" else "1",
            "ancilla_z_before": "0",
            "ancilla_z_after_first_coupling": "1" if protocol == "D" else "0",
            "readout_flip": "0",
        }
        extraction_rows.append(
            {
                "protocol": protocol,
                "histories": len(cases),
                "widths": sorted(widths),
                "witnesses": witnesses,
                "ideal_channel": extraction.certify_ideal(),
            }
        )
    result["extraction"] = extraction_rows

    # The factory preparation has many random records. Compare on matched
    # actual prefix branches, and separately certify their complete sign law.
    factory_rows = []
    source = (args.benchmark_dir / "quadcycle-noisy.stim").read_text()
    front = ConditionalPhase(source)
    assert front.first is not None
    selected_region = front.first.region
    shared = selected_region.shared
    for entry in manifest["cases"]:
        if entry.get("parent") != "quadcycle-noisy.stim":
            continue
        history = tuple(map(tuple, entry["history"]))
        prefix_history, tail_lines = selected_region.split_history(front.first.map_history(history))
        prefix_law(shared, prefix_history)
        lines = normalized_lines(shared, prefix_history)
        for seed in (17, 311):
            simulator = stim.TableauSimulator(seed=seed)
            simulator.set_num_qubits(front.model.num_qubits)
            simulator.do(stim.Circuit("\n".join(lines[: shared.first])))
            records = list(map(int, simulator.current_measurement_record()))
            signs = [simulator.peek_observable_expectation(p) for p in shared.probe_rows]
            if any(sign not in (-1, 1) for sign in signs):
                raise AssertionError("Unexpected conditional stabilizer span")
            actual_bits = sum(
                bit << i for i, bit in enumerate(records + [int(s == -1) for s in signs])
            )
            flips = 0
            for site, outcome in prefix_history:
                flips ^= shared.prefix_responses[site][outcome]
            candidate = front.rewrite(
                history, seed, choose_prefix=lambda _s, _i: actual_bits ^ flips
            )
            reference = (
                str(simulator.current_inverse_tableau().inverse().to_circuit())
                + "\n"
                + f"I {front.model.num_qubits - 1}\n"
                + "\n".join(f"MPAD {b}" for b in records)
                + "\n"
                + "\n".join(lines[shared.first :])
                + "\n"
                + "\n".join(tail_lines)
                + "\n"
            )
            programs = [compile_source(s) for s in (candidate.source, reference)]
            for program, _ in programs:
                sample = clifft.sample(program, shots=8, seed=9271, threads=1)
                sample_parities(source, sample)
                for record in sample.measurements.astype(int).tolist():
                    check(
                        programs[0][0],
                        record,
                        visible_probability(programs[1][0], record),
                        "factory_conditional_record",
                    )
            factory_rows.append(
                {"file": entry["file"], "seed": seed, "widths": [w for _, w in programs]}
            )
        print(entry["file"], "conditional reference passed", flush=True)
    result.update(
        factory=factory_rows,
        counts=counts,
        maximum_relative_error=max_relative,
        maximum_absolute_error=max_absolute,
        scope=(
            "Complete coarse D/E laws; selected raw/state queries; "
            "matched conditional factory records."
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
