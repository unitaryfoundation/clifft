"""Validate reuse of a phase-boundary prefix with exact physical fault corrections.

The native diagnostic constructs and plans every suffix offline. It does not
share executor state or add dynamic topology planning. Optional Merlin checks
compare the constructed programs with the original physical faulty circuits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import statistics
import subprocess
import tempfile
from collections import Counter
from importlib import import_module
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import stim
from analyze_bt27_phase_corrections import correction, region
from profile_bt27_fault_specialization import FIXTURE, source
from validate_bt27_fault_specialization import compare, probes


def read_samples(path: Path, shots: int, probe_count: int) -> Any:
    data = path.read_bytes()
    path.unlink()
    offset = shots * (126 + 72)
    if len(data) != offset + shots * probe_count * 8:
        raise AssertionError("Unexpected native sample dimensions")
    return SimpleNamespace(
        measurements=np.frombuffer(data, dtype="u1", count=shots * 126).reshape(shots, 126),
        detectors=np.frombuffer(data, dtype="u1", count=shots * 72, offset=shots * 126).reshape(
            shots, 72
        ),
        observables=np.empty((shots, 0), dtype="u1"),
        exp_vals=np.frombuffer(data, dtype="=f8", offset=offset).reshape(shots, probe_count),
    )


def plan_stats(path: Path, prefix_actions: int) -> dict[str, Any]:
    text = path.read_text()
    path.unlink()
    actions = re.findall(r"^  \d+ (.+)$", text, re.MULTILINE)
    prefix = "\n".join(actions[:prefix_actions])
    rotations = [
        a for a in actions[:prefix_actions] if "ROTATE_ACTIVE" in a or "PROMOTE_DORMANT" in a
    ]
    result: dict[str, Any] = {
        "semantic_plan_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "action_counts": dict(Counter(a.split()[2] for a in actions)),
    }
    if prefix_actions:
        result["prefix_action_sha256"] = hashlib.sha256(prefix.encode()).hexdigest()
        result["rotation_signs_constant"] = all(re.search(r"sign=[01]$", a) for a in rotations)
    return result


def summarize(rows: dict[str, Any]) -> dict[str, Any]:
    return {
        "patterns": len(rows),
        "raw_boundary_matches": sum(row["raw_boundary_matches"] for row in rows.values()),
        "prefix_matches": sum(row["prefix_matches"] for row in rows.values()),
        "shared_widths": dict(Counter(row["peak_width"] for row in rows.values())),
        "fresh_widths": dict(Counter(row["fresh_peak_width"] for row in rows.values())),
        "distinct_semantic_plans": len(
            {row["shared_plan"]["semantic_plan_sha256"] for row in rows.values()}
        ),
        "distinct_prefix_actions": len(
            {row["shared_plan"]["prefix_action_sha256"] for row in rows.values()}
        ),
        "median_seconds": {
            field: statistics.median(row[field] for row in rows.values())
            for field in (
                "correction_formula_seconds",
                "compose_seconds",
                "fresh_compile_seconds",
                "plan_seconds",
                "prepare_seconds",
                "sample_seconds",
                "fresh_sample_seconds",
            )
        },
    }


def write_output(path: Path, output: dict[str, Any]) -> None:
    metadata = {key: value for key, value in output.items() if key != "results"}
    # Keep individual cases readable without expanding repetitive telemetry
    # beyond the repository's research-artifact size budget.
    rows = ",\n".join(
        f"    {json.dumps(key)}: {json.dumps(value)}" for key, value in output["results"].items()
    )
    text = json.dumps(metadata, indent=2).removesuffix("\n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ',\n  "results": {\n' + rows + "\n  }\n}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--merlin", action="store_true")
    parser.add_argument(
        "--without-probes", action="store_true", help="Measure ordinary record sampling"
    )
    args = parser.parse_args()
    if not 16 <= args.shots <= 65536:
        parser.error("shots must be between 16 and 65536")
    study = json.loads(args.study.read_text())
    reference = study["reference"]
    if len(reference["detectors"]) != 72 or any(reference["detectors"]) or reference["observables"]:
        raise ValueError("The native diagnostic requires BT27's original zero reference")
    cases = {key: row for key, row in study["results"].items() if "faults" in row}
    if next(iter(cases)) != "identity":
        raise ValueError("The baseline pattern must appear first")
    merlin = import_module("merlin") if args.merlin else None
    original = source()
    prefix, body, suffix, polynomial = region(original)
    probe_list = [] if args.without_probes else probes()
    probe_text = "".join(f"EXP_VAL {p}\n" for p in probe_list)
    clifford = stim.Circuit(
        "\n".join(line for line in original.splitlines() if not line.startswith(("T ", "T_DAG ")))
    )
    converter = clifford.compile_m2d_converter(skip_reference_sample=True)
    started = perf_counter()
    results = {}
    formula_times = {}
    with tempfile.TemporaryDirectory(prefix="bt27-boundary-") as temp:
        directory = Path(temp)
        (directory / "prefix.stim").write_text("\n".join(prefix + body) + "\n")
        (directory / "full.stim").write_text(original + probe_text)
        (directory / "names.txt").write_text("\n".join(cases) + "\n")
        for i, (key, row) in enumerate(cases.items()):
            faults = tuple((q, p) for q, p in row["faults"])
            formula_start = perf_counter()
            gates, _, _ = correction(polynomial, faults)
            formula_times[key] = perf_counter() - formula_start
            (directory / f"{i}.correction").write_text("\n".join(gates + ["I 134"]) + "\n")
            (directory / f"{i}.moved").write_text(
                "\n".join(prefix + body + gates + suffix) + "\n" + probe_text
            )
            (directory / f"{i}.original").write_text(source(faults) + probe_text)
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
                    row["correction_formula_seconds"] = formula_times[key]
                    if key != list(cases)[i]:
                        raise AssertionError("Native output order changed")
                    shared = read_samples(
                        directory / f"{i}.shared.bin", args.shots, len(probe_list)
                    )
                    fresh = read_samples(directory / f"{i}.fresh.bin", args.shots, len(probe_list))
                    row["fresh_validation"] = compare(shared, fresh, converter, reference)
                    row["shared_plan"] = plan_stats(directory / f"{i}.plan", row["prefix_actions"])
                    row["fresh_plan"] = plan_stats(directory / f"{i}.fresh.plan", 0)
                    if merlin is not None:
                        physical = (directory / f"{i}.original").read_text()
                        independent = merlin.CircuitSampler(physical, seed=715591).sample(
                            args.shots
                        )
                        row["merlin_validation"] = compare(
                            shared, independent, converter, reference
                        )
                    results[key] = row
                    if len(results) % 16 == 0:
                        print(f"Validated {len(results)} reused-prefix patterns", flush=True)
                if process.wait() != 0:
                    raise RuntimeError("Native diagnostic failed")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    if results.keys() != cases.keys():
        raise AssertionError("Incomplete native sweep")
    output = {
        "study_sha256": hashlib.sha256(args.study.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "shots_per_pattern_per_sampler": args.shots,
        "pauli_probes": len(probe_list),
        "merlin_version": importlib.metadata.version("merlin-sim") if merlin is not None else None,
        "setup": setup,
        "summary": summarize(results),
        "limitations": (
            "Offline per-pattern composition and planning; finite sampled probes, not tomography. "
            "Timings are exploratory, include source I/O, and may overlap validation work. "
            "No shared executor state or runtime conditional-Clifford implementation."
        ),
        "elapsed_seconds": perf_counter() - started,
        "results": results,
    }
    write_output(args.output, output)
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
