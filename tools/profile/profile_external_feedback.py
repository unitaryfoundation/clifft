"""Reproduce bounded phase-limit experiments on pinned external generators.

The checked-in BT27 fixture does not require Merlin. Supply the pinned checkout
for BT81 and original target-scoring cases, and --ibm-input for a gauging control.
Large unreduced circuits are inspected without lowering or dense execution.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import clifft

MERLIN_REVISION = "097380fac1a3968ca47925146e211fe990f4c396"
FIXTURE = Path(__file__).resolve().parents[2] / "tests/fixtures/merlin_bt27.stim"


def bt27_source() -> str:
    return (
        "\n".join(line for line in FIXTURE.read_text().splitlines() if not line.startswith("#"))
        + "\n"
    )


def pre_ccz_noise(source: str, gate: str) -> str:
    lines = source.splitlines()
    first_t = next(i for i, line in enumerate(lines) if line.startswith("T "))
    lines.insert(first_t, f"{gate} " + " ".join(map(str, range(81))))
    return "\n".join(lines) + "\n"


def cases(merlin: Path | None, ibm: Path | None) -> dict[str, str]:
    from profile_symbolic_stabilizers import ccz_factory

    source = bt27_source()
    result = {
        "bt27_ideal": source,
        "bt27_depolarizing": pre_ccz_noise(source, "DEPOLARIZE1(0.001)"),
        "bt27_modified_z_only": pre_ccz_noise(source, "Z_ERROR(0.001)"),
    }
    if merlin is not None:
        revision = subprocess.check_output(
            ["git", "-C", str(merlin), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != MERLIN_REVISION:
            raise ValueError(f"Merlin must be at {MERLIN_REVISION}, got {revision}")
        if subprocess.check_output(
            ["git", "-C", str(merlin), "diff", "HEAD", "--", "benchmarks"], text=True
        ):
            raise ValueError("Merlin benchmark sources have uncommitted changes")
        sys.path.insert(0, str(merlin.resolve()))
        generator = importlib.import_module("benchmarks.protocols.code_switching")
        build = generator.build_code_switching_case
        assert build("bt27", p_phys=0, target_scoring=False).circuit == source
        assert (
            build("bt27", p_phys=0.001, target_scoring=False).circuit == result["bt27_depolarizing"]
        )
        result["bt27_target_scoring"] = build("bt27", p_phys=0, target_scoring=True).circuit
        for p, name in ((0, "ideal"), (0.001, "depolarizing")):
            result[f"bt81_{name}"] = build("bt81", p_phys=p, target_scoring=False).circuit
    if ibm is not None:
        result["ibm_gauging_d3"] = ibm.read_text()
    for name in ("reed_muller_measured", "cultivation_d5", "coherent_d5_r5"):
        result[name] = (FIXTURE.parent / f"{name}.stim").read_text()
    result["slicing"] = (
        "H 0 1\nMPP Z0*Z1\nCX rec[-1] 1\nR_Z(0.125) 0\nR_Z(-0.125) 1\nMPP X0*X1 Z0*Z1\n"
    )
    result["measurement_barriers"] = "H 0\n" + "T 0\nMX 0\n" * 1024
    result["ccz_skeleton"] = ccz_factory()
    return result


def manager(limit: int | None) -> tuple[clifft.HirPassManager, list[Any]]:
    phase = (
        clifft.PhasePolynomialPass()
        if limit is None
        else clifft.PhasePolynomialPass(max_variables=limit)
    )
    passes = [
        clifft.PeepholeFusionPass(),
        phase,
        clifft.RotationSimplificationPass(),
        clifft.StatevectorSqueezePass(),
    ]
    result = clifft.HirPassManager()
    for pass_ in passes:
        result.add(pass_)
    return result, passes


def profile(source: str, limit: int | None, repeats: int, shots: int) -> dict[str, Any]:
    times = []
    for trial in range(repeats + 1):
        hir = clifft.trace(clifft.parse(source))
        _, passes = manager(limit)
        row = []
        for pass_ in passes:
            single = clifft.HirPassManager()
            single.add(pass_)
            start = perf_counter()
            single.run(hir)
            row.append(perf_counter() - start)
        if trial:
            times.append(row)
    width = clifft.active_width_trace(hir)["peak"]
    phase = passes[1]
    result: dict[str, Any] = {
        "t_count": hir.num_t_gates,
        "peak_active_width": width,
        "pass_seconds": times,
        "phase": {
            name: getattr(phase, name)
            for name in (
                "blocks_reduced",
                "blocks_examined",
                "blocks_capped",
                "expansion_attempts",
                "blocks_expanded",
            )
            if hasattr(phase, name)
        },
        "compile_seconds": [],
        "sample_seconds": [],
    }
    # The cap bounds workspace even when comparing with an older implementation.
    if width <= 16:
        for trial in range(repeats + 1):
            pm, _ = manager(limit)
            start = perf_counter()
            program = clifft.compile(source, hir_passes=pm)
            elapsed = perf_counter() - start
            if trial:
                result["compile_seconds"].append(elapsed)
            start = perf_counter()
            sampled = clifft.sample(program, shots=shots, seed=53, threads=1)
            if trial:
                result["sample_seconds"].append(perf_counter() - start)
        result["sanity_sample"] = {
            "shots": shots,
            "detector_events": int(sampled.detectors.sum()),
            "observable_events": int(sampled.observables.sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path)
    parser.add_argument("--ibm-input", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.shots < 1:
        parser.error("repeats and shots must be positive")
    results: dict[str, dict[str, Any]] = {}
    for name, source in cases(args.merlin_checkout, args.ibm_input).items():
        results[name] = {}
        for limit in (None, 32, 64):
            print(name, limit, flush=True)
            results[name][str(limit)] = profile(source, limit, args.repeats, args.shots)
    args.output.write_text(
        json.dumps(
            {
                "pass_columns": ["peephole", "phase", "rotation", "squeeze"],
                "shots": args.shots,
                "profiles": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
