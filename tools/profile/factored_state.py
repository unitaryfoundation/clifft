"""Conservative product-component study of Clifft's existing semantic plans.

This is an offline cost model, not a simulator or a prediction of wall time.
It never splits a component except when a measured coordinate is removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path


def analyze(plan: dict) -> dict:
    width = plan["initial_width"]
    # An arbitrary supplied initial residual need not factor.
    components = [set(range(width))] if width else []
    peak_storage = max(1, sum(1 << len(c) for c in components))
    peak_component = width
    dense_visits = factor_visits = merge_writes = merges = 0
    trace = []

    for i, action in enumerate(plan["actions"]):
        if action["before"] != width:
            raise ValueError("discontinuous active widths")
        kind = action["kind"]
        support = action.get("support", 0)
        if support < 0 or support >> width:
            raise ValueError("support outside active coordinates")
        touched = [c for c in components if any(support & (1 << q) for q in c)]
        union = set().union(*touched)
        dense_visits += action["dense_passes"] * (1 << max(width, action["after"]))

        if kind == "promote":
            if action["after"] != width + 1:
                raise ValueError("invalid promotion")
            components.append({width})
            factor_visits += 2
        elif kind in {"rotate", "measure"}:
            if not touched:
                raise ValueError("active operation has empty support")
            if len(touched) > 1:
                components = [c for c in components if c not in touched] + [union]
                merge_writes += 1 << len(union)
                merges += 1
            factor_visits += action["dense_passes"] * (1 << len(union))
            # Include the merged input before measurement removes its pivot.
            peak_storage = max(peak_storage, sum(1 << len(c) for c in components))
            peak_component = max(peak_component, len(union))
            if kind == "measure":
                pivot = action["pivot"]
                if not support & (1 << pivot) or action["after"] != width - 1:
                    raise ValueError("invalid measurement pivot or width")
                components = [{q - (q > pivot) for q in c if q != pivot} for c in components]
                components = [c for c in components if c]
            elif action["after"] != width:
                raise ValueError("rotation changes width")
        elif kind == "expectation":
            # A Pauli expectation on a product is a product of local expectations.
            factor_visits += sum(1 << len(c) for c in touched)
            if action["after"] != width:
                raise ValueError("expectation changes width")
        elif kind == "classical":
            if action["after"] != width or action["dense_passes"]:
                raise ValueError("unrecognized state-changing action")
        elif kind == "unsupported_instrument":
            return {"eligible": False, "reason": "instrument or continuation boundary"}
        else:
            raise ValueError(f"unknown action: {kind}")

        width = action["after"]
        sizes = sorted((len(c) for c in components), reverse=True)
        storage = max(1, sum(1 << n for n in sizes))
        peak_storage = max(peak_storage, storage)
        peak_component = max(peak_component, max(sizes, default=0))
        if action["dense_passes"]:
            trace.append(
                {
                    "action": i,
                    "width": width,
                    "components": sizes,
                    "coordinates": [sorted(c) for c in components],
                }
            )

    return {
        "eligible": True,
        "peak_component_width": peak_component,
        "dense_peak_coefficients": 1 << plan["peak_width"],
        "factor_peak_live_coefficients": peak_storage,
        "dense_unfused_visits": dense_visits,
        "factor_unfused_visits": factor_visits,
        "merge_output_coefficients": merge_writes,
        "merges": merges,
        "visit_ratio_including_merge_outputs": (
            dense_visits / (factor_visits + merge_writes) if factor_visits + merge_writes else None
        ),
        "trace": trace,
    }


def study(binary: Path, path: Path, shots: int, repeats: int, postselect: bool) -> dict:
    samples = []
    for _ in range(repeats):
        result = subprocess.run(
            [str(binary), str(path), str(shots), "22", str(int(postselect))],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        samples.append(json.loads(result.stdout))
    plan = samples[0]
    structure = analyze(plan)
    raw_timings = [{k: v for k, v in s.items() if k != "actions"} for s in samples]
    timings = {
        key: statistics.median(s[key] for s in samples)
        for key in ("compile_ms", "parse_ms", "trace_ms", "plan_ms", "prepare_ms")
    }
    timings["shot_ms"] = (
        statistics.median(s["shot_ms"] for s in samples) if plan["shot_ms"] is not None else None
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "peak_width": plan["peak_width"],
        "num_qubits": plan["num_qubits"],
        "semantic_actions": len(plan["actions"]),
        "structure": structure,
        "median": timings,
        "runs": raw_timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--circuit", type=Path, action="append", default=[])
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--postselect", action="store_true")
    parser.add_argument("--exclude-one-round", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 1 or args.repeats < 1:
        parser.error("shots and repeats must be positive")
    workloads = []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        for workload in manifest["workloads"]:
            if args.exclude_one_round and workload.get("parameters", {}).get("rounds") == 1:
                continue
            artifact = workload["artifact"]
            path = (args.manifest.parent / artifact["path"]).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact["sha256"]:
                raise ValueError(f"artifact hash mismatch: {path}")
            workloads.append((workload["id"], path))
    workloads.extend((path.stem, path.resolve()) for path in args.circuit)
    if not workloads:
        parser.error("supply --manifest and/or --circuit")
    results = []
    for name, path in workloads:
        print(name, flush=True)
        results.append(
            {
                "id": name,
                **study(args.binary.resolve(), path, args.shots, args.repeats, args.postselect),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "schema": 1,
                "machine": platform.uname()._asdict(),
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "manifest_sha256": (
                    hashlib.sha256(args.manifest.read_bytes()).hexdigest()
                    if args.manifest
                    else None
                ),
                "postselect": args.postselect,
                "exclude_one_round": args.exclude_one_round,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
