"""Assess the frozen combined frontend across complete protocols and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import stim
from automatic_specialization import FaultModel
from compiled_prefix_reuse import compile_stages
from continuation_trace_reuse import ContinuationPhase, ContinuationWorker
from study_automatic_specialization import sample_parities, stress_histories
from study_chained_phase_specialization import branch_dependent_entry, measured_chain
from study_factory_frontend import sources
from study_factory_frontend import worker as baseline
from study_shared_phase_specialization import features

import clifft


def panel(args: Any) -> dict[str, dict[str, str]]:
    available = sources(args)
    groups = {
        "factory": [
            "quadcycle-ideal.stim",
            "quadcycle-noisy.stim",
            "steane-D.stim",
            "steane-E.stim",
        ],
        "distillation": ["15to1_direct_x", "15to1_scored", "bh_direct_x", "bh_scored"],
        "cultivation": ["cultivation_d3", "cultivation_d5"],
        "code_switching": ["bt27_scored", "bt27_direct_x", "bt81_scored", "bt81_direct_x"],
        "control": [
            "measured_coset",
            "measured_feedback",
            "noncommuting_control",
            "unsupported_rotation",
        ],
    }
    result = {
        name: {"family": family, "source": available[name], "origin": "existing_input"}
        for family, names in groups.items()
        for name in names
    }
    for family in ("distillation", "cultivation", "code_switching"):
        for name in groups[family]:
            result[name + "_ideal"] = {
                "family": family,
                "source": FaultModel(available[name]).render(()),
                "origin": "no_fault_materialization_of_" + name,
            }
    controls = {
        "chain_three_noisy": measured_chain(3, noisy=True),
        "branch_dependent": branch_dependent_entry(),
        "two_surviving_rotations": "H 0 1\nT 0 1\nH 0\nT 0\nMX 0 1\n",
        "clifford_only": "H 0\nM 0\n",
        "unsupported_later_rotation": "H 0\nT 0\nT_DAG 0\nH 0\nR_PAULI(.137) X0\nH 0\nM 0\n",
    }
    result.update(
        {
            n: {"family": "control", "source": s, "origin": "constructed_control"}
            for n, s in controls.items()
        }
    )
    return result


def native_sample(row: dict[str, Any]) -> Any:
    return SimpleNamespace(
        **{
            key: np.array([[int(c) for c in row[key]]], dtype=np.uint8)
            for key in ("measurements", "detectors", "observables")
        }
    )


def branch_info(branch: Any) -> dict[str, Any]:
    return {
        "route": branch.stop_reason,
        "regions": len(branch.steps),
        "raw_region_t": [s["residual_t"] for s in branch.steps],
        "carrier_stop": branch.carrier["stop_reason"] if branch.carrier else None,
        "rejection": branch.rejection,
    }


def combined(source: str, args: Any) -> dict[str, Any]:
    start = perf_counter()
    front = ContinuationPhase(source, args.exporter)
    frontend_ready = perf_counter()
    fast = front.continuation_info["eligible"]
    result: dict[str, Any] = {
        "first_region": front.first.metadata() if front.first else {"rejection": front.rejection},
        "preparation_reuse": front.reuse_info,
        "continuation_reuse": front.continuation_info,
        "phase_required": front.phase_required,
        "frontend_setup_seconds": frontend_ready - start,
    }
    with ExitStack() as stack:
        native = None
        if fast:
            native = stack.enter_context(
                ContinuationWorker(
                    args.native_worker,
                    front.optimized_prefix,
                    max_width=args.max_width,
                    phase=front.phase_required,
                    continuation=front.template_source,
                )
            )
            result["native_setup"] = native.setup
        result["setup_seconds"] = perf_counter() - start
        faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x7815)
        histories, times, records = [], [], []
        stages: dict[str, float] = defaultdict(float)
        widths: Counter[int] = Counter()
        counts: Counter[int] = Counter()
        routes: Counter[str] = Counter()
        regions: Counter[int] = Counter()
        residuals: Counter[str] = Counter()
        stops: Counter[str] = Counter()
        schedules: Counter[str] = Counter()
        native_peak = 0

        def attempt(history: Any, prefix_seed: int, seed: int, *, check: bool = False) -> Any:
            nonlocal native_peak
            begun = perf_counter()
            if native:
                payload = front.payload(history, prefix_seed)
                prepared = perf_counter()
                branch = front.rewrite(history, prefix_seed) if check else None
                reference = branch.source[len(front.optimized_prefix) :] if branch else None
                row = native.instantiate(
                    payload, seed, mode="coordinate-identity", reference=reference
                )
                ended = perf_counter()
                info = (
                    branch_info(branch)
                    if branch
                    else {
                        "route": "residual_magic",
                        "regions": 1,
                        "raw_region_t": [front.step["residual_t"]],
                        "carrier_stop": None,
                        "rejection": None,
                    }
                )
                info.update(
                    width=row["width"], final_t=row["t_count"], squeeze=row["squeeze_status"]
                )
                native_peak = max(native_peak, row["native_peak_kib"] * 1024)
                if check:
                    assert row["checked"] and row["optimized_checked"]
                    if row["width"] <= args.max_width:
                        assert row["plan_checked"]
                    info["checks"] = {
                        k: row[k] for k in ("checked", "optimized_checked", "plan_checked")
                    }
                timing = {"specialize": prepared - begun, "native_roundtrip": ended - prepared}
                timing.update({"native_" + k: v for k, v in row["stage_seconds"].items()})
                sample = native_sample(row) if row["width"] <= args.max_width else None
            else:
                branch = front.rewrite(history, prefix_seed)
                prepared = perf_counter()
                hir, compiled = compile_stages(branch.source, phase=front.phase_required)
                optimized = perf_counter()
                info = branch_info(branch) | {
                    "width": compiled["width"],
                    "final_t": hir.num_t_gates,
                }
                sample = None
                lowered = ended = optimized
                if compiled["width"] <= args.max_width:
                    program = clifft.lower(hir)
                    lowered = perf_counter()
                    sample = clifft.sample(program, shots=1, seed=seed, threads=1)
                    ended = perf_counter()
                timing = {
                    "specialize": prepared - begun,
                    "compile": optimized - prepared,
                    "lower": lowered - optimized,
                    "sample": ended - lowered,
                }
            if sample is not None:
                sample_parities(source, sample)
            return info, sample, timing, ended - begun

        for _ in range(args.shots):
            begun = perf_counter()
            history = front.model.draw(faults)
            prefix_seed, seed = quantum.getrandbits(64), quantum.getrandbits(64)
            drawn = perf_counter() - begun
            info, sample, timing, elapsed = attempt(history, prefix_seed, seed)
            times.append(drawn + elapsed)
            histories.append(history)
            stages["draw_and_seeds"] += drawn
            for key, value in timing.items():
                stages[key] += value
            widths[info["width"]] += 1
            counts[info["final_t"]] += 1
            routes[info["route"]] += 1
            regions[info["regions"]] += 1
            residuals[json.dumps(info["raw_region_t"])] += 1
            if info["carrier_stop"]:
                stops[info["carrier_stop"]] += 1
            if "squeeze" in info:
                schedules[info["squeeze"]] += 1
            if sample is not None:
                records.append(sample.measurements[0].copy())
        result.update(
            status="sampled" if len(records) == args.shots else "partial_width_budget",
            shots=args.shots,
            completed_shots=len(records),
            width_failures=args.shots - len(records),
            widths=dict(widths),
            final_t_counts=dict(counts),
            routes=dict(routes),
            regions=dict(regions),
            raw_region_t_counts=dict(residuals),
            carrier_stops=dict(stops),
            squeeze_status=dict(schedules),
            seconds_per_attempt=statistics.mean(times),
            maximum_seconds_per_attempt=max(times),
            median_seconds_per_attempt=statistics.median(times),
            stage_seconds_per_attempt={k: v / args.shots for k, v in stages.items()},
            history_sha256=hashlib.sha256(json.dumps(histories).encode()).hexdigest(),
            fault_weights=dict(Counter(map(len, histories))),
        )
        if len(records) == args.shots:
            arrays = np.stack(records)
            assert arrays.shape == (args.shots, front.model.num_records)
            observed = features(source, arrays)
            result.update(
                seconds_per_shot=result["seconds_per_attempt"],
                feature_means=observed.mean(axis=0).tolist(),
                feature_variances=observed.var(axis=0, ddof=1).tolist(),
                record_sha256=hashlib.sha256(arrays.astype("u1").tobytes()).hexdigest(),
            )
        # Audit fixed histories separately; they are never substituted for timed draws.
        selected = stress_histories(front.model)
        selected.append(
            tuple((i, len(s.replacements) - 1) for i, s in enumerate(front.model.sites))
        )
        selected = list(dict.fromkeys(selected))
        audits = []
        for i, history in enumerate(selected):
            info, _, _, _ = attempt(history, args.seed + i, 715, check=True)
            audits.append({"history": history, **info})
        result["stress"] = audits
        result["native_process_peak_bytes"] = native_peak
        if front.model.num_qubits <= 5:
            from validate_compiled_prefix_reuse import check as independent_check

            result["independent_instruments"] = [independent_check(front, h) for h in selected]
    return result


def run_child(command: list[str], timeout: float) -> dict[str, Any]:
    # Kill the entire isolated process group on timeout, including native helpers.
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return {"status": "timeout", "timeout_seconds": timeout}
    if process.returncode:
        return {"status": "error", "returncode": process.returncode, "stderr": stderr[-12000:]}
    return dict(json.loads(stdout))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark-dir", "merlin-checkout", "native-worker", "exporter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=271053)
    parser.add_argument("--max-width", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--worker", choices=("combined", "ordinary", "merlin"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.shots < 8 or not 0 <= args.max_width <= 16 or args.timeout <= 0:
        parser.error(
            "Require at least eight shots, a width budget at most sixteen and positive timeout"
        )
    if args.worker:
        source = args.source.read_text()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        row = combined(source, args) if args.worker == "combined" else baseline(source, args)
        row.update(
            process_peak_before_bytes=before,
            process_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        print(json.dumps(row), flush=True)
        return
    root = Path(__file__).resolve().parents[2]
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *Path(__file__).parent.glob("*.cpp"),
        *Path(__file__).parent.glob("*.h"),
    ]
    result: dict[str, Any] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "stim": stim.__version__,
        },
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "worker_sha256": hashlib.sha256(args.native_worker.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(args.exporter.read_bytes()).hexdigest(),
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="clifft-capability-") as directory:
        for name, entry in panel(args).items():
            if args.cases and name not in args.cases:
                continue
            source = entry["source"]
            model = FaultModel(source)
            path = Path(directory) / "input.stim"
            path.write_text(source)
            item: dict[str, Any] = {
                "family": entry["family"],
                "origin": entry["origin"],
                "source": source,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "physical_qubits": model.num_qubits,
                "records": model.num_records,
                "noise_sites": len(model.sites),
                "expected_faults": sum(1 - s.probabilities[0] for s in model.sites),
                "backends": {},
            }
            result["cases"][name] = item
            for backend in ("ordinary", "combined", "merlin"):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--source",
                    str(path),
                    "--worker",
                    backend,
                    "--shots",
                    str(args.shots),
                    "--seed",
                    str(args.seed),
                    "--max-width",
                    str(args.max_width),
                ]
                for key in (
                    "benchmark_dir",
                    "merlin_checkout",
                    "native_worker",
                    "exporter",
                    "output",
                ):
                    command += ["--" + key.replace("_", "-"), str(getattr(args, key).resolve())]
                row = run_child(command, args.timeout)
                item["backends"][backend] = row
                print(name, backend, row["status"], row.get("seconds_per_shot"), flush=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            candidate = item["backends"]["combined"]
            for backend in ("ordinary", "merlin"):
                row = item["backends"][backend]
                if row["status"] != "sampled" or candidate["status"] != "sampled":
                    continue
                a, b = np.array(candidate["feature_means"]), np.array(row["feature_means"])
                variance = (
                    np.array(candidate["feature_variances"]) + np.array(row["feature_variances"])
                ) / args.shots
                score = float(
                    np.max(
                        np.maximum(0, abs(a - b) - 4 / args.shots)
                        / np.maximum(np.sqrt(variance), 1e-12)
                    )
                )
                row["moment_screen"] = {"maximum_score": score, "threshold": 7, "passed": score < 7}
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
